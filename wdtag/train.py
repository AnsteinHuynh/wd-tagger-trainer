"""Training loop: DINOv3-B + OFTv2 + SinkSGD_ADV, BF16, resumable.

Checkpoints land in runs/<sha8(config)>/ every 200 steps.  A `.PAUSE` file
(also accepted as `.PAUSE.txt` / `.PUASE`) at the run-directory root or at the
workspace root makes the trainer checkpoint and exit cleanly at the next step;
re-running the same command resumes from the newest checkpoint and *continues
the data stream* where it stopped rather than replaying the epoch from the
beginning.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler

from .cache import ensure_cache
from .config import CHECKPOINT_EVERY, RUNS_ROOT, TrainConfig
from .data import CacheDataset, MultiHotCollator, label_priors, load_targets
from .labels import load_label_space
from .metrics import compute_metrics
from .model import DinoTagger

WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PAUSE_NAMES = (".PAUSE", ".PAUSE.txt", ".PUASE", ".PUASE.txt", "PAUSE")

# Checkpoint selection: CLI-only, like --compile-mode/--val-every-steps, so it
# never changes sha8.  `micro` is the default because macro-F1 on this label
# space is dominated by thousands of rare/unseen tags and demonstrably picks
# worse checkpoints (run fa51da74: macro selected step 2496 / micro 0.3800,
# while micro actually peaked at 0.3938 on step 5304).
SELECT_METRICS = {
    "micro": "micro_f1",
    "macro": "macro_f1",
    "supported_macro": "supported_macro_f1",
}
HEX8 = re.compile(r"^[0-9a-f]{8}$")


def best_value(best: dict, metric: str) -> float:
    """Best val score recorded for the chosen metric (-1.0 if unknown).

    Checkpoints written before --select-metric existed only carry
    `val_macro_f1`; that legacy key is still honoured for macro selection and
    is kept in every new `best` dict for compatibility.
    """
    key = f"val_{SELECT_METRICS[metric]}"
    if key in best:
        return float(best[key])
    if metric == "macro":
        return float(best.get("val_macro_f1", -1.0))
    return -1.0


def paused(run_dir: str) -> bool:
    for d in (run_dir, WS_ROOT):
        for n in PAUSE_NAMES:
            if os.path.exists(os.path.join(d, n)):
                return True
    return False


class ListSampler(Sampler):
    def __init__(self, idx: list[int]):
        self.idx = idx

    def __iter__(self):
        return iter(self.idx)

    def __len__(self) -> int:
        return len(self.idx)


def epoch_perm(n: int, seed: int, epoch: int) -> list[int]:
    g = torch.Generator().manual_seed(seed * 1000003 + epoch)
    return torch.randperm(n, generator=g).tolist()


def lr_at(step: int, total: int, base: float, warmup: int) -> float:
    if warmup > 0 and step < warmup:
        return base * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(1.0, max(0.0, progress))
    return base * (0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress)))


@torch.no_grad()
def evaluate(forward, model: DinoTagger, ds: CacheDataset, targets: np.ndarray,
             batch_size: int, device: str) -> tuple[dict, np.ndarray]:
    model.eval()
    n = len(ds)
    probs = np.empty((n, model.n_labels), dtype=np.float32)
    for start in range(0, n, batch_size):
        idx = list(range(start, min(start + batch_size, n)))
        take = len(idx)
        idx += [idx[-1]] * (batch_size - take)   # keep one static shape for compile
        x = torch.stack([ds[i][0] for i in idx]).to(device, non_blocking=True)
        logits = forward(x)
        probs[start:start + take] = torch.sigmoid(logits.float()[:take]).cpu().numpy()
    model.train()
    metrics = compute_metrics(probs >= 0.5, targets)
    return metrics, probs


def dump_debug_pictures(run_dir: str, ds: CacheDataset, indices: list[int],
                        manifest: list[dict], mode: str) -> None:
    """Save exactly what the trainer fed the network, for traceability."""
    from PIL import Image

    d = os.path.join(run_dir, "debugpics")
    os.makedirs(d, exist_ok=True)
    for i in indices:
        k, j = ds.index[i]
        shard = ds.shards[k]
        img = shard["images"][j].numpy().transpose(1, 2, 0)  # CHW u8 -> HWC
        key = shard["keys"][j]
        tar = os.path.splitext(shard["tars"][j])[0]
        name = f"{key}-{tar}-{mode}.png"
        Image.fromarray(img).save(os.path.join(d, name))
        manifest.append({"file": name, "key": key, "tar": shard["tars"][j],
                         "dataset_id": int(shard["ids"][j])})
    with open(os.path.join(d, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def save_ckpt(path: str, model: DinoTagger, opt, step: int, epoch: int,
              step_in_epoch: int, best: dict, cfg: TrainConfig) -> None:
    torch.save(
        {
            "adapter": model.adapter_state_dict(),
            "optimizer": opt.state_dict(),
            "step": step,
            "epoch": epoch,
            "step_in_epoch": step_in_epoch,
            "best": best,
            "config": cfg.to_dict(),
            "rng": {
                # keep every RNG state on the CPU: cuda states are ByteTensors and
                # map_location='cuda' would otherwise break set_state.
                "torch": torch.get_rng_state().cpu(),
                "cuda": [s.cpu() for s in torch.cuda.get_rng_state_all()],
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
        },
        path,
    )


def load_config(a: argparse.Namespace, ap: argparse.ArgumentParser) -> TrainConfig:
    """Resolve --run/--config into a TrainConfig.

    --run wins over --config.  The run's config.json carries `_sha8`; it must
    resolve back to the same directory, otherwise a resume would silently start
    a new run next to it.  An unpinned 8-hex run is pinned to its dirname here.
    """
    if a.run:
        run_path = os.path.normpath(a.run)
        cfg_path = os.path.join(run_path, "config.json")
        if not os.path.isfile(cfg_path):
            ap.error(f"--run {a.run}: no config.json found in that directory")
        if a.config:
            print(f"--run given; ignoring --config {a.config}", flush=True)
        cfg = TrainConfig.from_json(cfg_path)
        if not getattr(cfg, "_sha8_pin", None):
            base = os.path.basename(run_path)
            if not HEX8.match(base):
                ap.error(f"--run {a.run}: config.json has no _sha8 and the directory "
                         f"name is not an 8-hex run id")
            cfg._sha8_pin = base
            print(f"--run: config.json had no _sha8; pinned it to {base}", flush=True)
        resolved = os.path.join(RUNS_ROOT, cfg.sha8())
        if (os.path.normcase(os.path.abspath(resolved))
                != os.path.normcase(os.path.abspath(run_path))):
            ap.error(f"--run {a.run}: its config.json pins sha8 {cfg.sha8()}, which resolves "
                     f"to {resolved}; refusing to resume a different run directory")
        return cfg
    if a.config:
        return TrainConfig.from_json(a.config)
    ap.error("give --config FILE or --run DIR")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="",
                    help="TrainConfig JSON (required unless --run is given)")
    ap.add_argument("--run", default="",
                    help="run directory to resume, e.g. --run runs/fe92f4df; loads that "
                         "directory's config.json, whose _sha8 pin resolves back to it. "
                         "Takes precedence over --config.")
    ap.add_argument("--fresh", action="store_true", help="ignore existing checkpoints")
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--val-every-steps", type=int, default=0,
                    help="validate every N steps instead of once per epoch (0 = per epoch). "
                         "Deliberately a CLI flag, not a config field, so it does not change sha8.")
    ap.add_argument("--compile-mode", default="",
                    help="torch.compile mode override, e.g. 'max-autotune' (~1.4x faster than "
                         "'default' on a 4090, measured). CLI flag so sha8 is unaffected.")
    ap.add_argument("--select-metric", choices=tuple(SELECT_METRICS), default="micro",
                    help="validation metric that selects best_val.pt (default: micro). "
                         "CLI flag, not a config field, so sha8 is unaffected.")
    ap.add_argument("--init-from", default="",
                    help="warm-start adapter+head from another run dir (staged training)")
    a = ap.parse_args()

    cfg = load_config(a, ap)
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = "cuda"

    space = load_label_space()
    n_labels = len(space)
    run_dir = cfg.run_dir()
    if not getattr(cfg, "_sha8_pin", None):
        # Freeze this brand-new run's identity in its config.json.  Without the
        # pin, any later added/removed dataclass field would silently re-key it
        # and a resume would start a new directory instead of continuing.
        cfg._sha8_pin = cfg.sha8()
    log_path = os.path.join(run_dir, "train_log.jsonl")
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, indent=2)
    print(f"run dir: {run_dir}\nconfig sha8: {cfg.sha8()}", flush=True)
    print(f"checkpoint selection metric: {SELECT_METRICS[a.select_metric]} "
          f"(--select-metric {a.select_metric})", flush=True)

    ensure_cache("train", mode=cfg.mode)
    ensure_cache("valid", mode=cfg.mode)
    ensure_cache("test", mode=cfg.mode)

    ds_train = CacheDataset("train", cfg.mode, n_labels)
    ds_val = CacheDataset("valid", cfg.mode, n_labels)
    val_targets = load_targets("valid", cfg.mode, n_labels)

    model = DinoTagger(cfg, n_labels).to(device)
    priors = label_priors("train", cfg.mode, n_labels)
    model.init_head_bias(priors)
    counts = model.count_params()
    print(f"params: OFT {counts['oft']/1e6:.2f}M, head {counts['head']/1e6:.2f}M, "
          f"frozen backbone {counts['backbone']/1e6:.1f}M, {model.n_oft} OFT modules", flush=True)

    # torch.compile is worth ~1.6x here; mode="max-autotune" was measured at a
    # further 1.4x over "default" on a 4090 (Inductor autotunes the GEMM/bmm
    # kernels, which dominate this graph). Overridable via --compile-mode.
    forward = model
    _cmode = a.compile_mode or "default"
    if cfg.compile_model:
        try:
            t0 = time.time()
            forward = torch.compile(model, mode=_cmode)
            print(f"torch.compile(mode={_cmode}) ready in {time.time()-t0:.1f}s", flush=True)
        except Exception as e:  # never let a compile problem kill a run
            print(f"torch.compile unavailable ({type(e).__name__}: {e}); using eager", flush=True)
            forward = model

    oft_params, head_params = model.trainable_parameters()
    from adv_optm import SinkSGD_adv

    if cfg.compiled_optimizer:
        # The per-tensor update is ~2 ms of Python/dispatch per adapter tensor
        # (73 of them => ~88 ms/step, flat in element count). Compiling it costs
        # ~6 s once and brings the step to ~28 ms. fullgraph=True hard-fails on
        # Dynamo's default recompile limit, so raise it before touching anything.
        torch._dynamo.config.cache_size_limit = 1_000_000
        torch._dynamo.config.recompile_limit = 1_000_000
        torch._dynamo.config.accumulated_recompile_limit = 1_000_000
        torch._dynamo.config.force_parameter_static_shapes = False

    if cfg.freeze_oft:
        # stage 1: adapters stay at identity (Cayley(0) = I, so the rotation is a
        # numerical no-op) and only the head learns. Dropping them from the
        # optimizer also removes 73 per-tensor updates from every step.
        for p in oft_params:
            p.requires_grad_(False)
        groups = [{"params": head_params, "lr": cfg.lr_head}]
        print("freeze_oft: adapters held at identity, training the head only", flush=True)
    else:
        groups = [
            {"params": oft_params, "lr": cfg.lr_oft},
            {"params": head_params, "lr": cfg.lr_head},
        ]

    opt = SinkSGD_adv(
        groups,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
        sinkhorn_iterations=cfg.sinkhorn_iterations,
        orthogonal_sinkhorn=cfg.orthogonal_sinkhorn,
        stochastic_rounding=True,
        compiled_optimizer=cfg.compiled_optimizer,
    )

    steps_per_epoch = len(ds_train) // cfg.batch_size
    total_steps = steps_per_epoch * cfg.epochs

    # ---------------------------------------------------------------- resume
    start_epoch, start_step_in_epoch, global_step = 0, 0, 0
    best = {"val_macro_f1": -1.0, "step": -1}
    if not a.fresh:
        ckpts = sorted(
            (p for p in os.listdir(run_dir) if p.startswith("ckpt_step") and p.endswith(".pt")),
            key=lambda p: int(p.split("step")[1].split(".")[0]),
        )
        if ckpts:
            path = os.path.join(run_dir, ckpts[-1])
            ck = torch.load(path, map_location=device, weights_only=False)
            model.load_adapter_state_dict(ck["adapter"])
            opt.load_state_dict(ck["optimizer"])
            global_step = ck["step"]
            start_epoch = ck["epoch"]
            start_step_in_epoch = ck["step_in_epoch"]
            best = ck.get("best", best)
            rng = ck.get("rng") or {}
            if rng:
                torch.set_rng_state(rng["torch"].cpu().to(torch.uint8))
                torch.cuda.set_rng_state_all([s.cpu().to(torch.uint8) for s in rng["cuda"]])
                np.random.set_state(rng["numpy"])
                random.setstate(rng["python"])
            print(f"resumed {path}: step {global_step}, epoch {start_epoch}, "
                  f"batch {start_step_in_epoch} (continuing the data stream, not replaying)",
                  flush=True)
        else:
            print("no checkpoint found, starting fresh", flush=True)

    # staged training: warm-start the adapter/head from another run's checkpoint
    # (e.g. stage 1 trains the head with the adapters frozen at identity, stage 2
    # starts from it with a gentle lr_oft). Optimizer state and schedule are NOT
    # inherited -- stage 2 is a new run with its own warmup and cosine.
    if a.init_from:
        src = a.init_from
        cand = os.path.join(src, "best_val.pt")
        if not os.path.exists(cand):
            srcs = sorted(p for p in os.listdir(src)
                          if p.startswith("ckpt_step") and p.endswith(".pt"))
            if not srcs:
                raise SystemExit(f"--init-from {src} has no best_val.pt or ckpt_step*.pt")
            cand = os.path.join(src, srcs[-1])
        ick = torch.load(cand, map_location=device, weights_only=False)
        model.load_adapter_state_dict(ick["adapter"])
        print(f"warm-started adapter+head from {cand} (step {ick['step']}, "
              f"source val {ick.get('metrics', {}).get('micro_f1', float('nan')):.4f}). "
              f"optimizer state and run history are fresh.", flush=True)

    coll = MultiHotCollator(n_labels)
    t_start = time.time()
    t_step = time.time()
    debug_manifest: list[dict] = []
    _mfile = os.path.join(run_dir, "debugpics", "manifest.json")
    if os.path.exists(_mfile):
        with open(_mfile, encoding="utf-8") as f:
            debug_manifest = json.load(f)
    debug_done = len(debug_manifest)

    def run_val(step: int, ep: int) -> dict:
        """Validate, log, and keep best_val.pt. Callable mid-epoch or at epoch end."""
        nonlocal best
        t0 = time.time()
        metrics, _ = evaluate(forward, model, ds_val, val_targets, cfg.batch_size, device)
        rec = {"step": step, "epoch": ep, **metrics, "val_seconds": round(time.time() - t0, 1)}
        sel_key = SELECT_METRICS[a.select_metric]
        print(f"VAL step {step} (ep {ep}): macro_f1 {metrics['macro_f1']:.4f} "
              f"macro_mcc {metrics['macro_mcc']:.4f} micro_f1 {metrics['micro_f1']:.4f} "
              f"micro_mcc {metrics['micro_mcc']:.4f} "
              f"(supported_macro_f1 {metrics.get('supported_macro_f1', float('nan')):.4f}) "
              f"| best by {sel_key}", flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        raw = metrics.get(sel_key)
        score = float(raw) if raw is not None else float("nan")
        if math.isfinite(score) and score > best_value(best, a.select_metric):
            best = {
                # legacy key kept for old checkpoints/readers: macro selection
                # still reads it from checkpoints written before this change
                "val_macro_f1": metrics["macro_f1"],
                "val_micro_f1": metrics["micro_f1"],
                "val_supported_macro_f1": metrics.get("supported_macro_f1", -1.0),
                "select_metric": a.select_metric,
                "step": step,
                "epoch": ep,
            }
            torch.save({"adapter": model.adapter_state_dict(), "config": cfg.to_dict(),
                        "metrics": metrics, "step": step},
                       os.path.join(run_dir, "best_val.pt"))
            print(f"new best val {sel_key} {score:.4f} "
                  f"(selection metric: {a.select_metric}) -> best_val.pt", flush=True)
        return metrics

    for epoch in range(start_epoch, cfg.epochs):
        perm = epoch_perm(len(ds_train), cfg.seed, epoch)
        skip_batches = start_step_in_epoch if epoch == start_epoch else 0
        idx = perm[skip_batches * cfg.batch_size:]
        batches = [[idx[i + j] for j in range(cfg.batch_size)]
                   for i in range(0, len(idx) - cfg.batch_size + 1, cfg.batch_size)]
        loader = DataLoader(
            ds_train,
            batch_sampler=batches,
            num_workers=cfg.num_workers,
            pin_memory=True,
            persistent_workers=False,
            collate_fn=coll,
        )
        model.train()
        for bi, (x, y) in enumerate(loader):
            step_in_epoch = skip_batches + bi
            if debug_done < 100:
                need = min(100 - debug_done, len(batches[bi]))
                dump_debug_pictures(run_dir, ds_train, batches[bi][:need],
                                    debug_manifest, cfg.mode)
                debug_done += need
                print(f"debugpics: {debug_done}/100 written to {run_dir}\\debugpics", flush=True)
            if paused(run_dir):
                p = os.path.join(run_dir, f"ckpt_step{global_step:06d}.pt")
                # this batch has NOT been consumed yet, so resume at it
                save_ckpt(p, model, opt, global_step, epoch, step_in_epoch, best, cfg)
                print(f"PAUSE file detected -> checkpointed {p} and exiting cleanly", flush=True)
                return

            bases = (cfg.lr_head,) if cfg.freeze_oft else (cfg.lr_oft, cfg.lr_head)
            for gi, base in enumerate(bases):
                opt.param_groups[gi]["lr"] = lr_at(global_step, total_steps, base, cfg.warmup_steps)

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = forward(x)
            loss = F.binary_cross_entropy_with_logits(logits.float(), y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for g in opt.param_groups for p in g["params"]], cfg.grad_clip)
            opt.step()
            global_step += 1

            if global_step % cfg.log_every == 0:
                dt = time.time() - t_step
                t_step = time.time()
                ips = cfg.log_every * cfg.batch_size / max(dt, 1e-9)
                _lrs = "/".join(f"{g['lr']:.2e}" for g in opt.param_groups)
                print(f"step {global_step}/{total_steps} ep {epoch} loss {loss.item():.4f} "
                      f"lr {_lrs} "
                      f"{ips:.0f} img/s vram {torch.cuda.max_memory_allocated()/1e9:.1f}GB",
                      flush=True)

            if global_step % CHECKPOINT_EVERY == 0:
                p = os.path.join(run_dir, f"ckpt_step{global_step:06d}.pt")
                # batch `step_in_epoch` is done -> the next one to run is +1
                save_ckpt(p, model, opt, global_step, epoch, step_in_epoch + 1, best, cfg)
                print(f"checkpoint {p}", flush=True)

            if a.max_steps and global_step >= a.max_steps:
                p = os.path.join(run_dir, f"ckpt_step{global_step:06d}.pt")
                save_ckpt(p, model, opt, global_step, epoch, step_in_epoch + 1, best, cfg)
                print("max-steps reached, stopped", flush=True)
                return

            if a.val_every_steps and global_step % a.val_every_steps == 0:
                run_val(global_step, epoch)

        # ------------------------------------------------------------- valid
        del loader  # reap worker processes before measuring memory-driven work
        if not a.val_every_steps:
            run_val(global_step, epoch)
        p = os.path.join(run_dir, f"ckpt_step{global_step:06d}.pt")
        save_ckpt(p, model, opt, global_step, epoch + 1, 0, best, cfg)

    print(f"training done in {(time.time()-t_start)/60:.1f} min, best {best}", flush=True)
    # final evaluation on the test cache, as required
    from .evaluate import eval_run
    eval_run(run_dir, "test")


if __name__ == "__main__":
    main()


