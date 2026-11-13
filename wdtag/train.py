"""Training loop: DINOv3 + OFTv2 + SinkSGD_ADV, BF16, resumable. `.PAUSE`/`.PUASE` at the run
dir or workspace root exits at the next step; re-running resumes the stream.
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

from .cache import CACHE_ROOT, ensure_cache
from .config import CHECKPOINT_EVERY, RUNS_ROOT, TrainConfig
from .data import CacheDataset, MultiHotCollator, label_priors, load_targets
from .labels import load_label_space
from .metrics import compute_metrics
from .model import DinoTagger

WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PAUSE_NAMES = (".PAUSE", ".PAUSE.txt", ".PUASE", ".PUASE.txt", "PAUSE")

# Step-targeted pauses: `.PAUSE-6300` stops after step 6300 has finished and checkpointed,
# `.PAUSE-VALID` stops after the next validation. Both delete themselves when they fire, first
# to fire wins, and a target already behind the current step pauses at the next boundary.
PAUSE_STEP_RE = re.compile(r"^\.?P(?:AUSE|UASE)-(\d+)$")
PAUSE_VALID_RE = re.compile(r"^\.?P(?:AUSE|UASE)-VALID$")

# `micro` default: macro-F1 is dominated by rare tags (fa51da74: macro picked step 2496 where
# micro was 0.3800; micro actually peaked 0.3938 at step 5304).
SELECT_METRICS = {
    "micro": "micro_f1",
    "macro": "macro_f1",
    "supported_macro": "supported_macro_f1",
    # The scoring protocol's metric: macro F1 at the fixed 0.40 threshold, not at the best
    # threshold found inside the val subset. On the 1k subset those disagree enough to pick
    # a different checkpoint, which is why the main run selects on this one.
    "macro040": "macro_f1_at_0.40",
}
HEX8 = re.compile(r"^[0-9a-f]{8}$")


def best_value(best: dict, metric: str) -> float:
    """Best val score for the chosen metric, -1.0 if unknown; legacy `val_macro_f1` kept for
    macro."""
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


def pause_schedule(run_dir: str) -> tuple[list[tuple[str, int]], list[str]]:
    """The step-targeted pause files, re-read every step so one written mid-run is seen.

    Returns the numbered targets as (path, step) sorted by step, and the paths of any
    `.PAUSE-VALID` markers."""
    steps: list[tuple[str, int]] = []
    valids: list[str] = []
    for d in (run_dir, WS_ROOT):
        try:
            names = os.listdir(d)
        except OSError:
            continue
        for n in names:
            m = PAUSE_STEP_RE.match(n)
            if m:
                steps.append((os.path.join(d, n), int(m.group(1))))
            elif PAUSE_VALID_RE.match(n):
                valids.append(os.path.join(d, n))
    steps.sort(key=lambda t: t[1])
    return steps, valids


def announce_pause_schedule(run_dir: str, a, start_step: int) -> None:
    """Say at startup what the pause files mean, including the ones that cannot fire usefully."""
    steps, valids = pause_schedule(run_dir)
    for path, n in steps:
        where = ("already passed, so it pauses at the next step boundary" if n <= start_step
                 else f"after step {n}")
        print(f"[pause] {os.path.basename(path)} -> {where}", flush=True)
    if valids:
        if not a.val_every_steps:
            print("[pause] .PAUSE-VALID seen but no validation is scheduled, so it falls back "
                  "to pausing at the next step boundary", flush=True)
        else:
            print(f"[pause] .PAUSE-VALID -> after the next validation "
                  f"(every {a.val_every_steps} steps)", flush=True)
    if steps and valids:
        print("[pause] both a step target and .PAUSE-VALID are present; first to fire wins",
              flush=True)


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


def lr_at(step: int, total: int, base: float, warmup: int,
          schedule: str = "cosine", decay_steps: int = 0,
          start_step: int = 0) -> float:
    """Cosine over the whole schedule by default.

    `start_step` re-anchors the cosine so a run can be extended by whole epochs: phase is
    measured from the anchor rather than from zero, so raising `total` lengthens the tail
    instead of rewinding the curve and re-heating the rate. Set it to the step the current
    cycle began on, and leave it 0 for a run that is not restarting.

    "const" holds the peak after warmup and only anneals over the final `decay_steps`, which is
    what makes a run extendable: raising `total` slides the decay window later instead of
    stretching a cosine that has already spent its learning rate. Use it whenever the plan is
    "keep training while it keeps climbing", then stop with a decay tail short enough that its
    length does not depend on how long the run turned out to be.
    """
    if warmup > 0 and step < warmup and start_step == 0:
        return base * (step + 1) / warmup
    if schedule == "const":
        tail = max(0, int(decay_steps))
        if tail <= 0:
            return base
        start = max(warmup, total - tail)
        if step < start:
            return base
        progress = min(1.0, (step - start) / max(1, tail))
        return base * (0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress)))
    lo = max(warmup, int(start_step))
    progress = (step - lo) / max(1, total - lo)
    progress = min(1.0, max(0.0, progress))
    return base * (0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress)))


@torch.no_grad()
def evaluate(forward, model: DinoTagger, ds: CacheDataset, targets: np.ndarray,
             batch_size: int, device: str, feats: np.ndarray | None = None
             ) -> tuple[dict, np.ndarray]:
    model.eval()
    n = len(ds)
    probs = np.empty((n, model.n_labels), dtype=np.float32)
    for start in range(0, n, batch_size):
        idx = list(range(start, min(start + batch_size, n)))
        take = len(idx)
        idx += [idx[-1]] * (batch_size - take)   # keep one static shape for compile
        x = torch.stack([ds[i][0] for i in idx]).to(device, non_blocking=True)
        if feats is not None:
            fb = np.ascontiguousarray(feats[start:start + take])
            if fb.shape[0] < batch_size:
                # pad to the same static shape as x (repeat the last row; the padded
                # logits are sliced away below, same as the padded images)
                fb = np.concatenate(
                    [fb, np.repeat(fb[-1:], batch_size - fb.shape[0], axis=0)])
            f = torch.from_numpy(fb).to(device)
            logits = forward(x, f)
        else:
            logits = forward(x)
        probs[start:start + take] = torch.sigmoid(logits.float()[:take]).cpu().numpy()
    model.train()
    metrics = compute_metrics(probs >= 0.5, targets)
    return metrics, probs


class NestedValSet:
    """The nested validation cache: a .npy memmap whose 1k/5k/10k/50k subsets are prefixes of
    one array, so only a batch's pages are touched."""

    def __init__(self, root: str, n_images: int, n_labels: int):
        self.images = np.load(os.path.join(root, "images.npy"), mmap_mode="r")
        lab = np.load(os.path.join(root, "labels.npz"))
        self.indices = lab["indices"]
        self.offsets = lab["offsets"]
        self.n_labels = int(n_labels)
        self.n = min(int(n_images), int(self.images.shape[0]))

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        img = torch.from_numpy(np.ascontiguousarray(self.images[i]))
        y = np.zeros(self.n_labels, dtype=np.float32)
        y[self.indices[self.offsets[i]:self.offsets[i + 1]]] = 1.0
        return img, torch.from_numpy(y)

    def dense_targets(self) -> np.ndarray:
        out = np.zeros((self.n, self.n_labels), dtype=np.float32)
        for i in range(self.n):
            lo, hi = int(self.offsets[i]), int(self.offsets[i + 1])
            out[i, self.indices[lo:hi]] = 1.0
        return out


VAL_GRID = (0.25, 0.30, 0.40, 0.50)


def threshold_summary(probs: np.ndarray, targets: np.ndarray, chunk: int = 4096) -> dict:
    """Best global threshold by macro F1: our optimum sits nearer 0.30, where 0.50
    understates macro ~30%. Row chunks per (threshold, label); whole 50k x 12k matrices
    cost 16.8 s."""
    n_labels = probs.shape[1]
    T = len(VAL_GRID)
    tp = np.zeros((T, n_labels), np.int64)
    fp = np.zeros((T, n_labels), np.int64)
    fn = np.zeros((T, n_labels), np.int64)
    for s in range(0, probs.shape[0], chunk):
        p = probs[s:s + chunk]
        t = targets[s:s + chunk].astype(bool)
        for ti, thr in enumerate(VAL_GRID):
            pred = p >= thr
            tp[ti] += (pred & t).sum(0)
            fp[ti] += (pred & ~t).sum(0)
            fn[ti] += (~pred & t).sum(0)

    best = {"macro_f1_best": -1.0, "best_threshold": 0.5,
            "micro_f1_at_best": -1.0, "macro_f1_at_0.40": -1.0}
    for ti, thr in enumerate(VAL_GRID):
        f1 = 2 * tp[ti] / np.maximum(2 * tp[ti] + fp[ti] + fn[ti], 1e-12)
        macro = float(f1.mean())
        TP, FP, FN = float(tp[ti].sum()), float(fp[ti].sum()), float(fn[ti].sum())
        micro = 2 * TP / max(2 * TP + FP + FN, 1e-12)
        if abs(thr - 0.40) < 1e-9:
            best["macro_f1_at_0.40"] = macro
        if macro > best["macro_f1_best"]:
            best.update(macro_f1_best=macro, best_threshold=float(thr),
                        micro_f1_at_best=micro)
    return best


def dump_debug_pictures(run_dir: str, ds: CacheDataset, indices: list[int],
                        manifest: list[dict], mode: str) -> None:
    """Exactly what the trainer fed the network."""
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


def dump_stream_debug(run_dir: str, x, src: list[tuple[str, str]],
                      manifest: list[dict]) -> None:
    """Same contract as `dump_debug_pictures`, from the tars."""
    from PIL import Image

    d = os.path.join(run_dir, "debugpics")
    os.makedirs(d, exist_ok=True)
    for i, (key, tar) in enumerate(src):
        img = x[i].numpy().transpose(1, 2, 0)  # CHW u8 -> HWC
        name = f"{key}-{os.path.splitext(tar)[0]}-squash.png"
        Image.fromarray(img).save(os.path.join(d, name))
        manifest.append({"file": name, "key": key, "tar": tar, "dataset_id": None})
    with open(os.path.join(d, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def save_ckpt(path: str, model: DinoTagger, opt, step: int, epoch: int,
              step_in_epoch: int, best: dict, cfg: TrainConfig) -> None:
    # Write beside the target and rename: a 250 MB torch.save interrupted by a reaped job, a
    # power cut or a full disk leaves a truncated file, and because resume takes the highest
    # step number it would then take the corrupt one and refuse to start at all.
    tmp = path + ".tmp"
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
                # RNG state on CPU: cuda states are ByteTensors that map_location='cuda'
                # breaks set_state on.
                "torch": torch.get_rng_state().cpu(),
                "cuda": [s.cpu() for s in torch.cuda.get_rng_state_all()],
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
        },
        tmp,
    )
    os.replace(tmp, path)


class StepHelper:
    """Per-step concerns both loops share: pause, debug dumps, `log_every`, step checkpoints,
    `--max-steps`, validation. `step` returns "go", "stop" (caller returns) or "complete"
    (caller saves); a checkpoint stores batch + 1, a pause the unconsumed batch."""

    def __init__(self, model, opt, cfg, a, run_dir, ds_train, best, val, step_fn,
                 start_step: int, total_steps: int, stop_at: int = 0, accum: int = 1):
        self.model, self.opt, self.cfg, self.a = model, opt, cfg, a
        self.run_dir, self.ds_train = run_dir, ds_train
        self.best, self.val, self.step_fn = best, val, step_fn
        self.global_step, self.total_steps, self.stop_at = start_step, total_steps, stop_at
        self.ckpt_every, self.t_step = max(1, a.ckpt_every_steps), time.time()
        # images per optimiser step is batch * accum; the img/s line needs both
        self.accum = max(1, int(accum))
        # steps since the last log line: a val resets t_step mid-window, so the window's step
        # count is not always log_every.
        self.since_log = 0
        mfile = os.path.join(run_dir, "debugpics", "manifest.json")
        self.debug_manifest: list[dict] = []
        if os.path.exists(mfile):
            with open(mfile, encoding="utf-8") as f:
                self.debug_manifest = json.load(f)
        self.debug_done = len(self.debug_manifest)
        announce_pause_schedule(run_dir, a, start_step)

    def _save(self, epoch: int, batch: int) -> str:
        p = os.path.join(self.run_dir, f"ckpt_step{self.global_step:06d}.pt")
        save_ckpt(p, self.model, self.opt, self.global_step, epoch, batch, self.best(), self.cfg)
        return p

    def step(self, x, y, epoch=None, batch=0, src=None, indices=None, feats=None) -> str:
        # x and y are either one batch or a list of `grad_accum` micro-batches; the accumulated
        # gradient is the mean of the micro-batch gradients, which for an elementwise-mean loss is
        # the same number as one larger batch, not an approximation of it.
        group = x if isinstance(x, (list, tuple)) else [x]
        cursor = (0, 0) if epoch is None else (epoch, batch + 1)
        if paused(self.run_dir):
            # unconsumed, so a resume starts here
            if self.global_step == 0:
                # A .PAUSE left over from a previous stop would otherwise write a step-0
                # checkpoint on every accidental relaunch, 250-300 MB each time.
                print("PAUSE file detected before the first step -> exiting without saving",
                      flush=True)
                return "stop"
            p = self._save(*((0, 0) if epoch is None else (epoch, batch)))
            print(f"PAUSE file detected -> checkpointed {p} and exiting cleanly", flush=True)
            return "stop"
        if self.debug_done < 100:
            need = min(100 - self.debug_done, len(src if src is not None else indices))
            if src is not None:
                # with accumulation the group holds several micro-batches and the first one may be
                # smaller than `need` on its own
                xcat = group[0] if len(group) == 1 else torch.cat(list(group))[:need]
                dump_stream_debug(self.run_dir, xcat, src[:need], self.debug_manifest)
            else:
                dump_debug_pictures(self.run_dir, self.ds_train, indices[:need],
                                    self.debug_manifest, self.cfg.mode)
            self.debug_done += need
            print(f"debugpics: {self.debug_done}/100 written to {self.run_dir}\\debugpics",
                  flush=True)

        loss = self.step_fn(group, y, self.global_step, feats)
        gs = self.global_step = self.global_step + 1
        self.since_log += 1
        if gs % self.cfg.log_every == 0:
            dt, self.t_step = time.time() - self.t_step, time.time()
            # count the steps this window actually contains, not log_every: a val can sit inside
            ips = (self.since_log * self.cfg.batch_size * self.accum
                   / max(dt, 1e-9))
            self.since_log = 0
            lrs = "/".join(f"{g['lr']:.2e}" for g in self.opt.param_groups)
            ep = "" if epoch is None else f" ep {epoch}"
            print(f"step {gs}/{self.total_steps}{ep} loss {loss.item():.4f} lr {lrs} "
                  f"{ips:.0f} img/s vram {torch.cuda.max_memory_allocated()/1e9:.1f}GB",
                  flush=True)

        if gs % self.ckpt_every == 0:
            print(f"checkpoint {self._save(*cursor)}", flush=True)
        if self.a.max_steps and gs >= self.a.max_steps:
            self._save(*cursor)
            print("max-steps reached, stopped", flush=True)
            return "stop"
        if self.stop_at and gs >= self.stop_at:
            # Fall through to the final save + test eval, so a finished schedule ends evaluated.
            print(f"schedule complete at step {gs}/{self.stop_at}", flush=True)
            return "complete"
        valued = bool(self.a.val_every_steps and gs % self.a.val_every_steps == 0)
        if valued:
            self.val(gs, 0 if epoch is None else epoch)
            self.t_step = time.time()   # else the next window's img/s includes the val
            self.since_log = 0
        # Step-targeted pauses fire after the step and its validation have completed, so a
        # numbered pause on a validation step leaves both the checkpoint and its numbers.
        steps, valids = pause_schedule(self.run_dir)
        due_steps = [(p, n) for p, n in steps if gs >= n]
        fire_valid = bool(valids) and (valued or not self.a.val_every_steps)
        if due_steps or fire_valid:
            p = self._save(*cursor)
            for path, _ in due_steps:
                try:
                    os.remove(path)
                except OSError:
                    pass
            if fire_valid:
                for path in valids:
                    try:
                        os.remove(path)
                    except OSError:
                        pass
            why = (f"step target {min(n for _, n in due_steps)} reached" if due_steps
                   else ".PAUSE-VALID fired after a validation")
            print(f"[pause] {why} -> checkpointed {p} and exiting cleanly", flush=True)
            return "stop"
        return "go"


def load_config(a: argparse.Namespace, ap: argparse.ArgumentParser) -> TrainConfig:
    """Resolve --run/--config into a TrainConfig; --run wins. Its config.json `_sha8` must
    resolve back here, else a resume silently starts a new run; an unpinned 8-hex run gets
    pinned."""
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


def check_image_size(cfg, ap) -> None:
    """The cache tree, the streamer's ring buffer and the validator all bind SIZE at import,
    while the config carries the intended resolution. A mismatch would feed the wrong pixel
    count to a checkpoint whose positional embedding assumes the other one, so stop here with
    the exact command instead of failing several minutes into a resume."""
    from .cache import SIZE

    if cfg.image_size != SIZE:
        ap.error(f"config says image_size {cfg.image_size} but the process bound SIZE {SIZE} "
                 f"at import; rerun with WDTAG_SIZE={cfg.image_size} in the environment")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="",
                    help="TrainConfig JSON (required unless --run is given)")
    ap.add_argument("--run", default="",
                    help="run directory to resume, e.g. --run runs/fe92f4df; loads that "
                         "directory's config.json, whose _sha8 pin resolves back to it. "
                         "Takes precedence over --config.")
    ap.add_argument("--fresh", action="store_true", help="ignore existing checkpoints")
    ap.add_argument("--reset-optimizer", action="store_true",
                    help="restore weights but not optimizer state; needed when a fork "
                         "adds a parameter group, since torch validates group layout")
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--deterministic", action="store_true",
                    help="force reproducible kernels so a resume can be diffed against an "
                         "uninterrupted run. Disables flash and mem-efficient attention, whose "
                         "backward pass scatters with atomics, and cudnn autotune. Measured: "
                         "without it two identical runs differ in the gradients from step 1, so "
                         "no resume can ever be bit-identical; with it the whole run reproduces.")
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
    ap.add_argument("--stream", action="store_true",
                    help="stream training data straight from the tars instead of the cache. "
                         "Required for corpora too big to cache (5.3M images would be ~4.2 TB). "
                         "CLI flag, so sha8 and resumability are unaffected.")
    ap.add_argument("--stream-images", type=int, default=0,
                    help="with --stream: number of samples to train on (default: one full pass "
                         "over the split, 5,321,713 images)")
    ap.add_argument("--stream-workers", type=int, default=8,
                    help="decode threads for --stream (measured 105 img/s at 512 on the "
                         "5800X3D with 8, 104 with 12: the pool saturates near 12)")
    ap.add_argument("--log-every", type=int, default=0,
                    help="print a step line every N steps (default: the config's log_every). 1 "
                         "gives per-step loss, lr, img/s and vram; each line costs one host sync")
    ap.add_argument("--stream-prefetch", type=int, default=4,
                    help="batches the reader may run ahead of the consumer (default 4). One "
                         "pinned slot is batch x 3 x H x W bytes, 176 MB at batch 224 / 512, so "
                         "prefetch 12 with grad_accum 6 pins about 3.3 GB of RAM")
    ap.add_argument("--ckpt-every-steps", type=int, default=CHECKPOINT_EVERY,
                    help=f"checkpoint interval (default {CHECKPOINT_EVERY}). A full 5.3M pass is "
                         "~111k steps, where the default would write ~55 GB, so pass e.g. 5000.")
    ap.add_argument("--ckpt-every-epochs", type=int, default=0,
                    help="additionally checkpoint every N epochs (0 = never). The epoch-end save "
                         "used to be unconditional, which on a 208-step epoch doubled the "
                         "checkpoint count alongside --ckpt-every-steps: 49 x 639 MB = 30.6 GB per "
                         "41-minute 5k-step arm, and the writes cost 5-15 s each (~8%% of wall "
                         "time). Keep this 0 unless you specifically want epoch-aligned resume "
                         "points.")
    a = ap.parse_args()

    cfg = load_config(a, ap)
    check_image_size(cfg, ap)
    if getattr(a, "log_every", 0):
        # Per-step logging costs one host sync per step (loss.item and the vram reading), about
        # 30 ms on this box. The identity is pinned before the override, because the run dir is
        # sha8 of the config and a display-only field must not fork it.
        if not getattr(cfg, "_sha8_pin", None):
            cfg._sha8_pin = cfg.sha8()
        cfg.log_every = max(1, int(a.log_every))
        print(f"[log] printing every {cfg.log_every} step(s), overriding the config", flush=True)
    # images per optimiser step = batch_size * accum. Kept as a local because total_steps, the
    # resume skip and the loop grouping all need it and it must not drift between them.
    accum = max(1, int(getattr(cfg, "grad_accum", 1) or 1))
    if accum > 1:
        print(f"[accum] {accum} micro-batches of {cfg.batch_size} per step "
              f"(effective batch {cfg.batch_size * accum}): the same gradient as one larger "
              f"batch for an elementwise-mean loss, at {cfg.batch_size} images of VRAM",
              flush=True)
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    if getattr(a, "deterministic", False):
        # Reproducibility, for verifying resume rather than for training: flash and
        # mem-efficient attention backwards scatter with atomics, so two identical runs diverge
        # in the gradients from the very first step and no comparison can be made. The math
        # backend is slower and exact.
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        print("[deterministic] math attention, cudnn deterministic, autotune off", flush=True)
    device = "cuda"

    space = load_label_space()
    n_labels = len(space)
    run_dir = cfg.run_dir()
    if not getattr(cfg, "_sha8_pin", None):
        # Freeze this run's identity: unfixed, a later field change re-keys it and a resume
        # starts a new dir.
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
    # 50k nested memmap ~3 min; the 1k set's ~0.37 macro ceiling cannot rank models
    # differing in macro.
    nested_root = os.path.join(CACHE_ROOT, "valid_nested")
    val_want = int(getattr(cfg, "val_images", 0) or 0)
    if val_want > 1000 and os.path.exists(os.path.join(nested_root, "images.npy")):
        ds_val = NestedValSet(nested_root, val_want, n_labels)
        val_targets = ds_val.dense_targets()
        print(f"[val] nested validation cache: {len(ds_val):,} images from {nested_root}",
              flush=True)
    else:
        ds_val = CacheDataset("valid", cfg.mode, n_labels)
        val_targets = load_targets("valid", cfg.mode, n_labels)

    # A run whose frozen second view comes from a feature cache must validate on the same
    # frozen view, not on a stock tower or on None. The val cache holds the trained trunk's
    # pooled features for the same rows, in the same order.
    model = DinoTagger(cfg, n_labels).to(device)
    if getattr(cfg, "feat_cache", "") and getattr(cfg, "feat_val_cache", ""):
        vf = np.load(cfg.feat_val_cache, mmap_mode="r")
        if len(vf) < len(ds_val):
            raise SystemExit(f"feat_val_cache {cfg.feat_val_cache} has {len(vf)} rows but "
                             f"validation wants {len(ds_val)}")
        model._val_feats = vf
        print(f"[val] frozen view from {cfg.feat_val_cache}: {len(vf)} rows", flush=True)
    priors = label_priors("train", cfg.mode, n_labels)
    model.init_head_bias(priors)
    counts = model.count_params()
    print(f"params: OFT {counts['oft']/1e6:.2f}M, head {counts['head']/1e6:.2f}M, "
          f"unfrozen {counts['blocks']/1e6:.2f}M in {model.n_unfrozen} block(s), "
          f"backbone {counts['backbone']/1e6:.1f}M, {model.n_oft} OFT modules", flush=True)


    # torch.compile ~1.6x; Inductor autotunes the GEMM/bmm kernels this graph is made of.
    # The log states the mode outright: default, max-autotune or eager. "eager" is a decision
    # (compile_model=false) or a failure, and the two are distinguished so a slow run can be
    # diagnosed from its own log rather than from the config's intent.
    forward = model
    _cmode = a.compile_mode or "default"
    _opt = "compiled" if cfg.compiled_optimizer else "eager"
    if cfg.compile_model:
        try:
            t0 = time.time()
            forward = torch.compile(model, mode=_cmode)
            print(f"[compile] model={_cmode} optimizer={_opt}"
                  f"{'' if a.compile_mode else ' (default mode; --compile-mode max-autotune for a long run)'}"
                  f", ready in {time.time()-t0:.1f}s", flush=True)
        except Exception as e:  # never let a compile problem kill a run
            print(f"[compile] model=eager (torch.compile FAILED: {type(e).__name__}: {e}) "
                  f"optimizer={_opt}", flush=True)
            forward = model
    else:
        print(f"[compile] model=eager (compile_model=false) optimizer={_opt}", flush=True)

    # Per-tag rate matching. The head bias starts at logit(prior) and the loss walks it away from
    # that, which ends up with the model predicting 72.9 tags per image where the truth has 32.3;
    # at the fixed 0.40 threshold that over-prediction is the whole precision deficit. This holds
    # each tag's predicted rate against its label rate instead. Off unless bias_match_eta > 0.
    # See docs/calibration-at-fixed-threshold.md.
    cal_bias = torch.zeros(model.n_labels, device=device, dtype=torch.float32)
    _bias_eta = float(getattr(cfg, "bias_match_eta", 0.0) or 0.0)
    _bias_every = max(1, int(getattr(cfg, "bias_match_every", 50) or 50))
    _bias_clip = float(getattr(cfg, "bias_match_clip", 6.0) or 6.0)
    if _bias_eta > 0.0:
        _inner_forward = forward

        def forward(x, f=None, _f=_inner_forward):
            z = _f(x) if f is None else _f(x, f)
            return z + cal_bias.to(z.dtype)

        print(f"[cal] per-tag rate matching on: eta {_bias_eta} every {_bias_every} steps, "
              f"bias clipped to +/-{_bias_clip}", flush=True)

    oft_params, head_params, block_params = model.trainable_parameters()
    from adv_optm import SinkSGD_adv

    if cfg.compiled_optimizer:
        # ~2 ms of Python/dispatch per adapter tensor (73 => ~88 ms/step, flat in element
        # count); compiled once (~6 s) => ~28 ms. fullgraph=True trips the recompile limit:
        # raise it first.
        torch._dynamo.config.cache_size_limit = 1_000_000
        torch._dynamo.config.recompile_limit = 1_000_000
        torch._dynamo.config.accumulated_recompile_limit = 1_000_000
        torch._dynamo.config.force_parameter_static_shapes = False

    if cfg.freeze_oft:
        # Adapters at identity (Cayley(0) = I), out of the optimizer: -73 updates/step.
        # Unfrozen blocks = arm C (no adapter, linear head); omitting block_params once
        # trained the head only.
        for p in oft_params:
            p.requires_grad_(False)
        groups = []
        if block_params:
            groups.append({"params": block_params, "lr": cfg.lr_unfreeze})
        groups.append({"params": head_params, "lr": cfg.lr_head})
        print(f"freeze_oft: adapters held at identity; training "
              f"{'unfrozen blocks + ' if block_params else ''}head", flush=True)
    else:
        groups = [{"params": oft_params, "lr": cfg.lr_oft}]
        if block_params:
            groups.append({"params": block_params, "lr": cfg.lr_unfreeze})
        groups.append({"params": head_params, "lr": cfg.lr_head})
        # Which parameter groups the geometry switches reach. Empty means all of them, and
        # `optimizer_oft_only` remains the older spelling of ["oft"].
        scope = set(cfg.optimizer_switch_groups or ())
        if cfg.optimizer_oft_only:
            scope = {"oft"}
        if scope and scope != {"oft", "block", "head"}:
            # A per-row Sinkhorn on a matrix whose rows are tags forces every tag's row to equal
            # RMS and throws the frequency prior away; on a rotation it is the geometry itself.
            plain = {"normed_momentum": False, "snr_cond": False, "nesterov": False,
                     "spectral_normalization": False, "orthogonal_gradient": "disabled",
                     "orthogonal_sinkhorn": False, "geometric_wd": False,
                     "cautious_wd": False}
            names = ["oft"] + (["block"] if block_params else []) + ["head"]
            for name, g in zip(names, groups):
                if name not in scope:
                    g.update(plain)
            print(f"[opt] switches scoped to {sorted(scope)}, plain for "
                  f"{[n for n in names if n not in scope]}", flush=True)

    opt = SinkSGD_adv(
        groups,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
        sinkhorn_iterations=cfg.sinkhorn_iterations,
        orthogonal_sinkhorn=cfg.orthogonal_sinkhorn,
        normed_momentum=cfg.optimizer_normed_momentum,
        snr_cond=cfg.optimizer_snr_cond,
        nesterov=cfg.optimizer_nesterov,
        orthogonal_gradient=cfg.optimizer_orthogonal_gradient,
        state_precision=cfg.optimizer_state_precision,
        spectral_normalization=cfg.optimizer_spectral_normalization,
        geometric_wd=cfg.optimizer_geometric_wd,
        cautious_wd=cfg.optimizer_cautious_wd,
        stochastic_rounding=True,
        compiled_optimizer=cfg.compiled_optimizer,
    )
    if (cfg.optimizer_normed_momentum or cfg.optimizer_snr_cond or cfg.optimizer_nesterov
            or cfg.optimizer_state_precision != "auto"
            or cfg.optimizer_orthogonal_gradient != "disabled"):
        print(f"[opt] SinkSGD_adv normed_momentum={cfg.optimizer_normed_momentum} "
              f"snr_cond={cfg.optimizer_snr_cond} nesterov={cfg.optimizer_nesterov} "
              f"state_precision={cfg.optimizer_state_precision} "
              f"orthogonal_gradient={cfg.optimizer_orthogonal_gradient} "
              f"orthogonal_sinkhorn={cfg.orthogonal_sinkhorn} "
              f"spectral={cfg.optimizer_spectral_normalization} "
              f"geometric_wd={cfg.optimizer_geometric_wd} "
              f"cautious_wd={cfg.optimizer_cautious_wd} "
              f"weight_decay={cfg.weight_decay}",
              flush=True)


    from .losses import make_loss, class_balanced_weights

    loss_name = getattr(cfg, "loss", "bce") or "bce"
    _loss_fn = make_loss(loss_name)
    _class_weight = _ntr_bias = _pos_weight = None
    if loss_name != "bce":
        from .data import label_priors as _priors
        from .losses import ntr_bias_from_prior

        _pri = torch.as_tensor(_priors("train", cfg.mode, n_labels),
                               dtype=torch.float32, device=device)
        _cnt = _pri * 10000.0
        if loss_name == "cb_ntr":
            _class_weight = class_balanced_weights(_cnt, beta=cfg.cb_beta,
                                                   normalize="pos_mass")
            _ntr_bias = ntr_bias_from_prior(_pri, kappa=cfg.ntr_kappa)
            print(f"loss cb_ntr: weight {float(_class_weight.min()):.3f}.."
                  f"{float(_class_weight.max()):.3f} (beta {cfg.cb_beta}, pos_mass), "
                  f"lambda {cfg.ntr_lambda}, kappa {cfg.ntr_kappa}", flush=True)
        elif loss_name == "bce_pos_weight":
            # pos_weight needs IMAGE count, not positive count; set once len(ds_train)
            # is known.
            _pos_weight = None
        else:
            print(f"loss {loss_name}: gamma_pos 0.0, gamma_neg {cfg.loss_gamma}, clip 0.05, "
                  f"detach_focal", flush=True)

    # BUG FIXED 2026-10-22: `loss_gamma` NEVER passed, so ASL ran gamma_neg=4.0 against configs
    # saying 2.0; the phase-3 gamma sweep (g1.0/g3.0/g4.0) would have been 3 identical arms.
    _loss_kwargs = {}
    if loss_name in ("asl", "asl_ntr"):
        _loss_kwargs["gamma_neg"] = float(cfg.loss_gamma)
    if loss_name == "asl_ntr":
        _loss_kwargs["ntr_lambda"] = float(cfg.ntr_lambda)
    # Omission tolerance: scale down the loss on a negative the model already finds likely,
    # because human annotators leave out tags they should have added. Off unless configured.
    if getattr(cfg, "loss_neg_relax_thr", 0.0) > 0.0:
        _loss_kwargs["neg_relax_thr"] = float(cfg.loss_neg_relax_thr)
        _loss_kwargs["neg_relax_alpha"] = float(cfg.loss_neg_relax_alpha)
        print(f"loss: confident negatives above {cfg.loss_neg_relax_thr} scaled by "
              f"{1.0 - cfg.loss_neg_relax_alpha:.2f} (omission tolerance)", flush=True)

    use_stream = a.stream or bool(getattr(cfg, "stream", False))
    if use_stream:
        from .stream import KNOWN_TOTAL

        stream_images = (a.stream_images or getattr(cfg, "stream_images", 0)
                            or KNOWN_TOTAL["train"])
        # one pass, no epoch: total_steps is the whole pass, and one step now spends
        # batch_size * grad_accum images.
        total_steps = -(-stream_images // (cfg.batch_size * accum))
        steps_per_epoch = total_steps
    else:
        stream_images = 0
        steps_per_epoch = len(ds_train) // cfg.batch_size
        total_steps = steps_per_epoch * cfg.epochs
        if accum > 1:
            print("[accum] ignored on the cached-image path, which has its own epoch and batch "
                  "loop; only the tar stream accumulates", flush=True)

    # pos_weight_i = ((N - n_i)/n_i)**0.5, N = images; the total-positive form inflates ~6x
    # and gave a one-example label 568x.
    if loss_name == "bce_pos_weight" and _pos_weight is None:
        _n_img = float(stream_images or len(ds_train))
        _pos_weight = ((_n_img - _cnt) / _cnt.clamp_min(1.0)).clamp_min(0.0).sqrt()
        print(f"loss bce_pos_weight: n_images {_n_img:.0f}, pos_weight "
              f"{float(_pos_weight.min()):.3f}..{float(_pos_weight.max()):.3f}", flush=True)


    start_epoch, start_step_in_epoch, global_step = 0, 0, 0
    best = {"val_macro_f1": -1.0, "step": -1}
    if not a.fresh:
        ckpts = sorted(
            (p for p in os.listdir(run_dir) if p.startswith("ckpt_step") and p.endswith(".pt")),
            key=lambda p: int(p.split("step")[1].split(".")[0]),
        )
        if ckpts:
            # Newest first, but never die on one bad file: a truncated checkpoint used to make
            # a run unresumable even with three good ones sitting beside it.
            ck = None
            for name in reversed(ckpts):
                path = os.path.join(run_dir, name)
                try:
                    ck = torch.load(path, map_location=device, weights_only=False)
                    break
                except Exception as e:  # noqa: BLE001 - any load failure means "try older"
                    print(f"[resume] {name} unreadable ({type(e).__name__}: {e}); "
                          f"falling back to an older checkpoint", flush=True)
                    ck = None
            if ck is None:
                raise SystemExit(f"no readable checkpoint in {run_dir}")
            model.load_adapter_state_dict(
                ck["adapter"], strict=not getattr(a, "reset_optimizer", False))
            if getattr(a, "reset_optimizer", False):
                # A parameter group changes the group count and torch validates layout, so
                # state_dict cannot load; momentum from zero suits a new phase.
                print("[resume] --reset-optimizer: skipping optimizer state restore", flush=True)
            else:
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

    # staged: warm-start from another run (stage 1 head vs identity adapters, stage 2 gentle
    # lr_oft); optimizer state and schedule NOT inherited.
    init_from = a.init_from or getattr(cfg, "init_from", "")
    if init_from:
        src = init_from
        # A directory means "its best_val.pt", which is usually right but not always: a run whose
        # last checkpoint is the annealed one and whose best_val sits mid-anneal should be started
        # from the last one. Naming the file explicitly is the escape hatch.
        if src.endswith(".pt"):
            cand = src
        else:
            cand = os.path.join(src, "best_val.pt")
            if not os.path.exists(cand):
                srcs = sorted(p for p in os.listdir(src)
                              if p.startswith("ckpt_step") and p.endswith(".pt"))
                if not srcs:
                    raise SystemExit(f"--init-from {src} has no best_val.pt or ckpt_step*.pt")
                cand = os.path.join(src, srcs[-1])
        if not os.path.exists(cand):
            raise SystemExit(f"--init-from {cand} does not exist")
        ick = torch.load(cand, map_location=device, weights_only=False)
        model.load_adapter_state_dict(ick["adapter"])
        print(f"warm-started adapter+head from {cand} (step {ick['step']}, "
              f"source val {ick.get('metrics', {}).get('micro_f1', float('nan')):.4f}). "
              f"optimizer state and run history are fresh.", flush=True)

    coll = MultiHotCollator(n_labels)
    t_start = time.time()

    def run_val(step: int, ep: int) -> dict:
        """Validate, log, keep best_val.pt; mid-epoch or epoch end."""
        nonlocal best
        t0 = time.time()
        vf = getattr(model, "_val_feats", None)
        metrics, probs = evaluate(forward, model, ds_val, val_targets, cfg.batch_size,
                                  device, feats=vf)
        extra = threshold_summary(probs, val_targets)
        rec = {"step": step, "epoch": ep, **metrics, **extra,
               "val_seconds": round(time.time() - t0, 1), "val_images": len(ds_val)}
        sel_key = SELECT_METRICS[a.select_metric]
        print(f"VAL step {step} (ep {ep}): macro_f1 {metrics['macro_f1']:.4f} "
              f"macro_mcc {metrics['macro_mcc']:.4f} micro_f1 {metrics['micro_f1']:.4f} "
              f"micro_mcc {metrics['micro_mcc']:.4f} "
              f"(supported_macro_f1 {metrics.get('supported_macro_f1', float('nan')):.4f}) "
              f"| n={len(ds_val)} best_thr {extra['best_threshold']:.2f} "
              f"macro@bestthr {extra['macro_f1_best']:.4f} "
              f"macro@0.40 {extra['macro_f1_at_0.40']:.4f} "
              f"| best by {sel_key} in {rec['val_seconds']:.1f}s", flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        # threshold_summary's metrics live in `extra`, not `metrics`: read the merged record,
        # or a threshold-based metric resolves to NaN and nothing is ever selected.
        raw = rec.get(sel_key)
        score = float(raw) if raw is not None else float("nan")
        if math.isfinite(score) and score > best_value(best, a.select_metric):
            best = {
                # legacy key: macro selection from pre---select-metric ckpts.
                "val_macro_f1": metrics["macro_f1"],
                "val_micro_f1": metrics["micro_f1"],
                "val_supported_macro_f1": metrics.get("supported_macro_f1", -1.0),
                "val_macro_f1_at_0.40": extra["macro_f1_at_0.40"],
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

    def step_once(x, y, step, feats=None):
        """One optimiser step for both loops."""
        # mirror the param-group order above, incl. freeze_oft + blocks.
        if cfg.freeze_oft:
            bases = ((cfg.lr_unfreeze, cfg.lr_head) if block_params
                     else (cfg.lr_head,))
        else:
            bases = ((cfg.lr_oft, cfg.lr_unfreeze, cfg.lr_head) if block_params
                     else (cfg.lr_oft, cfg.lr_head))
        for gi, base in enumerate(bases):
            opt.param_groups[gi]["lr"] = lr_at(step, total_steps, base,
                                               cfg.warmup_steps, cfg.lr_schedule,
                                               cfg.decay_steps,
                                               getattr(cfg, "lr_cosine_start", 0))
        acc = max(1, int(getattr(cfg, "grad_accum", 1) or 1))
        micro = x if isinstance(x, (list, tuple)) else [x]
        ys = y if isinstance(y, (list, tuple)) else [y]
        opt.zero_grad(set_to_none=True)
        total = None
        p_sum = y_sum = None
        n_seen = 0
        # Backward per micro-batch on loss/acc, then one optimiser step: for a loss that averages
        # over elements this is the same gradient as the full batch, because the mean of the
        # micro-batch means is the mean. The dropout draw and the cast differ from a true large
        # batch, so equal is up to floating point and RNG, not bitwise.
        for mi, (xi, yi) in enumerate(zip(micro, ys)):
            xi = xi.to(device, non_blocking=True)
            yi = yi.to(device, non_blocking=True)
            fi = None
            if feats is not None:
                fi = feats[mi].to(device, non_blocking=True)
            logits = forward(xi) if fi is None else forward(xi, fi)
            if loss_name == "cb_ntr":
                loss = _loss_fn(logits.float(), yi, ntr_bias=_ntr_bias,
                                class_weight=_class_weight, gamma=cfg.loss_gamma,
                                ntr_lambda=cfg.ntr_lambda)
            elif loss_name == "bce_pos_weight":
                loss = _loss_fn(logits.float(), yi, pos_weight=_pos_weight)
            elif loss_name != "bce":
                loss = _loss_fn(logits.float(), yi, **_loss_kwargs)
            else:
                loss = F.binary_cross_entropy_with_logits(logits.float(), yi)
            (loss / acc).backward()
            total = loss.detach() if total is None else total + loss.detach()
            if _bias_eta > 0.0:
                with torch.no_grad():
                    ps = torch.sigmoid(logits.float()).sum(0)
                    ys = yi.float().sum(0)
                    p_sum = ps if p_sum is None else p_sum + ps
                    y_sum = ys if y_sum is None else y_sum + ys
                    n_seen += int(yi.shape[0])
        if total is None:
            raise RuntimeError("no micro-batches reached the step function")
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for g in opt.param_groups for p in g["params"]], cfg.grad_clip)
        opt.step()
        if _bias_eta > 0.0 and n_seen > 0 and step % _bias_every == 0:
            with torch.no_grad():
                pred_rate = p_sum / n_seen
                want_rate = y_sum / n_seen
                cal_bias.add_((want_rate - pred_rate) * _bias_eta * 8.0)
                cal_bias.clamp_(-_bias_clip, _bias_clip)
                if step % 1000 == 0:
                    print(f"[cal] predicted per image {float(p_sum.sum()) / n_seen:.1f} "
                          f"against true {float(y_sum.sum()) / n_seen:.1f}, "
                          f"bias median {float(cal_bias.median()):+.2f}", flush=True)
        return total / max(1, len(micro))

    h = StepHelper(model, opt, cfg, a, run_dir, ds_train, lambda: best, run_val,
                   step_once, global_step, total_steps,
                   stop_at=total_steps if use_stream else 0, accum=accum)

    if use_stream:

        from .stream import TarStream

        skip = global_step * cfg.batch_size * accum  # exact: every step spends the same images
        print(f"[stream] one pass over {stream_images:,} images in {total_steps:,} steps, "
              f"batch {cfg.batch_size}"
              + (f" x {accum} accumulated" if accum > 1 else "")
              + f", {a.stream_workers} decode threads"
              + (f", resuming by skipping {skip:,} samples" if skip else ""), flush=True)
        # stream_images may span traversals (2x corpus = warm restart): chain pass_index
        # 0,1,2,..; 0 capped training at one epoch. Order is pure f(seed, pass_index).
        corpus = KNOWN_TOTAL["train"]
        want_passes = max(1, -(-stream_images // corpus))
        # absolute samples consumed can exceed one corpus; split into (pass, offset) to
        # resume mid-epoch-N.
        start_pass = skip // corpus
        next_skip = skip % corpus
        if start_pass >= want_passes:
            print(f"[stream] checkpoint is at pass {start_pass} but stream_images only "
                  f"covers {want_passes} pass(es); raise stream_images to continue",
                  flush=True)

        def _passes():
            for p in range(start_pass, want_passes):
                sp = TarStream("train", seed=cfg.seed, pass_index=p,
                               batch_size=cfg.batch_size, n_labels=n_labels,
                               workers=a.stream_workers, prefetch=a.stream_prefetch, hold=accum,
                               skip=next_skip if p == start_pass else 0,
                               max_samples=int(min(corpus, stream_images - p * corpus)),
                               log=lambda m: print(m, flush=True))
                yield from sp.batches()

        model.train()
        fc = getattr(model, 'feat_cache', None)
        if fc is not None and (fc.seed, fc.pass_index) != (cfg.seed, start_pass):
            raise SystemExit(
                f"feat_cache was built at seed {fc.seed} pass {fc.pass_index} but this run "
                f"is seed {cfg.seed} pass {start_pass}; every row would pair an image with "
                f"another image's features")
        group_x, group_y, group_src, group_f = [], [], [], []
        for x, y, consumed, src in _passes():
            group_x.append(x)
            group_y.append(y)
            group_src.extend(src)
            if fc is not None:
                feat0 = consumed - len(x)
                block = fc.read(feat0, len(x))
                # keep the cache's fp32 precision: the head is fp32, and rounding the frozen
                # view to bf16 here would undo the fp32 cache's point
                group_f.append(torch.from_numpy(block).to(torch.float32))
            if len(group_x) < h.accum:
                continue
            r = h.step(group_x, group_y, src=group_src,
                       feats=group_f if fc is not None else None)
            group_x, group_y, group_src, group_f = [], [], [], []
            if r == "stop":
                return
            if r == "complete":
                break
        if group_x:
            # A stream whose length is not a whole number of optimiser steps still spends its
            # tail. That last step consumes fewer images than batch*accum, so the sample cursor a
            # later resume infers from global_step is slightly ahead of the truth; it only happens
            # once the stream is exhausted, where a resume is refused anyway.
            short = cfg.batch_size * (accum - len(group_x))
            if short:
                print(f"[accum] last step is short by {short} images (stream exhausted)",
                      flush=True)
            r = h.step(group_x, group_y, src=group_src,
                       feats=group_f if fc is not None else None)
            if r == "stop":
                return

        print(f"training done in {(time.time()-t_start)/60:.1f} min, best {best}", flush=True)
        p = os.path.join(run_dir, f"ckpt_step{h.global_step:06d}.pt")
        save_ckpt(p, model, opt, h.global_step, 0, 0, best, cfg)
        from .evaluate import eval_run
        eval_run(run_dir, "test")
        return

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
            if h.step(x, y, epoch=epoch, batch=step_in_epoch,
                      indices=batches[bi]) == "stop":
                return


        del loader  # reap worker processes before measuring memory-driven work
        if not a.val_every_steps:
            run_val(h.global_step, epoch)
            h.t_step = time.time()   # else the next window's img/s includes the val
            h.since_log = 0
        # OPT-IN: unconditional saves on a 208-step epoch interleaved with the 200-step
        # interval wrote 639 MB ~every 100 steps (30.6 GB per 5k arm).
        if a.ckpt_every_epochs and (epoch + 1) % a.ckpt_every_epochs == 0:
            p = os.path.join(run_dir, f"ckpt_step{h.global_step:06d}.pt")
            save_ckpt(p, model, opt, h.global_step, epoch + 1, 0, best, cfg)

    print(f"training done in {(time.time()-t_start)/60:.1f} min, best {best}", flush=True)
    # The streamed branch always saved a final ckpt here, the cached branch did not -- up to
    # 999 steps stale.
    p = os.path.join(run_dir, f"ckpt_step{h.global_step:06d}.pt")
    save_ckpt(p, model, opt, h.global_step, cfg.epochs, 0, best, cfg)
    from .evaluate import eval_run
    eval_run(run_dir, "test")


if __name__ == "__main__":
    main()







