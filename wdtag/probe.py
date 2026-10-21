"""Feature probe: a linear head on DINOv3-B features, frozen or adapted.

Two sources of features, one tool:

* **frozen** (default): the backbone with OFT at identity. This measures the
  linear ceiling of the pre-trained features, and is what the "no adapters at
  all" reference in REPORT.md came from.
* **adapted** (``--adapter-from RUN``): the same model after loading
  ``RUN/<ckpt>`` (OFT rotations plus the run's own head/norm). This is the more
  useful probe: joint training reshapes the features, so the head LR that is
  right for frozen features is ~16x too high at the end of a run. Features are
  extracted once from the checkpoint (backbone frozen, adapters loaded) and
  cached, so each LR arm still costs seconds.

The linear head is trained for ``--epochs`` with ``--lrs`` and scored with the
same macro/micro metrics as evaluation.

    python -m wdtag.probe                          # frozen features
    python -m wdtag.probe --adapter-from runs/fa51da74 --lrs 0.0005,0.001,0.002,0.004
    python -m wdtag.probe --adapter-from runs/fe92f4df --splits test   # cache only

Cache naming (both live in ``runs/_probe`` and never collide):

* frozen:  ``{split}_{mode}.npz``
* adapted: ``oft_{run}_{ckpt}_{split}.npz`` (e.g. ``oft_fa51da74_best_val_test.npz``)
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import RUNS_ROOT, DINO_DIR, TrainConfig
from .data import CacheDataset, load_targets
from .labels import load_label_space
from .metrics import compute_metrics
from .model import IMAGENET_MEAN, IMAGENET_STD, N_REGISTERS, DinoTagger

PROBE_DIR = os.path.join(RUNS_ROOT, "_probe")
ALL_SPLITS = ("train", "valid", "test")
POOLS = ("cls", "mean", "cls_mean")
FROZEN_LRS = "0.002,0.005,0.01,0.02,0.05"
ADAPTED_LRS = "0.0005,0.001,0.002,0.004,0.008,0.016"


# --------------------------------------------------------------------- caching
def adapter_tag(adapter: str, ckpt: str) -> str:
    """Stable cache tag for one checkpoint: ``<run-dir-name>_<ckpt-stem>``."""
    run = os.path.basename(os.path.normpath(adapter))
    stem = os.path.splitext(os.path.basename(ckpt))[0]
    return f"{run}_{stem}"


def cache_path(split: str, mode: str, adapter: str | None = None,
               ckpt: str = "best_val.pt") -> str:
    """Where one split's features live.

    The adapted name deliberately cannot collide with the frozen one (which is
    ``{split}_{mode}.npz``), and matches the caches earlier probe runs wrote.
    """
    if adapter is None:
        return os.path.join(PROBE_DIR, f"{split}_{mode}.npz")
    return os.path.join(PROBE_DIR, f"oft_{adapter_tag(adapter, ckpt)}_{split}.npz")


# ------------------------------------------------------------------ extraction
@torch.no_grad()
def _extract_frozen(split: str, mode: str, batch_size: int = 64) -> dict[str, np.ndarray]:
    """CLS / mean-patch / both features from the untouched backbone."""
    from transformers import DINOv3ViTModel

    backbone = DINOv3ViTModel.from_pretrained(DINO_DIR, dtype=torch.bfloat16).cuda().eval()
    ds = CacheDataset(split, mode, len(load_label_space()))
    n = len(ds)
    out = {k: np.empty((n, 768 * (2 if k == "cls_mean" else 1)), dtype=np.float16) for k in POOLS}
    # DINOv3ViTModel, like forward_features(), wants ImageNet-normalised input.
    mean_t = torch.tensor(IMAGENET_MEAN, device="cuda").view(1, 3, 1, 1)
    std_t = torch.tensor(IMAGENET_STD, device="cuda").view(1, 3, 1, 1)
    t0 = time.time()
    for start in range(0, n, batch_size):
        idx = list(range(start, min(start + batch_size, n)))
        imgs = torch.stack([ds[i][0] for i in idx]).cuda()
        # uint8 [0,255] -> ImageNet-normalised bf16 (same maths as model._prep).
        x = ((imgs.float().div_(255.0) - mean_t) / std_t).to(torch.bfloat16)
        o = backbone(pixel_values=x)
        cls = o.pooler_output.float()
        patch_mean = o.last_hidden_state[:, 1 + N_REGISTERS:, :].mean(1).float()
        out["cls"][start:start + len(idx)] = cls.cpu().numpy().astype(np.float16)
        out["mean"][start:start + len(idx)] = patch_mean.cpu().numpy().astype(np.float16)
        out["cls_mean"][start:start + len(idx)] = torch.cat([cls, patch_mean], -1).cpu().numpy().astype(np.float16)
        if (start // batch_size) % 20 == 0:
            print(f"  {split}: {start + len(idx)}/{n} ({time.time()-t0:.0f}s)", flush=True)
    del backbone
    torch.cuda.empty_cache()
    return out


@torch.no_grad()
def _extract_adapted(model: DinoTagger, split: str, mode: str,
                     batch_size: int = 64) -> dict[str, np.ndarray]:
    """Features from a trained adapter, using the run's own pooling."""
    pool = model.cfg.head_pool
    ds = CacheDataset(split, mode, len(load_label_space()))
    n = len(ds)
    hidden = model.backbone.config.hidden_size
    d = hidden * (2 if pool == "cls_mean" else 1)
    out = {pool: np.empty((n, d), dtype=np.float16)}
    t0 = time.time()
    for start in range(0, n, batch_size):
        idx = list(range(start, min(start + batch_size, n)))
        imgs = torch.stack([ds[i][0] for i in idx]).cuda()
        # forward_features() expects ALREADY-normalised input (the full forward is
        # head(norm(forward_features(_prep(x))))). Feeding raw uint8 here hands the
        # backbone 0-255 values and silently produces garbage features
        # (micro F1 ~0.15 instead of ~0.39) -- always go through _prep.
        f = model.forward_features(model._prep(imgs)).float()
        out[pool][start:start + len(idx)] = f.cpu().numpy().astype(np.float16)
        if (start // batch_size) % 25 == 0:
            print(f"  {split}: {start + len(idx)}/{n} ({time.time()-t0:.0f}s)", flush=True)
    return out


def build_adapter_model(adapter: str, ckpt: str = "best_val.pt") -> tuple[DinoTagger, dict]:
    """Rebuild the run's model and load ``<adapter>/<ckpt>`` into it."""
    cfg = TrainConfig.from_json(os.path.join(adapter, "config.json"))
    model = DinoTagger(cfg, len(load_label_space())).cuda().eval()
    state = torch.load(os.path.join(adapter, ckpt), map_location="cuda", weights_only=False)
    model.load_adapter_state_dict(state["adapter"])
    return model, state


@torch.no_grad()
def extract_features(split: str, mode: str = "squash", batch_size: int = 64, *,
                     adapter: str | None = None, ckpt: str = "best_val.pt",
                     model: DinoTagger | None = None) -> dict[str, np.ndarray]:
    """Load one split's features from cache, or extract and cache them.

    ``adapter`` is the run directory whose checkpoint shapes the features; when
    it is given, the model must be passed in (built once by ``main``) or is
    built here. Returns ``{pool_name: float16 array}``; the cache path is printed.
    """
    path = cache_path(split, mode, adapter, ckpt)
    if os.path.exists(path):
        with np.load(path) as d:
            feats = {k: d[k] for k in d.files}
        first = next(iter(feats.values()))
        print(f"features {split}: {path} (cached, {first.shape[1]}d)", flush=True)
        return feats

    if adapter is None:
        print(f"extracting frozen features for {split} ...", flush=True)
        feats = _extract_frozen(split, mode, batch_size)
    else:
        if model is None:
            model, _ = build_adapter_model(adapter, ckpt)
        print(f"extracting adapted features for {split} ({os.path.normpath(adapter)}/{ckpt}) ...",
              flush=True)
        feats = _extract_adapted(model, split, mode, batch_size)

    os.makedirs(PROBE_DIR, exist_ok=True)
    np.savez(path, **feats)
    print(f"features {split}: {path} (extracted, cached)", flush=True)
    return feats


# -------------------------------------------------------------------- head
def train_head(feats: dict, labels: torch.Tensor, pool: str, lr: float, epochs: int,
               wd: float, bs: int = 256, momentum: float = 0.95) -> nn.Module:
    from adv_optm import SinkSGD_adv

    d = feats[pool].shape[1]
    # No prior-bias init here, unlike training's init_head_bias(): keeping the
    # output bias at zero is what the numbers published in REPORT.md were
    # measured with, so leave it alone or the probe stops being comparable.
    head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, labels.shape[1])).cuda()
    opt = SinkSGD_adv(head.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
    g = torch.Generator().manual_seed(0)
    for ep in range(epochs):
        perm = torch.randperm(feats[pool].shape[0], generator=g)
        for i in range(0, len(perm) - bs + 1, bs):
            idx = perm[i:i + bs]
            x = torch.from_numpy(feats[pool][idx.numpy()].astype(np.float32)).cuda()
            y = labels[idx].cuda()
            loss = F.binary_cross_entropy_with_logits(head(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return head


@torch.no_grad()
def predict_head(head: nn.Module, feats: np.ndarray, bs: int = 1024) -> np.ndarray:
    head.eval()
    out = []
    for i in range(0, len(feats), bs):
        x = torch.from_numpy(feats[i:i + bs].astype(np.float32)).cuda()
        out.append(torch.sigmoid(head(x).float()).cpu().numpy())
    return np.concatenate(out)


# --------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Linear-head feature probe on frozen DINOv3-B or a trained OFT adapter.")
    ap.add_argument("--mode", default=None,
                    help="image geometry: squash or pad (default: the run config's mode "
                         "when --adapter-from is given, else squash)")
    ap.add_argument("--adapter-from", default=None, metavar="RUN",
                    help="run directory (e.g. runs/fa51da74) whose <ckpt> adapter shapes "
                         "the features; omit to probe the frozen backbone")
    ap.add_argument("--ckpt", default="best_val.pt",
                    help="checkpoint file inside --adapter-from (default: best_val.pt)")
    ap.add_argument("--splits", default="train,valid,test",
                    help="comma-separated splits to cache/extract; the LR sweep needs all "
                         "three (default: train,valid,test)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lrs", default=None,
                    help="comma-separated head learning rates (default: %s frozen, %s adapted)"
                         % (FROZEN_LRS, ADAPTED_LRS))
    a = ap.parse_args()

    splits = [s.strip() for s in a.splits.split(",") if s.strip()]
    bad = [s for s in splits if s not in ALL_SPLITS] or not splits
    if bad:
        ap.error(f"--splits must be a subset of {','.join(ALL_SPLITS)}")

    adapter = os.path.abspath(a.adapter_from) if a.adapter_from else None
    cfg = None
    if adapter is not None:
        cfg_path = os.path.join(adapter, "config.json")
        ckpt_path = os.path.join(adapter, a.ckpt)
        if not os.path.exists(cfg_path):
            ap.error(f"no config.json in {adapter}")
        if not os.path.exists(ckpt_path):
            ap.error(f"no checkpoint {a.ckpt} in {adapter}")
        cfg = TrainConfig.from_json(cfg_path)
        mode = a.mode or cfg.mode
    else:
        mode = a.mode or "squash"
    lrs = [float(x) for x in (a.lrs or (ADAPTED_LRS if adapter else FROZEN_LRS)).split(",")]

    src = f"adapted {os.path.normpath(adapter)}/{a.ckpt}" if adapter else "frozen backbone"
    print(f"probe: {src}; mode={mode}; splits={','.join(splits)}; epochs={a.epochs}", flush=True)

    # Only touch the GPU when a requested cache is missing, so repeating a sweep
    # over existing .npz features costs no feature extraction at all.
    model = None
    if adapter is not None and any(
            not os.path.exists(cache_path(s, mode, adapter, a.ckpt)) for s in splits):
        model, ck = build_adapter_model(adapter, a.ckpt)
        metrics = ck.get("metrics", {})
        print(f"loaded {os.path.normpath(adapter)}/{a.ckpt} step {ck.get('step')} "
              f"(its own val micro {metrics.get('micro_f1', float('nan')):.4f}, "
              f"macro {metrics.get('macro_f1', float('nan')):.4f})", flush=True)

    feats = {s: extract_features(s, mode, adapter=adapter, ckpt=a.ckpt, model=model)
             for s in splits}
    if model is not None:
        del model
        torch.cuda.empty_cache()

    if any(s not in feats for s in ALL_SPLITS):
        print(f"sweep skipped: needs train+valid+test features, got {','.join(splits)}",
              flush=True)
        return

    n_labels = len(load_label_space())
    ytr = torch.from_numpy(load_targets("train", mode, n_labels).astype(np.float32))
    yva = load_targets("valid", mode, n_labels)
    yte = load_targets("test", mode, n_labels)

    pools = [p for p in POOLS if p in feats["train"]]
    if not pools:
        ap.error(f"no known pool key in the train features (have {list(feats['train'])})")

    results = []
    for pool in pools:
        for lr in lrs:
            t0 = time.time()
            head = train_head(feats["train"], ytr, pool, lr, a.epochs, wd=0.0)
            vm = compute_metrics(predict_head(head, feats["valid"][pool]) >= 0.5, yva)
            tm = compute_metrics(predict_head(head, feats["test"][pool]) >= 0.5, yte)
            rec = {"pool": pool, "lr": lr, "mode": mode, "epochs": a.epochs,
                   "adapter_from": adapter, "ckpt": a.ckpt if adapter else None,
                   "val_macro_f1": vm["macro_f1"], "val_micro_f1": vm["micro_f1"],
                   "test_macro_f1": tm["macro_f1"], "test_micro_f1": tm["micro_f1"],
                   "test_macro_mcc": tm["macro_mcc"], "test_micro_mcc": tm["micro_mcc"],
                   "seconds": round(time.time() - t0, 1)}
            results.append(rec)
            print(f"{pool:9s} lr {lr:<6} val macro_f1 {vm['macro_f1']:.4f} "
                  f"micro_f1 {vm['micro_f1']:.4f} | test macro_f1 {tm['macro_f1']:.4f} "
                  f"micro_f1 {tm['micro_f1']:.4f} ({rec['seconds']}s)", flush=True)

    tag = adapter_tag(adapter, a.ckpt) if adapter else None
    os.makedirs(PROBE_DIR, exist_ok=True)
    out_path = os.path.join(PROBE_DIR,
                            f"probe_{('oft_' + tag + '_') if tag else ''}{mode}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"results written to {out_path}", flush=True)

    best = max(results, key=lambda r: r["val_macro_f1"])
    print("best by val macro_f1:", best, flush=True)
    best_micro = max(results, key=lambda r: r["val_micro_f1"])
    print(f"best by val micro_f1: pool {best_micro['pool']} lr {best_micro['lr']} -> "
          f"test micro {best_micro['test_micro_f1']:.4f}, macro {best_micro['test_macro_f1']:.4f}",
          flush=True)


if __name__ == "__main__":
    main()
