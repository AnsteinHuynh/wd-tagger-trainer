"""Frozen-feature probe: DINOv3-B features + a linear head.

Used to pick the head pooling / learning rate quickly (features are computed
once, then head training is instant), and to report a "linear probe" reference
next to the OFTv2 run.
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

from .config import RUNS_ROOT, DINO_DIR
from .data import CacheDataset, load_targets
from .labels import load_label_space
from .metrics import compute_metrics
from .model import IMAGENET_MEAN, IMAGENET_STD, N_REGISTERS

PROBE_DIR = os.path.join(RUNS_ROOT, "_probe")
POOLS = ("cls", "mean", "cls_mean")


@torch.no_grad()
def extract_features(split: str, mode: str = "squash", batch_size: int = 64) -> dict[str, np.ndarray]:
    path = os.path.join(PROBE_DIR, f"{split}_{mode}.npz")
    if os.path.exists(path):
        d = np.load(path)
        return {k: d[k] for k in d.files}

    from transformers import DINOv3ViTModel

    os.makedirs(PROBE_DIR, exist_ok=True)
    backbone = DINOv3ViTModel.from_pretrained(DINO_DIR, dtype=torch.bfloat16).cuda().eval()
    ds = CacheDataset(split, mode, len(load_label_space()))
    n = len(ds)
    out = {k: np.empty((n, 768 * (2 if k == "cls_mean" else 1)), dtype=np.float16) for k in POOLS}
    mean_t = torch.tensor(IMAGENET_MEAN, device="cuda").view(1, 3, 1, 1)
    std_t = torch.tensor(IMAGENET_STD, device="cuda").view(1, 3, 1, 1)
    t0 = time.time()
    for start in range(0, n, batch_size):
        idx = list(range(start, min(start + batch_size, n)))
        imgs = torch.stack([ds[i][0] for i in idx]).cuda()
        x = ((imgs.float().div_(255.0) - mean_t) / std_t).to(torch.bfloat16)
        o = backbone(pixel_values=x)
        cls = o.pooler_output.float()
        patch_mean = o.last_hidden_state[:, 1 + N_REGISTERS:, :].mean(1).float()
        out["cls"][start:start + len(idx)] = cls.cpu().numpy().astype(np.float16)
        out["mean"][start:start + len(idx)] = patch_mean.cpu().numpy().astype(np.float16)
        out["cls_mean"][start:start + len(idx)] = torch.cat([cls, patch_mean], -1).cpu().numpy().astype(np.float16)
        if (start // batch_size) % 20 == 0:
            print(f"  {split}: {start + len(idx)}/{n} ({time.time()-t0:.0f}s)", flush=True)
    np.savez(path, **out)
    del backbone
    torch.cuda.empty_cache()
    return out


def train_head(feats: dict, labels: torch.Tensor, pool: str, lr: float, epochs: int,
               wd: float, bs: int = 256, momentum: float = 0.95) -> nn.Module:
    from adv_optm import SinkSGD_adv

    d = feats[pool].shape[1]
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="squash")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lrs", default="0.002,0.005,0.01,0.02,0.05")
    a = ap.parse_args()

    space = load_label_space()
    n_labels = len(space)
    print("extracting features ...", flush=True)
    ftr = extract_features("train", a.mode)
    fva = extract_features("valid", a.mode)
    fte = extract_features("test", a.mode)
    ytr = torch.from_numpy(load_targets("train", a.mode, n_labels).astype(np.float32))
    yva = load_targets("valid", a.mode, n_labels)
    yte = load_targets("test", a.mode, n_labels)

    results = []
    for pool in POOLS:
        for lr in [float(x) for x in a.lrs.split(",")]:
            t0 = time.time()
            head = train_head(ftr, ytr, pool, lr, a.epochs, wd=0.0)
            vp = predict_head(head, fva[pool])
            vm = compute_metrics(vp >= 0.5, yva)
            tm = compute_metrics(predict_head(head, fte[pool]) >= 0.5, yte)
            rec = {"pool": pool, "lr": lr, "mode": a.mode, "epochs": a.epochs,
                   "val_macro_f1": vm["macro_f1"], "val_micro_f1": vm["micro_f1"],
                   "test_macro_f1": tm["macro_f1"], "test_micro_f1": tm["micro_f1"],
                   "test_macro_mcc": tm["macro_mcc"], "test_micro_mcc": tm["micro_mcc"],
                   "seconds": round(time.time() - t0, 1)}
            results.append(rec)
            print(f"{pool:9s} lr {lr:<6} val macro_f1 {vm['macro_f1']:.4f} "
                  f"micro_f1 {vm['micro_f1']:.4f} | test macro_f1 {tm['macro_f1']:.4f} "
                  f"micro_f1 {tm['micro_f1']:.4f} ({rec['seconds']}s)", flush=True)

    os.makedirs(PROBE_DIR, exist_ok=True)
    with open(os.path.join(PROBE_DIR, f"probe_{a.mode}.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    best = max(results, key=lambda r: r["val_macro_f1"])
    print("best by val macro_f1:", best, flush=True)


if __name__ == "__main__":
    main()


