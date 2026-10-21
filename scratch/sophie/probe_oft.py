"""Probe the head on *trained* OFT features instead of frozen-backbone features.

The frozen probe answers "best head LR when the adapter is identity". After joint
training the adapter has moved the features a long way (drift ~0.8), so the head
sees a different distribution, and -- more usefully -- this measures the *ceiling*
of head-only training at the operating point we actually reached: if joint
training can't beat this, the adapters bought nothing a better head couldn't.

Features are extracted once from the checkpoint (rotations loaded, backbone
frozen) and cached, so each LR arm costs seconds.

    python scratch/sophie/probe_oft.py --run runs/fa51da74
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

from wdtag.data import CacheDataset, load_targets
from wdtag.labels import load_label_space
from wdtag.metrics import compute_metrics
from wdtag.model import DinoTagger, N_REGISTERS
from wdtag.probe import PROBE_DIR, train_head

POOLS = ("cls_mean",)


def extract(model, split, mode, n_labels, batch_size=64):
    ds = CacheDataset(split, mode, n_labels)
    n = len(ds)
    out = {k: np.empty((n, 1536), dtype=np.float16) for k in POOLS}
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, n, batch_size):
            idx = list(range(start, min(start + batch_size, n)))
            imgs = torch.stack([ds[i][0] for i in idx]).cuda()
            f = model.forward_features(model._prep(imgs)).float()
            out["cls_mean"][start:start + len(idx)] = f.cpu().numpy().astype(np.float16)
            if (start // batch_size) % 25 == 0:
                print(f"  {split}: {start + len(idx)}/{n} ({time.time()-t0:.0f}s)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="best_val.pt")
    ap.add_argument("--lrs", default="0.0005,0.001,0.002,0.004,0.008,0.016")
    ap.add_argument("--epochs", type=int, default=30)
    a = ap.parse_args()

    ck = torch.load(os.path.join(a.run, a.ckpt), map_location="cuda", weights_only=False)
    cfg = None
    from wdtag.config import TrainConfig
    cfg = TrainConfig.from_json(os.path.join(a.run, "config.json"))
    n_labels = len(load_label_space())
    model = DinoTagger(cfg, n_labels).cuda().eval()
    model.load_adapter_state_dict(ck["adapter"])
    src = ck.get("metrics", {})
    print(f"loaded {a.run}/{a.ckpt} step {ck.get('step')} "
          f"(its own val micro {src.get('micro_f1', float('nan')):.4f}, "
          f"macro {src.get('macro_f1', float('nan')):.4f})", flush=True)

    tag = os.path.basename(a.run.rstrip("\\/")) + "_" + a.ckpt.replace(".pt", "")
    cached = os.path.join(PROBE_DIR, f"oft_{tag}_train.npz")
    if os.path.exists(cached):
        print(f"using cached features {cached}", flush=True)
        ftr = {k: np.load(cached)[k] for k in POOLS}
        fte = {k: np.load(cached.replace("_train", "_test"))[k] for k in POOLS}
        fva = {k: np.load(cached.replace("_train", "_valid"))[k] for k in POOLS}
    else:
        os.makedirs(PROBE_DIR, exist_ok=True)
        print("extracting features from the trained adapter ...", flush=True)
        ftr = extract(model, "train", cfg.mode, n_labels)
        fva = extract(model, "valid", cfg.mode, n_labels)
        fte = extract(model, "test", cfg.mode, n_labels)
        for nm, d in (("train", ftr), ("valid", fva), ("test", fte)):
            np.savez(os.path.join(PROBE_DIR, f"oft_{tag}_{nm}.npz"), **d)
    del model
    torch.cuda.empty_cache()

    ytr = torch.from_numpy(load_targets("train", cfg.mode, n_labels)).float()
    yva = torch.from_numpy(load_targets("valid", cfg.mode, n_labels)).bool()
    yte = torch.from_numpy(load_targets("test", cfg.mode, n_labels)).bool()

    print(f"\n{'lr_head':>9s} {'val micro':>10s} {'val macro':>10s} {'test micro':>11s} "
          f"{'test macro':>11s} {'secs':>6s}")
    best = None
    for lr in [float(x) for x in a.lrs.split(",")]:
        t0 = time.time()
        head = train_head(ftr, ytr, "cls_mean", lr, a.epochs, wd=0.0)
        with torch.no_grad():
            pv = head(torch.from_numpy(fva["cls_mean"].astype(np.float32)).cuda()).float()
            pt = head(torch.from_numpy(fte["cls_mean"].astype(np.float32)).cuda()).float()
        mv = compute_metrics((pv >= 0).cpu().numpy(), yva.numpy())
        mt = compute_metrics((pt >= 0).cpu().numpy(), yte.numpy())
        secs = time.time() - t0
        print(f"{lr:9.6f} {mv['micro_f1']:10.4f} {mv['macro_f1']:10.4f} "
              f"{mt['micro_f1']:11.4f} {mt['macro_f1']:11.4f} {secs:6.1f}", flush=True)
        if best is None or mv["micro_f1"] > best[1]:
            best = (lr, mv["micro_f1"], mt["micro_f1"], mt["macro_f1"])
    print(f"\nbest by val micro: lr {best[0]} -> test micro {best[2]:.4f}, macro {best[3]:.4f}")


if __name__ == "__main__":
    main()

