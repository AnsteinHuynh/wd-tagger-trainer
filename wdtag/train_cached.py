"""Train and score a head on cached features, so an experiment costs minutes instead of hours.

The production backbone is frozen at whatever checkpoint its features were cached from, so this
trains the projection only: linear, or a hidden layer, or an ensemble of several caches
concatenated. It reports the protocol number that matters, macro and micro F1 at the fixed 0.40
threshold, plus the predicted tags per image, which is the calibration diagnostic.

Usage:
  python -m wdtag.train_cached --train-feats I:/.cache/feat/train --val-feats I:/.cache/feat/valid \
      --out runs/cached-linear --head linear --epochs 3
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F


def parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="train a head on cached features")
    ap.add_argument("--train-feats", nargs="+", required=True, help="one or more train caches")
    ap.add_argument("--val-feats", nargs="+", required=True, help="the same number of val caches")
    ap.add_argument("--out", required=True)
    ap.add_argument("--head", default="linear", choices=("linear", "mlp"))
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--loss", default="asl", choices=("asl", "bce"))
    ap.add_argument("--gamma-neg", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--select", default="macro", choices=("macro", "micro"))
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--limit", type=int, default=0, help="debug: cap the rows used")
    return ap.parse_args()


def load_cache(dirs: list[str], limit: int = 0):
    feats, labels, metas = [], [], []
    for d in dirs:
        meta = json.load(open(os.path.join(d, "meta.json")))
        f = np.load(os.path.join(d, "feats.npy"), mmap_mode="r")
        z = np.load(os.path.join(d, "labels.npz"))
        n = f.shape[0] if not limit else min(limit, f.shape[0])
        feats.append(f)
        labels.append((z["indices"], z["offsets"], n, int(meta.get("dim", f.shape[1]))))
        metas.append(meta)
    dim = sum(m["dim"] for m in metas)
    n = min(l[2] for l in labels)
    return feats, labels, n, dim, metas


def dense_labels(entry, lo: int, hi: int, n_labels: int) -> torch.Tensor:
    idx, off, _n, _dim = entry
    y = np.zeros((hi - lo, n_labels), dtype=np.float32)
    for i in range(lo, hi):
        s, e = int(off[i]), int(off[i + 1])
        y[i - lo, idx[s:e]] = 1.0
    return torch.from_numpy(y)


def gather(feats, lo: int, hi: int, device: str) -> torch.Tensor:
    parts = []
    for f in feats:
        parts.append(torch.from_numpy(np.asarray(f[lo:hi], dtype=np.float32)))
    x = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
    return x.to(device, non_blocking=True)


def counts_at(probs: torch.Tensor, y: torch.Tensor, thr: float = 0.40):
    pred = probs >= thr
    tp = (pred & (y > 0.5)).sum(0)
    fp = (pred & (y <= 0.5)).sum(0)
    fn = (~pred & (y > 0.5)).sum(0)
    return tp, fp, fn


def report(tp, fp, fn, n_img: int, n_labels: int, tag: str) -> dict:
    den = 2 * tp + fp + fn
    sup = den > 0
    macro = float((2 * tp[sup] / den[sup]).mean())
    TP, FP, FN = int(tp.sum()), int(fp.sum()), int(fn.sum())
    TN = n_img * n_labels - TP - FP - FN
    micro = 2 * TP / max(1, 2 * TP + FP + FN)
    prec = TP / max(1, TP + FP)
    rec = TP / max(1, TP + FN)
    denom = float((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    mcc = (TP * TN - FP * FN) / max(1e-9, denom ** 0.5)
    per_img = (TP + FP) / max(1, n_img)
    print(f"{tag}: macro {macro:.4f} micro {micro:.4f} P {prec:.4f} R {rec:.4f} "
          f"MCC {mcc:.4f} tags/img {per_img:.1f}", flush=True)
    return {"macro": macro, "micro": micro, "precision": prec, "recall": rec,
            "mcc": mcc, "tags_per_image": per_img}


def main() -> int:
    a = parse()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = "cuda"
    os.makedirs(a.out, exist_ok=True)

    tr_f, tr_l, n_tr, dim, tr_meta = load_cache(a.train_feats, a.limit)
    va_f, va_l, n_va, dim_va, va_meta = load_cache(a.val_feats, a.limit)
    if dim != dim_va:
        raise SystemExit(f"dim mismatch: train {dim} val {dim_va}")
    n_labels = 12476
    print(f"train rows {n_tr} val rows {n_va} dim {dim} labels {n_labels}", flush=True)
    print(f"train sources {[m.get('ckpt') for m in tr_meta]}", flush=True)

    layers: list[torch.nn.Module] = [torch.nn.LayerNorm(dim)]
    if a.head == "mlp":
        layers += [torch.nn.Linear(dim, a.hidden), torch.nn.GELU(),
                   torch.nn.Dropout(a.dropout), torch.nn.Linear(a.hidden, n_labels)]
    else:
        layers += [torch.nn.Linear(dim, n_labels)]
    head = torch.nn.Sequential(*layers).to(device)
    n_par = sum(p.numel() for p in head.parameters())
    print(f"head {a.head}, {n_par / 1e6:.1f}M parameters", flush=True)

    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=a.wd)
    steps_per_epoch = max(1, n_tr // a.batch)
    total_steps = steps_per_epoch * a.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, a.warmup))
        * 0.5 * (1 + np.cos(np.pi * min(1.0, s / max(1, total_steps)))))

    best = -1.0
    step = 0
    t0 = time.time()
    for ep in range(a.epochs):
        order = np.random.permutation(n_tr)
        run_loss, seen = 0.0, 0
        for s in range(0, n_tr - a.batch + 1, a.batch):
            sel = np.sort(order[s:s + a.batch])
            lo, hi = int(sel[0]), int(sel[-1]) + 1
            x = gather(tr_f, lo, hi, device)[torch.from_numpy(sel - lo).to(device)]
            y = dense_labels(tr_l[0], lo, hi, n_labels)
            y = y[torch.from_numpy(sel - lo)].to(device)
            z = head(x)
            if a.loss == "asl":
                from .losses import asl
                loss = asl(z.float(), y, gamma_neg=a.gamma_neg)
            else:
                loss = F.binary_cross_entropy_with_logits(z.float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            run_loss += float(loss)
            seen += 1
            step += 1
            if step % a.log_every == 0:
                print(f"ep {ep} step {step}/{total_steps} loss {run_loss / seen:.4f} "
                      f"lr {sched.get_last_lr()[0]:.2e} "
                      f"{(step * a.batch) / max(1e-9, time.time() - t0):.0f} rows/s", flush=True)

        head.eval()
        tp = torch.zeros(n_labels, dtype=torch.long, device=device)
        fp = torch.zeros(n_labels, dtype=torch.long, device=device)
        fn = torch.zeros(n_labels, dtype=torch.long, device=device)
        with torch.no_grad():
            for s in range(0, n_va, 2048):
                e = min(s + 2048, n_va)
                x = gather(va_f, s, e, device)
                p = torch.sigmoid(head(x).float())
                y = dense_labels(va_l[0], s, e, n_labels).to(device)
                t, f_, n_ = counts_at(p, y)
                tp += t
                fp += f_
                fn += n_
        m = report(tp.cpu(), fp.cpu(), fn.cpu(), n_va, n_labels, f"epoch {ep}")
        head.train()
        score = m[a.select]
        if score > best:
            best = score
            torch.save({"head": head.state_dict(), "args": vars(a), "metrics": m},
                       os.path.join(a.out, "best_head.pt"))
            print(f"new best {a.select} {score:.4f} -> best_head.pt", flush=True)
        json.dump({"epoch": ep, "metrics": m, "best": best},
                  open(os.path.join(a.out, "metrics.json"), "w"), indent=1)

    print(f"done, best {a.select} {best:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
