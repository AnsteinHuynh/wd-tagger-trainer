"""Can a running per-tag bias controller recover the grid-searched calibration, online, without
ever looking at the set it is scored on?

Simulates training-time adaptation on the fit half: for each batch of images, compare each tag's
predicted positive rate with its label rate and nudge that tag's logit bias. Then freeze the bias
and score the held-out half. This is the shippable form of remedy three in
docs/calibration-at-fixed-threshold.md, tested without touching the trainer.
"""
import json

import numpy as np

CACHE = r"I:\.cache\wd-tagger-trainer\valid_nested"
PROBS = r"I:\.cache\wd-tagger-trainer\probs\459e2a02_008083.npy"
THR = np.float32(0.40)
TEMP = 0.25
FIT_LO, FIT_HI = 0, 25000
EVAL_LO, EVAL_HI = 25000, 50000
CHUNK = 2000
BATCH = 384

z = np.load(f"{CACHE}\\labels.npz")
idx, off = z["indices"], z["offsets"]
n_lab = json.load(open(f"{CACHE}\\meta.json"))["n_labels"]
probs = np.load(PROBS, mmap_mode="r")


def lg(lo, hi):
    p = np.asarray(probs[lo:hi], dtype=np.float32)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)) / TEMP


def labels(lo, hi):
    y = np.zeros((hi - lo, n_lab), dtype=np.float32)
    for i in range(lo, hi):
        y[i - lo, idx[off[i]:off[i + 1]]] = 1.0
    return y


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def eval_macro(bias):
    tp = np.zeros(n_lab, dtype=np.int64)
    fp = np.zeros(n_lab, dtype=np.int64)
    fn = np.zeros(n_lab, dtype=np.int64)
    for s in range(EVAL_LO, EVAL_HI, CHUNK):
        e = min(s + CHUNK, EVAL_HI)
        pred = sig(lg(s, e) + bias[None, :]) >= THR
        y = labels(s, e) > 0.5
        tp += (pred & y).sum(0)
        fp += (pred & ~y).sum(0)
        fn += (~pred & y).sum(0)
    d = 2 * tp + fp + fn
    sup = d > 0
    return float((2 * tp[sup] / d[sup]).mean())


print(f"baseline, temperature {TEMP}, no bias: eval macro {eval_macro(np.zeros(n_lab, dtype=np.float32)):.4f}")

for eta in (0.5, 1.0, 2.0):
    for epochs in (1, 2, 4):
        bias = np.zeros(n_lab, dtype=np.float32)
        for _ in range(epochs):
            for s in range(FIT_LO, FIT_HI, BATCH):
                e = min(s + BATCH, FIT_HI)
                p = sig(lg(s, e) + bias[None, :])
                y = labels(s, e)
                err = y.mean(0) - p.mean(0)
                bias += eta * err * 8.0
                bias = np.clip(bias, -6.0, 6.0)
        m = eval_macro(bias)
        print(f"  eta {eta:.1f} epochs {epochs}: eval macro {m:.4f}  "
              f"bias median {np.median(bias):+.2f} spread {bias.std():.2f}")
