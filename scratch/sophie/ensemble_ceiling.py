"""If a second member existed, how much could an ensemble of ours with it possibly score?

Uses our step-8083 probabilities and the bar's cached probabilities as a stand-in for a
strong decorrelated member, both on the same 50000-image nested set at the fixed 0.40.

Reports three things:
  simple blends at a few fixed weights, the honest achievable ensemble
  the per-tag oracle blend, an optimistic upper bound on any weighted blend
  the error correlation between the two, which is what predicts ensemble gain
"""
import json

import numpy as np

CACHE = r"I:\.cache\wd-tagger-trainer\valid_nested"
PROBS_DIR = r"I:\.cache\wd-tagger-trainer\probs"
OURS = rf"{PROBS_DIR}\459e2a02_008083.npy"
BAR = rf"{PROBS_DIR}\bar_convnextv2_huge.npy"
THR = np.float32(0.40)
N = 50000

z = np.load(f"{CACHE}\\labels.npz")
indices, offsets = z["indices"], z["offsets"]
n_lab = json.load(open(f"{CACHE}\\meta.json"))["n_labels"]


def dense_labels(n):
    y = np.zeros((n, n_lab), dtype=bool)
    for i in range(n):
        y[i, indices[offsets[i]:offsets[i + 1]]] = True
    return y


def macro_f1(prob, y, chunk=2000):
    tp = np.zeros(n_lab, dtype=np.int64)
    fp = np.zeros(n_lab, dtype=np.int64)
    fn = np.zeros(n_lab, dtype=np.int64)
    for s in range(0, len(prob), chunk):
        e = min(s + chunk, len(prob))
        p = prob[s:e] >= THR
        yy = y[s:e]
        tp += (p & yy).sum(0)
        fp += (p & ~yy).sum(0)
        fn += (~p & yy).sum(0)
    den = 2 * tp + fp + fn
    sup = den > 0
    return float((2 * tp[sup] / den[sup]).mean()), int(sup.sum())


y = dense_labels(N)
a = np.load(OURS, mmap_mode="r")[:N]
b = np.load(BAR, mmap_mode="r")[:N]

ma, _ = macro_f1(a, y)
mb, _ = macro_f1(b, y)
print(f"ours step 8083  macro {ma:.4f}")
print(f"bar             macro {mb:.4f}")

for w in (0.2, 0.3, 0.4, 0.5):
    blend = (w * np.asarray(a, dtype=np.float32) + (1 - w) * np.asarray(b, dtype=np.float32))
    m, _ = macro_f1(blend, y)
    print(f"blend ours {w:.1f} / bar {1-w:.1f}  macro {m:.4f}  delta over bar {m-mb:+.4f}")

best = np.zeros(n_lab, dtype=np.float32)
for w in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
    blend = w * np.asarray(a, dtype=np.float32) + (1 - w) * np.asarray(b, dtype=np.float32)
    tp = np.zeros(n_lab); fp = np.zeros(n_lab); fn = np.zeros(n_lab)
    for s in range(0, N, 2000):
        e = min(s + 2000, N)
        p = blend[s:e] >= THR
        yy = y[s:e]
        tp += (p & yy).sum(0); fp += (p & ~yy).sum(0); fn += (~p & yy).sum(0)
    den = 2 * tp + fp + fn
    sup = den > 0
    cur = np.zeros(n_lab)
    cur[sup] = 2 * tp[sup] / den[sup]
    best = np.maximum(best, cur.astype(np.float32))
    del blend, tp, fp, fn
sup = best > 0
print(f"per-tag oracle blend  macro {best[sup].mean():.4f}  ({sup.sum()} tags)")

err_a = (np.asarray(a) >= THR) != y
err_b = (np.asarray(b) >= THR) != y
fa = err_a.mean(0)
fb = err_b.mean(0)
rho = np.corrcoef(fa, fb)[0, 1]
print(f"per-tag error correlation ours vs bar: {rho:.3f}")
both_wrong = (err_a & err_b).mean(0)
either_wrong = (err_a | err_b).mean(0)
print(f"tags where either errs and both err: {(both_wrong/either_wrong).mean():.3f} shared")
