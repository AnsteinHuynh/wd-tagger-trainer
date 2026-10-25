"""Extend the calibration search: sharper temperatures, then a shrunk per-tag bias.

Fit on the first half of the 50k, judged on the second half, so nothing is fitted on the images
it is scored on. CPU only.
"""
import json

import numpy as np

CACHE = r"I:\.cache\wd-tagger-trainer\valid_nested"
PROBS = r"I:\.cache\wd-tagger-trainer\probs\459e2a02_008083.npy"
THR = np.float32(0.40)
FIT_LO, FIT_HI = 0, 25000
EVAL_LO, EVAL_HI = 25000, 50000
CHUNK = 5000

z = np.load(f"{CACHE}\\labels.npz")
idx, off = z["indices"], z["offsets"]
n_lab = json.load(open(f"{CACHE}\\meta.json"))["n_labels"]
probs = np.load(PROBS, mmap_mode="r")


def dense_labels(lo, hi):
    y = np.zeros((hi - lo, n_lab), dtype=bool)
    for i in range(lo, hi):
        y[i - lo, idx[off[i]:off[i + 1]]] = True
    return y


def probs_for(lo, hi, t=1.0, bias=None):
    p = np.asarray(probs[lo:hi], dtype=np.float32)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    lg = np.log(p / (1 - p)) / t
    if bias is not None:
        lg = lg + bias[None, :]
    return 1.0 / (1.0 + np.exp(-lg))


def macro_of(p, y):
    pred = p >= THR
    tp = (pred & y).sum(0)
    fp = (pred & ~y).sum(0)
    fn = (~pred & y).sum(0)
    den = 2 * tp + fp + fn
    sup = den > 0
    return float((2 * tp[sup] / den[sup]).mean()), sup


yfit = dense_labels(FIT_LO, FIT_HI)
yeval = dense_labels(EVAL_LO, EVAL_HI)

print("temperature grid, fit half")
best = (0.0, 1.0)
for t in (0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.8, 1.0):
    m, _ = macro_of(probs_for(FIT_LO, FIT_HI, t), yfit)
    print(f"  T={t:.2f}: {m:.4f}")
    if m > best[0]:
        best = (m, t)
t = best[1]
m_eval, sup = macro_of(probs_for(EVAL_LO, EVAL_HI, t), yeval)
m_base, _ = macro_of(probs_for(EVAL_LO, EVAL_HI, 1.0), yeval)
print(f"chosen T={t:.2f}  fit {best[0]:.4f}  eval base {m_base:.4f}  eval {m_eval:.4f}  "
      f"gain {m_eval - m_base:+.4f}")

print("\nper-tag bias, fitted on the fit half at the chosen T, shrunk toward zero")
pfit = probs_for(FIT_LO, FIT_HI, t)
peval = probs_for(EVAL_LO, EVAL_HI, t)
for lam in (0.5, 1.0, 2.0):
    for iters in (1,):
        bias = np.zeros(n_lab, dtype=np.float32)
        for _ in range(20):
            pred = peval = None
            lg = np.log(np.clip(pfit, 1e-6, 1 - 1e-6) / (1 - np.clip(pfit, 1e-6, 1 - 1e-6)))
            p = 1.0 / (1.0 + np.exp(-(lg + bias[None, :])))
            pred = p >= THR
            # one Newton step per tag toward matching the observed positive rate
            tp = (pred & yfit).sum(0)
            fp = (pred & ~yfit).sum(0)
            err = (yfit.sum(0) - tp - fp).astype(np.float32) / max(1, len(yfit))
            bias += lam * err * 4.0
            bias = np.clip(bias, -4.0, 4.0)
        m, _ = macro_of(probs_for(EVAL_LO, EVAL_HI, 1.0, bias), yeval)
        print(f"  lam {lam:.1f}: eval macro {m:.4f}  ({m - m_eval:+.4f} against the plain temperature)")
