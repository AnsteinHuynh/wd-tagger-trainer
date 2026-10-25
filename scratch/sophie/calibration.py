"""Can a temperature fitted on held-out data make the fixed 0.40 threshold land where the model's
own optimum is?

The protocol scores at 0.40 and cannot be changed. But the map from logits to probabilities can,
and the model's best global threshold has drifted to 0.50, which means at 0.40 it is predicting
more positives than its own calibration wants. A single temperature is one scalar fitted here on
the first half of the 50k and scored on the second half, so nothing is fitted on the images it is
judged on.

CPU only. Reads the cached probabilities of one checkpoint.
"""
import json
import math

import numpy as np

CACHE = r"I:\.cache\wd-tagger-trainer\valid_nested"
PROBS = r"I:\.cache\wd-tagger-trainer\probs\459e2a02_008083.npy"
THR = np.float32(0.40)
FIT_LO = 0
FIT_HI = 25000
EVAL_LO = 25000
EVAL_HI = 50000
CHUNK = 5000

z = np.load(f"{CACHE}\\labels.npz")
idx, off = z["indices"], z["offsets"]
n_lab = json.load(open(f"{CACHE}\\meta.json"))["n_labels"]

probs = np.load(PROBS, mmap_mode="r")


def counts(fn, lo, hi):
    tp = np.zeros(n_lab, dtype=np.int64)
    fp = np.zeros(n_lab, dtype=np.int64)
    fn_ = np.zeros(n_lab, dtype=np.int64)
    for s in range(lo, hi, CHUNK):
        e = min(s + CHUNK, hi)
        p = fn(s, e)
        pred = p >= THR
        for i in range(e - s):
            pos = idx[off[s + i]:off[s + i + 1]]
            true = np.zeros(n_lab, dtype=bool)
            true[pos] = True
            tp += pred[i] & true
            fp += pred[i] & ~true
            fn_ += ~pred[i] & true
    return tp, fp, fn_


def macro(tp, fp, fn_):
    den = 2 * tp + fp + fn_
    sup = den > 0
    return float((2 * tp[sup] / den[sup]).mean()), int(sup.sum())


def raw(s, e):
    return np.asarray(probs[s:e], dtype=np.float32)


def temp(t):
    def f(s, e):
        p = np.asarray(probs[s:e], dtype=np.float32)
        p = np.clip(p, 1e-6, 1 - 1e-6)
        lg = np.log(p / (1 - p)) / t
        return 1.0 / (1.0 + np.exp(-lg))
    return f


tp, fp, fn_ = counts(raw, FIT_LO, FIT_HI)
fit_macro, fit_tags = macro(tp, fp, fn_)
print(f"fit half, T=1.00: macro {fit_macro:.4f} over {fit_tags} tags")

best = (0.0, 1.0)
for t in (0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 1.0):
    tp, fp, fn_ = counts(temp(t), FIT_LO, FIT_HI)
    m, _ = macro(tp, fp, fn_)
    print(f"  fit half, T={t:.2f}: macro {m:.4f}")
    if m > best[0]:
        best = (m, t)

t = best[1]
print(f"\nchosen T={t:.2f} on the fit half (macro {best[0]:.4f})")
tp, fp, fn_ = counts(raw, EVAL_LO, EVAL_HI)
m1, tags = macro(tp, fp, fn_)
print(f"eval half, T=1.00: macro {m1:.4f} over {tags} tags")
tp, fp, fn_ = counts(temp(t), EVAL_LO, EVAL_HI)
m2, _ = macro(tp, fp, fn_)
print(f"eval half, T={t:.2f}: macro {m2:.4f}  gain {m2 - m1:+.4f}")


