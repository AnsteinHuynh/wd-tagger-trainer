"""Per-tag calibration, done properly this time.

For each tag, choose a logit bias from a grid that maximises that tag's own F1 on the fit half
of the 50k, shrink it toward zero by how few positives the tag has, then judge the whole thing
on the other half. Vectorised over tags, chunked over images, CPU only.

The point of the comparison is whether the global temperature is already capturing everything,
or whether the per-tag structure is real. It also answers a question about the threshold: if the
per-tag optimum varies wildly, a fixed 0.40 is leaving macro on the table that a trained
calibration could pick up.
"""
import json

import numpy as np

CACHE = r"I:\.cache\wd-tagger-trainer\valid_nested"
PROBS = r"I:\.cache\wd-tagger-trainer\probs\459e2a02_008083.npy"
THR = np.float32(0.40)
FIT_LO, FIT_HI = 0, 25000
EVAL_LO, EVAL_HI = 25000, 50000
CHUNK = 2500
GRID = np.arange(-3.0, 3.01, 0.25, dtype=np.float32)
TEMP = float(__import__("os").environ.get("TEMP_SCALE", "0.25"))

z = np.load(f"{CACHE}\\labels.npz")
idx, off = z["indices"], z["offsets"]
n_lab = json.load(open(f"{CACHE}\\meta.json"))["n_labels"]
probs = np.load(PROBS, mmap_mode="r")


def dense(lo, hi):
    y = np.zeros((hi - lo, n_lab), dtype=bool)
    for i in range(lo, hi):
        y[i - lo, idx[off[i]:off[i + 1]]] = True
    return y


def logits(lo, hi):
    p = np.asarray(probs[lo:hi], dtype=np.float32)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)) / TEMP


yfit = dense(FIT_LO, FIT_HI)
yeval = dense(EVAL_LO, EVAL_HI)

# counts per candidate bias, accumulated over chunks
tp = np.zeros((len(GRID), n_lab), dtype=np.int64)
fp = np.zeros((len(GRID), n_lab), dtype=np.int64)
fn = np.zeros((len(GRID), n_lab), dtype=np.int64)
pos = yfit.sum(0)

for s in range(0, len(yfit), CHUNK):
    e = min(s + CHUNK, len(yfit))
    lg = logits(FIT_LO + s, FIT_LO + e)
    yy = yfit[s:e]
    for gi, b in enumerate(GRID):
        pred = (1.0 / (1.0 + np.exp(-(lg + b)))) >= THR
        tp[gi] += (pred & yy).sum(0)
        fp[gi] += (pred & ~yy).sum(0)
        fn[gi] += (~pred & yy).sum(0)

den = 2 * tp + fp + fn
f1 = np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0)
best_gi = f1.argmax(0)
best_bias = GRID[best_gi]
fit_macro = float(f1[best_gi, np.arange(n_lab)][den[best_gi, np.arange(n_lab)] > 0].mean())
print(f"per-tag best bias on the fit half: macro {fit_macro:.4f}")

scale = np.clip(pos / 200.0, 0.0, 1.0).astype(np.float32)
shrunk = best_bias * scale
print(f"bias shrunk for tags under 200 positives; tags with 0 positives {(pos == 0).sum()}")

# evaluate on the held-out half
def eval_macro(bias):
    tp2 = np.zeros(n_lab, dtype=np.int64)
    fp2 = np.zeros(n_lab, dtype=np.int64)
    fn2 = np.zeros(n_lab, dtype=np.int64)
    for s in range(0, len(yeval), CHUNK):
        e = min(s + CHUNK, len(yeval))
        lg = logits(EVAL_LO + s, EVAL_LO + e)
        pred = (1.0 / (1.0 + np.exp(-(lg + bias[None, :].astype(np.float32))))) >= THR
        yy = yeval[s:e]
        tp2 += (pred & yy).sum(0)
        fp2 += (pred & ~yy).sum(0)
        fn2 += (~pred & yy).sum(0)
    d = 2 * tp2 + fp2 + fn2
    sup = d > 0
    return float((2 * tp2[sup] / d[sup]).mean())


m0 = eval_macro(np.zeros(n_lab, dtype=np.float32))
m1 = eval_macro(np.full(n_lab, -1.0, dtype=np.float32))
m2 = eval_macro(shrunk)
print(f"temperature {TEMP}")
print(f"eval half: uncalibrated {m0:.4f}  global bias -1.0 {m1:.4f}  per-tag shrunk on top of the temperature {m2:.4f}")
print(f"per-tag over global: {m2 - m1:+.4f}")
print(f"bias distribution: median {np.median(best_bias):.2f}, p10 {np.percentile(best_bias, 10):.2f}, "
      f"p90 {np.percentile(best_bias, 90):.2f}, spread {best_bias.std():.2f}")

