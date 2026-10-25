"""How wide is a 1k validation reading compared with a 50k one?

Bootstrap over tags for macro F1 and over images for micro F1, on the cached champion
probabilities, at the fixed 0.40 threshold. CPU only, reads the nested cache.
"""
import json

import numpy as np

CACHE = r"I:\.cache\wd-tagger-trainer\valid_nested"
PROBS = r"I:\.cache\wd-tagger-trainer\probs\2b6b81aa_012500.npy"
THR = np.float32(0.40)
B = 300
rng = np.random.default_rng(12345)

z = np.load(f"{CACHE}\\labels.npz")
indices, offsets = z["indices"], z["offsets"]
n_img = len(offsets) - 1
n_lab = json.load(open(f"{CACHE}\\meta.json"))["n_labels"]


def per_image_counts(prob, lo, hi):
    """TP, FP, FN per image for rows [lo, hi)."""
    tp = np.zeros(hi - lo, dtype=np.int64)
    fp = np.zeros(hi - lo, dtype=np.int64)
    fn = np.zeros(hi - lo, dtype=np.int64)
    for i in range(lo, hi):
        pos = indices[offsets[i]:offsets[i + 1]]
        pred = np.flatnonzero(prob[i - lo] >= THR)
        tp[i - lo] = np.intersect1d(pos, pred, assume_unique=True).size
        fp[i - lo] = pred.size - tp[i - lo]
        fn[i - lo] = pos.size - tp[i - lo]
    return tp, fp, fn


def macro_per_tag(prob, lo, hi):
    tp = np.zeros(n_lab, dtype=np.int64)
    fp = np.zeros(n_lab, dtype=np.int64)
    fn = np.zeros(n_lab, dtype=np.int64)
    for i in range(lo, hi):
        pos = indices[offsets[i]:offsets[i + 1]]
        pred = np.flatnonzero(prob[i - lo] >= THR)
        hit = np.intersect1d(pos, pred, assume_unique=True)
        tp[hit] += 1
        fp[pred] += 1
        fn[pos] += 1
    fp -= tp
    fn -= tp
    denom = 2 * tp + fp + fn
    supported = denom > 0
    f1 = np.zeros(n_lab, dtype=np.float64)
    f1[supported] = 2 * tp[supported] / denom[supported]
    return f1, supported


for name, n in (("1k", 1000), ("5k", 5000), ("10k", 10000), ("50k", 50000)):
    prob = np.load(PROBS, mmap_mode="r")[:n]
    tp, fp, fn = per_image_counts(prob, 0, n)
    micro = 2 * tp.sum() / (2 * tp.sum() + fp.sum() + fn.sum())
    micro_boot = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        t, p, f = tp[idx].sum(), fp[idx].sum(), fn[idx].sum()
        micro_boot.append(2 * t / (2 * t + p + f))
    f1, sup = macro_per_tag(prob, 0, n)
    vals = f1[sup]
    macro = vals.mean()
    macro_boot = [vals[rng.integers(0, len(vals), len(vals))].mean() for _ in range(B)]
    lo_m, hi_m = np.percentile(macro_boot, [2.5, 97.5])
    lo_u, hi_u = np.percentile(micro_boot, [2.5, 97.5])
    print(f"{name}: {int(sup.sum())} supported tags | macro {macro:.4f} "
          f"[{lo_m:.4f}, {hi_m:.4f}] width {hi_m-lo_m:.4f} | "
          f"micro {micro:.4f} [{lo_u:.4f}, {hi_u:.4f}] width {hi_u-lo_u:.4f}")
