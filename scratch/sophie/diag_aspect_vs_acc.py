"""Does squash distortion actually cost accuracy, per image?

The geometry audit produced test_aspect.csv in the same row order as
test_probs.npy, so we can split the 1,000 test images by aspect distortion and
see whether per-image F1 decays as the squash gets more extreme. This is the
direct test of "maybe the stretching is hurting us".

Distortion d = max(w,h)/min(w,h): an image with d=1.78 has one axis scaled 1.78x
more than the other when forced into 512x512.
"""
import numpy as np
import pandas as pd

from wdtag.data import load_targets
from wdtag.labels import load_label_space

RUN = r"runs\fe92f4df"          # lr_head 0.00012, best test micro (0.4142)
N = len(load_label_space())
p = np.load(rf"{RUN}\test_probs.npy")
y = load_targets("test", "squash", N).astype(bool)
asp = pd.read_csv("scratch/sophie/test_aspect.csv")
assert len(asp) == len(p) == 1000, (len(asp), len(p))

d = asp["aspect"].to_numpy()
d = np.maximum(d, 1 / d)        # distortion is symmetric
pred = p >= 0.5

tp = (pred & y).sum(1)
fp = (pred & ~y).sum(1)
fn = (~pred & y).sum(1)
f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
prec = tp / np.maximum(tp + fp, 1)
rec = tp / np.maximum(tp + fn, 1)

bins = [(1.0, 1.02), (1.02, 1.25), (1.25, 1.5), (1.5, 2.0), (2.0, 99)]
print(f"{'distortion d':>16s} {'n imgs':>7s} {'micro F1':>9s} {'prec':>6s} {'rec':>6s} "
      f"{'pred/img':>9s} {'true/img':>9s}")
for lo, hi in bins:
    m = (d >= lo) & (d < hi)
    if m.sum() == 0:
        continue
    T, F, Nn = tp[m].sum(), fp[m].sum(), fn[m].sum()
    mi = 2 * T / max(2 * T + F + Nn, 1)
    print(f"{lo:6.2f}-{hi:<9.2f} {m.sum():7d} {mi:9.4f} "
          f"{T/max(T+F,1):6.3f} {T/max(T+Nn,1):6.3f} "
          f"{pred[m].sum(1).mean():9.1f} {y[m].sum(1).mean():9.1f}")

print()
sq = d < 1.02
print(f"square-ish  (d<1.02): n={sq.sum():4d} micro F1 {(lambda T,F,Nn: 2*T/max(2*T+F+Nn,1))(tp[sq].sum(), fp[sq].sum(), fn[sq].sum()):.4f}")
w = d > 1.5
print(f"heavy warp (d>1.5) : n={w.sum():4d} micro F1 {(lambda T,F,Nn: 2*T/max(2*T+F+Nn,1))(tp[w].sum(), fp[w].sum(), fn[w].sum()):.4f}")

# per-image F1 vs d, ignoring images with very few tags (noisy target)
keep = y.sum(1) >= 5
r = np.corrcoef(d[keep], f1[keep])[0, 1]
print(f"\nper-image Pearson r(distortion, F1) = {r:+.3f}  (n={keep.sum()})")
r2 = np.corrcoef(d[keep], prec[keep])[0, 1]
r3 = np.corrcoef(d[keep], rec[keep])[0, 1]
print(f"                     r(distortion, precision) = {r2:+.3f}")
print(f"                     r(distortion, recall)    = {r3:+.3f}")
