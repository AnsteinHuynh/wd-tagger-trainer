"""Would aspect bucketing beat squash? Answer from the pad arm, for free.

Bucketing gives every image its own aspect: no distortion (like pad) and no wasted
canvas (like squash). The pad arm is a one-bucket bucketing experiment: it also
removes distortion, but pays padding whose amount depends on the image's aspect.
So decomposing pad-vs-squash by *padding fraction* tells us what bucketing would do:

  - on low-padding images (portraits) pad is essentially bucketing already, so if
    pad does not beat squash there, removing distortion buys nothing;
  - on high-padding images (landscape) any pad deficit is the cost of the border,
    which bucketing would not pay.

If pad >= squash on low-padding images, bucketing's upside is real. If pad < squash
even there, distortion is harmless and bucketing cannot help.

Usage: python scratch/sophie/diag_aspect_bucket.py
"""
import numpy as np
import pandas as pd

from wdtag.data import load_targets
from wdtag.labels import load_label_space

N = len(load_label_space())
SQUASH = r"runs\fe92f4df\test_probs.npy"   # squash, lr_head 0.00012, seed 1337
PAD = r"runs\d0ab7d29\test_probs.npy"      # pad,    lr_head 0.00012, seed 1337

y = load_targets("test", "squash", N).astype(bool)
p_sq = np.load(SQUASH)
p_pd = np.load(PAD)
asp = pd.read_csv("scratch/sophie/test_aspect.csv")
assert len(asp) == len(y) == 1000, (len(asp), len(y))

w = asp["w"].to_numpy().astype(np.float64)
h = asp["h"].to_numpy().astype(np.float64)
# Fraction of the padded square canvas that actually carries content. Under the pad
# arm, side = max(w,h), so utilisation = w*h / max(w,h)^2.
util = (w * h) / np.maximum(w, h) ** 2


def per_image(p):
    pred = p >= 0.5
    tp = (pred & y).sum(1)
    fp = (pred & ~y).sum(1)
    fn = (~pred & y).sum(1)
    return 2 * tp / np.maximum(2 * tp + fp + fn, 1)


def micro(p, m):
    pred = (p >= 0.5)[m]
    t = y[m]
    tp = (pred & t).sum(dtype=np.float64)
    fp = (pred & ~t).sum(dtype=np.float64)
    fn = (~pred & t).sum(dtype=np.float64)
    return 2 * tp / max(2 * tp + fp + fn, 1)


f_sq, f_pd = per_image(p_sq), per_image(p_pd)
delta = f_pd - f_sq

print("Padding fraction under the pad arm, bucketed. Low padding == what bucketing")
print("would give every image; high padding == the border cost bucketing would avoid.\n")
print(f"{'content utilisation':>21s} {'n':>5s} {'squash micro':>13s} {'pad micro':>10s} "
      f"{'delta':>8s} {'paired dF1/img':>15s} {'img win %':>10s}")
edges = [(0.90, 1.01), (0.75, 0.90), (0.60, 0.75), (0.0, 0.60)]
for lo, hi in edges:
    m = (util >= lo) & (util < hi)
    if m.sum() == 0:
        continue
    print(f"{lo:8.2f}-{hi:<12.2f} {m.sum():5d} {micro(p_sq, m):13.4f} {micro(p_pd, m):10.4f} "
          f"{micro(p_pd, m) - micro(p_sq, m):+8.4f} {delta[m].mean():+15.4f} "
          f"{100 * (delta[m] > 0).mean():9.1f}%")

print()
low = util >= 0.75
print(f"Low-padding images (util >= 0.75, n={int(low.sum())}): "
      f"squash {micro(p_sq, low):.4f} vs pad {micro(p_pd, low):.4f} "
      f"({micro(p_pd, low) - micro(p_sq, low):+.4f})")
print("\nInterpretation: if pad does not beat squash on the low-padding images, then")
print("removing distortion is worth ~nothing and bucketing has no headroom. If it wins")
print("there, bucketing would keep that win on every image instead of only the portraits.")
print(f"\nsanity: overall squash {micro(p_sq, np.ones(1000, bool)):.4f} "
      f"vs pad {micro(p_pd, np.ones(1000, bool)):.4f}")
