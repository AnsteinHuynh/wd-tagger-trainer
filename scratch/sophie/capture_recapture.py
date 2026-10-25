"""Capture-recapture on the unlabelled cells, and why its answer cannot be trusted here.

Treat the set of truly-present tags that the labels omit as a hidden population. Two readers
detect some of it: our model and the bar. Lincoln-Petersen estimates the population as
n1 * n2 / overlap. The estimator assumes the readers err independently; the run reports the
overlap and a second estimator so the violation is visible instead of buried.
"""
import json

import numpy as np

CACHE = r"I:\.cache\wd-tagger-trainer\valid_nested"
OURS = r"I:\.cache\wd-tagger-trainer\probs\459e2a02_014000.npy"
BAR = r"I:\.cache\wd-tagger-trainer\probs\bar_convnextv2_huge.npy"
CHUNK = 5000

z = np.load(f"{CACHE}\\labels.npz")
idx, off = z["indices"], z["offsets"]
n_lab = json.load(open(f"{CACHE}\\meta.json"))["n_labels"]
n_img = len(off) - 1
ours = np.load(OURS, mmap_mode="r")
bar = np.load(BAR, mmap_mode="r")

for t_ours, t_bar in ((0.40, 0.40), (0.40, 0.50), (0.50, 0.50), (0.50, 0.90)):
    a = b = both = 0
    for s in range(0, n_img, CHUNK):
        e = min(s + CHUNK, n_img)
        p = np.asarray(ours[s:e], dtype=np.float32) >= np.float32(t_ours)
        q = np.asarray(bar[s:e], dtype=np.float32) >= np.float32(t_bar)
        y = np.zeros((e - s, n_lab), dtype=bool)
        for i in range(s, e):
            y[i - s, idx[off[i]:off[i + 1]]] = True
        free = ~y
        pa, qb = p & free, q & free
        a += int(pa.sum())
        b += int(qb.sum())
        both += int((pa & qb).sum())
    if both == 0:
        print(f"ours {t_ours} bar {t_bar}: no overlap, estimator undefined")
        continue
    lp = a * b / both
    # Chapman, the less biased small-sample form
    chap = (a + 1) * (b + 1) / (both + 1) - 1
    print(f"ours {t_ours} bar {t_bar}: ours {a / n_img:.1f}/img, bar {b / n_img:.1f}/img, "
          f"overlap {both / n_img:.1f}/img -> population {lp / n_img:.1f}/img "
          f"(Chapman {chap / n_img:.1f})")
