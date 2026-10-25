"""Where do the extra predictions go?

Our model emits 72.9 tags per image at 0.40 while the labels contain 32.3. This splits that 72.9
three ways: labelled positives, unlabelled cells the bar also believes in (probable human
omission, or a tag the annotator left off), and unlabelled cells only we believe in (our own
over-prediction). The bar is used here as an instrument and nothing else.
"""
import json

import numpy as np

CACHE = r"I:\.cache\wd-tagger-trainer\valid_nested"
OURS = r"I:\.cache\wd-tagger-trainer\probs\459e2a02_014000.npy"
BAR = r"I:\.cache\wd-tagger-trainer\probs\bar_convnextv2_huge.npy"
THR = np.float32(0.40)
CHUNK = 5000

z = np.load(f"{CACHE}\\labels.npz")
idx, off = z["indices"], z["offsets"]
n_lab = json.load(open(f"{CACHE}\\meta.json"))["n_labels"]
n_img = len(off) - 1
ours = np.load(OURS, mmap_mode="r")
bar = np.load(BAR, mmap_mode="r")

labelled = 0
unlab_bar_hi = 0
unlab_bar_mid = 0
unlab_bar_lo = 0
ours_total = 0
bar_conf_cells = 0

for s in range(0, n_img, CHUNK):
    e = min(s + CHUNK, n_img)
    p = np.asarray(ours[s:e], dtype=np.float32)
    b = np.asarray(bar[s:e], dtype=np.float32)
    y = np.zeros((e - s, n_lab), dtype=bool)
    for i in range(s, e):
        y[i - s, idx[off[i]:off[i + 1]]] = True
    pred = p >= THR
    ours_total += int(pred.sum())
    labelled += int((pred & y).sum())
    free = pred & ~y
    unlab_bar_hi += int((free & (b >= np.float32(0.9))).sum())
    unlab_bar_mid += int((free & (b >= np.float32(0.5)) & (b < np.float32(0.9))).sum())
    unlab_bar_lo += int((free & (b < np.float32(0.5))).sum())
    bar_conf_cells += int((free & (b >= np.float32(0.5))).sum())

print(f"images {n_img}, true labels per image {off[-1] / n_img:.1f}")
print(f"our predictions at 0.40 per image            {ours_total / n_img:7.1f}")
print(f"  of which labelled positives                {labelled / n_img:7.1f}")
print(f"  unlabelled, bar >= 0.9                     {unlab_bar_hi / n_img:7.1f}")
print(f"  unlabelled, bar 0.5 to 0.9                 {unlab_bar_mid / n_img:7.1f}")
print(f"  unlabelled, bar < 0.5                      {unlab_bar_lo / n_img:7.1f}")
free = ours_total - labelled
print(f"unlabelled total {free / n_img:.1f}, of which the bar supports "
      f"{bar_conf_cells / max(1, free) * 100:.1f} percent")
