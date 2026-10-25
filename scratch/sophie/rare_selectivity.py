"""How selective is a rare-tag filter?

For several support thresholds, reports what fraction of images contain at least one such tag,
how many such positives they carry, and therefore how many images must be scanned to collect a
batch of curated examples. Decides the scan cost and the size of any extracted set.
"""
import numpy as np

CACHE = r"I:\.cache\wd-tagger-trainer\valid_nested"
z = np.load(f"{CACHE}\\labels.npz")
idx, off = z["indices"], z["offsets"]
n_img = len(off) - 1
sup = np.zeros(int(idx.max()) + 1, dtype=np.int64)
for i in range(n_img):
    sup[idx[off[i]:off[i + 1]]] += 1

print(f"{n_img} images, labels per image {off[-1] / n_img:.1f}")
print(f"{'support <':>10s} {'tags':>6s} {'images with one':>16s} {'such tags/img':>14s} "
      f"{'scan for 192k':>14s}")
for thresh in (5, 10, 25, 50, 100, 250):
    rare = sup < thresh
    rare[sup == 0] = False
    has, cnt = 0, 0
    for i in range(n_img):
        row = idx[off[i]:off[i + 1]]
        hit = int(rare[row].sum())
        if hit:
            has += 1
            cnt += hit
    frac = has / n_img
    need = 192000 / frac if frac else float("inf")
    print(f"{thresh:10d} {int(rare.sum()):6d} {frac:15.1%} {cnt / n_img:14.2f} "
          f"{need:14,.0f}")

avg_kb = 318 * 1024 ** 3 / 5321713 / 1024
print(f"\naverage training image is {avg_kb:.1f} KB on disk, so 192000 extracted is "
      f"{192000 * avg_kb / 1024 ** 2:.1f} GB, 384000 is {384000 * avg_kb / 1024 ** 2:.1f} GB")
