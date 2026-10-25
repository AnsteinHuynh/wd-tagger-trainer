"""Is the tail under-trained or under-representable?

Bins every supported tag by how many positives it has in the 50k and reports mean F1 for our
model and the bar. If our F1 climbs steeply with support and stays below the bar, the tail is
data-starved and resampling or class weighting is the cure. If our F1 is flat across the
well-supported bins, the deficit is not scarcity and no amount of oversampling will fix it.
"""
import numpy as np

CACHE = r"I:\.cache\wd-tagger-trainer\valid_nested"
OURS = r"I:\.cache\wd-tagger-trainer\probs\459e2a02_014000.npy"
BAR = r"I:\.cache\wd-tagger-trainer\probs\bar_convnextv2_huge.npy"
THR = np.float32(0.40)
CHUNK = 5000

z = np.load(f"{CACHE}\\labels.npz")
idx, off = z["indices"], z["offsets"]
n_img = len(off) - 1
ours = np.load(OURS, mmap_mode="r")
bar = np.load(BAR, mmap_mode="r")
n_lab = ours.shape[1]

tp_o = np.zeros(n_lab, dtype=np.int64); fp_o = np.zeros(n_lab, dtype=np.int64); fn_o = np.zeros(n_lab, dtype=np.int64)
tp_b = np.zeros(n_lab, dtype=np.int64); fp_b = np.zeros(n_lab, dtype=np.int64); fn_b = np.zeros(n_lab, dtype=np.int64)
sup = np.zeros(n_lab, dtype=np.int64)

for s in range(0, n_img, CHUNK):
    e = min(s + CHUNK, n_img)
    po = np.asarray(ours[s:e], dtype=np.float32) >= THR
    pb = np.asarray(bar[s:e], dtype=np.float32) >= THR
    y = np.zeros((e - s, n_lab), dtype=bool)
    for i in range(s, e):
        y[i - s, idx[off[i]:off[i + 1]]] = True
    sup += y.sum(0)
    tp_o += (po & y).sum(0); fp_o += (po & ~y).sum(0); fn_o += (~po & y).sum(0)
    tp_b += (pb & y).sum(0); fp_b += (pb & ~y).sum(0); fn_b += (~pb & y).sum(0)

def f1(tp, fp, fn):
    d = 2 * tp + fp + fn
    return np.where(d > 0, 2 * tp / np.maximum(d, 1), np.nan)

fo, fb = f1(tp_o, fp_o, fn_o), f1(tp_b, fp_b, fn_b)
edges = [1, 10, 50, 200, 1000, 5000, 20000, 100000]
print(f"{'support bin':>16s} {'tags':>6s} {'ours':>6s} {'bar':>6s} {'pred/img ours':>14s}")
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (sup >= lo) & (sup < hi)
    if m.sum() == 0:
        continue
    pred = float((tp_o[m] + fp_o[m]).sum()) / n_img
    print(f"{f'{lo}-{hi-1}':>16s} {int(m.sum()):6d} {np.nanmean(fo[m]):6.3f} "
          f"{np.nanmean(fb[m]):6.3f} {pred:14.2f}")

big = sup >= 20000
print(f"\nthe {int(big.sum())} most common tags: ours {np.nanmean(fo[big]):.3f}, "
      f"bar {np.nanmean(fb[big]):.3f}")
tiny = (sup >= 1) & (sup < 10)
print(f"the {int(tiny.sum())} tags with under 10 positives: ours {np.nanmean(fo[tiny]):.3f}, "
      f"bar {np.nanmean(fb[tiny]):.3f}")
print(f"correlation of our F1 with log support: "
      f"{np.corrcoef(np.log(sup[sup > 0]), fo[sup > 0])[0, 1]:.3f}")
