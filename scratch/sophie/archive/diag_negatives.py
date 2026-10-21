"""What is our model actually saying about *absence*?

BCE over a sigmoid is already a two-sided objective: every one of the 12,476
labels at every image contributes -log(p) if present and -log(1-p) if absent.
So "P(tag is NOT there)" = 1 - sigmoid(logit) is already computed. The question
is whether it means anything.
"""
import numpy as np
from wdtag.data import load_targets
from wdtag.labels import load_label_space

N = len(load_label_space())
run = r"runs\7ff9ee38"          # pad, lr_oft 0.0005, best micro F1 (0.3985)
p = np.load(rf"{run}\test_probs.npy").astype(np.float64)
y = load_targets("test", "pad", N).astype(bool)
np.clip(p, 1e-7, 1 - 1e-7, out=p)

pos, neg = p[y], p[~y]
n_pos, n_neg = pos.size, neg.size
loss_pos = -np.log(pos).sum()
loss_neg = -np.log(1 - neg).sum()

print(f"pairs: {n_pos:,} present / {n_neg:,} absent  ({n_pos / (n_pos + n_neg) * 100:.2f}% positive)")
print(f"mean p when present : {pos.mean():.4f}")
print(f"mean p when absent  : {neg.mean():.4f}   (median {np.median(neg):.4f})")
print()
print("BCE decomposition:")
print(f"  from PRESENT pairs: {loss_pos:12,.0f} nats  ({loss_pos / (loss_pos + loss_neg) * 100:5.1f}%)")
print(f"  from ABSENT  pairs: {loss_neg:12,.0f} nats  ({loss_neg / (loss_pos + loss_neg) * 100:5.1f}%)")
print(f"  the model is {n_neg / n_pos:.0f}x more often graded on absence than presence")
print()

# how many absent tags does it ever call present?
for t in (0.5, 0.3, 0.2, 0.1):
    frac_neg_above = (neg >= t).mean()
    frac_pos_above = (pos >= t).mean()
    print(f"  threshold {t:.2f}: {frac_pos_above*100:5.1f}% of present pairs pass, "
          f"{frac_neg_above*100:6.3f}% of absent pairs pass")
print()

# calibration: what does a score actually mean?
bins = [0, .02, .05, .1, .2, .3, .5, .7, .9, 1.0]
print("calibration (all 12.5M pairs):")
print("  score bin        n pairs      empirical P(present)")
for lo, hi in zip(bins[:-1], bins[1:]):
    m = (p >= lo) & (p < hi)
    if m.sum() == 0:
        continue
    print(f"  {lo:.2f}-{hi:.2f}   {m.sum():11,}   {y[m].mean()*100:6.2f}%"
          f"   (label says {m.sum()*y[m].mean():,.0f} real)")
