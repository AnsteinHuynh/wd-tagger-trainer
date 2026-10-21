"""Decompose the macro-F1 gap against convnextv2_huge by label coverage.

The label audit found that only 10,790 of 12,476 labels appear anywhere in the
10k training set, 7,908 appear in fewer than 10 images, and 217 labels present
in the test split never appear in train at all. The baseline was trained on the
full 5.3M corpus, so on those labels it has an opportunity we structurally do
not. This script isolates how much of the headline macro gap is that, versus a
real quality gap on labels we could learn.
"""
import numpy as np

from wdtag.data import load_targets
from wdtag.labels import load_label_space

N = len(load_label_space())
OURS = r"runs\fe92f4df\test_probs.npy"


def f1_per_label(p, y, thr=0.5):
    pred = p >= thr
    tp = (pred & y).sum(0).astype(np.float64)
    fp = (pred & ~y).sum(0).astype(np.float64)
    fn = (~pred & y).sum(0).astype(np.float64)
    den = 2 * tp + fp + fn
    return np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0)


def micro(p, y, thr=0.5):
    pred = p >= thr
    tp = (pred & y).sum(dtype=np.float64)
    fp = (pred & ~y).sum(dtype=np.float64)
    fn = (~pred & y).sum(dtype=np.float64)
    return 2 * tp / max(2 * tp + fp + fn, 1)


y = load_targets("test", "squash", N).astype(bool)
n_test = y.sum(0)
train_cnt = load_targets("train", "squash", N).astype(bool).sum(0)

p_ours = np.load(OURS)
p_bar = np.load(r"runs\_baseline\convnextv2_huge\squash\test_probs.npy")
assert p_ours.shape == p_bar.shape == (1000, N), (p_ours.shape, p_bar.shape)

f_ours = f1_per_label(p_ours, y)
f_bar = f1_per_label(p_bar, y)

subsets = {
    "all labels": np.ones(N, bool),
    "present in test": n_test > 0,
    "in train, >=1 image": train_cnt >= 1,
    "in train, >=10 images": train_cnt >= 10,
    "in train, <10 images": (train_cnt >= 1) & (train_cnt < 10),
    "NEVER in train (test-present)": (train_cnt == 0) & (n_test > 0),
    "never in train (all)": train_cnt == 0,
}

print(f"{'label subset':>32s} {'n':>6s} {'ours macF1':>11s} {'bar macF1':>10s} {'gap':>8s} "
      f"{'ours share':>10s}")
for name, m in subsets.items():
    n = int(m.sum())
    if n == 0:
        continue
    a, b = f_ours[m].mean(), f_bar[m].mean()
    print(f"{name:>32s} {n:6d} {a:11.4f} {b:10.4f} {b - a:+8.4f} {a / b if b else 0:9.1%}")

print()
print(f"micro F1 @0.5   ours {micro(p_ours, y):.4f}   bar {micro(p_bar, y):.4f}")
print(f"positives in test: {int(y.sum())}, of which on never-trained labels: "
      f"{int(y[:, (train_cnt == 0) & (n_test > 0)].sum())} "
      f"({y[:, (train_cnt == 0) & (n_test > 0)].sum() / y.sum():.2%})")
