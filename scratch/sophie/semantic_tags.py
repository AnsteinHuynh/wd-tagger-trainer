"""Per-tag scores for the tags people actually omit: rare, semantic, relational.

The claim to test is that annotators and the bar both skip unusual semantic tags, things like
eye modifiers and family relations, so a hard negative there teaches the model to suppress a
correct answer. This prints label support, our F1 and the bar's F1 for the tags matching a set
of relation and modifier patterns, so the claim is a row of numbers rather than a memory.
"""
import json
import re

import numpy as np

CACHE = r"I:\.cache\wd-tagger-trainer\valid_nested"
OURS = r"I:\.cache\wd-tagger-trainer\probs\459e2a02_014000.npy"
BAR = r"I:\.cache\wd-tagger-trainer\probs\bar_convnextv2_huge.npy"
THR = np.float32(0.40)
CHUNK = 5000

from wdtag.labels import load_label_space

names = [n for n in load_label_space().names]
z = np.load(f"{CACHE}\\labels.npz")
idx, off = z["indices"], z["offsets"]
n_lab = len(names)
n_img = len(off) - 1
ours = np.load(OURS, mmap_mode="r")
bar = np.load(BAR, mmap_mode="r")

tp_o = np.zeros(n_lab, dtype=np.int64); fp_o = np.zeros(n_lab, dtype=np.int64)
fn_o = np.zeros(n_lab, dtype=np.int64)
tp_b = np.zeros(n_lab, dtype=np.int64); fp_b = np.zeros(n_lab, dtype=np.int64)
fn_b = np.zeros(n_lab, dtype=np.int64)
support = np.zeros(n_lab, dtype=np.int64)

for s in range(0, n_img, CHUNK):
    e = min(s + CHUNK, n_img)
    po = np.asarray(ours[s:e], dtype=np.float32) >= THR
    pb = np.asarray(bar[s:e], dtype=np.float32) >= THR
    y = np.zeros((e - s, n_lab), dtype=bool)
    for i in range(s, e):
        y[i - s, idx[off[i]:off[i + 1]]] = True
    support += y.sum(0)
    tp_o += (po & y).sum(0); fp_o += (po & ~y).sum(0); fn_o += (~po & y).sum(0)
    tp_b += (pb & y).sum(0); fp_b += (pb & ~y).sum(0); fn_b += (~pb & y).sum(0)

def f1(tp, fp, fn):
    d = 2 * tp + fp + fn
    return np.where(d > 0, 2 * tp / np.maximum(d, 1), np.nan)

f1o, f1b = f1(tp_o, fp_o, fn_o), f1(tp_b, fp_b, fn_b)

patterns = ("mother", "daughter", "sister", "brother", "sibling", "twins", "family",
            "father", "son", "aunt", "cousin", "grandmother", "grandfather")
print(f"relation tags, support >= 5 in the 50k, ours against bar")
print(f"{'tag':38s} {'sup':>5s} {'ours':>6s} {'bar':>6s} {'pred_ours':>9s}")
for i, n in enumerate(names):
    if any(p in n.lower() for p in patterns) and support[i] >= 5:
        pred = int(tp_o[i] + fp_o[i])
        print(f"{n[:38]:38s} {support[i]:5d} {f1o[i]:6.3f} {f1b[i]:6.3f} {pred:9d}")

print("\nsymbol and modifier tags with support >= 5")
for i, n in enumerate(names):
    if re.search(r"[+\-=]|^[a-z]+_[a-z]+_[a-z]+$", n) and support[i] >= 5 and len(n) <= 24:
        print(f"{n[:38]:38s} {support[i]:5d} {f1o[i]:6.3f} {f1b[i]:6.3f} "
              f"{int(tp_o[i] + fp_o[i]):9d}")
