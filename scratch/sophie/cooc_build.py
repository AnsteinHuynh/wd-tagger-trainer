"""Co-occurrence structure of the labels, to relax negatives that the image plausibly implies.

Builds P(tag t | tag u) from the 50k validation labels, keeps the top partners per tag
above a floor, and writes a sparse log(1 - P) matrix so a training step can compute, for
every image, how strongly each absent tag is implied by the tags that are present.

The noisy-or over present tags is the plausibility of an unannotated tag: high plausibility
means a negative there is probably a human omission, not a true absence.

Writes scratch/sophie/cooc.npz
"""
import json

import numpy as np

CACHE = r"I:\.cache\wd-tagger-trainer\valid_nested"
TOPK = 24
FLOOR = 0.50      # P(t | u) must be at least this
LIFT = 2.0        # and at least this many times t's base rate, else u says nothing about t

z = np.load(f"{CACHE}\\labels.npz")
idx, off = z["indices"], z["offsets"]
n_img = len(off) - 1
n_lab = json.load(open(f"{CACHE}\\meta.json"))["n_labels"]

Y = np.zeros((n_img, n_lab), dtype=np.float32)
for i in range(n_img):
    Y[i, idx[off[i]:off[i + 1]]] = 1.0

count = Y.sum(0)
print(f"images {n_img}  labels {n_lab}  labels per image {Y.sum() / n_img:.1f}")
print(f"tags present at least once: {(count > 0).sum()}  median count {np.median(count[count > 0]):.0f}")

present = count / n_img
co = Y.T @ Y                      # 12476 x 12476 co-occurrence counts
np.fill_diagonal(co, 0)

base = np.maximum(count, 1)[None, :]      # P(t | u) = co[t, u] / count[u]
cond = co / base

rows, cols, vals = [], [], []
for u in range(n_lab):
    if count[u] < 20:
        continue
    p = cond[:, u]
    lift = p / np.maximum(present, 1e-9)
    order = np.argsort(-p)[:TOPK]
    keep = order[(p[order] >= FLOOR) & (lift[order] >= LIFT)]
    for t in keep:
        if t == u:
            continue
        rows.append(u)
        cols.append(int(t))
        vals.append(float(p[t]))

rows = np.asarray(rows, dtype=np.int32)
cols = np.asarray(cols, dtype=np.int32)
vals = np.asarray(vals, dtype=np.float32)
print(f"sparse partner entries {len(vals)}  ({len(vals) / max(1, n_lab):.1f} per tag)")

for t in (rows[:1] - 0):
    pass
top_u = np.argsort(-count)[:3]
for u in top_u:
    p = cond[:, u]
    order = np.argsort(-p)[1:6]
    mates = ", ".join(f"{int(t)}:{p[t]:.2f}" for t in order if t != u)
    print(f"tag {int(u)} (count {int(count[u])}) implies: {mates}")

np.savez(r"scratch\sophie\cooc.npz", rows=rows, cols=cols, vals=vals,
         count=count.astype(np.int64), n_labels=n_lab, images=n_img)
print("saved cooc.npz")

