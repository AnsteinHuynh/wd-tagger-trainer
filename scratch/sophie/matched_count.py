"""Matched-count prediction: take the top-k tags per image with k equal to the number of labels
that image actually has, then score. This is not a shippable model, it needs the answer to a
question the fixed threshold hides: if the count were right, how good is the ranking?

If macro at matched count is far above the threshold curve's best, the ranking is fine and the
whole gap is calibration plus count control.
"""
import json

import numpy as np

CACHE = r"I:\.cache\wd-tagger-trainer\valid_nested"
PROBS = r"I:\.cache\wd-tagger-trainer\probs\459e2a02_008083.npy"
CHUNK = 2000

z = np.load(f"{CACHE}\\labels.npz")
idx, off = z["indices"], z["offsets"]
n_lab = json.load(open(f"{CACHE}\\meta.json"))["n_labels"]
n_img = len(off) - 1
probs = np.load(PROBS, mmap_mode="r")

for extra in (0, -4, 4):
    tp = np.zeros(n_lab, dtype=np.int64)
    fp = np.zeros(n_lab, dtype=np.int64)
    fn = np.zeros(n_lab, dtype=np.int64)
    for s in range(0, n_img, CHUNK):
        e = min(s + CHUNK, n_img)
        p = np.asarray(probs[s:e], dtype=np.float32)
        k = np.array([off[i + 1] - off[i] for i in range(s, e)]) + extra
        k = np.clip(k, 1, n_lab)
        pred = np.zeros_like(p, dtype=bool)
        top = np.argpartition(-p, max(1, int(k.max())), axis=1)[:, :int(k.max())]
        for i in range(e - s):
            pred[i, top[i, :k[i]]] = True
        y = np.zeros_like(p, dtype=bool)
        for i in range(s, e):
            y[i - s, idx[off[i]:off[i + 1]]] = True
        tp += (pred & y).sum(0)
        fp += (pred & ~y).sum(0)
        fn += (~pred & y).sum(0)
    d = 2 * tp + fp + fn
    sup = d > 0
    f1 = 2 * tp[sup] / d[sup]
    TP, FP, FN = tp.sum(), fp.sum(), fn.sum()
    micro = 2 * TP / max(1, 2 * TP + FP + FN)
    prec = TP / max(1, TP + FP)
    rec = TP / max(1, TP + FN)
    print(f"k = truth {extra:+d}: macro {f1.mean():.4f}  micro {micro:.4f}  "
          f"precision {prec:.4f}  recall {rec:.4f}")
