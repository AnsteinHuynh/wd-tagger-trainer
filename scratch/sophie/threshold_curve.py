"""The threshold curve of production checkpoint 8083 on the 50k, so the cost of fixing 0.40 is a
number rather than an impression.

Reports macro F1, MCC, precision, recall and micro F1 at each threshold, plus the count of
predicted positives, which is the quantity the fixed threshold is really trading.
"""
import json

import numpy as np

CACHE = r"I:\.cache\wd-tagger-trainer\valid_nested"
PROBS = r"I:\.cache\wd-tagger-trainer\probs\459e2a02_008083.npy"
CHUNK = 5000

z = np.load(f"{CACHE}\\labels.npz")
idx, off = z["indices"], z["offsets"]
n_lab = json.load(open(f"{CACHE}\\meta.json"))["n_labels"]
n_img = len(off) - 1
probs = np.load(PROBS, mmap_mode="r")

THRS = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60)
tp = {t: np.zeros(n_lab, dtype=np.int64) for t in THRS}
fp = {t: np.zeros(n_lab, dtype=np.int64) for t in THRS}
fn = {t: np.zeros(n_lab, dtype=np.int64) for t in THRS}
tot_pos = {t: 0 for t in THRS}

for s in range(0, n_img, CHUNK):
    e = min(s + CHUNK, n_img)
    p = np.asarray(probs[s:e], dtype=np.float32)
    y = np.zeros((e - s, n_lab), dtype=bool)
    for i in range(s, e):
        y[i - s, idx[off[i]:off[i + 1]]] = True
    for t in THRS:
        pred = p >= np.float32(t)
        tot_pos[t] += int(pred.sum())
        tp[t] += (pred & y).sum(0)
        fp[t] += (pred & ~y).sum(0)
        fn[t] += (~pred & y).sum(0)

n_true = 0
for i in range(n_img):
    n_true += off[i + 1] - off[i]

print(f"{n_img} images, {n_true} true positives, {n_true / n_img:.1f} labels per image")
print("thr   macro    mcc     prec    rec     micro   predicted   per-image")
for t in THRS:
    d = 2 * tp[t] + fp[t] + fn[t]
    sup = d > 0
    f1 = 2 * tp[t][sup] / d[sup]
    prec = tp[t][sup] / np.maximum(tp[t][sup] + fp[t][sup], 1)
    rec = tp[t][sup] / np.maximum(tp[t][sup] + fn[t][sup], 1)
    mcc_num = tp[t] * 0 + 0
    TP, FP, FN = tp[t].sum(), fp[t].sum(), fn[t].sum()
    TN = n_img * n_lab - TP - FP - FN
    denom = np.sqrt(float((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)))
    mcc = ((TP * TN - FP * FN) / denom) if denom > 0 else 0.0
    micro = 2 * TP / max(1, 2 * TP + FP + FN)
    print(f"{t:.2f}  {f1.mean():.4f}  {mcc:.4f}  {prec.mean():.4f}  {rec.mean():.4f}  "
          f"{micro:.4f}  {tot_pos[t]:9d}  {tot_pos[t] / n_img:7.1f}")
