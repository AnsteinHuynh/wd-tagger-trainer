"""How often does a human annotator omit a tag the best tagger we have is sure about?

Streams training images, runs the bar over them, and counts per tag: cells the bar is
confident about, of which how many carry the label and how many are unannotated. The
unannotated share estimates the omission rate, the number a loss that tolerates missing
positives needs.

Verbose on purpose: every stage prints with a flush so a hang is distinguishable from work.
Writes scratch/sophie/omission_prior.npz
"""
from __future__ import annotations

import sys
import time

import numpy as np
import torch

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


N = int(sys.argv[1]) if len(sys.argv) > 1 else 50000
BATCH = 16
THRS = (0.5, 0.7, 0.9)

log(f"start, target {N} images, batch {BATCH}")

from wdtag.baseline import MEAN, STD, load_baseline
from wdtag.labels import load_label_space
from wdtag.stream import TarStream

log("imports done, loading label space")
n_labels = len(load_label_space())
log(f"n_labels {n_labels}, loading bar")

device = "cuda"
model = load_baseline(device)
model.eval()
log(f"bar loaded, vram {torch.cuda.memory_allocated() / 2**30:.1f} GiB")

mean_t = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
std_t = torch.tensor(STD, device=device).view(1, 3, 1, 1)

log("opening stream")
st = TarStream("train", seed=1337, pass_index=0, batch_size=BATCH, n_labels=n_labels,
               workers=12, prefetch=8, max_samples=N)
log("stream open, consuming")

conf = {t: np.zeros(n_labels, dtype=np.int64) for t in THRS}
conf_unlab = {t: np.zeros(n_labels, dtype=np.int64) for t in THRS}
labelled = np.zeros(n_labels, dtype=np.int64)

seen = 0
nbatch = 0
last = time.time()
t1 = time.time()
with torch.no_grad():
    for item in st.batches():
        images, labels = item[0], item[1]
        x = images if torch.is_tensor(images) else torch.from_numpy(images)
        x = x.to(device, non_blocking=True)
        y = labels if torch.is_tensor(labels) else torch.from_numpy(labels)
        y = y.to(device) > 0.5
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(((x.float() - mean_t) / std_t).to(torch.bfloat16))
        probs = torch.sigmoid(logits.float())
        labelled += y.sum(0).cpu().numpy()
        for t in THRS:
            p = probs >= t
            conf[t] += p.sum(0).cpu().numpy()
            conf_unlab[t] += (p & ~y).sum(0).cpu().numpy()
        seen += len(x)
        nbatch += 1
        if nbatch <= 3:
            log(f"batch {nbatch} shape {tuple(x.shape)}")
        if time.time() - last > 5:
            last = time.time()
            rate = seen / max(1e-9, time.time() - t1)
            log(f"{seen}/{N} images, {rate:.0f} img/s, eta {(N - seen) / max(rate, 1e-9):.0f}s")

out = {"n_images": seen, "labelled": labelled}
for t in THRS:
    key = f"{t:.1f}"
    out[f"conf_{key}"] = conf[t]
    out[f"conf_unlab_{key}"] = conf_unlab[t]
np.savez(r"scratch\sophie\omission_prior.npz", **out)
log(f"saved npz after {seen} images")

for t in THRS:
    c, cu = conf[t].sum(), conf_unlab[t].sum()
    log(f"thr {t:.1f}: confident cells {c}, unlabelled {cu} "
        f"= {cu / max(1, c):.3f} omission rate on confident cells")


