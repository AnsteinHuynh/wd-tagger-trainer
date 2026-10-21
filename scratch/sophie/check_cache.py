import time
import numpy as np
from wdtag.data import CacheDataset, load_targets
from wdtag.labels import load_label_space

space = load_label_space()
t0 = time.time()
for split in ("train", "valid", "test"):
    for mode in ("pad", "squash"):
        t = time.time()
        ds = CacheDataset(split, mode, len(space))
        img, lab = ds[0]
        print(f"{split}/{mode}: {len(ds)} imgs, open {time.time()-t:.2f}s, "
              f"img {tuple(img.shape)} {img.dtype}, labels {int(lab.sum())}, "
              f"meta {ds.meta(0)}")
        del ds
print("total %.1fs" % (time.time() - t0))

t0 = time.time()
tgt = load_targets("test", "pad", len(space))
print("test targets:", tgt.shape, "positives/image avg %.1f" % (tgt.sum(1).mean()),
      "labels with support %d" % tgt.any(0).sum(), "%.1fs" % (time.time() - t0))
