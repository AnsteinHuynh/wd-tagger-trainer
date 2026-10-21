import time, torch, numpy as np
from wdtag.data import CacheDataset, label_priors

ds = CacheDataset("train", "pad", 12476)
print("dataset", len(ds), flush=True)

# warm the mapping
_ = ds[0]

# 1) current path: per-item __getitem__ + stack
idx = torch.randperm(len(ds))[:32 * 12].tolist()
t0 = time.perf_counter()
for i in range(0, 32 * 12, 32):
    batch = torch.stack([ds[j][0] for j in idx[i:i + 32]])
t1 = time.perf_counter()
per_img = (t1 - t0) / (32 * 12)
print(f"current path: {per_img*1000:.2f} ms/img -> {1/per_img:.0f} img/s CPU-only", flush=True)

# 2) contiguous slice + dense label matrix
L = np.zeros((len(ds), 12476), dtype=np.uint8)
t0 = time.perf_counter()
for i in range(len(ds)):
    k, j = ds.index[i]
    s = ds.shards[k]
    lo, hi = int(s["ptr"][j]), int(s["ptr"][j + 1])
    L[i, s["pos"][lo:hi].numpy()] = 1
print("dense label matrix built in %.2fs, %.0f MB" % (time.perf_counter() - t0, L.nbytes / 1e6), flush=True)

starts = [i for i in range(0, 32 * 12, 32)]
t0 = time.perf_counter()
for st in starts:
    k, j = ds.index[st]
    s = ds.shards[k]
    imgs = s["images"][j:j + 32]
    labs = torch.from_numpy(L[st:st + 32])
t1 = time.perf_counter()
per_img = (t1 - t0) / (32 * 12)
print(f"contiguous slice: {per_img*1000:.2f} ms/img -> {1/per_img:.0f} img/s CPU-only", flush=True)

# 3) random gather from shard (no contiguity)
t0 = time.perf_counter()
for i in range(0, 32 * 12, 32):
    sel = idx[i:i + 32]
    ks = [ds.index[j] for j in sel]
    imgs = torch.stack([ds.shards[k]["images"][jj] for k, jj in ks])
t1 = time.perf_counter()
per_img = (t1 - t0) / (32 * 12)
print(f"gather w/ list idx: {per_img*1000:.2f} ms/img -> {1/per_img:.0f} img/s CPU-only", flush=True)
