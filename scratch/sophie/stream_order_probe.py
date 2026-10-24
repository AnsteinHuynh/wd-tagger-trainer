import hashlib, sys
from wdtag.stream import TarStream
sp = TarStream("train", seed=1337, pass_index=0, batch_size=4, n_labels=12476,
               workers=8, prefetch=4, skip=0, max_samples=10**9, log=lambda m: None)
h = hashlib.sha256()
n = 0
for x, y, consumed, src in sp.batches():
    h.update(str(tuple(src)).encode())
    h.update(x.numpy().tobytes())
    n += 1
    if n == 250:
        break
print("batches", n, "sha", h.hexdigest()[:16])

