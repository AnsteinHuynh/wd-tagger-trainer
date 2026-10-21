"""Recon: metadata keys, tag vocab, image sizes/formats. CPU + disk only."""
import io, json, os, sys, time, collections
import numpy as np
from PIL import Image
from wdtag.cache import build_plan, _read_samples, DATASET_ROOT, SPLITS

def shard_path(split, shard):
    return os.path.join(DATASET_ROOT, SPLITS[split]["subdir"], shard)

plan = build_plan("train")
print("train plan entries:", len(plan), "per_pos n:", plan[0].n, "total plan imgs:", sum(e.n for e in plan))
tplan = build_plan("test")
print("test plan entries:", len(tplan), "per_pos n:", tplan[0].n, "total:", sum(e.n for e in tplan))

e = plan[0]
t0 = time.time()
samples = _read_samples(shard_path("train", e.shard), e.start_frac, 400)
dt = time.time() - t0
print(f"read 400 samples from {e.shard} in {dt:.1f}s ({dt/400*1000:.1f} ms/img), avg blob {np.mean([len(s[1]) for s in samples])/1024:.0f} KiB")
key, blob, meta = samples[0]
print("key:", key)
print("meta keys:", sorted(meta.keys()))
print("meta id:", meta.get("id"))
print("num general:", len(meta.get("general_tags", [])), "num char:", len(meta.get("character_tags", [])))
print("rating:", meta.get("rating"))
print("first 25 general tags:", meta["general_tags"][:25])
# any width/height fields?
for k, v in meta.items():
    if not isinstance(v, list):
        print("  scalar:", k, "=", repr(v)[:120])
    else:
        print("  list:", k, "len", len(v), "sample", repr(v[:3])[:120])

im = Image.open(io.BytesIO(blob))
print("img format:", im.format, "mode:", im.mode, "size:", im.size)

sizes = collections.Counter()
fmts = collections.Counter()
for k, b, m in samples:
    with Image.open(io.BytesIO(b)) as i:
        sizes[i.size] += 1
        fmts[i.format] += 1
print("formats:", fmts.most_common())
print("distinct sizes among 400:", len(sizes))
for s, c in sizes.most_common(15):
    print("  ", s, c)
