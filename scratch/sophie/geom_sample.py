"""Stage 1: contiguous-only sampling of train/test tars. CPU + disk only.

Writes JSONL of {group,row,key,shard,w,h,meta_w,meta_h,general,character} to
scratch/sophie/geom_*.jsonl, plus pickled shard list.

Groups:
  train_cache : EXACT cache plan (build_plan('train')) -- 213 shards x 47 images,
                global row = the cache's image index (what the trainer sees).
  train_broad : 400 images from each of 40 shards spread over all 213.
  test_cache  : EXACT cache plan (build_plan('test')) truncated to 1000 rows,
                global row = cache image index (must match prediction order).
"""
import io, json, os, sys, time
import numpy as np
from PIL import Image
from wdtag.cache import build_plan, _read_samples, DATASET_ROOT, SPLITS

OUT = os.path.dirname(os.path.abspath(__file__))

def shard_path(split, shard):
    return os.path.join(DATASET_ROOT, SPLITS[split]["subdir"], shard)

def header_size(blob):
    with Image.open(io.BytesIO(blob)) as im:
        return im.size  # (w, h), header only

def dump(group, records, path):
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
            n += 1
    print(f"[{group}] wrote {n} rows -> {path}", flush=True)

def rec(group, row, key, shard, sz, meta):
    w, h = sz
    return dict(
        group=group, row=row, key=key, shard=shard, w=int(w), h=int(h),
        meta_w=meta.get("width"), meta_h=meta.get("height"),
        general=meta.get("general_tags", []), character=meta.get("character_tags", []),
        rating=meta.get("rating", []),
    )

t0 = time.time()

# ---------------- train_cache: exact plan the trainer uses ----------------
plan = build_plan("train")
recs = []
row = 0
for ei, e in enumerate(plan):
    samples = _read_samples(shard_path("train", e.shard), e.start_frac, e.n)
    for j, (key, blob, meta) in enumerate(samples):
        if row >= 10_000:
            break
        recs.append(rec("train_cache", row, key, e.shard, header_size(blob), meta))
        row += 1
    if row >= 10_000:
        break
    if (ei + 1) % 50 == 0:
        print(f"  train_cache {ei+1}/{len(plan)} entries, {row} rows, {time.time()-t0:.1f}s", flush=True)
dump("train_cache", recs, os.path.join(OUT, "geom_train_cache.jsonl"))
print(f"train_cache plan entries used, {row} rows, {time.time()-t0:.1f}s", flush=True)

# ---------------- train_broad: 400 x 40 shards ----------------
idxs = np.unique(np.linspace(0, len(plan) - 1, 100).round().astype(int))
recs = []
row = 0
for ei in idxs:
    e = plan[ei]
    samples = _read_samples(shard_path("train", e.shard), e.start_frac, 400)
    for key, blob, meta in samples:
        recs.append(rec("train_broad", row, key, e.shard, header_size(blob), meta))
        row += 1
dump("train_broad", recs, os.path.join(OUT, "geom_train_broad.jsonl"))
print(f"train_broad: {len(idxs)} shards {[plan[i].shard for i in idxs][:5]}... {row} rows, {time.time()-t0:.1f}s", flush=True)

# ---------------- test_cache: exact plan, first 1000 rows ----------------
tplan = build_plan("test")
recs = []
row = 0
for e in tplan:
    samples = _read_samples(shard_path("test", e.shard), e.start_frac, e.n)
    for key, blob, meta in samples:
        if row >= 1000:
            break
        recs.append(rec("test_cache", row, key, e.shard, header_size(blob), meta))
        row += 1
    if row >= 1000:
        break
dump("test_cache", recs, os.path.join(OUT, "geom_test_cache.jsonl"))
print(f"done in {time.time()-t0:.1f}s", flush=True)
