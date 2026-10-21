import time, collections
from wdtag.cache import _read_samples, _decode, build_plan, DATASET_ROOT, SPLITS
from wdtag.labels import load_label_space, tag_indices
import os

sp = load_label_space()
print("labels:", len(sp), "rating:", sp.n_rating)

t0 = time.time()
s = _read_samples(os.path.join(DATASET_ROOT, "val", "00001.tar"), 0.0, 20)
print("read 20 @0.0: %.2fs" % (time.time() - t0))
t0 = time.time()
s2 = _read_samples(os.path.join(DATASET_ROOT, "val", "00001.tar"), 0.5, 20)
print("read 20 @0.5: %.2fs" % (time.time() - t0), "first key @0.5:", s2[0][0], "first key @0.0:", s[0][0])

unk = collections.Counter()
npos = []
for key, blob, meta in s:
    idx, unknown = tag_indices(meta, sp)
    unk.update(unknown)
    npos.append(len(idx))
print("positives per image: min %d avg %.1f max %d" % (min(npos), sum(npos) / len(npos), max(npos)))
print("unknown tag names:", unk.most_common(10), "total", sum(unk.values()))

t0 = time.time()
a = _decode(s[0][1], "pad"); b = _decode(s[0][1], "squash")
print("decode both arms: %.3fs" % (time.time() - t0), a.shape, a.dtype, b.shape)

plan = build_plan("valid")
print("valid plan: %d entries x %d imgs = %d" % (len(plan), plan[0].n, sum(e.n for e in plan)))
plan = build_plan("train")
print("train plan: %d entries x %d imgs = %d" % (len(plan), plan[0].n, sum(e.n for e in plan)))
plan = build_plan("test")
print("test plan: %d entries x %d imgs = %d" % (len(plan), plan[0].n, sum(e.n for e in plan)))
