"""Build a *padded-square* cache, as a one-off experiment.

The main pipeline is squash-only on purpose. This script exists solely to re-run
the padding arm at the corrected learning rate, so the comparison that deleted
the arm in the first place can be checked at the operating point we actually
ship. It writes the same shard format to <cache>/<split>/pad/.

    python scratch/sophie/build_pad_cache.py --split all
"""
import argparse
import io
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image

from wdtag.cache import (_read_samples, _shard_name, _validate, build_plan,
                         CACHE_ROOT, DATASET_ROOT, IMAGES_PER_SHARD, SIZE, SPLITS)
from wdtag.labels import load_label_space, tag_indices

PAD = "pad"


def decode_pad(blob: bytes) -> np.ndarray:
    im = Image.open(io.BytesIO(blob))
    if im.mode != "RGB":
        im = im.convert("RGB")
    w, h = im.size
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    canvas.paste(im, ((side - w) // 2, (side - h) // 2))
    return np.asarray(canvas.resize((SIZE, SIZE), Image.BILINEAR), dtype=np.uint8)


def build(split: str, workers: int = 12) -> None:
    t0 = time.time()
    space = load_label_space()
    plan = build_plan(split)
    total = SPLITS[split]["n"]
    n_shards = (total + IMAGES_PER_SHARD - 1) // IMAGES_PER_SHARD
    offsets = np.cumsum([0] + [e.n for e in plan])
    d = os.path.join(CACHE_ROOT, split, PAD)
    os.makedirs(d, exist_ok=True)
    for f in os.listdir(d):
        if f.endswith(".tmp"):
            os.remove(os.path.join(d, f))

    pool = ThreadPoolExecutor(max_workers=workers)
    for k in range(n_shards):
        count = min(IMAGES_PER_SHARD, total - k * IMAGES_PER_SHARD)
        path = _shard_name(split, PAD, k, total)
        if os.path.exists(path) and _validate(path, count):
            print(f"[pad:{split}] shard {k+1}/{n_shards} present", flush=True)
            continue
        buf = np.empty((count, SIZE, SIZE, 3), dtype=np.uint8)
        pos_lists, ids, keys, srcs = [], [], [], []
        lo, hi = k * IMAGES_PER_SHARD, k * IMAGES_PER_SHARD + count
        entries = [i for i in range(len(plan)) if offsets[i] < hi and offsets[i + 1] > lo]
        tk = time.time()
        for ei in entries:
            e = plan[ei]
            samples = _read_samples(os.path.join(DATASET_ROOT, SPLITS[split]["subdir"], e.shard),
                                    e.start_frac, e.n)
            arrays = list(pool.map(lambda s: decode_pad(s[1]), samples))
            for j, (key, _blob, meta) in enumerate(samples):
                g = offsets[ei] + j
                if g < lo or g >= hi:
                    continue
                buf[g - lo] = arrays[j]
                idx, _ = tag_indices(meta, space)
                pos_lists.append(idx)
                ids.append(int(meta["id"]))
                keys.append(key)
                srcs.append(e.shard)
            del arrays
        assert len(keys) == count, f"shard {k}: {len(keys)} != {count}"
        imgs = torch.from_numpy(buf.transpose(0, 3, 1, 2)).contiguous()
        ptr = torch.tensor([0] + list(np.cumsum([len(p) for p in pos_lists])), dtype=torch.int64)
        pos = torch.tensor([i for p in pos_lists for i in p], dtype=torch.int64)
        tmp = path + ".tmp"
        torch.save({"images": imgs, "pos": pos, "ptr": ptr,
                    "ids": torch.tensor(ids, dtype=torch.int64), "keys": keys, "tars": srcs,
                    "split": split, "mode": PAD}, tmp)
        os.replace(tmp, path)
        print(f"[pad:{split}] shard {k+1}/{n_shards} written ({count} imgs, "
              f"{time.time()-tk:.1f}s, {time.time()-t0:.0f}s total)", flush=True)
    pool.shutdown(wait=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="all")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    for s in (["valid", "test", "train"] if a.split == "all" else [a.split]):
        build(s, a.workers)
    print("pad cache done")
