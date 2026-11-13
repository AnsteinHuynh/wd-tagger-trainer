"""Sampling + building of the 512x512 image caches.

Layout (temporary, rebuilt when absent): `I:/.cache/wd-tagger-trainer/{train,valid,test}/squash/shard#####-{total}.pt`.

Bilinear resize, aspect ratio not preserved ("squash"); padded-square tied it statistically once adapter LR was right (the private experiment log), so only squash is built. Shard ~0.5 GB (625 x 3x512x512 uint8) of u8 pixels + sparse CSR labels; `.tmp` then atomic rename, so an interrupted build resumes at the first missing/invalid shard.
"""
from __future__ import annotations

import io
import json
import os
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from .labels import LabelSpace, load_label_space, tag_indices

# Machine-local roots come from the environment so nothing user-specific ships in the file.
CACHE_ROOT = os.environ.get("WDTAG_CACHE_ROOT", os.path.join(".cache", "wd-tagger-trainer"))
DATASET_ROOT = os.environ.get(
    "WDTAG_DATASET_ROOT",
    os.path.join("datasets", "animetimm-danbooru-wdtagger-v4-w640-ws-full", "repo"),
)
# Resolution is read at import so every module that does `from .cache import SIZE` binds the
# same value. It must match the run's `image_size`; train.py refuses to start otherwise.
# The shard path carries no resolution, so a second size would silently overwrite the first
# one's 100.3 GB: 512 keeps its tree and every other size gets its own.
SIZE = int(os.environ.get("WDTAG_SIZE", "512"))
if SIZE != 512:
    CACHE_ROOT = CACHE_ROOT + f"-{SIZE}"
IMAGES_PER_SHARD = 625  # 625 * 3*512*512 B = 0.49 GB; at 384 the same shard is 0.28 GB
MODE = "squash"         # default geometry; "pad" is an experiment-only alternative

# split -> (subdir, n_images, positions, per-position count)
# Sampling is contiguous from each tar's start: mid-tar seeking costs an HDD seek per
# member (~0.4 ms) and bottlenecks a busy disk, while the first few MB are sequential.
# Shard order is shuffled, so starts are already spread out.
SPLITS = {
    "train": dict(subdir="train", n=10_000, positions=(0.0,), per_pos=None),
    "valid": dict(subdir="val", n=1_000, positions=(0.0,), per_pos=None),
    "test": dict(subdir="test", n=1_000, positions=(0.0,), per_pos=None),
}


@dataclass(frozen=True)
class PlanEntry:
    shard: str        # tar file name
    start_frac: float  # position in the tar (0..1 of bytes) to start collecting
    n: int            # images to take


def build_plan(split: str) -> list[PlanEntry]:
    """Deterministic plan: header seeks, no full scans."""
    spec = SPLITS[split]
    shards = sorted(
        f for f in os.listdir(os.path.join(DATASET_ROOT, spec["subdir"])) if f.endswith(".tar")
    )
    n_total, positions = spec["n"], spec["positions"]
    per_pos = (n_total + len(shards) * len(positions) - 1) // (len(shards) * len(positions))
    plan = []
    for s in shards:
        for p in positions:
            plan.append(PlanEntry(shard=s, start_frac=p, n=per_pos))
    return plan


def _read_samples(tar_path: str, start_frac: float, n: int) -> list[tuple[str, bytes, dict]]:
    """`n` consecutive (image, json) pairs from `start_frac`."""
    size = os.path.getsize(tar_path)
    out: list[tuple[str, bytes, dict]] = []
    pending: dict[str, bytes] = {}
    with tarfile.open(tar_path, "r:") as tf:
        for m in tf:
            if start_frac > 0 and tf.offset < start_frac * size:
                continue  # seek forward over members; tarfile skips the payload
            if not m.isfile():
                continue
            stem, ext = os.path.splitext(os.path.basename(m.name))
            if ext == ".webp" or ext == ".jpg" or ext == ".png":
                pending[stem] = tf.extractfile(m).read()
            elif ext == ".json" and stem in pending:
                meta = json.loads(tf.extractfile(m).read().decode("utf-8"))
                out.append((stem, pending.pop(stem), meta))
                if len(out) >= n:
                    break
    if len(out) < n:
        raise RuntimeError(f"{tar_path}: wanted {n} samples at frac {start_frac}, got {len(out)}")
    return out


def _decode(blob: bytes) -> np.ndarray:
    im = Image.open(io.BytesIO(blob))
    if im.mode != "RGB":
        im = im.convert("RGB")
    # bilinear resize; PIL's antialias default differs between versions, be explicit
    im = im.resize((SIZE, SIZE), Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


def _shard_name(split: str, mode: str, idx: int, total: int) -> str:
    return os.path.join(CACHE_ROOT, split, mode, f"shard{idx:05d}-{total}.pt")


def _validate(path: str, expect_n: int) -> bool:
    try:
        obj = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    except Exception:
        return False
    ok = (
        isinstance(obj, dict)
        and obj["images"].shape == (expect_n, 3, SIZE, SIZE)
        and obj["images"].dtype == torch.uint8
        and obj["ptr"].numel() == expect_n + 1
    )
    return ok


def build_split(split: str, mode: str = MODE, workers: int = 12, log=None) -> None:
    """Build missing shards for `split`; idempotent, resumable."""
    if log is None:
        log = lambda *a, **k: print(*a, **k, flush=True)
    t_start = time.time()
    space = load_label_space()
    plan = build_plan(split)
    total = SPLITS[split]["n"]
    n_shards = (total + IMAGES_PER_SHARD - 1) // IMAGES_PER_SHARD

    # global image offsets of each plan entry
    offsets = np.cumsum([0] + [e.n for e in plan])

    d = os.path.join(CACHE_ROOT, split, mode)
    os.makedirs(d, exist_ok=True)
    for f in os.listdir(d):
        if f.endswith(".tmp"):
            os.remove(os.path.join(d, f))

    todo = []
    for k in range(n_shards):
        count = min(IMAGES_PER_SHARD, total - k * IMAGES_PER_SHARD)
        path = _shard_name(split, mode, k, total)
        if os.path.exists(path) and _validate(path, count):
            continue
        todo.append((k, count, path))
    if not todo:
        log(f"[cache:{split}] all {n_shards} shards present, nothing to do")
        return
    log(f"[cache:{split}] building {len(todo)}/{n_shards} shards "
        f"({total} images, {len(plan)} plan entries)")

    pool = ThreadPoolExecutor(max_workers=workers)
    for k, count, path in todo:
        buf = np.empty((count, SIZE, SIZE, 3), dtype=np.uint8)
        pos_lists: list[list[int]] = []
        ids: list[int] = []
        keys: list[str] = []
        srcs: list[str] = []
        lo, hi = k * IMAGES_PER_SHARD, k * IMAGES_PER_SHARD + count

        # which plan entries overlap this output shard
        entries = [i for i in range(len(plan)) if offsets[i] < hi and offsets[i + 1] > lo]
        t0 = time.time()
        for ei in entries:
            e = plan[ei]
            samples = _read_samples(os.path.join(DATASET_ROOT, SPLITS[split]["subdir"], e.shard),
                                    e.start_frac, e.n)
            arrays = list(pool.map(lambda s: _decode(s[1]), samples))
            for j, (key, _blob, meta) in enumerate(samples):
                g = offsets[ei] + j
                if g < lo or g >= hi:
                    continue
                local = g - lo
                buf[local] = arrays[j]
                idx, _unknown = tag_indices(meta, space)
                pos_lists.append(idx)
                ids.append(int(meta["id"]))
                keys.append(key)
                srcs.append(e.shard)
            del arrays

        assert len(keys) == count, f"shard {k}: got {len(keys)} images, expected {count}"
        # NHWC -> NCHW without a copy of the whole buffer
        imgs = torch.from_numpy(buf.transpose(0, 3, 1, 2)).contiguous()
        ptr = torch.tensor([0] + list(np.cumsum([len(p) for p in pos_lists])), dtype=torch.int64)
        pos = torch.tensor([i for p in pos_lists for i in p], dtype=torch.int64)
        tmp = path + ".tmp"
        torch.save(
            {"images": imgs, "pos": pos, "ptr": ptr,
             "ids": torch.tensor(ids, dtype=torch.int64), "keys": keys, "tars": srcs,
             "split": split, "mode": mode},
            tmp,
        )
        os.replace(tmp, path)
        log(f"[cache:{split}] shard {k+1}/{n_shards} written ({count} imgs, "
            f"{time.time()-t0:.1f}s, {time.time()-t_start:.0f}s total)")
    pool.shutdown(wait=True)


def cache_ready(split: str, mode: str) -> bool:
    total = SPLITS[split]["n"]
    n_shards = (total + IMAGES_PER_SHARD - 1) // IMAGES_PER_SHARD
    for k in range(n_shards):
        count = min(IMAGES_PER_SHARD, total - k * IMAGES_PER_SHARD)
        if not _validate(_shard_name(split, mode, k, total), count):
            return False
    return True


def ensure_cache(split: str, mode: str = MODE, workers: int = 12, log=print) -> None:
    build_split(split, mode=mode, workers=workers, log=log)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="all", choices=["all", "train", "valid", "test"])
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    splits = ["valid", "test", "train"] if a.split == "all" else [a.split]
    for s in splits:
        ensure_cache(s, workers=a.workers)
    print("done")

