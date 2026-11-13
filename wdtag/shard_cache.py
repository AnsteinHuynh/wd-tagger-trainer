"""Cut a monolithic feature cache into one shard per tar, verified byte for byte.

The cache is 31 GB of rows in the order the streamer wrote them. That order is a pure function of
(seed, pass_index) over the tar manifest, so every row range belongs to a known tar and a different
order is a different list of the same ranges. Splitting it per tar makes the order explicit: training
on tar 22 then 15 then 12 is three files to read in that order, and seed 2 is a different permutation
of the same files.

Each shard is self-contained, so it can be read, trained on, or thrown away alone:

    tar_00019.npy        25000 x 3072 fp16, memmap-able
    tar_00019.npz        indices int32, offsets int64 relative to the shard
    shards.json          what exists, in what order, with the blake2b of each features file

shards.json is rewritten after every shard, so a half-finished split is usable and resumable rather
than a pile of untrusted files. Every shard is hashed against the source slice as it is written, and
the source is only deleted when `--delete-source` is passed and every shard matched.

Usage:
  python -m wdtag.shard_cache --cache ./cache/.../siglip2-cmss-train --out ./cache/.../siglip2-cmss-shards
  python -m wdtag.shard_cache --check ./cache/.../siglip2-cmss-shards
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np


def _hash(arr: np.ndarray) -> str:
    h = hashlib.blake2b(digest_size=16)
    h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def _hash_file(path: str, dtype, shape) -> str:
    a = np.load(path, mmap_mode="r")
    if tuple(a.shape) != tuple(shape) or a.dtype != dtype:
        raise SystemExit(f"{path} is {a.shape} {a.dtype}, expected {shape} {dtype}")
    h = hashlib.blake2b(digest_size=16)
    for s in range(0, a.shape[0], 8192):
        h.update(np.ascontiguousarray(a[s:s + 8192]).tobytes())
    return h.hexdigest()


def split(args) -> int:
    from .manifest import ranges

    cache = args.cache
    meta = json.load(open(os.path.join(cache, "meta.json")))
    seed, pid = int(meta["seed"]), int(meta.get("pass_index", 0))
    written = int(meta["written"])
    split_name = meta["split"]
    dim = int(meta["dim"])
    feats = np.load(os.path.join(cache, "feats.npy"), mmap_mode="r")
    dtype = feats.dtype

    off_path = os.path.join(cache, "labels_offsets.i64")
    idx_path = os.path.join(cache, "labels_indices.i32")
    if not (os.path.isfile(off_path) and os.path.isfile(idx_path)):
        raise SystemExit("this cache has no append-only labels; re-cache with the current writer")
    offs = np.fromfile(off_path, dtype=np.int64)[:written]
    idx = np.fromfile(idx_path, dtype=np.int32)[:int(offs[-1])]

    os.makedirs(args.out, exist_ok=True)
    manifest_path = os.path.join(args.out, "shards.json")
    done = {}
    if os.path.isfile(manifest_path):
        done = {s["tar"]: s for s in json.load(open(manifest_path)).get("shards", [])}
        print(f"resuming a split with {len(done)} shards already recorded", flush=True)

    order = ranges(split_name, seed, pid)
    covered = 0
    t0 = time.time()
    for name, start, rows in order:
        if start + rows > written:
            print(f"stopping: {name} starts at {start} but the cache holds {written} rows", flush=True)
            break
        stem = os.path.splitext(name)[0]
        fpath = os.path.join(args.out, f"{stem}.npy")
        lpath = os.path.join(args.out, f"{stem}.npz")
        prev = done.get(name)
        if prev and os.path.isfile(fpath) and prev.get("rows") == rows:
            covered = max(covered, start + rows)
            continue

        block = np.asarray(feats[start:start + rows])
        digest = _hash(block)
        np.save(fpath, block)
        got = _hash_file(fpath, dtype, (rows, dim))
        if got != digest:
            raise SystemExit(f"{fpath} does not match the source slice; stopping before anything is deleted")

        # offs[i] is the END of row i, so the start of a range is the PREVIOUS end, and row 0 needs a
        # leading zero. Without it the shard's labels are shifted one row, which trains fine and is
        # silently wrong, and the label totals were what exposed it.
        full = np.concatenate([[0], offs])
        s, e = int(full[start]), int(full[start + rows])
        np.savez(lpath, indices=idx[s:e], offsets=full[start:start + rows + 1] - full[start])
        done[name] = {"tar": name, "feats": f"{stem}.npy", "labels": f"{stem}.npz",
                      "rows": rows, "source_start": start, "labels_total": int(e - s),
                      "blake2b": digest}
        json.dump({"schema": 1, "dim": dim, "dtype": str(dtype), "split": split_name,
                   "seed": seed, "pass_index": pid, "source": cache,
                   "shards": [done[n] for n, _, _ in order if n in done]},
                  open(manifest_path, "w"), indent=1)
        covered = start + rows
        rate = covered / max(1e-9, time.time() - t0)
        print(f"{name}: rows {start}..{start + rows} ({rows}), {e - s} labels, "
              f"{rate / 1e6:.1f} M rows/s", flush=True)

    label_total = sum(s["labels_total"] for s in done.values())
    if len(done) == len(order):
        if label_total != int(offs[-1]):
            raise SystemExit(f"label totals disagree: shards {label_total}, source {int(offs[-1])}")
    total = sum(s["rows"] for s in done.values())
    print(f"shards {len(done)}, {total} rows of {written}, dim {dim}, {dtype}")
    if args.delete_source and total == written:
        if os.path.dirname(os.path.abspath(args.out)) != os.path.dirname(os.path.abspath(cache)):
            print("refusing to delete a source in a different directory", file=sys.stderr)
            return 1
        for p in ("feats.npy", "labels_indices.i32", "labels_offsets.i64"):
            os.remove(os.path.join(cache, p))
        print(f"deleted the monolithic copy in {cache}; the shards carry the same bytes")
    elif args.delete_source:
        print("source kept: the split does not yet cover every row")
    return 0


def check(out: str) -> int:
    manifest_path = os.path.join(out, "shards.json")
    man = json.load(open(manifest_path))
    bad = 0
    total = 0
    for s in man["shards"]:
        got = _hash_file(os.path.join(out, s["feats"]), np.dtype(man["dtype"]),
                         (s["rows"], man["dim"]))
        ok = got == s["blake2b"]
        total += s["rows"]
        if not ok:
            bad += 1
            print(f"MISMATCH {s['tar']}")
    print(f"checked {len(man['shards'])} shards, {total} rows, {bad} mismatched")
    return 0 if bad == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="split a feature cache into per-tar shards")
    ap.add_argument("--cache", help="directory holding feats.npy, labels_*.i*, meta.json")
    ap.add_argument("--out", help="shard directory")
    ap.add_argument("--check", metavar="DIR", help="re-hash an existing shard directory")
    ap.add_argument("--delete-source", action="store_true",
                    help="remove the monolithic copy once every shard has been verified")
    args = ap.parse_args()
    if args.check:
        return check(args.check)
    if not args.cache or not args.out:
        ap.error("--cache and --out are required unless --check is used")
    return split(args)


if __name__ == "__main__":
    raise SystemExit(main())


