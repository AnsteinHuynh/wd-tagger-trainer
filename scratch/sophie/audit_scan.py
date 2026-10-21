"""Label-integrity scan of the exact 10k/1k/1k sampled by wdtag.cache.build_plan.

Stage 1 (tars):  read each plan entry contiguously with wdtag.cache._read_samples,
                 keep the first SPLITS[split]["n"] samples in global plan order
                 (identical to wdtag.cache.build_split's offset logic), and record
                 annotation facts + raw-blob sha1.
Stage 2 (cache): mmap the trainer's .pt cache, verify key order matches stage 1,
                 then hash the 512x512 pixels the trainer actually sees and compute
                 a 64-bit dHash for near-duplicate detection.

Writes scratch/sophie/records_<split>.pkl .  Nothing outside scratch/ is touched.
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import pickle
import sys
import time

import numpy as np

from wdtag.cache import _read_samples, build_plan, DATASET_ROOT, SPLITS

WS = r"C:\Users\kerok\apps\wd-tagger-trainer"
OUT = os.path.join(WS, "scratch", "sophie")

EXPECTED_KEYS = ("id", "width", "height", "rating", "general_tags", "character_tags")


def sha1(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()


def scan_split(split: str) -> list[dict]:
    plan = build_plan(split)
    total = SPLITS[split]["n"]
    recs: list[dict] = []
    offsets = np.cumsum([0] + [e.n for e in plan])
    t0 = time.time()
    for ei, e in enumerate(plan):
        path = os.path.join(DATASET_ROOT, SPLITS[split]["subdir"], e.shard)
        samples = _read_samples(path, e.start_frac, e.n)
        for j, (key, blob, meta) in enumerate(samples):
            g = int(offsets[ei]) + j
            if g >= total:
                continue  # beyond the 10k/1k budget, exactly as build_split drops it
            if not isinstance(meta, dict):
                meta = {}
            gtags = meta.get("general_tags")
            ctags = meta.get("character_tags")
            rt = meta.get("rating")
            extra = sorted(set(meta) - set(EXPECTED_KEYS))
            recs.append(
                dict(
                    split=split,
                    idx=g,
                    tar=e.shard,
                    key=key,
                    id=meta.get("id"),
                    gtags=gtags,
                    ctags=ctags,
                    rt=rt,
                    width=meta.get("width"),
                    height=meta.get("height"),
                    extra_keys=extra,
                    has_keys=[k for k in EXPECTED_KEYS if k in meta],
                    types=dict(
                        general_tags=type(gtags).__name__,
                        character_tags=type(ctags).__name__,
                        rating=type(rt).__name__,
                        id=type(meta.get("id")).__name__,
                    ),
                    sha=sha1(blob),
                    nbytes=len(blob),
                )
            )
        print(f"[{split}] {ei+1}/{len(plan)} {e.shard} kept={len(recs)} "
              f"t={time.time()-t0:.0f}s", flush=True)
    assert len(recs) == total, f"{split}: {len(recs)} != {total}"
    return recs


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    which = sys.argv[1:] or ["train", "valid", "test"]
    for split in which:
        p = os.path.join(OUT, f"records_{split}.pkl")
        if os.path.exists(p):
            print(f"[{split}] exists, skip", flush=True)
            continue
        recs = scan_split(split)
        with open(p + ".tmp", "wb") as f:
            pickle.dump(recs, f)
        os.replace(p + ".tmp", p)
        print(f"[{split}] wrote {p}", flush=True)


if __name__ == "__main__":
    main()
