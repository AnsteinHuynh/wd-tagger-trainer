"""Stage 2: cross-check the trainer's .pt cache against stage-1 tar records and
hash the pixels the trainer actually sees (512x512 squash uint8).

Adds to each split: pix_sha (sha1 of the 512x512x3 uint8 buffer) and dhash (uint64).
"""
from __future__ import annotations

import hashlib
import os
import pickle
import time

import numpy as np
from PIL import Image

from wdtag.cache import SPLITS, _shard_name, IMAGES_PER_SHARD

WS = r"C:\Users\kerok\apps\wd-tagger-trainer"
OUT = os.path.join(WS, "scratch", "sophie")
CACHE_ROOT = r"I:\.cache\wd-tagger-trainer"


def dhash64(arr_chw: np.ndarray) -> int:
    im = Image.fromarray(arr_chw.transpose(1, 2, 0)).convert("L").resize((9, 8), Image.BILINEAR)
    a = np.asarray(im, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()  # 64 bools
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v


def main() -> None:
    for split in ("train", "valid", "test"):
        rp = os.path.join(OUT, f"records_{split}.pkl")
        pp = os.path.join(OUT, f"pixels_{split}.pkl")
        if os.path.exists(pp):
            print(split, "pixels exist, skip", flush=True)
            continue
        with open(rp, "rb") as f:
            recs = pickle.load(f)
        total = SPLITS[split]["n"]
        n_shards = (total + IMAGES_PER_SHARD - 1) // IMAGES_PER_SHARD
        pix_sha: list[str] = []
        dsh: list[int] = []
        keys_cache: list[str] = []
        posn: list[int] = []
        t0 = time.time()
        for k in range(n_shards):
            cnt = min(IMAGES_PER_SHARD, total - k * IMAGES_PER_SHARD)
            p = _shard_name(split, "squash", k, total)
            s = __import__("torch").load(p, map_location="cpu", mmap=True, weights_only=False)
            assert s["images"].shape[0] == cnt, (split, k)
            keys_cache.extend(s["keys"])
            ptr, pos = s["ptr"].numpy(), s["pos"].numpy()
            posn.extend((ptr[1:] - ptr[:-1]).tolist())
            for j in range(cnt):
                a = s["images"][j].numpy()
                pix_sha.append(hashlib.sha1(a.tobytes()).hexdigest())
                dsh.append(dhash64(a))
            print(f"[{split}] cache shard {k+1}/{n_shards} t={time.time()-t0:.0f}s", flush=True)
        assert keys_cache == [r["key"] for r in recs], f"{split}: cache key order <> tar order"
        with open(pp + ".tmp", "wb") as f:
            pickle.dump(dict(pix_sha=pix_sha, dhash=dsh, cache_pos=posn,
                             key_match=True), f)
        os.replace(pp + ".tmp", pp)
        print(f"[{split}] pixel hashes written, keys identical to tar scan", flush=True)


if __name__ == "__main__":
    main()
