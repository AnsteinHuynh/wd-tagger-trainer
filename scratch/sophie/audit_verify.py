"""Verification of two surprising results.

1. OOV self-test: inject a fake tag and confirm tag_indices flags it; measure how
   much of the 12,476-name space the sample actually uses.
2. dHash near-duplicates: are the train<->test pairs at Hamming <= 4 really the
   same picture, or just similar composition? Second independent descriptor
   (63-bit DCT pHash) + pixel MAE on the 512x512 cache tensors.
"""
from __future__ import annotations

import collections
import hashlib
import itertools
import os
import pickle

import numpy as np
from PIL import Image

from wdtag.cache import IMAGES_PER_SHARD, SPLITS, _shard_name
from wdtag.labels import load_label_space, tag_indices

WS = r"C:\Users\kerok\apps\wd-tagger-trainer"
OUT = os.path.join(WS, "scratch", "sophie")
space = load_label_space()

DCT = np.cos(np.pi * (2 * np.arange(32)[:, None] + 1) * np.arange(32)[None, :] / 64.0) \
    * np.sqrt(2.0 / 32.0)
DCT[0] /= np.sqrt(2.0)


def phash63(a: np.ndarray) -> int:
    im = Image.fromarray(a.transpose(1, 2, 0)).convert("L").resize((32, 32), Image.BILINEAR)
    d = DCT @ np.asarray(im, dtype=np.float64) @ DCT.T
    low = d[:8, :8].ravel()[1:]
    mask = low > np.median(low)
    v = 0
    for b in mask:
        v = (v << 1) | int(b)
    return v


def gray128(a: np.ndarray) -> np.ndarray:
    im = Image.fromarray(a.transpose(1, 2, 0)).convert("L").resize((128, 128), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32)


def popcount(x: np.ndarray) -> np.ndarray:
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x).astype(np.int32)
    x = np.ascontiguousarray(x)
    return np.unpackbits(x.view(np.uint8), axis=-1).sum(axis=-1).astype(np.int32)


def load_split(split: str):
    with open(os.path.join(OUT, f"records_{split}.pkl"), "rb") as f:
        recs = pickle.load(f)
    with open(os.path.join(OUT, f"pixels_{split}.pkl"), "rb") as f:
        px = pickle.load(f)
    return recs, px


def cache_rows(split: str, idxs: list[int]):
    """Fetch 512x512 uint8 arrays for global indices of a split."""
    total = SPLITS[split]["n"]
    want = collections.defaultdict(list)
    for g in idxs:
        want[g // IMAGES_PER_SHARD].append(g)
    got = {}
    for k, gs in want.items():
        s = __import__("torch").load(_shard_name(split, "squash", k, total),
                                     map_location="cpu", mmap=True, weights_only=False)
        for g in gs:
            got[g] = s["images"][g % IMAGES_PER_SHARD].numpy()
    return got


# ---------------------------------------------------------------- OOV self-test
print("== OOV self-test ==", flush=True)
fake = {"general_tags": ["1girl", "totally_fake_tag_xyz"], "character_tags": ["fake_char_xyz"],
        "rating": ["sensitive"]}
idx, unk = tag_indices(fake, space)
print("tag_indices on injected meta ->", idx, unk, flush=True)
assert unk == ["totally_fake_tag_xyz", "fake_char_xyz"], "OOV detection is broken!"
assert idx == [1, 4], idx  # 'sensitive' -> 1, '1girl' -> 4
print("OOV detection works; label space index 1='sensitive', 4='1girl'", flush=True)

used = collections.Counter()
for split in SPLITS:
    recs, _ = load_split(split)
    for r in recs:
        for t in (r["gtags"] or []) + (r["ctags"] or []) + (r["rt"] or []):
            used[t] += 1
print(f"distinct tag names used across all 12,000 images: {len(used)}; "
      f"of them in space: {sum(1 for t in used if t in space.name2idx)}", flush=True)
print(f"space names never used anywhere in the sample: "
      f"{len(space.name2idx) - sum(1 for t in used if t in space.name2idx)}", flush=True)
print("top 5 by images:", used.most_common(5), flush=True)

# ------------------------------------------------------------ perceptual hashes
print("\n== pHash over cache tensors ==", flush=True)
for split in SPLITS:
    pp = os.path.join(OUT, f"phash_{split}.pkl")
    if os.path.exists(pp):
        continue
    recs, px = load_split(split)
    ph = []
    total = SPLITS[split]["n"]
    n_shards = (total + IMAGES_PER_SHARD - 1) // IMAGES_PER_SHARD
    for k in range(n_shards):
        cnt = min(IMAGES_PER_SHARD, total - k * IMAGES_PER_SHARD)
        s = __import__("torch").load(_shard_name(split, "squash", k, total),
                                     map_location="cpu", mmap=True, weights_only=False)
        for j in range(cnt):
            ph.append(phash63(s["images"][j].numpy()))
        print(f"  [{split}] {k+1}/{n_shards}", flush=True)
    assert len(ph) == total
    with open(pp + ".tmp", "wb") as f:
        pickle.dump(ph, f)
    os.replace(pp + ".tmp", pp)

dh, ph, recs = {}, {}, {}
for split in SPLITS:
    recs[split], px = load_split(split)
    dh[split] = np.array(px["dhash"], dtype=np.uint64)
    with open(os.path.join(OUT, f"phash_{split}.pkl"), "rb") as f:
        ph[split] = np.array(pickle.load(f), dtype=np.uint64)
    print(f"{split}: distinct dHash {len(set(dh[split].tolist()))}/{len(dh[split])}, "
          f"distinct pHash {len(set(ph[split].tolist()))}/{len(ph[split])}", flush=True)

# ------------------------------------------------------------ candidate pairs
print("\n== candidate near-dup pairs ==", flush=True)
for a, b in (("train", "test"), ("train", "valid"), ("test", "valid")):
    da, db = dh[a], dh[b]
    pa, pb = ph[a], ph[b]
    Dh = popcount(db[:, None] ^ da[None, :])
    ii, jj = np.where(Dh <= 6)
    if not len(ii):
        print(f"{a}|{b}: no dHash candidates", flush=True)
        continue
    ia, ib = [int(x) for x in jj], [int(x) for x in ii]
    imgs_a = cache_rows(a, sorted(set(ia)))
    imgs_b = cache_rows(b, sorted(set(ib)))
    ga = {g: gray128(v) for g, v in imgs_a.items()}
    gb = {g: gray128(v) for g, v in imgs_b.items()}
    rows = []
    for x, y in zip(ia, ib):
        ma, mb = ga[x], gb[y]
        mae = float(np.abs(ma - mb).mean())
        corr = float(np.corrcoef(ma.ravel(), mb.ravel())[0, 1])
        rows.append(dict(mae=mae, corr=corr, dham=int(Dh[y, x]),
                         pham=int(popcount(pb[y] ^ pa[x])),
                         a_key=recs[a][x]["key"], a_tar=recs[a][x]["tar"], a_idx=x,
                         b_key=recs[b][y]["key"], b_tar=recs[b][y]["tar"], b_idx=y,
                         same_tags=recs[a][x]["idx"] == recs[b][y]["idx"]))
    rows.sort(key=lambda r: (r["mae"], r["pham"]))
    print(f"\n{a}|{b}: {len(rows)} dHash<=6 pairs. pixel MAE (0=identical): "
          f"min {rows[0]['mae']:.3f} median {np.median([r['mae'] for r in rows]):.3f} "
          f"max {max(r['mae'] for r in rows):.3f}", flush=True)
    print(f"  pairs with MAE<2: {sum(1 for r in rows if r['mae'] < 2)}; "
          f"MAE<5: {sum(1 for r in rows if r['mae'] < 5)}; "
          f"pHash Hamming<=8 among them: {sum(1 for r in rows if r['pham'] <= 8)}", flush=True)
    for r in rows[:12]:
        print(f"  MAE={r['mae']:6.2f} corr={r['corr']:+.3f} dH={r['dham']} pH={r['pham']:2d} "
              f"sameTags={r['same_tags']} {a}:{r['a_key']}@{r['a_tar']} "
              f"{b}:{r['b_key']}@{r['b_tar']}", flush=True)
    with open(os.path.join(OUT, f"neardup_{a}_{b}.pkl"), "wb") as f:
        pickle.dump(rows, f)

# ------------------------------------------------------------ within split
print("\n== within-split near duplicates ==", flush=True)
for split in SPLITS:
    d = dh[split]
    n = len(d)
    cnt4 = cnt2 = 0
    pairs = []
    for lo in range(0, n, 256):
        hi = min(n, lo + 256)
        D = popcount(d[lo:hi, None] ^ d[None, :])  # (chunk, n)
        for r in range(hi - lo):
            row = D[r, lo + r + 1:]  # strictly upper triangle, no self
            cnt4 += int((row <= 4).sum())
            cnt2 += int((row <= 2).sum())
            for off in np.where(row <= 2)[0][:5]:
                pairs.append((int(row[off]), lo + r, lo + r + 1 + int(off)))
    print(f"{split}: {cnt4} pairs with dHash<=4, {cnt2} pairs with dHash<=2; "
          f"examples: {pairs[:5]}", flush=True)
