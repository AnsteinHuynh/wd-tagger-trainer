"""Render every cross-split pair with cosine >= 0.80 as a side-by-side contact
sheet, so the band where a hard-cropped duplicate could hide is checked by eye
(a 90% centre crop scores 0.68-0.87 with this descriptor)."""
from __future__ import annotations

import json
import os
import pickle

import numpy as np
from PIL import Image

from wdtag.cache import IMAGES_PER_SHARD, SPLITS, _shard_name

WS = r"C:\Users\kerok\apps\wd-tagger-trainer"
OUT = os.path.join(WS, "scratch", "sophie")
FU = os.path.join(OUT, "for-user")
TILE = 384


def get(split: str, i: int) -> np.ndarray:
    total = SPLITS[split]["n"]
    k = i // IMAGES_PER_SHARD
    s = __import__("torch").load(_shard_name(split, "squash", k, total),
                                 map_location="cpu", mmap=True, weights_only=False)
    return s["images"][i % IMAGES_PER_SHARD].numpy()


recs = {}
for split in ("train", "valid", "test"):
    with open(os.path.join(OUT, f"records_{split}.pkl"), "rb") as f:
        recs[split] = pickle.load(f)
D = {s: np.load(os.path.join(OUT, f"desc_{s}.npy")) for s in ("train", "valid", "test")}

listing = {}
for a, b in (("train", "test"), ("train", "valid"), ("test", "valid")):
    S = D[a] @ D[b].T
    ii, jj = np.where(S >= 0.80)
    order = np.argsort(-S[ii, jj])
    pairs = [(int(ii[o]), int(jj[o]), float(S[ii[o], jj[o]])) for o in order]
    listing[f"{a}|{b}"] = [
        dict(cos=c, a_key=recs[a][i]["key"], a_tar=recs[a][i]["tar"],
             b_key=recs[b][j]["key"], b_tar=recs[b][j]["tar"],
             same_tags=recs[a][i]["idx"] == recs[b][j]["idx"]) for i, j, c in pairs]
    print(f"{a}|{b}: {len(pairs)} pairs >= 0.80", flush=True)
    if not pairs:
        continue
    canvas = Image.new("RGB", (2 * TILE, TILE * len(pairs)), (0, 0, 0))
    for row, (i, j, c) in enumerate(pairs):
        for col, (split, idx) in enumerate(((a, i), (b, j))):
            im = Image.fromarray(get(split, idx).transpose(1, 2, 0)).resize((TILE, TILE),
                                                                            Image.BILINEAR)
            canvas.paste(im, (col * TILE, row * TILE))
    p = os.path.join(FU, f"similar-pairs-{a}-{b}.png")
    canvas.save(p)
    print(f"  wrote {p} {canvas.size}", flush=True)

with open(os.path.join(OUT, "pairs_ge80.json"), "w", encoding="utf-8") as f:
    json.dump(listing, f, indent=1)
