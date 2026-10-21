"""Exhaustive content-similarity check (no threshold guessing).

Descriptor: 32x32 grayscale thumbnail of the trainer's 512x512 squash tensor,
z-scored per image (kills brightness/contrast differences from re-encoding) and
L2-normalised.  Cosine similarity is then computed for EVERY pair:
  train(10,000) x test(1,000)   = 10,000,000 pairs
  train(10,000) x valid(1,000)  = 10,000,000 pairs
  test(1,000)   x valid(1,000)  =  1,000,000 pairs
  within train / valid / test   = 49,995,000 / 499,500 / 499,500 pairs
A duplicated (even re-encoded, rescaled or re-cropped-hard) picture would show
up as cosine ~0.99+ / MAE ~0.  Also repeats the train/test search on horizontally
flipped test images (mirrored reposts).
"""
from __future__ import annotations

import os
import pickle
import time

import numpy as np
from PIL import Image

from wdtag.cache import IMAGES_PER_SHARD, SPLITS, _shard_name

WS = r"C:\Users\kerok\apps\wd-tagger-trainer"
OUT = os.path.join(WS, "scratch", "sophie")


def thumb(a: np.ndarray, size: int = 32) -> np.ndarray:
    im = Image.fromarray(a.transpose(1, 2, 0)).convert("L").resize((size, size), Image.BILINEAR)
    v = np.asarray(im, dtype=np.float32).ravel()
    v = v - v.mean()
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else v


def build(split: str) -> np.ndarray:
    total = SPLITS[split]["n"]
    n_shards = (total + IMAGES_PER_SHARD - 1) // IMAGES_PER_SHARD
    out = np.zeros((total, 32 * 32), dtype=np.float32)
    for k in range(n_shards):
        cnt = min(IMAGES_PER_SHARD, total - k * IMAGES_PER_SHARD)
        s = __import__("torch").load(_shard_name(split, "squash", k, total),
                                     map_location="cpu", mmap=True, weights_only=False)
        for j in range(cnt):
            out[k * IMAGES_PER_SHARD + j] = thumb(s["images"][j].numpy())
    return out


def main() -> None:
    D = {}
    for split in SPLITS:
        p = os.path.join(OUT, f"desc_{split}.npy")
        if os.path.exists(p):
            D[split] = np.load(p)
        else:
            t0 = time.time()
            D[split] = build(split)
            np.save(p, D[split])
            print(f"[{split}] descriptor {D[split].shape} in {time.time()-t0:.0f}s", flush=True)

    def report(name: str, A: np.ndarray, B: np.ndarray, same: bool) -> None:
        S = A @ B.T  # cosine, [na, nb]
        if same:
            np.fill_diagonal(S, -2.0)
        flat = np.argsort(S, axis=None)[::-1][:6]
        na = A.shape[0]
        print(f"\n{name}: {A.shape[0]*B.shape[0]:,} pairs; max cos {S.max():.4f}; "
              f"p99.99 {np.percentile(S, 99.99):.4f}", flush=True)
        for f in flat:
            i, j = divmod(int(f), B.shape[0])
            print(f"   cos={S[i, j]:.4f} a[{i}] b[{j}]", flush=True)
        return S

    with open(os.path.join(OUT, "records_train.pkl"), "rb") as f:
        rt = pickle.load(f)
    with open(os.path.join(OUT, "records_test.pkl"), "rb") as f:
        rte = pickle.load(f)
    with open(os.path.join(OUT, "records_valid.pkl"), "rb") as f:
        rv = pickle.load(f)
    R = {"train": rt, "test": rte, "valid": rv}

    res = {}
    for a, b in (("train", "test"), ("train", "valid"), ("test", "valid")):
        S = report(f"{a}|{b}", D[a], D[b], same=False)
        top = np.argsort(S, axis=None)[::-1][:10]
        res[f"{a}|{b}"] = [
            dict(cos=float(S[i, j]), a_key=R[a][i]["key"], a_tar=R[a][i]["tar"],
                 b_key=R[b][j]["key"], b_tar=R[b][j]["tar"],
                 same_tags=R[a][i]["idx"] == R[b][j]["idx"])
            for i, j in (divmod(int(f), D[b].shape[0]) for f in top)
        ]

    # mirrored test/valid: flip the descriptor's 32x32 grid horizontally
    for a, b in (("train", "test"), ("train", "valid")):
        F = D[b].reshape(-1, 32, 32)[:, :, ::-1].reshape(len(D[b]), -1)
        S = report(f"{a}|{b} (b mirrored)", D[a], F, same=False)
        res[f"{a}|{b}_mirror"] = [
            dict(cos=float(S[i, j]), a_key=R[a][i]["key"], a_tar=R[a][i]["tar"],
                 b_key=R[b][j]["key"], b_tar=R[b][j]["tar"])
            for i, j in (divmod(int(f), D[b].shape[0])
                         for f in np.argsort(S, axis=None)[::-1][:10])
        ]

    for split in SPLITS:
        S = report(f"{split} within", D[split], D[split], same=True)
        top = np.argsort(S, axis=None)[::-1][:5]
        res[f"{split}_within"] = [
            dict(cos=float(S[i, j]), a_key=R[split][i]["key"], a_tar=R[split][i]["tar"],
                 b_key=R[split][j]["key"], b_tar=R[split][j]["tar"],
                 same_tags=R[split][i]["idx"] == R[split][j]["idx"])
            for i, j in (divmod(int(f), D[split].shape[0]) for f in top)
        ]

    with open(os.path.join(OUT, "similarity.json"), "w", encoding="utf-8") as f:
        import json
        json.dump(res, f, indent=1, default=str)


if __name__ == "__main__":
    main()
