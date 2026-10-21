"""Calibrate the similarity descriptor: what score does a *genuine* duplicate get?

Take 5 real training images from the cache and degrade them the way a repost
would be degraded (JPEG recompression at q=90/50/20, downscale-reupload,
90% centre crop, horiz. flip), then score each variant against its original with
the exact descriptor used for the 10M-pair cross-split search.
"""
from __future__ import annotations

import io
import os
import pickle

import numpy as np
from PIL import Image

from wdtag.cache import IMAGES_PER_SHARD, SPLITS, _shard_name

WS = r"C:\Users\kerok\apps\wd-tagger-trainer"
OUT = os.path.join(WS, "scratch", "sophie")


def thumb_from_pil(im: Image.Image, size: int = 32) -> np.ndarray:
    v = np.asarray(im.convert("L").resize((size, size), Image.BILINEAR), dtype=np.float32).ravel()
    v = v - v.mean()
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else v


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b)


def jpeg(im: Image.Image, q: int) -> Image.Image:
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def main() -> None:
    total = SPLITS["train"]["n"]
    picks = [(0, 3), (1, 17), (7, 42), (12, 100), (15, 600)]
    res = {k: [] for k in ("jpeg90", "jpeg50", "jpeg20", "reupload256", "crop90", "flip")}
    for k, j in picks:
        s = __import__("torch").load(_shard_name("train", "squash", k, total),
                                     map_location="cpu", mmap=True, weights_only=False)
        im = Image.fromarray(s["images"][j].numpy().transpose(1, 2, 0))
        base = thumb_from_pil(im)
        W, H = im.size
        res["jpeg90"].append(cos(base, thumb_from_pil(jpeg(im, 90))))
        res["jpeg50"].append(cos(base, thumb_from_pil(jpeg(im, 50))))
        res["jpeg20"].append(cos(base, thumb_from_pil(jpeg(im, 20))))
        res["reupload256"].append(cos(base, thumb_from_pil(jpeg(im.resize((256, 256), Image.BILINEAR), 85))))
        crop = im.crop((int(W * 0.05), int(H * 0.05), int(W * 0.95), int(H * 0.95)))
        res["crop90"].append(cos(base, thumb_from_pil(crop)))
        res["flip"].append(cos(base, thumb_from_pil(im.transpose(Image.FLIP_LEFT_RIGHT))))
    out = {}
    for k, v in res.items():
        out[k] = dict(min=float(np.min(v)), median=float(np.median(v)), max=float(np.max(v)))
        print(f"{k:12s} cos min {out[k]['min']:.4f} median {out[k]['median']:.4f} "
              f"max {out[k]['max']:.4f}", flush=True)
    with open(os.path.join(OUT, "calibration.json"), "w", encoding="utf-8") as f:
        import json
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
