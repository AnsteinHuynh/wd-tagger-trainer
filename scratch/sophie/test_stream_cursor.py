"""One purpose: prove the stream's resume cursor is exact.

Two independent TarStream readers over the same (seed, pass_index): one starting at zero and one
starting after `skip` samples, as a resumed run does. If resume is correct the skipped reader's
batch k equals the fresh reader's batch k + skip/batch, byte for byte, for every k. Anything else
means a pause silently repeats or drops images.

    python scratch/sophie/test_stream_cursor.py
"""

import numpy as np
import torch

from wdtag.stream import TarStream

BATCH = 8
SKIP = 800  # 100 steps at batch 8, the point a 100-step pause would resume from
N = 12


def batches(skip: int, n: int):
    sp = TarStream("train", seed=1337, pass_index=0, batch_size=BATCH,
                   n_labels=12476, workers=8, prefetch=2, skip=skip,
                   max_samples=10 ** 9, log=lambda m: None)
    out = []
    for x, y, consumed, src in sp.batches():
        out.append((x.clone(), y.clone(), consumed, list(src)))
        if len(out) >= n:
            break
    return out


def main() -> None:
    fresh = batches(0, SKIP // BATCH + N)
    resumed = batches(SKIP, N)

    bad = 0
    for k, (xr, yr, cr, sr) in enumerate(resumed):
        xf, yf, cf, sf = fresh[SKIP // BATCH + k]
        img_ok = torch.equal(xf, xr)
        lab_ok = torch.equal(yf, yr)
        src_ok = sr == sf
        # `consumed` counts from this reader's start (it excludes the skipped samples), so the
        # two readers must report the same value for the same batch rather than one offset by skip.
        off_ok = cr == cf
        if not (img_ok and lab_ok and src_ok and off_ok):
            bad += 1
            print(f"batch {k}: images {img_ok} labels {lab_ok} sources {src_ok} "
                  f"consumed {cr} vs {cf}")
            if not img_ok:
                d = (xf.float() - xr.float()).abs()
                print(f"   image max abs diff {d.max().item():.1f} over {int((d > 0).sum())} px")
    print(f"{N - bad}/{N} resumed batches match the fresh stream exactly "
          f"(skip {SKIP} samples = {SKIP // BATCH} steps)")
    print("first source fresh   ", fresh[SKIP // BATCH][3][:2])
    print("first source resumed ", resumed[0][3][:2])
    print("consumed fresh", fresh[SKIP // BATCH][2], "resumed", resumed[0][2])


if __name__ == "__main__":
    main()
