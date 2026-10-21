"""Do the tags you'd actually put in a negative prompt exist in the raw
danbooru annotations, even though our 12,476-label space dropped them?

Samples entries from the start of a few train tars (same contiguous-read trick
the cache builder uses) and counts tag occurrences in general_tags.
"""
import collections
import os
import sys

from wdtag.cache import DATASET_ROOT, SPLITS, _read_samples, build_plan

WANT = ["lowres", "highres", "absurdres", "bad anatomy", "bad hands", "bad feet",
        "worst quality", "low quality", "jpeg artifacts", "blurry", "watermark",
        "signature", "artist name", "text", "english text", "speech bubble",
        "translated", "multiple views", "cropped", "border", "comic", "sketch",
        "realistic", "photo", "3d", "screencap", "official art", "twitter username",
        "web address", "bad proportions", "extra digits", "missing fingers",
        "fused fingers", "monochrome", "greyscale", "manga", "traditional media",
        "vector trace", "game cg", "anime screenshot"]

plan = build_plan("train")
n_tar, per_tar = 6, 400
counts = collections.Counter()
imgs = 0
tot_tags = 0
for e in plan[:n_tar]:
    samples = _read_samples(os.path.join(DATASET_ROOT, SPLITS["train"]["subdir"], e.shard),
                            e.start_frac, per_tar)
    for _key, _blob, meta in samples:
        imgs += 1
        tags = meta.get("general_tags") or []
        tot_tags += len(tags)
        for t in tags:
            if t in WANT:
                counts[t] += 1

print(f"sampled {imgs} images from {n_tar} tars, {tot_tags/imgs:.1f} general tags/image\n")
print(f"{'tag':22s} {'% of images':>12s}")
for t in WANT:
    c = counts.get(t, 0)
    if c:
        print(f"{t:22s} {c/imgs*100:11.2f}%")
absent = [t for t in WANT if counts.get(t, 0) == 0]
print(f"\nnot seen at all in {imgs} images ({len(absent)}): {', '.join(absent[:12])}")
print(f"\nany listed tag present in an image: "
      f"{sum(1 for _ in [0]) and ''}", end="")
