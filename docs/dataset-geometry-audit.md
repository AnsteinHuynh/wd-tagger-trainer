# Dataset geometry audit — `wdtaggerv4` (w640 webdataset)

Scope: geometric-tampering traces (stretch, padding, bar removal) in the source images, and the true aspect-ratio distribution before our pipeline squashes it to 512×512.

Constraint compliance: CPU + disk only, no GPU, no writes outside `scratch/sophie/`, nothing under `wdtag/`, `configs/`, `runs/` modified, `I:\.cache\wd-tagger-trainer` not read at all. 55000
image reads (51000 header-only + 4000 full decodes), every one contiguous from a shard start.

| sample | what it is | n |
|---|---|---|
| `train_broad` | 400 consecutive images from each of 100 train shards (evenly over the 213 sorted tars) | 40000 |
| `train_cache` | the exact 10000 trainer images, `CacheDataset` index order | 10000 |
| `test_cache` | the exact 1000 test images, prediction order (`per_pos = 84`, truncated at 1000) | 1000 |
| `bar_pixels` | 4000 decoded images (10 shards × 400) for the pixel bar check | 4000 |

Replication verified: all 800 `debugpics/manifest.json` entries across the 8 real runs are in the replicated `train_cache` rows — 800/800 (`key`, `tar`) pairs, 0 missing. Metadata `width`/`height` equals the PIL header in 51000/51000 images; all images are WEBP.

## Headline

1. No tag describes stretch/aspect. All 12476 names scanned: `aspect`, `disproportion`, `squished`, `squashed`, `elongat`, `fattened`, `messed up` → zero matches. Only geometry can detect it.
2. Not 640 wide. `w640` = min side ≤ 640: `min(w,h)==640` 85.1%, `<640` 14.9%, `>640` none; width exactly 640 for 65.2% (`h==640` 24.9%). Upstream resize preserved aspect.
3. Median portrait 3:4. aspect p50 0.750, p95 1.60; 70.4% portrait, 24.2% landscape, 5.4% square; 6.5% within ±2% of square.
4. Squash to 512² distorts the median by 1.41× (p95 1.82; 41% scale mismatch). 19.96% of the 10000 training images (1996) >1.5×, 8.62% >1.75×, 2.88% >2×.
5. Padding is source art, not ours: `letterboxed` 0.67%, `pillarboxed` 0.08%, `border` 1.89% / `white_border` 1.12%. Pixel check: `letterboxed` images ~4× base rate for symmetric flat strips (47.4% vs 12.0%). Nothing removes bars.

## 1. What the images actually are

`w==640` 65.23%, `w<640` 11.44%, `w>640` 23.33%, `min(w,h)==640` 85.06%; width p50 640 px, height p50 841 px, height max 8231 px; 4067 distinct aspects in 40k. Full table:
[what the images are](../../../docs/REPORTS/geom-what-images-are.md).

## 2. Task A — prowling the annotations

`tags.csv` vocabulary scan (12476 labels) plus a 40000-image sample, `exact`/`token`/`sub` match classes. Five distortion tag families top out at 0.122%; 12 user terms score zero everywhere:
[tag forensics](../../../docs/REPORTS/geom-tag-forensics.md).

## 3. Task B — the true aspect-ratio distribution

Aspect percentiles, square-deviation buckets, named aspects, distortion factor, letterbox waste and applied scale factors:
[aspect distribution](../../../docs/REPORTS/geom-aspect-distribution.md).

## 4. Task C — joinable tables

Three per-row `w,h,aspect,key,shard` CSVs in cache index order, plus `test_probs.npy` row alignment:
[joinable tables](../../../docs/REPORTS/geom-joinable-tables.md).

## 5. What this means for training on squashed 512×512

Median stretched 31–41%, a long distorted tail, a consistent not shifted bias, squash trading 26.7% blank canvas for 1.33× anisotropy, and no label can police geometry:
[training implications](../../../docs/REPORTS/geom-training-implications.md).

## 6. Exact commands

Seven scripts, contiguous header-only reads except the 4000-image pixel check:
[commands](../../../docs/REPORTS/geom-commands.md).
