# Validation pipeline (separate from training)

How we score models, and why it is a separate process.

## The protocol: fixed 0.40, eight numbers

`convnextv2_huge.dbv4-full` publishes exactly this shape, and it is the only comparison that counts (AGENTS.md). Filled in below with the bar's published Validation numbers = our target:

| | F1 | MCC | P | R |
|---|---|---|---|---|
| Macro@0.40 | 0.580 | 0.584 | 0.626 | 0.556 |
| Micro@0.40 | 0.697 | 0.696 | 0.692 | 0.701 |

No threshold grids, no "best", no per-tag tuning. `--thr` defaults to `0.40` and a bare value means one threshold. The `lo:hi:step` grid form still exists for diagnostics, but reporting anything at a different threshold is not comparable and wastes time.

Why fixed 0.40 matters: macro and micro optima diverge sharply on this model — macro peaks near 0.05, micro near 0.28, and 0.40 is neither. At 0.40 our model is precision-heavy (micro P 0.7068 vs R 0.5417). Quoting a model's own best threshold flatters it by ~38% relative on macro and makes the bar comparison meaningless.

## Why it must not run alongside training

One training step already holds ~21 GB of the 24.5 GB card, so a large validation batch cannot coexist with it. Training keeps only a 1k in-run check (`val_images: int = 1000` in `config.py`) for monitoring. The real 50k evaluation is:

```
python -m wdtag.validate --run runs/4a13a120 --batch-size 256     # our model
python -m wdtag.validate --bar --sizes 1000,5000,10000,50000      # the bar
```

Cold 50k pass = 3.4 min; rescoring from cached probabilities = ~2 s.

## Batch size differs per model — do not use one default

Measured on the RTX 4090 at 512px (our model is 129M params; the bar is 692.6M):

| model | batch | img/s | peak VRAM |
|---|---|---|---|
| ours | 128 | 202 | 8.1 GB |
| ours | 256 | 246 | 7.5 GB |
| ours | 384 | 258 | 11.0 GB |
| ours | 512 | — | maxes VRAM |
| bar | 64 | 61 | 17.0 GB |
| bar | 128 | — | 23.7 GB (96%, too tight) |

`--batch-size` therefore defaults to 256 for our model and 64 for `--bar`. Using one default is how the card got maxed for no gain.

Forward-only and with-counting now measure the same (244 vs 246 img/s), because the threshold accounting happens on the GPU.

## The nested cache

`cache/wd-tagger-trainer\valid_nested\` — `images.npy` (50000 x 3 x 512 x 512 uint8,
39.3 GB memmap), `labels.npz` (sparse `indices`/`offsets`), `meta.json`.

The 1k / 5k / 10k / 50k subsets are prefixes of the same array, so they are nested by construction and one pass yields all of them. Verified: 32.4 / 32.1 / 32.1 / 32.3 positives per image and 4669 / 8973 / 10809 / 12454 supported labels.

Batches straddle subset boundaries (128 does not divide 1000), so each subset counts only rows below its own size. Counting whole batches silently truncates every subset to a multiple of the batch — earlier numbers were computed on 992 images while labelled 1000.

## Probability cache

`cache/wd-tagger-trainer\probs\{run}_{step}.npy`, float16, 50000 x 12476 = 1.25 GB,
plus a `.json` sidecar recording how many rows are valid.

The sidecar is not optional. `open_memmap` allocates the full 50000 rows but only rows below the requested size are ever written, so a cache produced by a 5000-image run was silently reused for a 50000-image run and scored 45000 rows of zeros — macro 0.0371 instead of 0.2594. `--recompute` forces a rebuild.

## Two bugs worth remembering

1. `.npy` has a 128-byte header. A raw `f.read()` from offset 0 reads `\x93NUMPY`, not pixels, so every image is shifted by 128 bytes. It does not crash — it silently cost 0.0019 macro (0.1019 → 0.0991). `ImageChunkReader` skips the header via `np.lib.format.read_magic` and validates dtype.
2. A 39.3 GB memmap is a RAM trap. `np.load(mmap_mode="r")` showed 35.9 GB working set with only 6.7 GB private — mapped file pages, reclaimable but counted — on a 47.9 GB box, leaving 0.1 GB free. `ImageChunkReader` reads through one reused buffer instead: working set 35.9 → 3.1 GB, free RAM 0.1 → 32.1 GB, at the same speed.

## Metrics are computed from counts, not from a prediction matrix

Per-`(threshold, label)` TP/FP/FN are accumulated on the GPU, and all eight metrics are derived from those counts. This avoids materialising a 50k x 12476 prediction matrix (2.5 GB) and a dense target matrix (2.5 GB).

Caution: micro F1 is the harmonic mean of pooled P and R, but macro F1 is the mean of per-tag F1 and is not the harmonic mean of macro P and macro R. Do not mix them.

## Outputs

| path | contents |
|---|---|
| `runs/{sha8}/validation.json` | canonical report: thresholds x subsets x 8 metrics |
| `runs/{sha8}/compare_bar_040.json` | ours vs bar at 0.40, with ratios |
| `cache/wd-tagger-trainer\bar_validation.json` | the bar scored on our cache |

## Reference numbers (checkpoint 086241, 50k, at 0.40)

| | ours | bar | ours/bar |
|---|---|---|---|
| Macro F1 | 0.2815 | 0.5579 | 50.4% |
| Macro MCC / P / R | 0.3062 / 0.4608 / 0.2266 | 0.5658 / 0.6201 / 0.5333 | 54 / 74 / 43% |
| Micro F1 | 0.6133 | 0.6886 | 89.1% |
| Micro MCC / P / R | 0.6179 / 0.7068 / 0.5417 | 0.6880 / 0.7033 / 0.6746 | 90 / 100.5% / 80% |

Micro is flat at 89.1–89.5% across every subset size, so 1k is reliable for tracking micro. Macro drifts 57.7% → 50.4% with subset size, because small subsets cannot support the rare labels that dominate macro — only judge macro on the full 50k.

## Still open

The nested cache is the val split. AGENTS.md asks for validation and test; a 50k nested test cache has not been built (12 tars / 295926 images, ~39.3 GB, ~1 h).

## Which checkpoint gets scored

`python -m wdtag.validate --run <dir>` scores the highest step number among `ckpt_step*.pt`. It does not touch `best_val.pt`, and the run's `--select-metric` has no effect on it, because that flag only decides which file the training loop copies to `best_val.pt`. So every number in the private experiment log that came from a bare `--run` is a final-checkpoint number, and the only way to score a different artifact is to name it, `--ckpt best_val.pt`.
