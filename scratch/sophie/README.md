# scratch/sophie

Exploration scratch space for the DINOv3-B/16 + OFTv2 tagger. Only scripts with
ongoing value live at this level; the rest is in `archive/`.

## Kept scripts

- `build_pad_cache.py` — rebuilds the padded-square image cache (`<cache>/<split>/pad/`) for the padding-vs-squash geometry comparison.
- `val_curves.py` — prints every run's per-epoch validation micro F1 curve, its peak epoch, and the drop since the peak.
- `drift_vs_steps.py` — measures OFT rotation drift vs step count and lr from the per-200-step checkpoints.
- `diag_aspect_vs_acc.py` — buckets the 1,000 test images by squash distortion and reports per-bucket micro F1 (reads `test_aspect.csv` and a run's `test_probs.npy`).
- `bench_batch.py` — measures img/s and peak VRAM at several batch sizes on the real training step.
- `bench_compile_modes.py` — measures ms/image for eager vs `torch.compile(default)` vs `max-autotune` on the real training step.
- `pin_sha8.py` — one-off migration: pins each existing run's directory name into its `config.json` (`_sha8`) so a `TrainConfig` field change cannot re-key old runs.

## Kept data (do not move — these cost ~55,000 tar reads to produce)

- `test_aspect.csv` — width/height/aspect for all 1,000 test images; row order matches `test_probs.npy`.
- `train_sample_aspect.csv` — same fields for the 10,000 training images the trainer sees.
- `train_broad_aspect.csv` — same fields for the 40,000-image broad sample.

## archive/

One-off diagnostics — benchmarks, probes, the geometry-audit pipeline, and
rendered dumps — kept for provenance and **not expected to run again**. Paths
inside them (and the command examples in `for-user/dataset-geometry-audit.md`)
refer to their old top-level names; prepend `archive/` to re-run one. The old
`probe_oft.py` is gone because its capability now lives in
`wdtag/probe.py --adapter-from RUN`.
