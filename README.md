# wd-tagger-trainer

Train an anime tagger on `wdtaggerv4` with a DINOv3 ViT-B/16 backbone adapted by OFTv2 (Orthogonal Fine-Tuning) and optimised with SinkSGD_ADV, then score it against `convnextv2_huge.dbv4-full` on a 50k validation subset at a fixed 0.40 threshold.

## Where this stands

We do not beat the bar yet. Measured on 50k validation images at 0.40, ours from the long run at checkpoint 086241 (`runs/4a13a120`), the bar from `wdtag.validate --bar` on the same images:

| | ours | bar | ours/bar |
|---|---|---|---|
| macro F1 @0.40 | 0.2815 | 0.5579 | 50.4% |
| macro MCC / P / R | 0.3062 / 0.4608 / 0.2266 | 0.5658 / 0.6201 / 0.5333 | 54 / 74 / 43% |
| micro F1 @0.40 | 0.6133 | 0.6886 | 89.1% |
| micro MCC / P / R | 0.6179 / 0.7068 / 0.5417 | 0.6880 / 0.7033 / 0.6746 | 90 / 100.5 / 80% |

Micro is within 11% of the bar and micro precision already exceeds it; the gap is entirely macro, and macro is where the rare tags live. The 10k-era comparison (whose numbers are in `ARM-RESULTS.md`) is not the current objective.

`ARM-RESULTS.md` is the central, hand-maintained experiment log and the only place results live: every head, loss and unfreeze arm, the 10k benchmark definition, the noise-floor arm, and the findings that outlive any single arm. Read it before designing an arm so nothing is re-run. `REPORT.md` and `wdtag/report.py` are deprecated: a generated second report drifted out of date, so it is kept for history and no longer maintained.

## Train

```powershell
$env:PYTHONPATH="C:\Users\kerok\apps\wd-tagger-trainer"
C:\Python314\python.exe -u -m wdtag.train --config configs\heads\phase3\01_u12-C-L-asl-g4.0.json --select-metric macro
```

Reference settings, all measured (`docs/best-settings.md` has the evidence and the traps): batch 48, 512px squash, 12 unfrozen blocks, OFT `block_size` 128 with exact Cayley (`cayley_terms 0`), `head_fp32`, `compile_model` and `compiled_optimizer` on, checkpoint every 1000 steps, `--select-metric macro`. Throughput at that config is 116.3 img/s compiled against 92.8 eager; a frozen backbone reaches 160 img/s, so roughly 30% of throughput is the deliberate cost of training the backbone.

Runs are named by the sha8 of their whole config and written to `runs/<sha8>/`. Adding a config field changes the hash of every unpinned config, so never look a run up by re-hashing: read the `run dir` line from the log, or `_sha8` in `<run>/config.json`.

Every run is pausable and resumable: create `.PAUSE` (or `.PUASE`) in the run directory or the workspace root and the trainer checkpoints and exits cleanly at the next step. Resume with `--run runs/<sha8>`; the epoch permutation is a pure function of `(seed, epoch)` and the checkpoint stores the next batch, so a resumed run never re-shows an image it has already trained on. The first 100 training images are written as PNGs to `runs/<sha8>/debugpics/` with a `manifest.json` so any picture can be traced back to its source tar.

## Score

```powershell
C:\Python314\python.exe -m wdtag.validate --run runs/4a13a120 --batch-size 256
C:\Python314\python.exe -m wdtag.validate --bar --sizes 1000,5000,10000,50000
```

Fixed threshold 0.40, eight numbers, no per-tag tuning. Batch 256 for our model, 64 for the 692.6M-parameter bar; the batch differs per model because 512 maxes the card. Read results from `<run>/validation.json` -> `subsets["50000"]`, never from console text: the validator prints an `at 0.40:` block once per nested size and scraping returns the 1000-image numbers as if they were 50k. A cold 50k pass is ~3.4 min and rescoring from the probability cache is ~2 s.

## Layout

```
wdtag/
  config.py    TrainConfig, sha8 naming, measured defaults
  cache.py     tar sampling + 512x512 shard builder (resumable, atomic)
  data.py      mmap dataset, collator, label priors
  labels.py    the 12476-label space of the bar, tag -> index mapping
  oft.py       OFTv2 adapters (ported from OneTrainer's PeftType.OFT_2)
  model.py     DINOv3-B + OFTv2 + multi-label head, unfreeze, block checkpointing
  losses.py    bce, bce_pos_weight, asl, cb_ntr, asl_ntr
  metrics.py   macro/micro F1, MCC, P, R, threshold sweep
  train.py     StepHelper + both training loops (cache and tar stream), resume, .PAUSE
  validate.py  the 50k pipeline, fixed 0.40, probability cache
  report.py    DEPRECATED, kept for history; results go in ARM-RESULTS.md
configs/heads/  one JSON per arm, phase3/ holds the current ladder
runs/           one directory per config hash
```

## Data

Images come from the webdataset tars in `X:\...\wdtaggerv4\repo\{train,test,val}`, 5321713 train images across 213 shards. Mid-tar seeking costs one HDD seek per member, so sampling is contiguous from the start of each shard; the shard order is already shuffled, so shard starts are spread across the corpus. Everything is materialised once as 512x512 uint8 shards of ~0.5 GB. The cache is disposable: it is validated at run start, missing or truncated shards are rebuilt, and shards are written to `.tmp` and atomically renamed so an interrupted build resumes at the first bad shard. The full corpus would cache to 4.18 TB, which is why training can also stream straight from the tars (`--stream`).

Squash and pad tie within the seed spread and squash is the default.

## Label coverage, before interpreting macro

The annotations are clean: no train/test leakage, no out-of-vocabulary tags, no empty label vectors, no duplicate keys. What is scarce is coverage. In the 10k benchmark only 10790 of 12476 labels appear in training, 7908 appear in fewer than 10 images, and 217 labels present in test never appear in train at all. 62% of the vocabulary never occurs in the test split and scores 0 for both models, so the headline macro figure dilutes both sides equally. The collapse is in the 1-9 image bucket, where we score 0.0009 against a bar that saw those tags hundreds of times. That is a data-coverage wall and most of the headline gap. Detail in `docs/label-integrity-audit.md`.

## Memory

Opening all train shards in one process costs ~8.8 GB of private commit on Windows, and each extra opening of the same file costs another copy, so every shard is opened once per process through a module-level cache (`data.open_shard`), workers are disabled (`num_workers=0`) because loading is not the bottleneck, and `CacheDataset.__getstate__` strips mapped tensors so a spawned worker cannot receive a pickled copy. Commit stays flat at ~55 GB across epochs instead of climbing ~18 GB per epoch.

## Open questions

Whether the macro gap can be closed at all on this dataset, and whether the 10k-era ranking of heads and losses still holds at full scale. The candidate long run is the best head times the best loss times the best freeze depth, and there is no evidence yet about how any two of those three interact.
