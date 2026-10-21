# wd-tagger-trainer

Train an anime tagger on a 10,000-image subset of `wdtaggerv4` with a **DINOv3 ViT-B/16**
backbone adapted by **OFTv2** (Orthogonal Fine-Tuning) and optimised with **SinkSGD_ADV**,
then score it against `convnextv2_huge.dbv4-full` on a fixed 1,000-image test split.

## Result summary

| stage | artifact |
|---|---|
| baseline bar | `runs/_baseline/convnextv2_huge/squash/eval_test.json` |
| our runs | `runs/<sha8(config)>/eval_test.json` |
| comparison | `REPORT.md` (from `python -m wdtag.report`) |

## Layout

```
wdtag/
  cache.py     tar sampling + 512x512 .pt shard builder (resumable)
  data.py      mmap dataset, collator, label statistics
  labels.py    the 12,476-label space of the baseline, tag -> index mapping
  oft.py       OFTv2 adapters (ported from OneTrainer's PeftType.OFT_2)
  model.py     DINOv3-B + OFTv2 + multi-label head
  infer.py     batched inference over a cached split
  metrics.py   macro/micro F1, MCC, P, R + per-label threshold sweep
  train.py     training loop (checkpoints, .PAUSE, resume, debugpics)
  evaluate.py  final evaluation on the test cache
  baseline.py  convnextv2_huge inference
  probe.py     frozen-feature probe (fast hyperparameter sweeps)
  report.py    builds REPORT.md
configs/       squash.json (canonical), squash_lr5e4.json, smoke.json
runs/          one directory per config hash
```

## Data

Images come from the webdataset tars in `X:\...\wdtaggerv4\repo\{train,test,val}`.
Because mid-tar seeking costs one HDD seek per member (~0.4 ms/entry, tens of minutes
for a full sweep), sampling is **contiguous from the start of every shard**: 47 images
from each of the 213 train shards, 84 from each of the 12 val/test shards. The shard
order inside these tars is shuffled, so shard starts are already spread across the corpus.

Everything is materialised once as 512x512 `uint8` shards of ~0.5 GB:

```
I:\.cache\wd-tagger-trainer\{train,valid,test}\squash\shard#####-{total}.pt
{ "images": u8 [N,3,512,512], "pos": i64, "ptr": i64,   # labels as CSR
  "ids": i64, "keys": [...], "tars": [...], "split", "mode" }
```

Images are bilinear-resized straight to 512x512 with the aspect ratio not preserved.
A padded-square variant was evaluated too: it is statistically tied with squash once the adapter LR is right (the first comparison, at a 4x too-high LR, made padding look 10% worse). Only the squash cache is built by default; scratch/sophie/build_pad_cache.py builds the pad one for experiments.

The cache is treated as disposable: it is validated at run start and any missing or
truncated shard is rebuilt. Shards are written to `.tmp` and atomically renamed, so an
interrupted build resumes at the first bad shard.

## Training

```powershell
$env:PYTHONPATH="C:\Users\kerok\apps\wd-tagger-trainer"
C:\Python314\python.exe -u -m wdtag.train --config configs\squash.json
```

* adapter: OFTv2 on every `Linear` (73 modules: 6 per block x 12 blocks, plus the
  patch-embed conv), `block_size=32`, `oft_scaled=True` → 1.30 M parameters.
  At init the rotation is exactly identity, so the model starts equal to the frozen base.
* head: `LayerNorm(1536) -> Linear(12476)` over `[LayerNorm(CLS) ; mean(patch tokens)]`,
  bias initialised to the log-odds of each tag's training prior.
* optimizer: `SinkSGD_ADV`, momentum 0.95, lr 0.002 for the adapters / 0.004 for the head,
  warmup 50 steps then cosine to 5%. The adapter LR matters enormously: 0.02 is
  destructive, and the useful range is ~0.0005-0.002 (see REPORT.md).
* 8 epochs over the 10,000 images (312 steps/epoch at batch 32), best checkpoint by
  validation macro-F1. The val curve is flat past epoch 5.
* staged training is available: `freeze_oft: true` holds the adapters at identity while
  the head learns, then `--init-from runs/<sha8>` warm-starts the joint phase.

### Resumability

* Checkpoints: `runs/<sha8>/ckpt_step{step:06d}.pt` every 200 steps, plus `best_val.pt`.
* Pause: create `.PAUSE` (or `.PUASE`) in the run directory **or** the workspace root.
  The trainer checkpoints and exits cleanly at the next step; re-running resumes.
* Resume continues the data stream: the epoch permutation is a pure function of
  `(seed, epoch)` and the checkpoint stores the next batch to consume, so a resumed run
  never re-shows images it has already trained on.
* Debug: the first 100 images fed to the network are written as individual PNGs to
  `runs/<sha8>/debugpics/{dataset_id}-{source_tar}-{mode}.png` with a `manifest.json`.

## Evaluation

`runs/<sha8>/eval_test.json` holds macro/micro F1, MCC, precision and recall at a fixed
0.5 threshold, plus the same again with per-label thresholds tuned on the 1,000-image
validation split, plus a per-category (rating/general/character) breakdown.

Conventions are identical for our model and the baseline:

* a label with no positive and no predicted positive contributes 0 to every average, so
  macro is taken over all 12,476 labels (7,779 of them have no positive in 1k test images);
  `supported_macro_*` is reported alongside for the 4,697 labels that do.
* micro pools TP/FP/FN over the whole matrix; micro MCC is the correlation over it.

## Measured performance (RTX 4090, batch 32, 512x512, BF16)

| variant | ms/step | img/s |
|---|---|---|
| eager | 359 | 89 |
| `torch.compile(mode="default")` | 290 | 110 |
| `max-autotune-no-cudagraphs` (215 s compile) | 259 | 124 |

Per-stage breakdown of an eager step: forward 156 ms, backward 243 ms, **optimizer
88 ms**, data 10 ms, H2D 4 ms. The optimizer cost is per-parameter Python/launch
overhead across its 73 tiny adapter tensors (~1.2 ms each) and is *not* affected by any
of its own options: `stochastic_rounding=False`, `sinkhorn_iterations=3`, fp32 states and
`compiled_optimizer=True` all measure identically, and `compiled_optimizer=True`
additionally hard-crashes with `FailOnRecompileLimitHit` (`fullgraph=True` plus
per-shape recompiles).

Data loading is deliberately trivial: pre-decoded uint8 shards, memory-mapped, no
worker processes (see below), normalisation on the GPU in BF16.

### Memory notes

Opening all 16 train shards in one process costs ~8.8 GB of private commit on Windows, and
each extra opening of the same file costs another copy, so:

* every shard is opened once per process through a module-level cache (`data.open_shard`);
* `num_workers=0` — workers would each pay their own copy, and loading is not the bottleneck;
* `CacheDataset.__getstate__` strips mapped tensors so a spawned worker can never receive
  a pickled copy of the cache.

With those, commit stays flat at ~55 GB across epochs instead of climbing ~18 GB per epoch.

## Limitations

* 10,000 images against 12,476 labels: the long tail is unlearnable at this scale, so
  macro-F1 over the full label space is dominated by tags we cannot predict. The baseline
  was trained on 5.3 M images and is expected to win; the interesting number is how much
  OFTv2 buys over a frozen linear probe.
* Per-label thresholds tuned on 1,000 validation images overfit sparse labels; the
  tuned table is reported for completeness next to the fixed-0.5 table, not instead of it.


