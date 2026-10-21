# wd-tagger-trainer

Train an anime tagger on a 10,000-image subset of `wdtaggerv4` with a **DINOv3 ViT-B/16**
backbone adapted by **OFTv2** (Orthogonal Fine-Tuning) and optimised with **SinkSGD_ADV**,
then score it against `convnextv2_huge.dbv4-full` on a fixed 1,000-image test split.

## Result summary

Our best run (`fe92f4df`: squash, `lr_oft 0.0005`, `lr_head 0.00012`, seed 1337, 8 epochs)
scores on the fixed 1,000-image test split:

| model | micro F1 @0.5 | macro F1 @0.5 | tuned macro F1 |
|---|---|---|---|
| our best run | **0.4142** | 0.0223 | **0.0331** |
| `convnextv2_huge.dbv4-full` (the bar) | 0.6800 | 0.1938 | 0.2008 |

We do **not** beat the bar. The baseline saw 5.3 M images for these 12,476 labels; we
trained on 10,000. The gap is data scale, not a pipeline bug. `REPORT.md` has the full
tables, the per-category breakdown and the open questions.

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
  score.py     shared scorecard (fixed 0.5 + validation-tuned thresholds)
  train.py     training loop (checkpoints, .PAUSE, resume, debugpics)
  evaluate.py  final evaluation on the test cache
  baseline.py  convnextv2_huge inference
  probe.py     frozen-feature probe (fast hyperparameter sweeps)
  report.py    builds REPORT.md
configs/       best.json (current recipe: lr_oft 0.0005, lr_head 0.00012); squash.json predates the sweep
configs/experiments/  one JSON per sweep arm
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

Images are bilinear-resized straight to 512x512 with the aspect ratio not preserved
("squash"). **Geometry is settled: the padded-square arm ties, and squash is the better
default.** At the corrected head learning rate and matched on `lr_oft 0.0005`, `lr_head
0.00012`, seed 1337, pad scores 0.4078 vs squash 0.4142 micro — squash ahead on all four
metrics (micro, macro, tuned micro, tuned macro) by 0.002-0.008, which is inside the seed
spread. Independently, per-image accuracy does not correlate with squash distortion
(r = +0.047 over the 1,000 test images; heavily warped images score, if anything, better).
The earlier apparent pad *win* (0.3985 vs 0.3888) was an artifact of the oversized head
learning rate, so it is not evidence for padding. `mode` remains configurable
(`squash` / `pad`), but squash is the default and the only variant built by default; the
pad cache is an experiment artifact (`scratch/sophie/build_pad_cache.py`).

The cache is treated as disposable: it is validated at run start and any missing or
truncated shard is rebuilt. Shards are written to `.tmp` and atomically renamed, so an
interrupted build resumes at the first bad shard.

### Label coverage (read this before interpreting macro F1)

The annotations are clean — audited over the exact 10k/1k/1k sets: **no leakage** between
train and test (0 shared keys, 0 byte-identical images, and an exhaustive 10,000,000-pair
thumbnail search whose best match, 0.8691, was visually confirmed to be a different
picture), **no out-of-vocabulary tags** (0 of 322,944 train tag occurrences), **no images
with zero positive labels**, and no duplicate tags, keys or target vectors. Mean 32.29
positives per image (median 30, p95 58).

What *is* scarce is coverage. Only **10,790 of 12,476 labels appear at all** in the 10k
training set; **7,908 appear in fewer than 10 images**, and only **2,882 have 10 or more**.
**217 labels present in the test split never appear in train**, so those columns are an
automatic zero for us while the full-corpus baseline scores on them. The macro gap
decomposes accordingly (best run vs bar):

| label subset | labels | bar | ours | ours, as % of bar |
|---|---|---|---|---|
| every label in the vocabulary | 12,476 | 0.1938 | 0.0223 | 11.5% |
| labels that occur in the test split | 4,697 | 0.5147 | 0.0591 | 11.5% |
| labels with >=10 training images | 2,882 | 0.4546 | 0.0939 | 20.7% |
| labels with 1-9 training images | 7,908 | 0.1260 | 0.0009 | 0.7% |
| labels never in train, present in test | 217 | 0.5091 | 0.0000 | 0.0% |

62% of the vocabulary never occurs in the test split and scores 0 for *both* models, so the
headline macro figure dilutes both sides equally; the second row is the real standing on
labels that can be scored at all, and the third row is the fairest "learnable" comparison.
The collapse is in the fourth row — labels seen 1-9 times in training, where we score
essentially nothing against a baseline that saw them hundreds of times. That is a
data-coverage wall, and most of the headline gap. Detail:
`scratch/sophie/for-user/label-integrity-audit.md`, reproduced by
`scratch/sophie/diag_bar_gap.py`.

## Training

```powershell
$env:PYTHONPATH="C:\Users\kerok\apps\wd-tagger-trainer"
C:\Python314\python.exe -u -m wdtag.train --config configs\best.json
```

* adapter: OFTv2 on every `Linear` (73 modules: 6 per block x 12 blocks, plus the
  patch-embed conv), `block_size=32`, `oft_scaled=True` → 1.30 M parameters.
  At init the rotation is exactly identity, so the model starts equal to the frozen base.
* head: `LayerNorm(1536) -> Linear(12476)` (19.18 M parameters) over
  `[LayerNorm(CLS) ; mean(patch tokens)]`, bias initialised to the log-odds of each tag's
  training prior.
* optimizer: `SinkSGD_ADV`, momentum 0.95, warmup 50 steps then cosine to 5%. The current
  recipe is `lr_oft 0.0005` / `lr_head 0.00012`; both are load-bearing, see below.
* 8 epochs over the 10,000 images (312 steps/epoch at batch 32), best checkpoint by
  validation macro-F1. A 20-epoch arm at the old head LR was flat from epoch 14; longer
  schedules have not been re-tested at `lr_head 0.00012`.
* two runtime flags:
  * `--compile-mode {default,max-autotune}` — `torch.compile` mode; defaults to `default`.
    `max-autotune` is ~1.2x faster end-to-end and costs minutes of one-off autotuning at
    startup.
  * `--val-every-steps N` — validate every N steps instead of once per epoch (`0`/omitted
    means per epoch). Essential for long runs where one epoch is thousands of steps.
  * Both are deliberately CLI flags rather than config fields. `sha8` hashes the config and
    names the run directory, so adding them to the config would change the run name and
    break resumability of existing runs.
* staged training is available: `freeze_oft: true` holds the adapters at identity while
  the head learns, then `--init-from runs/<sha8>` warm-starts the joint phase.

### Learning rates

The `lr_head` sweep is the largest single lever we found. Matched arms (squash,
`lr_oft 0.0005`, seed 1337, 8 epochs):

| `lr_head` | micro F1 @0.5 | macro F1 @0.5 | tuned macro F1 |
|---|---|---|---|
| 0.004 (old) | 0.3888 | 0.0246 | 0.0277 |
| 0.00025 | 0.4048 | 0.0249 | 0.0327 |
| 0.00012 (best) | **0.4142** | 0.0223 | **0.0331** |
| 0.00006 | 0.4068 | 0.0183 | 0.0326 |

* The old 0.004 was **16-33x too high** — 16x the 0.00025 bracket, 33x the 0.00012 optimum
  — and the old 0.002 default (also the frozen probe's pick) was ~16x too high. Lowering
  `lr_head` from 0.004 to 0.00012 raises micro F1 6.5% and tuned macro F1 ~20%.
* There is a small micro/macro trade inside the bracket: 0.00012 maximises micro and tuned
  macro, while 0.00025 buys fixed-threshold macro (0.0249 vs 0.0223) at a micro cost
  (0.4048 vs 0.4142). At 0.00006 micro is past its peak and fixed-threshold macro
  collapses to 0.0183.
* `lr_oft` is bracketed on **both** sides at the corrected head LR — 0.0002 gives 0.4027,
  0.0005 gives 0.4142, 0.002 gives 0.4100 micro — so 0.0005 is an interior optimum, not a
  plateau edge. 0.02 is catastrophic and destroys the features (micro F1 0.1427), and
  `lr_oft = 0` is the frozen-probe floor at 0.3279.
* **Do not pick `lr_head` from the frozen-feature probe.** It suggests 0.002-0.005 because
  that is optimal for fixed features; the adapter later reshapes those features and moves
  the optimum down to 0.00012. Re-run the probe on trained-adapter features before trusting
  a head LR from it.

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

| variant | ms/image | img/s | ms/step |
|---|---|---|---|
| eager | ~11.2 | ~89 | ~360 |
| `torch.compile(mode="default")` | ~8.5 | ~118 | ~271 |
| `torch.compile(mode="max-autotune")` | 5.9-6.6 | 150-170 | ~190-211 |

`max-autotune` is ~1.2x faster end-to-end than `default` and pays minutes of one-off
autotuning at startup; select it with `--compile-mode max-autotune`.

The eager optimizer update costs ~88 ms: SinkSGD_ADV dispatches once per tensor, and
there are 73 tiny adapter tensors (~1.2 ms each). That cost is launch overhead, not
arithmetic — `stochastic_rounding` and `sinkhorn_iterations` do not move it. Compiling the
update, with Dynamo's recompile limits raised (`train.py` does this; otherwise
`fullgraph=True` hard-fails with `FailOnRecompileLimitHit`), brings it to ~28 ms. Data
loading is deliberately trivial: pre-decoded uint8 shards, memory-mapped, no worker
processes (see below), normalisation on the GPU in BF16. Data and H2D are ~10 ms and ~4 ms
per step and are not the bottleneck.

### Memory notes

Opening all 16 train shards in one process costs ~8.8 GB of private commit on Windows, and
each extra opening of the same file costs another copy, so:

* every shard is opened once per process through a module-level cache (`data.open_shard`);
* `num_workers=0` — workers would each pay their own copy, and loading is not the bottleneck;
* `CacheDataset.__getstate__` strips mapped tensors so a spawned worker can never receive
  a pickled copy of the cache.

With those, commit stays flat at ~55 GB across epochs instead of climbing ~18 GB per epoch.

## Limitations

* 10,000 images against 12,476 labels is 0.8 images per label: the long tail is not
  learnable at this scale, and macro-F1 over the full label space is dominated by tags we
  cannot predict. Our best micro F1 is 0.4142 and tuned macro 0.0331, against the bar's
  0.6800 and 0.1938 (0.2008 tuned). The baseline's 5.3 M training images are the
  difference; OFTv2 does buy a real +26% relative micro F1 over a frozen probe of the same
  backbone (0.4142 vs 0.3279).
* Per-label thresholds tuned on 1,000 validation images overfit sparse labels; the
  tuned table is reported for completeness next to the fixed-0.5 table, not instead of it.
