# MAIN RUN — the recipe, the budget, and the gates before it

The full-corpus run that chases the bar. Everything here is measured or computed from measurements in `docs/benchmark.md`, `docs/best-settings.md` and the private experiment log; where it is extrapolation it says so.

## What it is

SigLIP2-B/16-512 as the adapted tower, OFTv2 at block size 128 on every Linear and the patch-embedding conv, the last block unfrozen, cls pooling, ASL with gamma_neg 2.0, bf16 with an fp32 head, two passes over the 5,321,713 training images.

That recipe is arm 16 `u1-C-L-asl(g2.0)-siglip2b` in the taxonomy, which scored macro 0.0755 and micro 0.4375 on the 10k benchmark at 50k scoring, and it is the best pair of numbers any arm has produced.

## The command

```
python -m wdtag.train --config configs/main/01_u1-C-L-asl-g2.0-siglip2b-main.json --select-metric macro040 --val-every-steps 10000 --ckpt-every-steps 5000
```

One re-run of exactly that command resumes from the newest checkpoint in the run directory, and `.PAUSE` at the run directory or at the workspace root makes it checkpoint and exit at the next step.

Three of those four flags are load-bearing and not in the config file. `--select-metric macro040` picks the checkpoint by the protocol's metric. `--val-every-steps 10000` is required because in stream mode one epoch is the whole schedule, so per-epoch validation would give one number at the end and nothing to select from. `--ckpt-every-steps 5000` is required because the compiled default is 1000, which would write 221 checkpoints at 248 MB, 55 GB of disk and a write every four minutes; 5000 gives 44 files and 11 GB, and a reaped job still loses at most 5000 steps, about 35 minutes.

## Step and hour arithmetic

5,321,713 images at batch 48 is 110,869 steps per pass, so two passes is 221,739 steps. In stream mode the trainer takes one pass as the whole schedule, `total_steps = ceil(stream_images / batch)`, so two passes is expressed as `stream_images: 10643426` rather than `epochs: 2`, and the pass index chains across the corpus boundary so no image repeats before the second traversal starts.

Throughput is now measured rather than estimated for this tower on the real tar: the 600k gate ran at 114 img/s at one unfrozen block, streaming, batch 48, 8 decode threads, 15.8 GB. That makes two passes 93,400 seconds, or 26 hours, and one pass 13 hours. The paused full-corpus run measured 83 img/s at twelve unfrozen blocks with the old recipe, which is consistent with 114 at one, and the 10k cache figure of 138 img/s is the same tower reading from RAM instead of from a tar.

## The config, field by field

| field | value | why |
|---|---|---|
| `backbone_dir` | `models\siglip2-base-patch16-512` | the tower, +21% macro over DINOv3 at identical recipe |
| `stream` | true | the corpus lives in tar shards; exact resume by consumed-sample cursor |
| `stream_images` | 10643426 | two passes |
| `batch_size` | 48 | 15.8 GB measured at this tower in stream-free mode, so the stream build fits |
| `unfreeze_blocks` | 1 | 2.2% behind 12 blocks in macro on the old tower at 18% more speed; untested on SigLIP2 |
| `head_pool` | cls | +58% macro over the four-part pooling in the new recipe |
| `head_hidden` | 0 | a 2048 bottleneck lost 19.5% macro in the real recipe |
| `head_fp32` | true | bf16 on a 12476-way head stalls late in cosine |
| `loss` | asl | |
| `loss_gamma` | 2.0 | brackets 4.0 and 1.0, peak macro, +11.9% micro at equal macro |
| `lr_oft` | 5e-4 | unchanged from every arm; cosine over 221,739 steps holds it near peak for far longer than the arms did |
| `lr_unfreeze` | 3e-5 | |
| `lr_head` | 1.2e-4 | |
| `warmup_steps` | 2000 | 0.9% of the schedule; the arms used 50 because their schedule was 5225 steps |
| `block_size` | 128 | best of 32/64/128 |
| `cayley_terms` | 0 | exact solve, ~9% slower, numerically exact |
| `compile_model` | true | `default` for exploratory, `max-autotune` for anything this long |
| `compiled_optimizer` | true | mandatory with 76 adapter tensors |
| `ckpt_every_steps` | 5000 | CLI flag, not a config field: 44 checkpoints at about 248 MB, 11 GB total |
| `val_images` | 50000 | the largest nested subset, and the protocol's number |
| `seed` | 1337 | |

## Validation and selection

`--val-every-steps 10000` gives 22 validations across two passes at about 170 seconds each, which is 1.7% of wall time. `--val-every-steps` is required in stream mode: without it validation fires once per epoch and the epoch is the whole schedule, so there would be exactly one number at the end and nothing to select a best checkpoint from.

`--select-metric macro040` selects `best_val.pt` on macro F1 at the fixed 0.40 threshold, which is the number the bar comparison uses. This is a 2026-10-22 addition and it fixes two defects in the old path, both recorded in the private experiment log at 18:42; it is unit-verified but its end-to-end smoke test is still owed, so it must not be the flag the main run depends on until that test passes.

At 50000 validation images all but 22 of the 12476 labels have support, so all-labels macro and supported-labels macro converge and the ambiguity that makes the 1k numbers misleading disappears.

## Pause, resume, checkpoints

`.PAUSE` or `.PAUSE.txt` or `.PUASE` or `.PUASE.txt` or `PAUSE`, at the run directory or the workspace root, exits cleanly at the next step with a checkpoint written at the unconsumed batch. Resume is the same command again. Each checkpoint stores the stream cursor as `(pass_index, batch)`, so resume reads tar headers only and never decodes an image it already saw.

Two failsafes worth knowing. The run directory is the sha8 of the config, so editing any field of that config file re-keys the run and a resume would start a fresh directory instead of continuing; there is a guard that refuses to resume a different directory, and it should be left alone. And `ckpt_every_epochs` must stay 0, because in stream mode the epoch is the whole pass and an epoch save would write a second checkpoint at the same instant.

## What the numbers should look like, and how much to trust that

The bar is macro 0.5579 and micro 0.6886 at 0.40 on the 50k. The paused full-corpus run, with the old recipe and the old tower at 0.78 of a pass, scored 0.2815 and 0.6133.

Today's 10k ratios, if they composed, would give macro 0.35 to 0.45 and micro 0.62 to 0.68. They do not compose and that is not a scaling law: the tower swap is worth +21%, the gamma change +11.9% micro, the pooling change +58% macro over the old four-part head, and those multipliers were each measured at 10k images where the model is data-starved and where the ranking of recipes is not guaranteed to survive thirty times more data.

Since the 600k gate ran, there is a better anchor than composed ratios. Its 50k score was macro 0.2733 after 600000 images, and its internal curve was still climbing at the end, from 0.0680 at 60000 images to 0.2278 at 540000 with no bend. Fit loosely on that shape, 0.16 of macro per decade of images on the 10k subset, another 0.95 of a decade to 5.3M images adds roughly 0.15, and the 50k score has been running about 0.045 above the 10k subset at equal checkpoints, so the first pass should land near 0.42 macro. That is a loose fit on a schedule that has not been run to completion and it carries the usual warning, but it is an anchor rather than a hope, and it puts the bar's 0.5579 inside the range a second pass could reach.

Micro is the weaker half of that projection. The gate's 0.4791 came with micro recall 0.7503 and precision 0.3519, which is a model whose probabilities are still spread out at the fixed threshold rather than a model that is badly ranked. Sharpening that is what more data buys, and it is the number to watch at the ten-thousand-step marks.

So the honest expectation is that the first full pass lands somewhere between the paused run and the bar on macro, with micro plausibly at or above the bar, and that the run should be judged by where the 50k macro curve is at the end of pass one against where it was at the end of pass two, not by hope.

## Gates before launch, in order

One. The `--select-metric macro040` smoke test. Passed 2026-10-22 at 20:15 in `runs/671e6989`, at batch 4 so it could not take the card from anything else: both mid-epoch validations selected, and the 0.40 number differed from the best-threshold number at the same step, which is the whole reason the protocol needs its own metric. Nothing owed.

Two. Mean pooling instead of the MAP pooler, tested 2026-10-22 21:20 in `runs/c8c22df5`: macro 0.0731 and micro 0.4315 against the pooler's 0.0755 and 0.4375, so -3.2% and -1.4%. The pooler is not discarding tag detail at the level this benchmark resolves, `head_pool: cls` stands, and the config below is unchanged.

Three. The mid-scale gate. Run 2026-10-23 in `runs/2b6b81aa` at 600000 images: macro 0.2733 and micro 0.4791 after 93.5 minutes, which is 97% of the paused full-corpus run's macro from 14% of its images, with the curve still climbing. The twelve-block half, `runs/77a135b2`, is five to eight times behind at matched steps because `unfreeze_last_blocks` freezes the rotation inside each unfrozen block, so it is not a capacity comparison at all; u1 stands and `lr_unfreeze` is the knob that would make that arm meaningful.

Four. Then the main run, as one harness background job, watched at the 10000-step marks.
## What is still unknown about this tower

Whether freeze depth 1 is optimal for features that are semantic rather than structural, which needs its own arm at 13 minutes per unfrozen block.

Whether a larger SigLIP2, `siglip2-large-patch16-512` at 24 blocks and 1024 wide, converts parameters into macro the way the bar cashes 692.6M of ConvNeXt into 0.5579. At 33 img/s and batch 20 the same estimation puts one pass near 45 hours, so it is a two-day decision, not an arm.

Whether the corpus stream feeds this tower at the rate the cache does; the first thousand steps of the main run will answer it and nothing before will.
