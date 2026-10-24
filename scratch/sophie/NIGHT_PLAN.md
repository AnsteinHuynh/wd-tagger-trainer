# Night of 20-21 Oct: state, predictions, and the plan to beat ConvNeXt

## Where we are

**Run `5f491a2f` is training** (started ~22:30 from step 0, finishes ~08:30):

| setting | value |
|---|---|
| backbone | DINOv3-ViT-B/16, 85.7M, **frozen** |
| adapter | OFTv2, 1.30M, fp32 master weights |
| head | `cls_mean_max_std` 3,072-d, **fp32**, 38.34M params |
| trainable | 39.64M (5.7% of the bar's 692.6M) |
| batch | 48, ~149 img/s, 16.9 GB |
| schedule | 110,870 steps, cosine with a **5% floor** (ends at lr_head 6.0e-6, lr_oft 2.5e-5) |
| checkpoints | every 5,000 steps, eval-validated on 1,000 images |

Config: `configs/experiments/_wide_fp32head.json` (sha8 `5f491a2f`).

## Predictions for the morning

Fitted from the two measured points of the baseline (5,000 and 10,000 steps), nudged for the wide+fp32 head being ~0.003 ahead at matched steps:

| metric | predicted final | bar (squash) | fraction of bar |
|---|---|---|---|
| micro F1 | 0.56 - 0.61 (centre 0.58) | 0.6800 | ~85% |
| macro F1 | 0.08 - 0.12 (centre 0.10) | 0.1938 | ~52% |

Two forces cancel in that range: saturation pulls the log-linear fit down, cosine decay pulls it up.

## Why this will almost certainly NOT beat ConvNeXt

The bar fine-tunes **692.6M parameters on the same 5.3M images we are using**. We are adapting **39.64M** — and, critically, the 85.7M-param feature extractor is **frozen for the entire run**. Its representation at step 110,870 is identical to its representation at step 1. Everything we gain comes from a linear head and a 1.3M-param rotation fitted on top of a fixed feature space.

Micro needs +20% relative to close. Macro needs **+100%**. No amount of the same data closes that; the missing ingredient is feature adaptation.

Probabilities: micro ~15%, macro ~8%, both ~5%.

## The ladder to actually beat it

| step | experiment | why | cost |
|---|---|---|---|
| 1 | **Unfreeze the last 4-6 transformer blocks** | lets features specialise to the tag distribution, which is where macro lives | ~+20-30% step time, one 9.5 h pass |
| 2 | Raise the LR floor to 10% (Chinchilla/GPT-3) | only matters at the tail; cannot be A/B'd cheaply | free, bundle into step 1 |
| 3 | ViT-L/16 (`I:\Models\dinov3-vitl16...`, 303.1M, batch 20, 27.2 img/s) | better frozen features, but 3.3x slower — ~54 h per pass, so only if steps 1-2 plateau | expensive |
| 4 | Second pass over the corpus | cheap if the run ends still climbing | 9.5 h |

Step 1 is the one that matters. Steps 3-4 are escalation, not defaults.

## Open design questions for unfreezing (need the user, not for the bot to guess)

1. **Which blocks, and does OFT stay on them?** Recommendation: for the last N blocks, *drop* the OFT wrapper and fine-tune the block fully (weights + LayerNorms). Keeping both a rotation and full weights on the same layer is redundant and confusing.
2. **What LR for the unfrozen blocks?** This is the risky number. SinkSGD_ADV is *normalised*: step size ≈ lr regardless of gradient magnitude, so each weight moves `lr` per step. For a weight of ~0.05 with a random-walk drift of `lr·sqrt(N)`, over 111k steps: lr 5e-5 gives ~33% of weight scale, lr 1e-5 gives ~6%. Start conservative (1e-5 to 2e-5) — destroying DINOv3's features is the failure mode.
3. **Does the head need to move too?** It is already at its optimum LR (1.2e-4), so leave it.

## OFT saturation: measured, and it is a real constraint

Measured by `scratch/sophie/oft_saturation.py` (rebuilds each block's skew matrix, applies the same 5-term Neumann series as `oft.py`, compares against the exact Cayley).

**Representational headroom: fine.** The median block has rotated **27° of a possible 180°**, max 78°. OFT is using ~15% of its rotation range and theta is still growing, so the adapter is nowhere near its expressive limit.

**Numerical accuracy: degrading.** The 5-term truncated Neumann series is only valid for small `||Q||`, and the tail has already left that regime:

| | step 5,020 | step 10,096 |
|---|---|---|
| `||Q||` p50 / p99 / max | 0.186 / 0.547 / 0.706 | 0.238 / 0.608 / **0.811** |
| relative error vs exact Cayley p50 / max | 2.2e-4 / 17.5% | 7.7e-4 / **35%** |
| blocks with >1% / >5% / >20% error | 99 / 26 / 0 | 179 / 53 / 3 |
| cond of applied matrix, max | 1.015 | 1.060 |

The median block is accurate to 0.08%, but the worst blocks apply a rotation that is 35% wrong, and the bad-block count grows with training. Extrapolating the same trend to 110,870 steps puts the median `||Q||` near 0.7, where the 5-term series loses 10-20%.

**Why it matters for the goal:** theta grows with training time, so the more we train, the more of the adapter runs where its own forward pass is inaccurate. This plausibly explains why the `lr_oft` bracket peaked at 5e-4 — a higher LR grows theta faster into the invalid region, so the bracket may have been measuring the truncation limit rather than the LR optimum. Raising `num_cayley_neumann_terms` from 5 to 10 costs **<0.2%** of step FLOPs (~0.9 of ~505 GFLOP) and removes the cap; it is worth bundling with the unfreeze experiment, and then re-bracketing `lr_oft` upward.

Overnight, do NOT restart the run for this (at 10k steps only 2% of blocks exceed 5% error). Instead, record the real curve: run `oft_saturation.py --append` on each new checkpoint so the growth is measured rather than extrapolated.

## Nested-validation study (the important one)

Scored the bar and our trajectory on nested valid subsets (1k ⊂ 5k ⊂ 10k ⊂ 50k), same images for every model, via `scratch/sophie/nested_eval.py` (one streaming pass, per-label TP/FP/FN/TN counts, snapshots at each size). Results in `runs/_nested_eval.json`.

**The eval-size effect is enormous.** Same bar model, same nested images, only the subset size changes -- and macro moves 3x, because only 37% of labels occur in 1,000 images:

| images | ceiling (labels seen) | bar macro | bar % of ceiling |
|---|---|---|---|
| 1,000 | 0.373 | 0.1831 | 49% |
| 5,000 | 0.725 | 0.3768 | 52% |
| 10,000 | 0.863 | 0.4559 | 53% |
| 50,000 | 0.998 | 0.5493 | 55% |

So a 1,000-image macro number is a *different quantity* from a large-set macro number. Compare models only at matched eval size, or in fraction-of-ceiling terms.

**At 50,000 images (ceiling 0.998):**

| model | micro F1 | micro MCC | macro F1 | % of ceiling |
|---|---|---|---|---|
| convnextv2_huge | 0.6816 | 0.6827 | 0.5493 | 55% |
| ours 5k | 0.4780 | 0.5095 | 0.1021 | 10% |
| ours 10k | 0.5073 | 0.5313 | 0.1222 | 12% |
| ours 15k | 0.5080 | 0.5349 | 0.1301 | 13% |
| ours 20k | 0.5156 | 0.5420 | 0.1371 | 14% |
| ours 25k | 0.5370 | 0.5554 | 0.1599 | 16% |

**Our macro trajectory is NOT flat** -- at 50k resolution it rises 0.1021 -> 0.1599, +57% relative, over steps 5k->25k. The apparent plateau I diagnosed from the 1,000-image val (16% of ceiling for three readings) was pure sampling noise. Lesson: 1k-image macro has enough variance to fake a plateau; judge progress on >=10k images.

Micro is nearly eval-size independent (bar 0.6717 -> 0.6816, ours 0.5268 -> 0.5370) because it is dominated by common tags. Macro is the metric that responds to subset size.

**Gap ratios (stable across sizes):** micro we are at ~78-79% of the bar, macro at ~29-36%. Note the macro ratio is *worse* at 50k (29%) than at 1k (36%) -- the bar is better at rare labels than the small-set measurement flattered, so the macro gap is real, not dilution.

**Projection to step 110,870** (log-linear on the 50k-image points): micro ~0.59, macro ~0.21 on a 50k eval, i.e. roughly micro 0.59 / macro 0.07 on a 1,000-image test. Still well short of the bar's 0.6800 / 0.1938, but micro would be ~87% of bar, not 81%.

## Momentum study (3 arms to matched steps)

SinkSGD_ADV is *normalised*: the step is ~lr regardless of gradient magnitude. So momentum does not scale the step the way it does in SGD (`lr/(1-beta)`). It only shapes direction. But that hides a trap: **with a fixed step size, memory is distance.** A direction that stays true for the length of the memory carries the weights `horizon x lr`.

For the head (lr 1.2e-4, weights ~0.013):

| momentum | horizon | stride `h x lr` | vs weight scale |
|---|---|---|---|
| 0.95 | 20 steps | 0.0024 | 18% |
| 0.99 | 100 | 0.012 | ~100% |
| 0.998 | 500 | 0.060 | **4.6x** |

Measured at step 5,000 (b=128, exact Cayley, fp32 head, wide head), single 1,000-image val:

| arm | micro | macro | supported macro | macro recall |
|---|---|---|---|---|
| 0.95 | **0.4793** | 0.0501 | 0.1353 | 0.1165 |
| 0.99 | 0.4760 | **0.0503** | **0.1357** | **0.1188** |
| 0.998 | **0.3642** | **0.0220** | 0.0594 | 0.0483 |

0.998 is a wipeout: -24% micro, -56% macro. The stride arithmetic predicted exactly this. 0.95 and 0.99 are a coin flip at 5k, so both were extended to 10,000 for a second paired point (same 1,000 images for every arm, so differences are paired, not independent).

**Block-size verdict**: at step 5,000, b=128 (micro 0.4793 / macro 0.0501 / supported 0.1353) beat the paused b=32 run (0.4721 / 0.0494 / 0.1334) on all three. At step 10,000 arm 1 reached micro 0.5091 / macro 0.0585 / supported 0.1581, versus the old b=32 cls_mean+bf16-head baseline at 10k of 0.4923 / 0.0523 / 0.1411 -- i.e. +0.0168 / +0.0062 / +0.0170. The stacked changes (block 128 + exact Cayley + fp32 head + wide head) are real.

## Safe commands

Resume the running job by **directory**, never by config path (adding config fields changes the hash of an unpinned file and would start a fresh run dir):

    python -u -m wdtag.train --run runs/5f491a2f --ckpt-every-steps 5000 \
        --val-every-steps 5000 --compile-mode max-autotune

Frozen-feature backbone probe (after training, keeps the GPU busy):

    python -u -m wdtag.probe --backbone-dir "I:\Models\dinov3-vitl16-pretrain-lvd1689m" \
        --backbone-tag l16 --extract-batch 8 --splits train,valid,test

## Known hazards

- `torch.cuda.mem_get_info()` read from a *separate* process reports ~2.5 GB in use while the trainer holds 16.9 GB. Do not use it to judge liveness; read the step lines.
- 1,000-image macro has a standard error around **±0.02**, so treat any macro delta below that as noise. This is why the wide-head win (+0.0053 micro, +0.0143 supported) is suggestive rather than proven, and why 3 consecutive declines are the switch signal.
- A change of head width or any input shape invalidates the Triton autotune cache and costs ~5-7 min of recompilation at startup.