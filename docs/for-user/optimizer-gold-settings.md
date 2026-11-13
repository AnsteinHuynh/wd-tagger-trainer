# Gold-star optimizer settings, tested

If you are a future agent and you are about to import a starred optimizer stack from a sister project, read this first. The bundle everyone means when they say gold star is not a blessing, it is a hypothesis, and on this model it lost by an order of magnitude in three separate configurations. What follows is the prescriptive version: what to use, what to avoid, why, and how to measure the next switch so you do not repeat our mistakes.

The short verdict. Adopt the scoping mechanism and two of the eight switches, reject four, and never enable spectral normalisation. Everything below is measured on this repo's own model rather than inherited.

## Use these settings

| setting | value | grade | evidence |
| --- | --- | --- | --- |
| `optimizer_switch_groups` | ["oft", "block"] | M | best measured scope, 0.0741 macro at step 1000 on the 50k |
| `optimizer_normed_momentum` | true, scoped | M | part of the couple, worthless without scoping |
| `optimizer_snr_cond` | true, scoped | M | cannot run without normed momentum, measured only as a pair |
| `optimizer_state_precision` | fp32 | I | 0.1498 against 0.1490 for bf16, inside noise, kept for drift safety |
| `compiled_optimizer` | true | M | 88 ms of dispatch per step down to about 28 ms |
| `momentum` | 0.95 | H | never actually trained at another value |
| `optimizer_nesterov` | false | M | 0.0677 against 0.0676 baseline, neutral, so no reason to pay for it |
| `orthogonal_sinkhorn` | false | M | neutral on score and 6 percent of throughput |
| `optimizer_orthogonal_gradient` | disabled | M | neutral on score and 5 to 9 percent of throughput |
| `optimizer_spectral_normalization` | false | M | harmful at every calibration, see the section below |
| `cayley_terms` | 0, exact | M | one block sits at ||Q|| 0.898 where the Neumann series is 58 percent wrong |
| `weight_decay` | 0, untested | — | arms 54 to 56 were built and never ran |

## The rule that decides everything: scope by group

Three things train: the 70 OFTv2 adapter tensors, the one unfrozen block, and the 12476-way head. The frozen backbone never reaches the optimizer. The geometry switches are safe on the first two and poisonous on the third, and that single fact explains every failure this project had with them.

`optimizer_switch_groups` takes a subset of "oft", "block" and "head". Empty means all of them. `optimizer_oft_only` is the older spelling of ["oft"] and still works. Excluded groups get explicit plain overrides for every switch, so an excluded group is genuinely plain rather than plainly-missing-one-override.

Every cell measured at the same step and same seed, and displacement matched rather than rate matched: the plain row runs at lr_oft 5e-4 while every switched row runs at 2.61e-4, because the couple moves the weights 1.916x further at equal rate and the two should spend the same step length. Scored on the fixed 0.40 protocol over 50000 images:

| adapters | head | block | 50k macro | 50k micro | arm |
| --- | --- | --- | --- | --- | --- |
| off | off | off, lr 5e-4 | 0.0596 | 0.3463 | 82 |
| on | on | on | 0.0085 | 0.4277 | 46 |
| off | on | off | 0.0097 | 0.3830 | 79 |
| off | off | on | 0.0693 | 0.3640 | 80 |
| on | off | off | 0.0684 | 0.3784 | 71 |
| on | off | on | 0.0741 | 0.3859 | 81 |

The head-only row is the proof rather than the inference: with the couple on the head alone, macro falls from 0.0596 to 0.0097 while micro moves 0.3463 to 0.3830. Macro destroyed, micro intact, which is a deleted frequency prior and not a damaged model. A per-row Sinkhorn on 12476 tag rows forces every weight row to unit RMS, and our tag frequencies span four orders of magnitude, so the prior that says rare tags need a different scale is the first thing it throws away. The row-norm coefficient of variation collapses from 0.255 to 0.021. `scratch/sophie/head_rows.py` reproduces it.

The block likes the couple. Channel rows in a 768 by 768 projection are interchangeable in the way adapter blocks are, and block only beats plain on both metrics. The best configuration measured is adapters plus block, worth 0.0145 macro and 0.0396 micro over plain, and 0.0057 macro over adapters alone — the last of which is inside the 0.0093 wide bootstrap interval at 50k, so treat that specific margin as promising rather than proved.

## What each switch actually does to the step

Measured with `scratch/sophie/measure_rung_displacement.py` over the live u1 inventory, weights restored between probes so parameter norm cannot drift the answer. Ratios are against the plain stack.

| stack | displacement | lr multiplier to match | note |
| --- | --- | --- | --- |
| plain, bf16 state | 1.000 | 1.000 | reference |
| plain, fp32 state | 1.000 | 1.000 | state precision is scale neutral |
| normed momentum plus snr | 1.916 | 0.522 | uniform throttle up, no shape gating |
| plus orthograd iterative | 1.916 | 0.522 | neutral to seven digits in the inventory |
| plus orthogonal sinkhorn | 1.916 | 0.522 | neutral to seven digits in the inventory |
| spectral, untagged | 0.0151 | 66.2 | shape gated |
| spectral, tagged with `_is_oft` | 0.0229 | 43.7 | better and still catastrophic |

The couple on a 768 by 768 block tensor measures 1.400, which is why the production block rate is divided by 1.400 rather than 1.916 when the scope includes blocks.

Two lessons live in that table. A switch can be judged by score alone only if its displacement is near 1.0; orthograd and sinkhorn qualify, spectral does not and never will. And the couple is a throttle as much as a direction, so any arm testing it must state whether it is displacement matched.

## The four that fail, and how

Spectral normalisation. Three calibrations, all at step 1000 of the same schedule: plain 0.0559 macro and 0.4386 micro; untagged at 5e-4, 0.0004 and 0.1788; tagged with `_is_oft`, 0.0007 and 0.2066; tagged with per-group rate compensation 2.19e-2, 4.13e-3 and 1.6e-3, 0.0078 and 0.3802. The tag improves the step size 76x and the score by 0.0003. Compensating each shape's throttle separately recovers another 0.0071 and still lands 7x below plain. It is structural: a (6, 8128) OFT factor, a (12476, 768) head and a (768,) bias need three different multipliers and one learning rate cannot supply them.

The couple everywhere. 0.0085 against 0.0559 at step 1000 in-run, and matching the displacement does not rescue it, 0.0067 for the matched arm against the same 0.0559. The head is the poison, proven by the head-only row above.

Orthogonal gradient. Both modes neutral to three decimals on both metrics, 0.0686 iterative and 0.0687 flattened against 0.0684 control, and it costs 6.7 and 8.9 percent of throughput. On a rank-6 factor there is no large matrix whose singular directions need flattening, so the theory has nothing to grip.

Orthogonal sinkhorn. Neutral when scoped, 0.0679 against 0.0676 in-run, and it costs 5.9 percent of throughput. It is expensive rather than harmful, which makes it the easiest thing here to leave off.

## Struck pages, kept in the book

The bundle scored 0.0114 macro against plain 0.1490 on our own arms 48 and 42, thirteen times worse, while the sister project's own matched pairs were ties — 24.41 against 24.82 and 39.605 against 39.282 on their metric. So the bundle is not proven bad in general; it is proven bad here, on a 12476-way head with four orders of magnitude of tag frequency. Do not re-litigate the general claim, and do not re-run the bundle unscoped either; the mechanism is the head and it is now measured four ways.

The first version of this file said spectral throttled the adapters 66x. That was an artifact of measuring the untagged path. The tag is worth 76x, the true throttle is 43.7x, and the earlier number is struck rather than deleted because the mistake is the lesson.

The first verdicts on nesterov and orthogonal sinkhorn were wrong, and wrong in an instructive way. Both arms were scoped with `optimizer_oft_only`, whose plain override covered only normed momentum, snr, spectral and orthograd, so nesterov and sinkhorn leaked onto the head and the block. Nesterov looked like a 0.0007 macro loss and sinkhorn looked like 0.0101, and both were the head reacting. With the override extended to all eight switches, both are neutral. If you add a switch, add it to the plain override in the same commit or your scoped arms are not scoped.

`state_precision: factored` was ranked as the first thing to test on the theory that smaller state buys a bigger micro-batch. Then the bytes were counted: the whole optimizer state is about 80 MB against 21.2 GB of activations, so factored form would save 0.36 percent of VRAM and cannot buy a single image. It is lossy, since it stores a rank-1 outer product with one sign bit per element, reconstructed every step. Do not test it here.

## The measurement rules that make any of this meaningful

One instrument at a time. In-run validation is a 10000-image subset; the 50k protocol is a separate `wdtag.validate` process; they are not comparable and every table in this file states which one it used.

Know the noise floor before comparing arms. Bootstrapped 95 percent intervals on the champion's cached probabilities at 0.40: at 1000 images macro is 0.0158 wide and micro 0.0127; at 5000, 0.0121 and 0.0053; at 10000, 0.0109 and 0.0042; at 50000, 0.0093 and 0.0016. Macro barely tightens with more images because its uncertainty is about tags, not pictures, and the supported tag count grows from 5732 to 12466 across those subsets. At 50k, two arms within 0.009 macro are the same arm. `scratch/sophie/ci_bootstrap.py`.

One arm at a time, launched by hand, read before the next one. Never score while training runs, on a 24.5 GB card a second process pushes 21.2 GB to 23.98 GB and the trainer collapses to 17 img/s.

Keep the exact Cayley. At the production checkpoint the median block sits at rotation angle 31.5 degrees, p99 68.2, max 83.8, and the largest ||Q|| is 0.898 in the head's own adapter. Under the five-term Neumann series that block would apply a matrix 58 percent wrong, silently. `scratch/sophie/oft_saturation.py`.

## How to test the next switch

Write the arm at the configured learning rate first, with the scope you intend to ship, and state its displacement. Score it on the 50k only if it is inside 0.009 macro of the control, otherwise the in-run number has already answered. Check throughput from the log's img/s rather than assuming, since three of the "free" switches cost 6 to 9 percent. If it wins, rerun at matched displacement to separate geometry from throttle. Never adopt on a single seed unless the margin exceeds the interval above.

## What is still open

Weight decay has never been trained here, at any value, and on a five-million-image pass that is the largest untested knob on the board. `centered_wd` is the natural regulariser for an adapter that should stay near identity and is not even plumbed. Momentum 0.95 is inherited from the sisters and interacts with normed momentum, because renormalising the buffer changes how much history survives the decay. `sinkhorn_iterations` 5 has never been varied. A scoped arm with the couple on adapters and block is the configuration production now runs, so any new switch should be tested on top of that baseline rather than on the old adapters-only control.
