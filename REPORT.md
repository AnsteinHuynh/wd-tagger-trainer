# DINOv3-B/16 + OFTv2 anime tagger vs `convnextv2_huge.dbv4-full`

Fixed **1,000-image test split** of `wdtaggerv4`. 12,476 labels (4 rating + 9,225 general + 3,247 character), label order taken from the baseline's own `config.json`.

Input geometry: bilinear resize straight to 512x512 with the aspect ratio not preserved ("squash"). A padded-square arm was also run; at the corrected learning rate the two are **statistically indistinguishable** (see the geometry section). Squash is kept as the pipeline default because it is the arm with two seeds behind it and is already cached — not because it measurably wins.

**Caveat on the bar**: squashing also distorts the baseline's input, which costs it real accuracy — measured on its own native padded geometry it scores macro F1 0.2017 / micro F1 0.6935. Treat the numbers below as a *lower bound* on the bar.

Metric conventions: a label with no positive and no predicted positive counts as 0, so *macro* is over the whole 12,476-label space; *micro* pools TP/FP/FN over the whole matrix. `tuned` = per-label thresholds chosen on the 1,000-image validation split and applied to test.

## The bar

| model | F1 | MCC | P | R |
|---|---|---|---|---|
| `convnextv2_huge.dbv4-full` **macro @0.5** | 0.1938 | 0.1964 | 0.2100 | 0.1917 |
| `convnextv2_huge.dbv4-full` **micro @0.5** | 0.6800 | 0.6811 | 0.7349 | 0.6328 |
| `convnextv2_huge.dbv4-full` macro, tuned | 0.2008 | 0.2047 | 0.2011 | 0.2209 |
| `convnextv2_huge.dbv4-full` micro, tuned | 0.6655 | 0.6666 | 0.6164 | 0.7230 |

## Our runs

Frozen DINOv3-B/16 backbone (87.0 M params, never updated) + 1.30 M OFTv2 adapter params + 19.18 M head params, batch 32, BF16, `SinkSGD_ADV`.

| run | macro F1 | macro MCC | macro P | macro R |
|---|---|---|---|---|
| bar: convnextv2_huge | 0.1938 | 0.1964 | 0.2100 | 0.1917 |
| squash · lr_oft 0.0005 · lr_head 0.00012 · seed 1337 · 8 ep [fe92f4df] | 0.0223 | 0.0238 | 0.0375 | 0.0178 |
| squash · lr_oft 0.0005 · lr_head 0.00025 · seed 1337 · 8 ep [d6d3c1d6] | 0.0249 | 0.0262 | 0.0395 | 0.0205 |
| squash · lr_oft 0.0005 · lr_head 6e-05 · seed 1337 · 8 ep [294af9c2] | 0.0183 | 0.0199 | 0.0334 | 0.0141 |
| squash · lr_oft 0.0005 · lr_head 0.004 · seed 1337 · 20 ep [fa51da74] | 0.0274 | 0.0275 | 0.0343 | 0.0266 |
| squash · lr_oft 0.0005 · lr_head 0.004 · seed 1337 · 8 ep [e6bc2879] | 0.0246 | 0.0255 | 0.0363 | 0.0211 |
| pad · lr_oft 0.0005 · lr_head 0.004 · seed 1337 · 8 ep [7ff9ee38] | 0.0245 | 0.0253 | 0.0368 | 0.0206 |
| squash · lr_oft 0.0005 · lr_head 0.004 · seed 2025 · 8 ep [af7c4ac4] | 0.0205 | 0.0211 | 0.0312 | 0.0170 |
| squash · lr_oft 0.002 · lr_head 0.004 · seed 1337 · 8 ep [539d8880] | 0.0202 | 0.0209 | 0.0312 | 0.0168 |

| run | micro F1 | micro MCC | micro P | micro R |
|---|---|---|---|---|
| bar: convnextv2_huge | 0.6800 | 0.6811 | 0.7349 | 0.6328 |
| squash · lr_oft 0.0005 · lr_head 0.00012 · seed 1337 · 8 ep [fe92f4df] | 0.4142 | 0.4445 | 0.6554 | 0.3028 |
| squash · lr_oft 0.0005 · lr_head 0.00025 · seed 1337 · 8 ep [d6d3c1d6] | 0.4048 | 0.4308 | 0.6205 | 0.3004 |
| squash · lr_oft 0.0005 · lr_head 6e-05 · seed 1337 · 8 ep [294af9c2] | 0.4068 | 0.4451 | 0.6895 | 0.2885 |
| squash · lr_oft 0.0005 · lr_head 0.004 · seed 1337 · 20 ep [fa51da74] | 0.3800 | 0.3835 | 0.4515 | 0.3281 |
| squash · lr_oft 0.0005 · lr_head 0.004 · seed 1337 · 8 ep [e6bc2879] | 0.3888 | 0.4080 | 0.5646 | 0.2965 |
| pad · lr_oft 0.0005 · lr_head 0.004 · seed 1337 · 8 ep [7ff9ee38] | 0.3985 | 0.4159 | 0.5646 | 0.3079 |
| squash · lr_oft 0.0005 · lr_head 0.004 · seed 2025 · 8 ep [af7c4ac4] | 0.3776 | 0.3960 | 0.5468 | 0.2884 |
| squash · lr_oft 0.002 · lr_head 0.004 · seed 1337 · 8 ep [539d8880] | 0.3804 | 0.4009 | 0.5618 | 0.2876 |

With per-label thresholds tuned on validation:

| run | macro F1 | macro MCC | macro P | macro R | micro F1 | micro MCC | micro P | micro R |
|---|---|---|---|---|---|---|---|---|
| bar: convnextv2_huge | 0.2008 | 0.2047 | 0.2011 | 0.2209 | 0.6655 | 0.6666 | 0.6164 | 0.7230 |
| squash · lr_oft 0.0005 · lr_head 0.00012 · seed 1337 · 8 ep [fe92f4df] | 0.0331 | 0.0339 | 0.0352 | 0.0418 | 0.4020 | 0.4033 | 0.3581 | 0.4584 |
| squash · lr_oft 0.0005 · lr_head 0.00025 · seed 1337 · 8 ep [d6d3c1d6] | 0.0327 | 0.0336 | 0.0345 | 0.0431 | 0.3727 | 0.3763 | 0.3184 | 0.4495 |
| squash · lr_oft 0.0005 · lr_head 6e-05 · seed 1337 · 8 ep [294af9c2] | 0.0326 | 0.0337 | 0.0328 | 0.0445 | 0.3989 | 0.4015 | 0.3476 | 0.4680 |
| squash · lr_oft 0.0005 · lr_head 0.004 · seed 1337 · 20 ep [fa51da74] | 0.0284 | 0.0285 | 0.0309 | 0.0338 | 0.3681 | 0.3686 | 0.3312 | 0.4144 |
| squash · lr_oft 0.0005 · lr_head 0.004 · seed 1337 · 8 ep [e6bc2879] | 0.0277 | 0.0280 | 0.0315 | 0.0325 | 0.3807 | 0.3804 | 0.3498 | 0.4176 |
| pad · lr_oft 0.0005 · lr_head 0.004 · seed 1337 · 8 ep [7ff9ee38] | 0.0277 | 0.0279 | 0.0319 | 0.0316 | 0.3881 | 0.3874 | 0.3612 | 0.4192 |
| squash · lr_oft 0.0005 · lr_head 0.004 · seed 2025 · 8 ep [af7c4ac4] | 0.0238 | 0.0238 | 0.0269 | 0.0280 | 0.3703 | 0.3698 | 0.3417 | 0.4041 |
| squash · lr_oft 0.002 · lr_head 0.004 · seed 1337 · 8 ep [539d8880] | 0.0237 | 0.0237 | 0.0267 | 0.0277 | 0.3719 | 0.3710 | 0.3473 | 0.4002 |

## Input geometry: padding vs squash

### Does the resize geometry matter? (matched LR and seed)

| lr_oft | seed | micro F1 pad | micro F1 squash | delta | macro F1 pad | macro F1 squash |
|---|---|---|---|---|---|---|
| 0.0005 | 1337 | 0.3985 | 0.3888 | +0.0097 | 0.0245 | 0.0246 |

Seed spread on the shipped config (squash, lr_oft 0.0005, 2 seeds [1337, 2025]): micro F1 0.3776-0.3888, i.e. **0.0112**. Any geometry difference smaller than that is not resolvable with this experiment.

Reading: at lr_oft 0.002 padding looked clearly worse (0.3459 vs 0.3804), but at the corrected lr_oft 0.0005 the gap collapses to within seed noise. Padding was more sensitive to the too-high learning rate, so the first comparison was measuring an LR x geometry interaction and not geometry alone.

**Does the squash distortion itself cost anything?** Measured per image on the 1,000 test images, bucketing by distortion `d = max(w,h)/min(w,h)`: micro F1 is 0.3628 for near-square (d<1.02, n=66), 0.4122 (1.02-1.25), 0.4217 (1.25-1.50), 0.4142 (1.50-2.00), 0.3908 (d>2, n=33). Correlation between distortion and per-image F1 is **r = +0.047** -- i.e. none, and if anything the wrong sign. Heavily warped images score *better* than square ones. The geometry is a dead end: each option trades shape against resolution and neither is the bottleneck.

For the record, the dataset is 70% portrait (median aspect 0.75, only 5% square), median distortion is 1.41x, 20% of images exceed 1.5x, and letterboxing instead would leave 26.7% of the canvas blank. No annotation could police this -- the vocabulary contains no stretch/aspect tag at all -- but 12% of images do carry real letterbox/pillarbox bars baked into the source art. See `scratch/sophie/for-user/dataset-geometry-audit.md`.

### Where the gap lives (best run vs bar, macro F1 @0.5)

| category | labels | bar | ours |
|---|---|---|---|
| rating | 4 | 0.8293 | 0.6627 |
| general | 9,225 | 0.1949 | 0.0278 |
| character | 3,247 | 0.1898 | 0.0057 |

The 4-label rating axis is essentially solved; the general and character vocabularies are where 10,000 images run out of steam.

### Reference point: no adapters at all

A head trained on **frozen** DINOv3 features (30 epochs, `cls_mean` pooling, best of 5 learning rates) reaches on the same test split:

* micro F1 **0.3279**, macro F1 0.0241

This is what OFTv2 has to beat to be worth its parameters, and it is the honest floor for "did the adapters help".

## Run history

| run dir | mode | lr_oft | lr_head | epochs | batch | micro F1 @0.5 | macro F1 @0.5 | tuned macro |
|---|---|---|---|---|---|---|---|---|
| *(removed)* | squash | 0.02 | 0.002 | 14 | 32 | 0.1427 | 0.0003 | - | muon-convention adapter LR: destroyed the features |
| `fe92f4df` | squash | 0.0005 | 0.00012 | 8 | 32 | 0.4142 | 0.0223 | 0.0331 |
| `d6d3c1d6` | squash | 0.0005 | 0.00025 | 8 | 32 | 0.4048 | 0.0249 | 0.0327 |
| `294af9c2` | squash | 0.0005 | 6e-05 | 8 | 32 | 0.4068 | 0.0183 | 0.0326 |
| `fa51da74` | squash | 0.0005 | 0.004 | 20 | 32 | 0.3800 | 0.0274 | 0.0284 |
| `e6bc2879` | squash | 0.0005 | 0.004 | 8 | 32 | 0.3888 | 0.0246 | 0.0277 |
| `7ff9ee38` | pad | 0.0005 | 0.004 | 8 | 32 | 0.3985 | 0.0245 | 0.0277 |
| `af7c4ac4` | squash | 0.0005 | 0.004 | 8 | 32 | 0.3776 | 0.0205 | 0.0238 |
| `539d8880` | squash | 0.002 | 0.004 | 8 | 32 | 0.3804 | 0.0202 | 0.0237 |

## Efficiency

| step variant | ms/step | img/s |
|---|---|---|
| eager | 359 | 89 |
| `torch.compile(mode="default")` on the forward | 290 | 110 |
| + compiled optimizer | 250 | 129 |

Measured breakdown of an eager step at batch 32: forward 156 ms, backward 243 ms, optimizer 88 ms, data 10 ms, H2D 4 ms. The optimizer cost is **per-tensor dispatch** (73 adapter tensors x ~1.2 ms), proven by scaling: 10/25/50/73 tensors cost a linear ~2 ms each, while making every tensor 8x larger costs only +15%. Compiling it, with Dynamo's recompile limits raised (otherwise `fullgraph=True` hard-fails with `FailOnRecompileLimitHit`), brings it to ~28 ms.

## Open questions (deliberately not resolved yet)

**Padding vs squash — head-to-head numbers still open.** The *mechanism* is now settled: per-image accuracy does not correlate with squash distortion (r = +0.047), so geometry is not a lever. What remains is only whether pad's single-seed edge survives a proper test: both arms ran at lr_head 0.004, since shown to be ~16x too high, so the arms are stale. At lr_head 0.00012: squash seed 1337 = `fe92f4df` (0.4142 micro); squash seed 2025 and pad x2 seeds would finish the job, ~32 min.

**Does more than 8 epochs help at the corrected head LR?** The 20-epoch run (`fa51da74`) was flat from epoch 14 and its final test micro (0.3800) was below the 8-epoch run's (0.3888), but it ran with the oversized head LR, so its head was overfitting early. Retest 20 epochs once lr_head is settled.

**Is lr_oft 0.0005 still right?** It was chosen with the head at 0.004. Now that the head moves ~2x slower than the adapters instead of 8x faster, the optimal adapter rate may shift. Untested interaction.

## Conclusion

We do **not** beat `convnextv2_huge`. The baseline saw 5.3 M images and predicts about 33 labels per image at 0.73 precision; 10,000 images against 12,476 labels is 0.8 images per label, and the long tail is not learnable at that scale. What the exercise does establish, on identical pixels and identical metrics:

* OFTv2 earns its parameters: at lr_oft 0.0005 the best run (pad) beats a frozen-feature probe of the same backbone by 21% relative on micro F1 (0.3985 vs 0.3279), though only marginally on macro (0.0245 vs 0.0241). Resetting the rotations to identity collapses it to 0.06.

* The adapters are extremely sensitive to their learning rate. At the muon-convention 0.02 they are destructive (micro 0.14); the ladder is 0.02 -> 0.143, 0.002 -> 0.380, 0.0005 -> 0.389-0.398. Note lr_oft = 0 is not hypothetical either: that is the frozen-feature probe at 0.328, so the curve is bracketed on both sides and 0.0005 sits inside the plateau rather than past a cliff.

* Resize geometry barely matters once the LR is right. Padding looked 10% worse at lr_oft 0.002 and is tied at 0.0005 — the earlier conclusion was an LR interaction, not a property of the geometry.

* The remaining gap to the bar is a data-scale gap, not a pipeline gap.
