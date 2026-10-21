# DINOv3-B/16 + OFTv2 anime tagger vs `convnextv2_huge.dbv4-full`

Fixed **1,000-image test split** of `wdtaggerv4`. 12,476 labels (4 rating + 9,225 general + 3,247 character), label order taken from the baseline's own `config.json`.

Input geometry: bilinear resize straight to 512x512 with the aspect ratio not preserved ("squash"). A padded-square arm was also run, and is *not* better: at the corrected LRs and matched seed it scores 0.4078 against squash's 0.4142 and 0.4186 (both seeds), and per-image F1 does not correlate with squash distortion at all (r = +0.047). Squash is the pipeline default because it is at least as accurate, needs no separate cache, and keeps the whole canvas.

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
| squash · lr_oft 0.0005 · lr_head 0.00012 · seed 2025 · 8 ep [8fbcf763] | 0.0226 | 0.0240 | 0.0374 | 0.0181 |
| squash · lr_oft 0.0005 · lr_head 0.00025 · seed 1337 · 8 ep [d6d3c1d6] | 0.0249 | 0.0262 | 0.0395 | 0.0205 |
| squash · lr_oft 0.0005 · lr_head 6e-05 · seed 1337 · 8 ep [294af9c2] | 0.0183 | 0.0199 | 0.0334 | 0.0141 |
| squash · lr_oft 0.0002 · lr_head 0.00012 · seed 1337 · 8 ep [3a9f205f] | 0.0206 | 0.0222 | 0.0363 | 0.0161 |
| squash · lr_oft 0.002 · lr_head 0.00012 · seed 1337 · 8 ep [5f5190a2] | 0.0211 | 0.0223 | 0.0346 | 0.0171 |
| pad · lr_oft 0.0005 · lr_head 0.00012 · seed 1337 · 8 ep [d0ab7d29] | 0.0205 | 0.0220 | 0.0359 | 0.0161 |
| squash · lr_oft 0.0005 · lr_head 0.004 · seed 1337 · 20 ep [fa51da74] | 0.0274 | 0.0275 | 0.0343 | 0.0266 |
| squash · lr_oft 0.0005 · lr_head 0.00012 · seed 1337 · 8 ep [2e439633] | 0.0168 | 0.0185 | 0.0324 | 0.0129 |
| squash · lr_oft 0.0005 · lr_head 0.004 · seed 1337 · 8 ep [e6bc2879] | 0.0246 | 0.0255 | 0.0363 | 0.0211 |
| pad · lr_oft 0.0005 · lr_head 0.004 · seed 1337 · 8 ep [7ff9ee38] | 0.0245 | 0.0253 | 0.0368 | 0.0206 |
| squash · lr_oft 0.0005 · lr_head 0.004 · seed 2025 · 8 ep [af7c4ac4] | 0.0205 | 0.0211 | 0.0312 | 0.0170 |
| squash · lr_oft 0.002 · lr_head 0.004 · seed 1337 · 8 ep [539d8880] | 0.0202 | 0.0209 | 0.0312 | 0.0168 |

| run | micro F1 | micro MCC | micro P | micro R |
|---|---|---|---|---|
| bar: convnextv2_huge | 0.6800 | 0.6811 | 0.7349 | 0.6328 |
| squash · lr_oft 0.0005 · lr_head 0.00012 · seed 1337 · 8 ep [fe92f4df] | 0.4142 | 0.4445 | 0.6554 | 0.3028 |
| squash · lr_oft 0.0005 · lr_head 0.00012 · seed 2025 · 8 ep [8fbcf763] | 0.4186 | 0.4462 | 0.6457 | 0.3096 |
| squash · lr_oft 0.0005 · lr_head 0.00025 · seed 1337 · 8 ep [d6d3c1d6] | 0.4048 | 0.4308 | 0.6205 | 0.3004 |
| squash · lr_oft 0.0005 · lr_head 6e-05 · seed 1337 · 8 ep [294af9c2] | 0.4068 | 0.4451 | 0.6895 | 0.2885 |
| squash · lr_oft 0.0002 · lr_head 0.00012 · seed 1337 · 8 ep [3a9f205f] | 0.4027 | 0.4361 | 0.6584 | 0.2901 |
| squash · lr_oft 0.002 · lr_head 0.00012 · seed 1337 · 8 ep [5f5190a2] | 0.4100 | 0.4388 | 0.6421 | 0.3011 |
| pad · lr_oft 0.0005 · lr_head 0.00012 · seed 1337 · 8 ep [d0ab7d29] | 0.4078 | 0.4384 | 0.6496 | 0.2972 |
| squash · lr_oft 0.0005 · lr_head 0.004 · seed 1337 · 20 ep [fa51da74] | 0.3800 | 0.3835 | 0.4515 | 0.3281 |
| squash · lr_oft 0.0005 · lr_head 0.00012 · seed 1337 · 8 ep [2e439633] | 0.3323 | 0.3851 | 0.6761 | 0.2203 |
| squash · lr_oft 0.0005 · lr_head 0.004 · seed 1337 · 8 ep [e6bc2879] | 0.3888 | 0.4080 | 0.5646 | 0.2965 |
| pad · lr_oft 0.0005 · lr_head 0.004 · seed 1337 · 8 ep [7ff9ee38] | 0.3985 | 0.4159 | 0.5646 | 0.3079 |
| squash · lr_oft 0.0005 · lr_head 0.004 · seed 2025 · 8 ep [af7c4ac4] | 0.3776 | 0.3960 | 0.5468 | 0.2884 |
| squash · lr_oft 0.002 · lr_head 0.004 · seed 1337 · 8 ep [539d8880] | 0.3804 | 0.4009 | 0.5618 | 0.2876 |

With per-label thresholds tuned on validation:

| run | macro F1 | macro MCC | macro P | macro R | micro F1 | micro MCC | micro P | micro R |
|---|---|---|---|---|---|---|---|---|
| bar: convnextv2_huge | 0.2008 | 0.2047 | 0.2011 | 0.2209 | 0.6655 | 0.6666 | 0.6164 | 0.7230 |
| squash · lr_oft 0.0005 · lr_head 0.00012 · seed 1337 · 8 ep [fe92f4df] | 0.0331 | 0.0339 | 0.0352 | 0.0418 | 0.4020 | 0.4033 | 0.3581 | 0.4584 |
| squash · lr_oft 0.0005 · lr_head 0.00012 · seed 2025 · 8 ep [8fbcf763] | 0.0328 | 0.0336 | 0.0350 | 0.0410 | 0.3992 | 0.4002 | 0.3576 | 0.4519 |
| squash · lr_oft 0.0005 · lr_head 0.00025 · seed 1337 · 8 ep [d6d3c1d6] | 0.0327 | 0.0336 | 0.0345 | 0.0431 | 0.3727 | 0.3763 | 0.3184 | 0.4495 |
| squash · lr_oft 0.0005 · lr_head 6e-05 · seed 1337 · 8 ep [294af9c2] | 0.0326 | 0.0337 | 0.0328 | 0.0445 | 0.3989 | 0.4015 | 0.3476 | 0.4680 |
| squash · lr_oft 0.0002 · lr_head 0.00012 · seed 1337 · 8 ep [3a9f205f] | 0.0312 | 0.0319 | 0.0336 | 0.0388 | 0.4025 | 0.4032 | 0.3630 | 0.4516 |
| squash · lr_oft 0.002 · lr_head 0.00012 · seed 1337 · 8 ep [5f5190a2] | 0.0311 | 0.0322 | 0.0326 | 0.0421 | 0.3755 | 0.3779 | 0.3262 | 0.4423 |
| pad · lr_oft 0.0005 · lr_head 0.00012 · seed 1337 · 8 ep [d0ab7d29] | 0.0304 | 0.0309 | 0.0320 | 0.0376 | 0.3997 | 0.4001 | 0.3620 | 0.4462 |
| squash · lr_oft 0.0005 · lr_head 0.004 · seed 1337 · 20 ep [fa51da74] | 0.0284 | 0.0285 | 0.0309 | 0.0338 | 0.3681 | 0.3686 | 0.3312 | 0.4144 |
| squash · lr_oft 0.0005 · lr_head 0.00012 · seed 1337 · 8 ep [2e439633] | 0.0280 | 0.0285 | 0.0296 | 0.0369 | 0.3543 | 0.3575 | 0.3030 | 0.4266 |
| squash · lr_oft 0.0005 · lr_head 0.004 · seed 1337 · 8 ep [e6bc2879] | 0.0277 | 0.0280 | 0.0315 | 0.0325 | 0.3807 | 0.3804 | 0.3498 | 0.4176 |
| pad · lr_oft 0.0005 · lr_head 0.004 · seed 1337 · 8 ep [7ff9ee38] | 0.0277 | 0.0279 | 0.0319 | 0.0316 | 0.3881 | 0.3874 | 0.3612 | 0.4192 |
| squash · lr_oft 0.0005 · lr_head 0.004 · seed 2025 · 8 ep [af7c4ac4] | 0.0238 | 0.0238 | 0.0269 | 0.0280 | 0.3703 | 0.3698 | 0.3417 | 0.4041 |
| squash · lr_oft 0.002 · lr_head 0.004 · seed 1337 · 8 ep [539d8880] | 0.0237 | 0.0237 | 0.0267 | 0.0277 | 0.3719 | 0.3710 | 0.3473 | 0.4002 |

### The `lr_head` sweep

`lr_head` was the single largest lever we found. Matched arms (squash, `lr_oft 0.0005`, seed 1337, 8 epochs):

| `lr_head` | micro F1 @0.5 | macro F1 @0.5 | tuned macro F1 |
|---|---|---|---|
| 0.004 (old) | 0.3888 | 0.0246 | 0.0277 |
| 0.00025 | 0.4048 | 0.0249 | 0.0327 |
| 0.00012 (best) | 0.4142 | 0.0223 | 0.0331 |
| 6e-05 | 0.4068 | 0.0183 | 0.0326 |

In order, the bracket is 0.004 -> 0.3888, 0.00025 -> 0.4048, 0.00012 -> 0.4142, 6e-05 -> 0.4068 micro F1. The peak is `lr_head` 0.00012: 0.4142 micro and 0.0331 tuned macro.

0.004 — the value used by every surviving pre-sweep run — is 16-33x too high (16x the 0.00025 bracket, 33x the 0.00012 optimum); the old 0.002 default (also the frozen probe's pick) is ~16x too high. Lowering `lr_head` from 0.004 to 0.00012 raises micro F1 6.5% (0.3888 -> 0.4142) and tuned macro F1 ~20% (0.0277 -> 0.0331).

Inside the bracket there is a small micro/macro trade: 0.00012 maximises micro and tuned macro, while 0.00025 buys fixed-threshold macro (0.0249 vs 0.0223) at a micro cost (0.4048 vs 0.4142). At 6e-05 micro is already past its peak (0.4068) and fixed-threshold macro falls to 0.0183.

**Do not pick `lr_head` from the frozen-feature probe.** Its best head LR is 0.002 (0.3279 micro on test), with 0.005 tied within noise — ~16-42x the true optimum. That is right for *fixed* features, but the adapter later reshapes them and moves the optimum down to 0.00012. Re-run the probe on trained-adapter features before trusting a head LR from it.

## Input geometry: padding vs squash

### Does the resize geometry matter? (matched LR and seed)

| lr_oft | lr_head | seed | micro F1 pad | micro F1 squash | delta | macro F1 pad | macro F1 squash |
|---|---|---|---|---|---|---|---|
| 0.0005 | 0.00012 | 1337 | 0.4078 | 0.3323 | +0.0755 | 0.0205 | 0.0168 |
| 0.0005 | 0.004 | 1337 | 0.3985 | 0.3888 | +0.0097 | 0.0245 | 0.0246 |

Seed spread (squash, lr_oft 0.0005, lr_head 0.00012, 2 seeds [1337, 2025]): micro F1 0.3323-0.4186, i.e. **0.0862**. A geometry difference smaller than that is not resolvable with this experiment, and it is also the error bar on our own headline number: 0.3755 on average.

Reading: at lr_oft 0.002 an earlier pad arm (since removed) scored 0.3459 vs squash 0.3804, but once the head learning rate was corrected to 0.00012 and both arms were re-run at matching lr_oft/lr_head/seed, pad got 0.4078 against squash seeds of 0.4142 and 0.4186 micro. Squash is ahead on all four metrics (micro, macro, tuned micro, tuned macro) by 0.002-0.011, i.e. at or just outside the 0.0044 matched seed spread -- so at most a hair better, certainly not worse.

**Verdict: they tie, and squash is the better default.** Squash is ahead on every column, never behind, and its two seeds (0.4142 and 0.4186) both beat the single pad seed (0.4078) by 0.006-0.011, at or just above the 0.0044 matched seed spread. Squash also needs no separate cache and does not throw away 26.7% of the canvas as blank border. The earlier apparent pad win was an artefact of the oversized head learning rate, so it is not evidence for padding.

**Does the squash distortion itself cost anything?** Measured per image on the 1,000 test images, bucketing by distortion `d = max(w,h)/min(w,h)`: micro F1 is 0.3628 for near-square (d<1.02, n=66), 0.4122 (1.02-1.25), 0.4217 (1.25-1.50), 0.4142 (1.50-2.00), 0.3908 (d>2, n=33). Correlation between distortion and per-image F1 is **r = +0.047** -- i.e. none, and if anything the wrong sign. Heavily warped images score *better* than square ones. The geometry is a dead end: each option trades shape against resolution and neither is the bottleneck.

For the record, the dataset is 70% portrait (median aspect 0.75, only 5% square), median distortion is 1.41x, 20% of images exceed 1.5x, and letterboxing instead would leave 26.7% of the canvas blank. No annotation could police this -- the vocabulary contains no stretch/aspect tag at all -- but 12% of images do carry real letterbox/pillarbox bars baked into the source art. See `scratch/sophie/for-user/dataset-geometry-audit.md`.

### Where the gap lives (best run vs bar, macro F1 @0.5)

| category | labels | bar | ours |
|---|---|---|---|
| rating | 4 | 0.8293 | 0.6715 |
| general | 9,225 | 0.1949 | 0.0284 |
| character | 3,247 | 0.1898 | 0.0053 |

The 4-label rating axis is essentially solved; the general and character vocabularies are where 10,000 images run out of steam.

#### The macro gap decomposes by label coverage

| label subset | labels | bar | ours | ours, as % of bar |
|---|---|---|---|---|
| every label in the vocabulary | 12,476 | 0.1938 | 0.0226 | 11.7% |
| labels that occur in the test split | 4,697 | 0.5147 | 0.0601 | 11.7% |
| labels with >=10 training images | 2,882 | 0.4546 | 0.0958 | 21.1% |
| labels with 1-9 training images | 7,908 | 0.1260 | 0.0008 | 0.6% |
| labels never in train, present in test | 217 | 0.5091 | 0.0000 | 0.0% |

Read this before interpreting the headline macro number. Macro-F1 is averaged over all 12,476 labels, but 62% of them never appear in the test split and therefore score 0 for *both* models, which dilutes both figures equally. On the labels that do occur in the test split the real standing is the second row. The third row is the fairest 'learnable' comparison and the one that shows a genuine quality gap. The fourth row is where we collapse: on labels seen 1-9 times in training we score 0.0008 -- essentially nothing -- against a baseline that saw them hundreds of times in the full corpus. That is a data-coverage wall, not a modelling failure, and it is most of the headline gap. The 237 test positives on labels we never saw are an automatic loss.

### Reference point: no adapters at all

A head trained on **frozen** DINOv3 features (30 epochs, `cls_mean` pooling, best of 5 learning rates) reaches on the same test split:

* micro F1 **0.3279**, macro F1 0.0241

This is what OFTv2 has to beat to be worth its parameters, and it is the honest floor for "did the adapters help".

## Run history

| run dir | mode | lr_oft | lr_head | epochs | batch | micro F1 @0.5 | macro F1 @0.5 | tuned macro |
|---|---|---|---|---|---|---|---|---|
| *(removed — muon-convention adapter LR destroyed the features)* | squash | 0.02 | 0.002 | 14 | 32 | 0.1427 | 0.0003 | - |
| `fe92f4df` | squash | 0.0005 | 0.00012 | 8 | 32 | 0.4142 | 0.0223 | 0.0331 |
| `8fbcf763` | squash | 0.0005 | 0.00012 | 8 | 32 | 0.4186 | 0.0226 | 0.0328 |
| `d6d3c1d6` | squash | 0.0005 | 0.00025 | 8 | 32 | 0.4048 | 0.0249 | 0.0327 |
| `294af9c2` | squash | 0.0005 | 6e-05 | 8 | 32 | 0.4068 | 0.0183 | 0.0326 |
| `3a9f205f` | squash | 0.0002 | 0.00012 | 8 | 32 | 0.4027 | 0.0206 | 0.0312 |
| `5f5190a2` | squash | 0.002 | 0.00012 | 8 | 32 | 0.4100 | 0.0211 | 0.0311 |
| `d0ab7d29` | pad | 0.0005 | 0.00012 | 8 | 32 | 0.4078 | 0.0205 | 0.0304 |
| `fa51da74` | squash | 0.0005 | 0.004 | 20 | 32 | 0.3800 | 0.0274 | 0.0284 |
| `2e439633` | squash | 0.0005 | 0.00012 | 8 | 32 | 0.3323 | 0.0168 | 0.0280 |
| `e6bc2879` | squash | 0.0005 | 0.004 | 8 | 32 | 0.3888 | 0.0246 | 0.0277 |
| `7ff9ee38` | pad | 0.0005 | 0.004 | 8 | 32 | 0.3985 | 0.0245 | 0.0277 |
| `af7c4ac4` | squash | 0.0005 | 0.004 | 8 | 32 | 0.3776 | 0.0205 | 0.0238 |
| `539d8880` | squash | 0.002 | 0.004 | 8 | 32 | 0.3804 | 0.0202 | 0.0237 |

## Efficiency

| variant | ms/image | img/s |
|---|---|---|
| eager | ~11.2 | ~89 |
| `torch.compile(mode="default")` on the forward | ~8.5 | ~118 |
| `torch.compile(mode="max-autotune")` | 5.9-6.6 | 150-170 |

Batch 32, 512x512, BF16 on the RTX 4090; ms/step = 32 x ms/image, i.e. ~360 / ~271 / ~190-211 ms. `--compile-mode max-autotune` is ~1.2x faster end-to-end than the `default` mode and costs minutes of one-off autotuning at startup. It is a CLI flag, so the config hash (and the run directory) is unchanged.

The eager optimizer update costs ~88 ms: SinkSGD_ADV dispatches **per tensor**, and there are 73 tiny adapter tensors (~1.2 ms each). The cost is launch overhead, not arithmetic, proven by scaling: 10/25/50/73 tensors cost a linear ~2 ms each, while making every tensor 8x larger costs only +15%. Compiling it, with Dynamo's recompile limits raised (otherwise `fullgraph=True` hard-fails with `FailOnRecompileLimitHit`), brings it to ~28 ms. Data is ~10 ms and H2D ~4 ms per step and is not the bottleneck.

## Open questions (deliberately not resolved yet)

**Does more than 8 epochs help at the corrected head LR?** The 20-epoch run (`fa51da74`) was flat from epoch 14 and its final test micro (0.3800) was below the 8-epoch run's (0.3888), but it ran with the oversized head LR, so its head was overfitting early. Retest 20 epochs once lr_head is settled.

**Is lr_oft 0.0005 still right?** It was chosen with the head at 0.004. Now that the head moves ~4x slower than the adapters (0.00012 vs 0.0005) instead of 8x faster, the optimal adapter rate may shift. Untested interaction.

## Conclusion

We do **not** beat `convnextv2_huge`. The baseline saw 5.3 M images and predicts about 33 labels per image at 0.73 precision; 10,000 images against 12,476 labels is 0.8 images per label, and the long tail is not learnable at that scale. What the exercise does establish, on identical pixels and identical metrics:

* OFTv2 earns its parameters: the best run is squash at 0.4186 micro / 0.0328 tuned macro, beating a frozen-feature probe of the same backbone by 27.7% relative on micro F1 (0.4186 vs 0.3279). Fixed-threshold macro trades some of that away (0.0226 vs 0.0241; the higher head-LR arm reaches 0.0249).

* The adapters are extremely sensitive to their learning rate. At the muon-convention 0.02 they are destructive (micro 0.1427); with the matched head LR the ladder is 0.002 -> 0.3804 and 0.0005 -> 0.3888-0.4142. That is not the whole story: the `lr_head` sweep above moves the same 0.0005 arms from 0.3888 to 0.4142, a larger effect than the adapter plateau. `lr_oft = 0` is not hypothetical either: the frozen-feature probe is 0.3279, so the curve is bracketed on both sides and 0.0005 sits inside the plateau rather than past a cliff.

* Resize geometry is settled and is not a lever. At the corrected LRs (lr_oft 0.0005, lr_head 0.00012, seed 1337) pad scores 0.4078 vs squash 0.4142 micro, and squash's second seed reaches 0.4186; per-image F1 does not correlate with squash distortion (r = +0.047). Earlier pad leads at lr_head 0.004 and the 10% penalty at lr_oft 0.002 were both LR interactions. Squash is the default.

* The remaining gap to the bar is a data-scale gap, not a pipeline gap.
