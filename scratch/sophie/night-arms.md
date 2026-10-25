# Night arms, 2026-10-23 into 10-24

base-384, u1-C-L-asl(g2.0), 200000 images, batch 48, lr_oft 5e-4, exact Cayley, seed 1337, one variable per row. Scored on the 50k at 0.40 unless the row says in-run, which is the 10000-image validation at step 1000 with every arm on the same schedule.

| arm | variable | steps | in-run macro at step 1000 | 50k macro | 50k micro | note |
|---|---|---:|---:|---:|---:|---|
| 42 accum 1 | baseline | 4167 | 0.0558 | 0.1490 | 0.4359 | 10.6 min of training, 9.4 GB, still climbing at every check |
| 45 ntm | normed_momentum | killed 1655 | 0.0079 | not scored | not scored | sevenfold collapse at the quarter mark, stopped |
| 46 sinkv | normed_momentum + snr_cond | killed 1363 | 0.0085 | not scored | not scored | same collapse as 45, stopped at the quarter mark |
| 48 gold145 | six switches, lr x1.45 | 4167 | 0.0066 | 0.0114 | 0.4166 | full gold stack, compensated lr, 13x worse macro than arm 1 |
| 49 r0 plainfp32 | fp32 optimizer state only | 4167 | 0.0559 | 0.1498 | 0.4369 | +0.0008 macro over arm 1 at bf16 state, so fp32 is neutral to slightly positive |
| 59 r1m matched | ntm+snr at lr x0.522, displacement parity | killed 1000 | 0.0067 | not scored | not scored | matched throttle changes nothing, so the geometry is the problem and not the scale |
| 51 r2 orthograd | + orthograd iterative | queued | | | | |
| 52 r3 sinkhorn | + orthogonal_sinkhorn | queued | | | | |
| 53 r4 spectral | + spectral_normalization | queued | | | | |
| 54 w0 wd1e5 | + weight_decay 1e-5 | queued | | | | |
| 55 w1 geowd | + geometric_wd | queued | | | | |
| 56 w2 cautiouswd | + cautious_wd | queued | | | | |
| 57 r1b oftonly | ntm+snr scoped to the adapters only | 4167 | 0.0631 | 0.1555 | 0.4358 | +0.0057 macro over plain, micro flat, still 1.9x hotter on the adapters |
| 62 r1bm control | same, adapter lr divided by 1.916 | killed 1000 | 0.0427 | not scored | not scored | also halved the head lr, so it is not the clean control it was meant to be |

The step-1000 in-run number is the screen: same 48000 images seen, same 24% of the cosine, same seed and order, so a collapse at that point is a property of the arm and not of the schedule.

Arm 42's 50k at 0.40, for the record: macro F1 0.1490, MCC 0.1605, P 0.1716, R 0.1883, and micro F1 0.4359, MCC 0.4699, P 0.3153, R 0.7061.

It beats base-512 at effective 1344, which read 0.1304 at 271488 images, and that is the update-count argument from the other side: 4167 updates at effective 48 beat 202 updates at effective 1344 on a third fewer images.

## Head row norms, plain against gold

Measured with `scratch/sophie/head_rows.py` on `head.1.weight`, 12476 rows, norm per row.

| checkpoint | mean | cv | max/min | p1 | p50 | p99 |
|---|---:|---:|---:|---:|---:|---:|
| arm 42 step 1000 | 0.8925 | 0.2550 | 3.42 | 0.5985 | 0.8226 | 1.5845 |
| arm 42 final | 0.9981 | 0.2628 | 4.67 | 0.5703 | 0.9466 | 1.6013 |
| gold 48 step 1000 | 0.7945 | 0.0916 | 1.56 | 0.6616 | 0.8017 | 0.9249 |
| gold 48 final | 1.0920 | 0.3360 | 3.57 | 0.6656 | 0.9794 | 1.9893 |

The head is measurably flattened early under the geometry switches, cv 0.0916 against 0.2550 at step 1000 and max/min 1.56 against 3.42, which is exactly what a per-row Sinkhorn on a 12476-row matrix should do to per tag scale. By step 4167 the spread has not only returned but overshot, cv 0.3360 against 0.2628, so the flattening is a first thousand steps that never comes back rather than a permanent distortion. That is the shape of a slow start under a cosine, and the schedule does not forgive a slow start. It is not by itself proof the head is the culprit, which is what rung 57 tests by scoping the switches to the adapters.

## Production, base-512 full corpus, 2026-10-24

`63_u1-C-L-asl-g2.0-siglip2b512-5.3m-b64x6-E384-prod`, run dir `62902e8d`, 13859 steps at effective 384, plain SinkSGD with fp32 state, lr 1.2e-3 / 2.88e-4 / 7.2e-5, warmup 300, cosine, 20.9 GB, 87 to 94 img/s.

| step | images seen | val macro 10k | val micro 10k | macro at 0.40 |
|---:|---:|---:|---:|---:|
| 1000 | 384000 | 0.1836 | 0.5226 | 0.1837 |
| 2000 | 768000 | 0.2309 | 0.5387 | 0.2211 |
| 3000 | 1152000 | 0.2565 | 0.5444 | 0.2437 |

For scale, the champion `2b6b81aa` was base-512 at effective 48 and finished 600000 images on 50k macro 0.2733, and arm 42, base-384 at effective 48, read 0.0558 in-run at its own step 1000 where production reads 0.1836 at 384000 images. The comparison is not like for like on images seen, which is the entire point: one update of 384 images is not eight updates of 48, and the curve will say which is better over the next five thousand steps.


## Step displacement per switch

`scratch/sophie/rung_step_size.py`, one 768x768 tensor and one 12476x768 tensor, lr 5e-4, 50 steps of the same fixed gradient so the momentum buffer reaches steady state, displacement is the Frobenius norm of the final step divided by the plain stack's. The authoritative measurement over the live inventory is `scratch/sophie/rung-displacement.md`; this table is the same finding on two shapes and agrees on the ordering.

| stack | 768x768 | 12476x768 |
|---|---:|---:|
| plain | 1.000 | 1.000 |
| nesterov only | 1.000 | 1.000 |
| orthograd iterative only | 1.000 | 1.000 |
| orthogonal sinkhorn only | 1.000 | 1.000 |
| normed momentum only | 0.923 | 0.923 |
| normed momentum + snr | 1.576 | 1.576 |
| plus orthograd | 1.388 | 1.388 |
| plus orthogonal sinkhorn | 1.388 | 1.388 |
| plus spectral normalization | 0.018 | 0.029 |
| spectral normalization only | 0.018 | 0.029 |

Three readings. Nesterov, orthograd and orthogonal sinkhorn are scale neutral to three decimals on these shapes, so any score difference they produce is direction, not throttle, and that is what makes them clean things to test. The ntm plus snr couple is a 1.58x step on top of the plain stack, so arms 45 and 46 were never at matched throttle with arm 1 and their collapse has to be read with that in mind. Spectral normalization divides the step by 35 to 55 on a random gradient and by roughly 750 on a rank-one one, so it is not a refinement, it is a rescaling, and any run with it at our learning rate is effectively frozen. That is a much stronger claim than dubious.

The persistent-gradient caveat matters for interpreting ntm: with a fixed direction the momentum buffer converges toward unit norm, which is its most favourable case. In a real run the gradient direction changes every step and the buffer stays smaller, so the honest statement is that ntm plus snr is at most 1.58x and probably below it.



## The common checkpoint at step 1000

Every arm below runs the same 200000-image schedule so step 1000 is 48000 images and 24 percent of the cosine. The instrument is the in-run 10000-image subset, macro over supported labels in parentheses. Plain is arm 49 at 0.0559 (0.0644).

| arm | stack | macro | supported macro | micro |
| 49 | plain, fp32 state | 0.0559 | 0.0645 | 0.4386 |
| 42 | plain, bf16 state | 0.0558 | 0.0644 | 0.4394 |
| 45 | normed momentum only, 5e-4 | 0.0079 | 0.0091 | 0.4193 |
| 46 | normed momentum + snr, 5e-4 | 0.0085 | 0.0098 | 0.4277 |
| 59 | normed momentum + snr, matched 2.61e-4 | 0.0067 | 0.0078 | 0.4003 |
| 48 | six switches at lr x 1.45 | 0.0066 | 0.0076 | 0.3996 |
| 57 | couple scoped to the 70 adapters, 5e-4 | 0.0631 | 0.0729 | 0.4601 |
| 62 | couple scoped, every rate halved | 0.0427 | 0.0492 | 0.4638 |
| 64 | spectral, 5e-4, untagged | 0.0004 | 0.0005 | 0.1788 |
| 64b | spectral, 5e-4, tagged | 0.0007 | 0.0008 | 0.2066 |
| 70 | spectral, tagged, per-group rate compensated | 0.0078 | 0.0090 | 0.3802 |
| 71 | couple scoped, adapter rate matched | pending | pending | pending |

Spectral is finished as a question. Untagging the fix cost it a factor of 76 in step size and the tag recovered only 0.0003 of macro, and compensating each shape's throttle with its own learning rate recovered a further 0.0071 without ever coming within 7x of plain. A switch that needs a 44x learning rate on the adapters, 34x on the head and 1x on the biases cannot be driven by one learning rate, and here even a per-group approximation loses badly. It does not pay, and the mechanism is now understood well enough that nobody has to wonder whether the failure was a configuration mistake.

The couple scoped to the adapters at the original rate beat plain by 0.0072 macro at this checkpoint, and the two ways to explain that are a hotter adapter or the geometry. Arm 71 holds the adapter rate at 2.61e-4 so displacement matches plain, leaves the head at 1.2e-4, and answers it.

