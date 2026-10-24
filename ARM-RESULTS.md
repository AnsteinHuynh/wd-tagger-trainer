# ARM RESULTS — every experiment, one place

Central record of arm experiments, and the only place results are written. `REPORT.md` and `wdtag/report.py` are deprecated: a generated second report drifted out of date and split the record in two, so this file is the single destination for findings, comparisons, probes that never became runs, and anything a future reader needs to avoid re-running an arm

## Scoring protocol

Every comparable number is macro F1 and micro F1 at a fixed global threshold of 0.40 on the 50000-image nested subset, and macro is the objective

Read `validation.json` -> `subsets["50000"]` and index `thresholds` by hand. No collector script exists, and `wdtag.validate` prints an `at 0.40:` block once per nested size (1000, 5000, 10000, 50000), so scraping stdout gives the 1000-image numbers as if they were 50k

`wdtag.validate --run <dir>` without `--ckpt` scores the highest-numbered `ckpt_step*.pt`, not `best_val.pt`, so every arm number here is a final-checkpoint number unless labelled best_val; the best artifact is scored with `--ckpt best_val.pt --out <dir>/validation_bestval.json`. A run stopped early gets its pause checkpoint scored, so pausing costs steps, not the artifact

`best_val.pt` is selected on micro by default, which on h1 coincided with the macro peak (step 1040). `--select-metric macro` maps to `macro_f1` and `macro040` to `macro_f1_at_0.40` (the protocol threshold, untuned); `micro` / `macro` / `supported_macro` also exist. Phase-3 best_val numbers are macro-selected and not comparable to the first sweep's micro-selected ones; within phase 3 they are consistent, and the final-checkpoint column is comparable across all arms

In-run 1k numbers are trajectory only: 1k is reliable for micro, not macro, and no in-run val should be used to rank arms. A 50k pass at batch 24 took 346 s (145 img/s) while another arm trained, so two scoring passes per arm add about 12 min to 44 min of training

## The 10k head benchmark (defined 2026-10-22)

Head/loss variants in ~45 min per arm instead of ~9 h on the full corpus

| property | value |
|---|---|
| training data | the cached shard set: 10000 images (`CacheDataset("train","squash")`) |
| steps/epoch at batch 48 | 209 |
| epochs | 25 -> 5225 steps, capped at 5000 by `--max-steps` |
| batch | 48 |
| in-run val | 1000 images, grid 0.25/0.30/0.40/0.50 |
| separate scoring | `python -m wdtag.validate --run <dir> --batch-size 256` on the 50k cache |
| backbone | DINOv3 ViT-B/16 frozen, OFT block 128, exact Cayley, 12 blocks unfrozen |
| LRs | lr_oft 5e-4, lr_unfreeze 3e-5, lr_head 1.2e-4 |
| determinism | sample order is a pure function of `(seed, pass_index)`, so every arm sees identical images at identical steps |
| noise floor | arm `h6_control_seed2025` duplicates the control at seed 2025 |

`epochs=8` stops the run at 1664 steps, silently short of `--max-steps 5000`. Check `step N/total`: the total must exceed the intended cap. The first h1 run was invalid for this and was re-run

`--max-steps` exits without the end-of-run validation, so `best_val.pt` comes only from epoch-end validations; the authoritative 50k number is the separate `wdtag.validate` call

`--ckpt-every-steps 0` does not disable checkpointing: it writes one every step. Use a large number to thin them

Trainer behaviours confirmed by smoke test (`scratch/sophie/smoke/*.json`, batch 2, 2 steps, EXIT=0): both loss paths wire up at real scale, `head 38.34M, unfrozen 85.06M in 12 block(s), 73 OFT modules` (head back to 38.34M after `class_bias` was removed), `pos_weight` computed with `n_images 10000`

## Arms

Final checkpoint / best_val where the two differ. One change from the control per arm unless stated

| arm | sha8 | change | 50k macro@0.40 | 50k micro@0.40 | finding |
|---|---|---:|---:|---:|---|
| h1_control_cls_mean_max_std | 470e6fa9 | control: u12, cls_mean_max_std, linear, BCE, 38.34M head, 123.46M trainable | 0.0247 / 0.0272 | 0.4372 / 0.4385 | 1k peak 0.0298 at step 1040; final 30% below its own peak; 12454 supported labels, macro P 0.0781 R 0.0199 at best_val |
| h2_cls_mean | c3e2744d | `head_pool: cls_mean`, head 19.16M | 0.0253 / 0.0262 | 0.4395 / 0.4429 | a tie, inside the band |
| h3_cls | 8afb7e5e | `head_pool: cls`, head 9.60M | 0.0258 / 0.0285 | 0.4397 / 0.4450 | +4.8% on best_val, the only head result above the band under BCE |
| h4_mlp1024 | d14c3bfd | `head_hidden: 1024`, head 15.94M | 0.0245 / 0.0254 | 0.4226 / 0.4225 | regression: no macro, micro down |
| h5_mlp2048 | 04aa7438 | `head_hidden: 2048`, head 31.86M | 0.0232 / 0.0234 | 0.4214 / 0.4216 | -14% best_val, a real regression |
| h6_control_seed2025 | 1e312196 | seed 2025 only | 0.0252 / 0.0264 | — | noise floor: spread 2.0% final, 3.0% best_val |
| h7_mlp512 | — | `head_hidden: 512`, ~8M head | 0.0244 | 0.4186 | MLP family flat-to-worse |
| h8_mlp4096 | — | `head_hidden: 4096`, ~63M head | 0.0219 | 0.4197 | monotone: wider is worse |
| h9_cb_ntr | ab446356 | loss CB-NTR β=0.9 λ=2 κ=0.05, 38.34M head | 0.0306 | 0.4197 | +24% macro, -4% micro; macro P 0.0756 R 0.0239 |
| h10_asl | 7ee12110 | loss ASL γ⁻=4, γ⁺=0, clip 0.05, detached focal | 0.0410 / 0.0410 | 0.4177 / 0.4170 | +51% best_val, +66% final; macro R doubles, macro P -11%, micro -4.9% |
| h11_pos_weight | 2608eb83 | loss `pos_weight=((N−n)/n)^0.5` | 0.0068 / 0.0097 | 0.2572 / 0.2681 | -64% best_val; 1k peak 0.0119 at step 1456, half its own peak |
| u12-C-L-asl-g4.0 (arm 01) | 83231ec3 | h10 with `head_pool: cls` | 0.0646 | 0.4048 | +57.6% macro, -3.1% micro; MCC 0.0669, P 0.0764, R 0.0720 |
| u1-C-L-asl-g4.0 | b9ed94fb | `unfreeze_blocks` 12 -> 1 | 0.0632 | 0.3713 | -2.2% macro inside band, -8.3% micro for 130 img/s and 16.8 GB |
| u1-C-H2048-asl-g4.0 | f8ff5c91 | +2048 hidden, head 27.14M | 0.0509 | 0.4166 | -19.5% macro, +12.2% micro, the highest single-tower micro measured |
| u1-C-L-asl-g2.0 | ce692296 | `loss_gamma` 4.0 -> 2.0 | 0.0624 | 0.4156 | -1.3% macro inside band, +11.9% micro; the operating point |
| u1-C-L-asl-g1.0 | 83a01eed | `loss_gamma` 1.0 | 0.0575 | 0.4335 | -7.8% macro, +4.3% micro; gamma axis closed at 2.0 |
| u1-C-L-asl-g4.0+siglipb | 966249d9 | +frozen SigLIP2-B, one LayerNorm(1536) over the concat | 0.0546 | 0.3652 | -13.6% macro, -1.6% micro |
| u1-C-L-asl-g2.0+siglipb | 03612c5c | +frozen SigLIP2-B, per-tower norm | 0.0615 | 0.4159 | -1.4% macro, +0.07% micro: a tie, not a loss |
| u1-C-L-asl-g2.0-siglip2base | 66e841ba | `backbone_dir` SigLIP2-B/16-512 | 0.0755 | 0.4375 | +21.0% macro, +5.3% micro; MCC 0.0786, P 0.1010, R 0.0730, 138 img/s, 15.8 GB |
| u1-M-L-asl-g2.0-siglip2base | c8c22df5 | `head_pool: mean` over 1024 patch tokens | 0.0731 | 0.4315 | -3.2% macro at the band edge, -1.4% micro; MAP pooler stands |
| 18 u1 600k | 2b6b81aa | 600000 images, u1 cls asl 2.0 | 0.2733 | 0.4791 | 97% of the paused run's macro from 14% of its images; curve still climbing |
| 19 u12 600k | 77a135b2 | same, `unfreeze_blocks: 12` | 0.1031 | 0.4054 | 2.65x worse macro, 15% worse micro, +8% wall, +2.7 GB; parameterisation confound |
| 23 u0 600k | b6e293f0 | `unfreeze_blocks: 0` | 0.2694 | 0.4729 | -1.4% macro, -1.3% micro: freeze depth closed |
| 24 384 600k | 2a5c9bad | SigLIP2-B 384 checkpoint at 384 input | 0.2553 | 0.4723 | -6.6% macro, -1.4% micro for 1.45x speed, 9.1 GB |
| 27 large const 600k | f63f6f54 | SigLIP2-L batch 20, const schedule, `decay_steps` 3000 | 0.2331 | 0.4915 | -14.7% macro vs u1 base, +2.6% micro; MCC 0.2462, P 0.2701, R 0.2636; micro P 0.3679 R 0.7400 |
| a4674fbd large 240k accum | a4674fbd | large, batch 24 x2 accum, effective 48, 240000 images | 0.2197 | 0.4593 | +45% macro and +4.5% micro vs f63f6f54 at matched 240000 images (0.1516 / 0.4397); MCC 0.2344, P 0.2355, R 0.2781; micro MCC 0.4980, P 0.3294, R 0.7586 |
| 29 base b24 | 48de8bf9 | base `batch_size` 48 -> 24 | not run | — | staged control for the batch mechanism |
| 26 large cosine 600k | 62de5705 | SigLIP2-L cosine, batch 20 | not scored | — | paused at step 1535, superseded by f63f6f54, orphaned |
| c8f2a635 large 600k accum | c8f2a635 | large batch 24 x2 accum, 600000 images | pending | — | launched 17:22, 21.1 GB, about 3.9 h |
| c75d34d4 | c75d34d4 | `lr_unfreeze` 1e-5 | not run | — | staged, parked: it tunes a knob that governs 1.4% of the result |
| 20142533 | 20142533 | `lr_unfreeze` 1e-4 | not run | — | staged, parked with c75d34d4 |
| b05cbca4 | b05cbca4 | third learning-rate continuation | partial | — | stopped cleanly at step 3597 of 20000, resumable |
| h3 replicate | — | `u12-C-L-bce-seed2025` | not run | — | staged to test whether h3's +4.8% survives a second seed |
| bar head | — | `u12-M-L-bce`, mean pooling only | not run | — | the bar's exact head, cheap insurance, never run |
| 33 large 384 b24x2 | e3c45603 | SigLIP2-L native 384, batch 24 x2 accum, no checkpointing | paused | — | 57-60 img/s at 12.6 GB |
| 34 large 384 b48 | 56fa4c17 | same at batch 48, no checkpointing | unusable | — | 23.5 GB and 3 img/s, the near-ceiling collapse |
| 35 large 384 b48 ckpt | 1fee2c4f | same at batch 48, `grad_checkpoint_blocks: true` (all 24) | paused at 902 | — | 54-55 img/s at 4.0 GB, checkpoint save present |
| 36 large 384 b48 ckpt3 | 8854c266 | same at batch 48, `grad_checkpoint_blocks: 3` | live | — | 68-70 img/s at 22.1 GB, 25% faster than 35 |

Orphaned and non-arm directories: `9cc009da` and `cf852a8c` hold partial logs reaped at step 624 with no step checkpoint and no `validation.json` (the gamma change and its pre-gamma twin); `671e6989` is a smoke directory, batch 4, 205 steps, seed 4242, no score worth quoting; `feb17953` is a dead hash from a doubled-backslash path that names nothing

11 arm configs hash distinctly, so no two arms can share a run. `runs/<sha8>` hashes the whole config including defaults, so a new field moves the sha8 of every config that omits it: h1 runs in `470e6fa9` while its config now hashes to `54b2ce30`; `10_u1-C-L-asl-g4.0.json` was `b9ed94fb` when it ran and is now `da67aabc`; the two-tower config is `966249d9`; adding `siglip_dir` and the defaults change each moved hashes again. Never look a run up by re-hashing its config: read `run dir:` from the log or `_sha8` from `<run>/config.json`. `lr_schedule` / `decay_steps` and `grad_accum` are dropped from the hashed blob at their defaults, so all 55 (later 58) existing run directories reproduce their pinned sha8

## Per-arm precision and recall (50k, threshold 0.40)

| arm | macro P | macro R | micro P | micro R |
|---|---:|---:|---:|---:|
| h1 final | 0.0763 | 0.0179 | 0.6396 | 0.3321 |
| h1 best_val | 0.0781 | 0.0199 | — | — |
| h9 control | 0.4523 | 0.0179 | — | — |
| h9 cb_ntr | 0.0756 | 0.0239 | — | — |
| h10 final | 0.0696 | 0.0371 | 0.4088 | 0.4271 |
| h10 best_val | 0.0696 | 0.0373 | — | 0.4170 |
| u12-C-L-asl-g4.0 | 0.0764 | 0.0720 | 0.3448 | 0.4899 |
| u1-C-L-asl-g4.0 | 0.0663 | 0.0791 | 0.2919 | 0.5101 |
| u1-C-H2048-asl-g4.0 | 0.0830 | 0.0479 | 0.3725 | 0.4726 |
| u1-C-L-asl-g2.0 | 0.0924 | 0.0591 | 0.3837 | 0.4533 |
| u1-C-L-asl-g1.0 | 0.1073 | 0.0474 | 0.4573 | 0.4120 |
| arm 13 two towers g4.0 | 0.0693 | 0.0610 | 0.2974 | 0.4733 |
| arm 14 two towers g2.0 | 0.0960 | 0.0570 | — | — |
| SigLIP2-B u1 g2.0 | 0.1010 | 0.0730 | 0.4061 | 0.4742 |
| SigLIP2-B u1 mean pool | 0.1021 | 0.0700 | 0.4075 | 0.4585 |
| 2b6b81aa u1 600k | 0.2960 | 0.3213 | 0.3519 | 0.7503 |
| 77a135b2 u12 600k | 0.1304 | 0.1237 | 0.2965 | 0.6408 |
| b6e293f0 u0 600k | 0.2958 | 0.3151 | 0.3466 | 0.7438 |
| 2a5c9bad u1 384 600k | 0.2823 | 0.2975 | 0.3488 | 0.7315 |
| f63f6f54 large 600k | 0.2701 | 0.2636 | 0.3679 | 0.7400 |
| a4674fbd large 240k accum | 0.2355 | 0.2781 | 0.3294 | 0.7586 |

Wall clock and throughput, all measured: a 5200-step 10k arm is 40-45 min at 99-109 img/s; u1-C-L-asl-g4.0 35.3 min at 130 img/s and 16.8 GB; arm 13 44.1 min at 106 and 17.1 GB; arm 14 44 min at 106 and 17.1 GB; u1-C-L-asl-g2.0 35.1 min at 130; SigLIP2-B arm 33.9 min at 138 and 15.8 GB; mean-pool arm 34.0 min at 136 and 15.8 GB; gate 2b6b81aa 93.5 min at 114 and 15.8 GB; 77a135b2 101 min at 100 and 18.5 GB; b6e293f0 75 min at 118 and 15.8 GB; f63f6f54 276 min, then 491.8 s to score at 102 img/s, batch 128; a4674fbd 108.3 min at 43 and 21.1 GB; c8f2a635 about 3.9 h; the paused main run 15.8 GB at 116-122 img/s. h10 used 20.6 GB and arm 01 20.4 GB. The g1.0 arm was staged at 108 img/s and 16.8 GB and scored at 130

MCC where recorded, macro and micro: h10 0.0440, arm 01 0.0669, u1 g2 0.0661, SigLIP2-B 0.0786, f63f6f54 0.2462 and 0.5201, a4674fbd 0.2344 and 0.4980

First-sweep 1k peaks, trajectory only: h1 0.0298 at step 1040, h2 0.0267 at 832, h3 0.0262 at 1456, h4 0.0217 at 1872, h7 0.0184, h10 0.0312 at 1248, h11 0.0119 at 1456. h1's last 1k reading was 0.0208

Checkpoint naming and sizes: `ckpt_step*.pt` at the step number, `best_val.pt` for the selection artifact, 231 MB with the linear `cls` head and 639 MB with `cls_mean_max_std`. `runs/4a13a120` paused at 86241 with `.PAUSE`; arm 01 relaunched from `ckpt_step000137.pt` with no sample replay; a resume after a pause at step 151 skips 1812 samples, 151 groups of twelve. Scored runs keep `ckpt_step005000.pt` and `ckpt_step005200.pt` beside `best_val.pt`

Per-checkpoint probability dumps sit at `I:\.cache\wd-tagger-trainer\probs\<run8>_<ckpt>.npy`: 28 files, 1247.6 MB each, 50000 x 12476 float16, with `_005000` and `_best_val` for the scored runs. So a per-tag comparison of two arms needs no inference, only the same 50000 images and two probability matrices. Trainable parameters: h3 94.71M with a 9.60M head, h4 101.05M with 15.94M, h5 116.98M with 31.86M. BALMS is arXiv 2109.04712

## Reference points (same 50k cache, threshold 0.40)

| model | macro F1 | micro F1 | condition |
|---|---:|---:|---|
| bar `convnextv2_huge` | 0.5579 | 0.6886 | 692.6M parameters, head 35.15M; macro P 0.6201 R 0.5333, micro P 0.7033 |
| paused full run `4a13a120` @086241 | 0.2815 | 0.6133 | DINOv3 u12 CMSS asl 4.0, 4138368 images, 77% of a pass; macro P 0.4608 R 0.2266, micro P 0.7068 R 0.5417 |
| gate `2b6b81aa` | 0.2733 | 0.4791 | 600000 images; macro P 0.2960 R 0.3213, micro P 0.3519 R 0.7503 |
| main run `3da79e14` best_val step 10000 | 0.1775 | 0.5164 | 480000 images, LR still at peak; best threshold 0.50 |
| per-tag oracle ceiling | 0.3879 | — | per-tag threshold oracle on the 50k |
| bar head size | — | — | 35.15M, avgpool -> LayerNorm -> dropout -> Linear(2816, 12476) |

The 10k arms are not comparable to these absolute numbers (1/50th of the validation distribution, far less training), only to each other

## Findings that outlive any single arm

### Macro@0.40 peaks early then declines, and the 10k benchmark measured overfitting

h1's mid-training curve, 1000-image val, `--max-steps 5000`:

  | step | 208 | 416 | 624 | 832 | 1040 | 1248 | 1456 | 1664 | 1872 | 2080 | 2288 | 2496 |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|
  | macro@0.40 | .0122 | .0234 | .0263 | .0256 | .0298 | .0278 | .0272 | .0261 | .0252 | .0235 | .0248 | .0236 |
  | micro F1 | .3263 | .3905 | .4155 | .4092 | .4313 | .4234 | .4243 | .4256 | .4238 | .4220 | .4291 | .4262 |

Macro falls 21% relative from its peak by step 2080 while micro stays flat: the tail gets squeezed as the head overfits 10k images. Step-to-step wiggle is ±0.0015

Consequences. A 5000-step arm is ~5x past its macro optimum, so scoring the final checkpoint understates every arm and may rank arms by overfitting resistance rather than head quality. Score a mid-training checkpoint too: the default cadence is every 1000 steps. Extending inconclusive arms to 10000 steps is suspect, because more steps lower macro at this scale. This is a property of the 10k subset, not the 5.3M corpus: at 10k images a 38M-parameter head memorises fast

h10 ASL showed no late decline: peak moved from step 1040 to 4784, final == best_val. Macro-peaks-early-then-decays is a property of BCE on a long-tailed label set, not of the benchmark, and the full 600k gate never bent, so the 10k freeze-depth answer was measured in the wrong regime

### The macro deficit is tail-concentrated, and head capacity is not the cause

Per-tag F1 gap to the bar against training frequency, on the 50k cache: 0.322 unseen, 0.319 for 10-99, 0.190 for 100-999, 0.126 for 1k-9.9k, 0.055 for >=10k. About 97% of the total macro gap is tags with under 1000 training examples

Our per-tag-oracle threshold ceiling is 0.3879 against the bar's 0.5579 at a fixed 0.40, so no calibration, threshold or rescaling trick closes the gap; it lives in the scores

Head capacity is not the differentiator: ours (38.34M) is larger than the bar's (35.15M)

### ASL's gain is entirely in the tail, none in the head

Tag bins by positives in the 50k subset, `probs/470e6fa9` against `probs/7ee12110`, threshold 0.40:

| bin (val positives) | tags | BCE F1 | ASL F1 | delta |
|---|---:|---:|---:|---:|
| 1-9 | 4641 | 0.0002 | 0.0028 | +0.0027 |
| 10-49 | 5093 | 0.0034 | 0.0189 | +0.0155 (5.6x) |
| 50-199 | 1682 | 0.0413 | 0.0906 | +0.0492 (2.2x) |
| 200-999 | 779 | 0.1471 | 0.1834 | +0.0363 |
| 1000+ | 259 | 0.4100 | 0.4098 | -0.0001 |

2636 of 12454 tags improved (21.2%), essentially every bit in the 10-999 band; the 259 frequent tags are unchanged to four decimal places. Top movers: label 11171 (10 positives) 0.000 -> 0.667; label 9579 (29) 0.000 -> 0.619; label 9736 (17) 0.000 -> 0.571. The bar's 0.5579 cannot be reached by improving frequent tags, already at 0.41

Bin-mass arithmetic, macro being a mean over 12454 tags:

| bin | tags | ASL F1 | contribution to 0.0410 macro | share |
|---|---:|---:|---:|---:|
| 1-9 | 4641 | 0.0028 | 0.00104 | 2.5% |
| 10-49 | 5093 | 0.0189 | 0.00773 | 18.9% |
| 50-199 | 1682 | 0.0906 | 0.01224 | 29.8% |
| 200-999 | 779 | 0.1834 | 0.01147 | 28.0% |
| 1000+ | 259 | 0.4098 | 0.00852 | 20.8% |

The 1-9 bin is 37% of the tags and 2.5% of the score: 5x there is worth +0.0042, the same as +5% on the 50-199 bin's 1682 tags. The mass is 10-999, where ASL already went; judge gamma by the 10-199 bins. With ~9 positives a tag's prior is ~1e-6, so a calibrated posterior can only cross 0.40 at a likelihood ratio >1e6, and the 1-9 bin may simply not be learnable from 9 examples, which would make label-text or feature initialisation (DECAF/CLIP-Decoder) the honest spend rather than loss shaping

### The head axis is inert under BCE and decisive under ASL

Under BCE four head variants span 0.0232-0.0285 macro and 0.4214-0.4450 micro, a 5% band against a 3% seed band; MLP heads are monotone worse (linear 0.0272, H512 0.0244, H1024 0.0245, H2048 0.0232, H4096 0.0219), and only h3's cls pooling clears the band at +4.8%

Under ASL at gamma 4 the same cls swap is +57.6% (0.0646 against 0.0410), the largest single-change effect measured. The distinction is BCE-axis versus ASL-axis, and anyone reading only the BCE sweep will assume the axis is inert

The mechanism is recall: macro P 0.0696 -> 0.0764 while macro R doubles 0.0371 -> 0.0720, and micro P falls 0.4088 -> 0.3448. At step 1040 of 5200, arm 01's 1000-image val read macro 0.0372 against h10's 0.0306 at the same step and 0.0316 at h10's peak of step 1248, with micro 0.2988 against 0.3037; over the first five checkpoints, 208 to 1040, macro led at all five and micro was ahead at 416 by 0.027 and behind from 624 to 1040 by 0.005 to 0.011. The run's loss at step 1140 was 0.0011. Macro rises steeply with subset size on that arm, 0.0407 at 1k, 0.0544 at 5k, 0.0604 at 10k, 0.0646 at 50k, where h10 was almost flat (0.0315 to 0.0410)

A capacity bottleneck is not a fix either: H2048 under ASL loses 19.5% macro and gains 12.2% micro

### Gamma is bracketed at three points and 2.0 is the peak

| gamma | macro F1 | macro P | macro R | micro F1 | micro P | micro R |
|---:|---:|---:|---:|---:|---:|---:|
| 4.0 | 0.0632 | 0.0663 | 0.0791 | 0.3713 | 0.2919 | 0.5101 |
| 2.0 | 0.0624 | 0.0924 | 0.0591 | 0.4156 | 0.3837 | 0.4533 |
| 1.0 | 0.0575 | 0.1073 | 0.0474 | 0.4335 | 0.4573 | 0.4120 |

One clean monotone trade: as gamma falls, precision and micro rise, recall and macro fall. 1.0 costs 7.8% macro and buys 4.3% micro; 3.0 would interpolate to roughly macro 0.0628 and micro 0.393 and dominate neither. The bar is 0.5579 macro and 0.6886 micro and we are about twice as far below on macro, so the trade is the wrong direction and 2.0 stands. Three arms, 105 minutes, axis closed. At 12476 labels with 0.26% positives the bulk is what micro measures, and it is also what the bar wins on micro with 0.7033 precision

### Class weighting and pos_weight are dead ends at this label density

The same mechanism explains h9: macro precision collapses 0.452 -> 0.076 while recall rises 0.018 -> 0.024, NTR's fixed kappa=0.05 buying tail positives at every step

Plain BCE gives positives 0.259% of the loss mass; CB beta=0.9 drops that to 0.063%, beta=0.999 to 0.019% -- a 4x and 14x cut in positive signal, which is the 0.0097 collapse. Under mean-1 normalisation the near-empty labels get the largest weights on their negatives. `pos_weight` (0.68-100 over 10000 images) protects the 1686 zero-positive labels, which was the point, but over-corrects: 1k peak 0.0119, last 1k 0.0058, final macro 0.0068, micro 0.2572

### Freeze depth is closed, but only at this parameterisation

| arm | unfrozen | live rotations | live base weights | macro F1 | micro F1 | img/s | VRAM |
|---|---:|---:|---:|---:|---:|---:|---|
| `2b6b81aa` u1 | 1 | 5.17M | 7.09M | 0.2733 | 0.4791 | 114 | 15.8 GB |
| `b6e293f0` u0 | 0 | 5.61M | 0 | 0.2694 | 0.4729 | 118 | 15.8 GB |
| `77a135b2` u12 | 12 | 0.34M | 85.05M | 0.1031 | 0.4054 | 100 | 18.5 GB |

u0 vs u1: -1.4% macro, -1.3% micro, inside the band. The last unfrozen block is 7.09M parameters and 12.6% of the trainable set, worth 1.4% of macro and nothing in micro; unfreezing 12 blocks was paying 11.5M parameters and 2 hours per pass for a difference this benchmark cannot see. u0 leads for the first two thousand steps, is level by five thousand, and is 2% behind from then on. The frozen-plus-adapter arm was predicted near 110-120 img/s rather than the 160 of a tower with no adapters, because a rotation inside a frozen block still needs a backward through it; measured 118 img/s at 15.8 GB

u12 is 2.65x worse at 600000 images and was 2.2% apart at 10k; its curve never threatened to converge, running five to eight times behind from the first validation and still at 0.36 of u1's midpoint at its own end. Against it, u1 is 18% faster, 3.6 GB lighter, and a full pass drops from 13.4 h to 11.4 h. The cause is the parameterisation, not the tower: `unfreeze_last_blocks` deliberately freezes the OFT rotation inside every block it unfreezes, so u12 trains 85.05M base weights at `lr_unfreeze` 3e-5 with 0.34M live OFT, while u1 trains 7.09M base weights at 3e-5 plus all 5.17M rotations at 5e-4 -- a seventeen times smaller effective step, not more capacity at the same step size. Any future freeze-depth arm must first make the two paths comparable, by `lr_unfreeze` near 3e-4 or by leaving rotations trainable inside unfrozen blocks. Until then the ledger says only what it measured: this parameterisation, at two scales, loses. This also re-reads the paused run: `4a13a120` reached 0.2815 macro over 4.14M images at 3e-5 with no rotation in its twelve unfrozen blocks, so part of what looked like a capacity ceiling was a step-size ceiling

### The tower: SigLIP2 beats DINOv3, Qwen loses, and the large tower loses on the tail

SigLIP2-B at the champion recipe is +21.0% macro and +5.3% micro over DINOv3-B, and faster and leaner (138 img/s, 15.8 GB against 130 and 16.8), because it has no cls token and no registers and its pooler is a small attention head. The champion chain at 50k and 0.40: h10 u12-CMSS-g4.0 0.0410, u12-C-L-g4.0 0.0646, u1-C-L-g4.0 0.0632, u1-C-L-g2.0 0.0624, SigLIP2-B u1-C-L-g2.0 0.0755, which is +84% macro and +4.7% micro over the original control

Qwen3.8's vision tower is the weakest of the three and dilutes any pair it joins. Strict load of 460730096 parameters, 512px, merged pooling, 5120 dimensions, 60k images in 19 minutes. At equal head width SigLIP2 0.0457 beats DINOv3 0.0380 beats Qwen 0.0291 macro. The concat pair drops from 0.0508 to 0.0425 macro with Qwen added at H768, and SigLIP2+Qwen without DINOv3 collapses to 0.0389, below SigLIP2 alone; 5120 of 6656 input dimensions are Qwen's. Cost avoided: 54.5 GB of cached vectors and about 25 hours of extraction at 512px, 12 at 384

The large tower at 600k images with batch 20 scores 0.2331 macro, 14.7% behind base u1 and 13.5% behind u0, while winning micro by 2.6% and 4.0%. At 3.4x the parameters and 2.4x the wall clock per image it does not earn that budget

Equal-image lead, in-run 10000-image prefix: 22.1% at 100000 images, 15.1% at 150000, 9.0% at 200000, then deficits of 2.5% at 250000, 8.7% at 300000, 13.7% at 350000, 18.2% at 400000 and 23.2% at 450000. Base-u0 interpolated on the same axis: 0.0743 at 60000, 0.0920 macro and 0.4661 micro at 100000, 0.0992 at 120000, 0.0909 in the later reading, 0.1083 at 150000, 0.1244 at 200000, 0.1860 at 400000, 0.2002 at 450000

The large tower's macro@0.40 went 0.0832, 0.1110, 0.1247, 0.1356 at 50k, 100k, 150k, 200k, so increments of 0.0278, 0.0137, 0.0109 per 50000 images, while base-u0 over the same span increments by 0.0174 and 0.0161 and holds its rate; from 400000 on the large tower adds about 0.004 per 50000 where base adds 0.017. The projection written at 200000 images was about 0.168 in-run, call it 0.18, projecting by the measured 1.20x to 0.216 at 50k; the same rule put the annealed endpoint at 0.17 to 0.18 in-run, which is 0.24 at 50k

The large tower has higher precision and lower recall at both levels (macro P 0.2701 against 0.2960, R 0.2636 against 0.3213). Tag-level comparison, `b6e293f0_012500` against `f63f6f54_030000`, same 50000 images at 0.40:

| support | labels | base u0 F1 | large F1 |
|---|---:|---:|---:|
| under 5 | 1472 | 0.1687 | 0.0900 |
| 5-20 | 6027 | 0.2508 | 0.1858 |
| 20-50 | 2257 | 0.3057 | 0.2997 |
| 50-200 | — | — | +0.0236 |
| 200-1000 | — | — | +0.0286 |
| 1000-5000 | — | — | +0.0263 |
| above 5000 | — | — | +0.0102 |

The large tower is better on 4319 labels and worse on 4694, better on every bucket from support 50 up, and it loses catastrophe on the 7499 labels below support 20, which is 60% of the label space and therefore 60% of the macro average

Precision/recall split of that deficit: support 0-5 base P 0.2116 R 0.1682, large P 0.1155 R 0.0957; support 5-20 base 0.3297 / 0.2426, large 0.2736 / 0.1731; support 20-50 large precision +11% and recall -13%; from support 50 up large precision +12% to +15% with recall level. A collapse in both directions is a model that never learned the tag, not a conservative one

The batch arithmetic points at the mechanism: at batch 20 a tag in 20 of 50000 images has 0.008 expected positives per update, support 50 has 0.02, support 200 has 0.08. Those tags are absent from almost every batch, and ASL normalises by the tag's positive count inside the batch, so their gradient is zero most updates and enormous on the rare update that contains one. Base at batch 48 is only 2.4x less starved. The prediction from the control: if base at batch 24 collapses to about 0.09 in the support 0-5 bucket, batch is the mechanism; if it holds near 0.16 and loses only a little, capacity per rare label is. Macro weights labels equally, so the sensitivity is the bucket sizes: 1472 plus 6027 labels is 7499 of 12454, 60.2% of the average; 20-50 adds 18.1%, everything at 50 or above is 21.8%. A 0.01 fall in the low-support mean F1 moves macro by 0.006, and a 0.065 wide gap on the 7499 rare labels is not something a lower learning rate closes

The frozen tower is not a worse tower: a frozen-feature probe with the same head and the same 10000 images gives SigLIP2-B 0.0452 / 0.1593 and SigLIP2-L 0.0630 / 0.1880, so the large tower is +39% macro and +18% micro before any adaptation. The bar buys its 0.5579 macro with 692.6M parameters, and we had compared an 86M tower against a 93M tower and found the second better without ever asking whether the family continues to pay

### Images per update, not capacity, was the large tower's deficit

Gradient accumulation with an elementwise-mean reduction is exact, because the mean of equal-sized micro-batch means is the whole-batch mean, which is not true of BALMS's `sum/n_pos`. Verified: one batch of 12 against three of 4 on the same twelve images gives loss agreement to 1.2e-5 relative, gradient cosine 0.999999, rms difference 1.4e-3, the residual being bf16 rounding. The first attempt read cosine 0.947 because `head_dropout` 0.1 drew one mask for the single batch and three for the micro-batches

`a4674fbd`, the large tower at batch 24 with two accumulated micro-batches, 240000 images, is +45% macro and +4.5% micro against `f63f6f54` at matched 240000 images (0.2197 / 0.4593 against 0.1516 / 0.4397). Buckets: support 0-5 0.0423 against 0.0833 (+97%), 5-20 0.0981 against 0.1783 (+82%), 20-50 0.1993 against 0.2866 (+44%), 50-200 +19.5%, 200-1000 +9.2%, 1000-5000 +3.9%, above 5000 +0.1%. Monotone in rarity and exactly the inverse of the tower's loss, which was on the rare labels too

Against base at matched images that is 11% ahead at 48000, 21% at 96000, 18% at 144000, 19% at 192000 with micro 4% to 8% ahead; against the batch-20 arm it is 10% ahead at 100000 and 23% at 200000

### 600k-image trajectories

u1 gate `2b6b81aa`, in-run 10000-image prefix. The gate was 600000 images in one pass at 12500 steps, measured at 114 img/s rather than the 95 to 110 assumed, so 88 minutes for u1 rather than 100 and two full passes are 26 hours rather than 28 to 32:

| step | images | macro | micro | macro@0.40 |
|---:|---:|---:|---:|---:|
| 1250 | 60k | 0.0680 | 0.4572 | 0.0696 |
| 2500 | 120k | 0.0970 | 0.4702 | 0.0946 |
| 3750 | 180k | 0.1192 | 0.4886 | 0.1172 |
| 5000 | 240k | 0.1416 | 0.5099 | 0.1388 |
| 6250 | 300k | 0.1627 | 0.5111 | 0.1586 |
| 7500 | 360k | 0.1831 | 0.5299 | 0.1794 |
| 8750 | 420k | 0.2001 | 0.5372 | 0.1968 |
| 10000 | 480k | 0.2162 | 0.5497 | 0.2151 |
| 11250 | 540k | 0.2278 | 0.5531 | 0.2276 |

u0 against u1, same 600000 images, same in-run cache:

| step | u0 macro | u1 macro | u0 micro | u1 micro |
|---:|---:|---:|---:|---:|
| 1250 | 0.0740 | 0.0680 | 0.4617 | 0.4572 |
| 2500 | 0.1019 | 0.0970 | 0.4684 | 0.4702 |
| 3750 | 0.1211 | 0.1192 | 0.4882 | 0.4886 |
| 5000 | 0.1414 | 0.1416 | 0.5065 | 0.5099 |
| 6250 | 0.1597 | 0.1627 | 0.5096 | 0.5111 |
| 8750 | 0.1958 | 0.2001 | 0.5280 | 0.5372 |
| 11250 | 0.2223 | 0.2278 | 0.5464 | 0.5531 |

u12 `77a135b2` at step 2920 of 12500, far behind from the first validation:

| step | u1 macro | u12 macro | u1 micro | u12 micro |
|---:|---:|---:|---:|---:|
| 1250 | 0.0680 | 0.0086 | 0.4572 | 0.3383 |
| 2500 | 0.0970 | 0.0179 | 0.4702 | 0.3748 |

Large at batch 20, `f63f6f54`, equal-image points: 50000 macro 0.0842 micro 0.4753, 100000 0.1121 and 0.4949, 150000 0.1264 and 0.5015, 200000 0.1371 and 0.5099, 250000 0.1379 and 0.4925, 300000 0.1452, 400000 0.1521, 450000 0.1537. Micro peaked near 200000 and drifted: 0.4753 at 50000, 0.5099 at 200000, then 0.4925, 0.5124, 0.5091, 0.4990, 0.5078, 0.5009 with the last at 500000. Step 25000 reads 0.1597 macro and 0.5009 micro, step 27500 0.1645 and 0.5323

Like for like at matched images, batch 20 against batch 24 x2:

| images | batch 20 macro | batch 20 micro | accum macro | accum micro |
|---:|---:|---:|---:|---:|
| 48000 / 50000 | 0.0842 | 0.4753 | 0.0823 | 0.4825 |
| 96000 / 100000 | 0.1121 | 0.4949 | 0.1229 | 0.5047 |
| 144000 / 150000 | 0.1264 | 0.5015 | 0.1432 | 0.5161 |
| 192000 / 200000 | 0.1371 | 0.5099 | 0.1685 | 0.5129 |

The accumulation arm's first reading, step 1000 of 5000 and 48000 images: macro 0.0823, micro 0.4800, 10809 labels with any positive, against `f63f6f54`'s 0.0832 at 50000 images, so level rather than better. At step 4280 of 5000, 42 img/s, its learning rate was already in the tail at 9.92e-5. At 192000 against 200000 images it is 22.9% ahead on macro and level on micro; at 96000 against 100000 it was 9.6%, so the advantage grows with images rather than settling, which is what starvation of signal per update looks like and not what a capacity limit looks like. If the 9.6% lead carried to 240000 the arm would sit near 0.154 against `f63f6f54`'s 0.140 at the same budget

Base's own curve steepens late: it gains 0.0626 macro between 240000 and 540000 images while the batch-20 tower gained 0.0081, so a lead measured at 192000 images is not a lead at 600000, which is exactly what happened to the batch-20 arm

### Model geometry and static cost

DINOv3-B/16 at u1: OFT 4.88M over 73 modules, head 9.60M, unfrozen 7.09M, backbone 91.03M, state 231 MB. SigLIP2-B/16 at u1: OFT 5.17M over 76 modules, head 9.60M, unfrozen 7.09M, backbone 99.13M, state 248 MB. The 76 modules are 72 block Linears, the patch-embedding conv, and three inside the MAP head: two MLP Linears and the attention output projection

SigLIP2's `cls` vector is the output of a trainable-attention pooler, `SiglipMultiheadAttentionPoolingHead`, about 2.3M parameters, whose base weights stay frozen while its rotations adapt. `SiglipVisionModel` in transformers 5.16.1 has children `embeddings`, `encoder`, `post_layernorm`, `head`, with 12 blocks at `encoder.layers`, hidden 768, intermediate 3072, patch 16, image 512; `last_hidden_state` is (B, 1024, 768) with no cls token and `pooler_output` is (B, 768). The patch offset is 5 for DINOv3 and 0 for SigLIP2, and SigLIP2's input normalisation is 0.5/0.5, which makes `_prep` produce `(x/255 - 0.5)/0.5`

SigLIP2-L: 3.53 GB of safetensors, 24 blocks, 1024 wide, intermediate 4096, 16 heads, 512px, 148 OFT modules, OFT 14.48M, head 12.79M, backbone 331.2M, 19.2 GB at batch 20. Loss was 0.0053 at step 200 with the learning rate still warming, which is the healthy shape and not the NaN the abandoned ViT-L probe produced at `lr_oft` 5e-4. Warmup is 300 rather than the doc's 4000, because 4000 was 1.5% of a 266000-step run and this gate is 30000 steps. Its cosine version ran at 39 img/s, the const version at 40-41 and 18.6 GB; `lr_oft` 2e-4 and `lr_head` 7e-5 were scaled for depth, width and batch from base's 5e-4 and 1.2e-4

Two-tower frozen: head in_features 1536, head 19.18M, 31.14M trainable (4.88M OFT + 7.09M unfrozen block + 19.18M head), state dict 288 MB and 269 tensors with zero `siglip.` keys, and a clean round trip. The frozen tower is excluded from checkpoints deliberately, since saving 93.5M dead weights would double every file. Two towers both adapting: per tower 4876800 rotations on the eleven frozen blocks plus 7088640 in the unfrozen one, both towers 23930880, head 19178684, total about 43.1M. A frozen ViT-L probe tower is 24 blocks, 1024 hidden, 145 OFT modules, 1024-dimensional cls features, 0.9 GB at batch 4. Under SigLIP2 the trainable adapter count rises by 0.29M, to 76 modules against DINOv3's 73, and its `cls` pooling vector is not a bare LayerNorm but the output of the trainable-attention pooler. Validating the wide tower costs 195 s for the 10000-image prefix against 38 s on base, so twelve validations across the gate are 39 minutes on top of 3.9 hours of stepping

### The main run, measured so far

`runs/3da79e14`, sha8 `3da79e14`, two passes, 221739 steps over 10643426 images at batch 48 with 8 decode threads:

```
python -m wdtag.train --config configs\main\01_u1-C-L-asl-g2.0-siglip2b-main.json --select-metric macro040 --val-every-steps 10000 --ckpt-every-steps 5000
```

Launch confirmed `params: OFT 5.17M, head 9.60M, unfrozen 7.09M in 1 block(s), backbone 99.1M, 76 OFT modules`, `loss asl: gamma_neg 2.0`, 116-122 img/s at 15.8 GB, expected wall 24 to 26 hours. It selects on 50k every 10000 steps, 22 times; one such validation costs 284.7 s measured, not the 190 estimated, so 105 minutes or 4.4% of wall clock. Checkpoints every 5000 steps, 44 files at about 248 MB, 11 GB

First 50k validation at step 10000, 480000 images, learning rate still at peak (4.98e-4 of 5e-4): macro 0.1792, micro 0.5164, macro@0.40 0.1775, best threshold 0.50, so the operating point was not yet distorted. That is not comparable to the 600k gate's 0.2733 at 540000 images, because the gate's cosine spanned 12500 steps and had annealed to a seventh of peak by 90% of its run

Paused deliberately at step 15373, checkpointed as `ckpt_step015373.pt` and exited within one step of the `.PAUSE` file appearing, with checkpoints at 5000, 10000, 15000 and 15373. `best_val.pt` from step 10000 holds macro@0.40 0.1775 and micro 0.5164. `runs/3da79e14/PAUSED.md` holds the reason, the resume command and the warning about the `.PAUSE` file

The `--select-metric macro040` switch needed two fixes: the threshold summary's values live in `extra`, so `metrics.get(sel_key)` would have resolved to NaN and selected nothing forever, and the `best` dict lacked a key for a 0.40 threshold score. A unit test on a hand-built 4x2 matrix returns 1.0 at every threshold and resolves all four metrics. The end-to-end smoke passed at batch 4, `runs/671e6989`, seed 4242, 205 steps, `--val-every-steps 100`: both validations fired mid-epoch and both selected, `new best val macro_f1_at_0.40 0.0023` at step 100 and 0.0044 at step 200, against a best-threshold macro of 0.0033 at the same step, which is why the protocol needs its own metric. A first attempt reused arm 16's config, hashed to arm 16's directory, saw step 5200 against `--max-steps 215` and exited without training

### The schedule, not the tower, explained the 600k large-tower plateau

The large gate held `lr_oft` 2e-4 for 90% of its 30000 steps and annealed only in the final 3000, while every base gate ran a cosine from step one, so every comparison of the two was annealed against plateaued. It does not anneal until step 27000 of 30000. The constant schedule exists for exactly this: the peak holds and only the last `decay_steps` fall to the 5% floor, so raising `stream_images` slides the anneal rather than stretching it, 30000 steps to 60000 keeping 2e-4 at step 30000 and annealing 57000 to 60000. The accumulation arm's 5000 steps carry a 1500-step tail starting at step 3500. Inside the tail, at step 27500 with the learning rate at 1.59e-4 and still falling, macro@0.40 read 0.1645 against 0.1597 at step 25000 and micro 0.5323 against 0.5009: one validation window of annealing bought 3.0% macro and 6.3% micro after fourteen windows of flat learning rate bought nothing. The final 2500 steps of decay lifted macro from 0.1645 to an implied 0.194 in-run at step 30000, about 18% of macro on their own: 0.2331 official divided by the 1.20 transfer factor. Training loss across the stall was 0.0040 at 200000 images against 0.0032 at 600000, noisier than that swing, so the model was neither undertrained nor overfitting. The comparison at matched budget is also unfair in the other direction: `f63f6f54` at step 12000 was still at the constant peak with 18000 steps to run. A 3000-step tail on 30000 steps was 10%; the follow-up if anyone wants it is a tail at 30% to 40% of the schedule, which is a schedule change and not a tower change

### Cached-feature probes rank representations, not recipes

Same features, 10k train cache, 4000 steps, Adam 1e-3, batch 48, gamma 4.0, bias initialised at the label priors, scored on the 50k at 0.40:

| features | linear macro | +2048 macro | linear micro | +2048 micro |
|---|---:|---:|---:|---:|
| DINOv3-B/16 frozen, cls, 768 | 0.0407 | 0.0475 | 0.1674 | 0.2305 |
| SigLIP2-B/16-512 frozen, MAP, 768 | 0.0452 | 0.0577 | 0.1593 | 0.2206 |
| concat, 1536 | 0.0597 | 0.0586 | 0.2065 | 0.2614 |

Linear-head detail: DINOv3 0.0407 macro, MCC 0.0533, P 0.0280, R 0.1471, micro 0.1674; SigLIP2 0.0452, 0.0669, 0.0283, 0.2144, 0.1593; concat 0.0597, 0.0739, 0.0437, 0.1680, 0.2065

A bottleneck is worth +16.7% macro on DINOv3 alone and +27.7% on SigLIP2 alone, so the linear head limited a single tower. On the concat it is flat on macro (-1.8%, inside the band) and +26.6% micro, so the +32% the concat showed over the better single tower was largely a linear head having too few degrees of freedom. SigLIP2-B alone at H2048 reaches 0.0577, 91% of the fine-tuned single-tower arm at 0.0632, with no adapter and no unfrozen block

The probe has now failed to transfer three times: concat +32% over the better single tower became +1.6% once either tower had a bottleneck, then -1.4% (a tie) once the first tower was fine-tuned, then -13.6% for a frozen second tower under the real recipe. Adaptation is worth more than a second tower's features. Read these as rankings, not scores: the head recipe is Adam, 4000 steps, no dropout, no image pipeline, so they cannot be compared to the arms

The concat is +46.7% over DINOv3 alone and +32.1% over SigLIP2 alone, an order of magnitude outside the 3% band. The two frozen towers fail in opposite directions, SigLIP2 better on macro (+11.1%) and worse on micro (-4.8%), which is what a caption-supervised tower and a self-supervised one should do. After adaptation the margin over the cached concat is about 7.3% macro (0.0632 against 0.0586), at 130 img/s against a cached pipeline that runs at hundreds to thousands of images per second. All three frozen rows are also precision-poor at the fixed threshold, macro P 0.028 to 0.044 against R 0.147 to 0.214, so part of every figure is an operating point rather than a ranking

Twelve-row ranking at 0.40, all with the same head recipe, H meaning one hidden layer of that width:

| features | head | macro F1 | micro F1 |
|---|---|---:|---:|
| SigLIP2-B/16-512 | H2048 | 0.0577 | 0.2206 |
| DINOv3 + SigLIP2 | linear | 0.0597 | 0.2065 |
| DINOv3 + SigLIP2 | H2048 | 0.0586 | 0.2614 |
| DINOv3 + SigLIP2 | H768 | 0.0508 | 0.2139 |
| SigLIP2 | H768 | 0.0457 | 0.1676 |
| DINOv3 | H2048 | 0.0475 | 0.2305 |
| DINOv3 | linear | 0.0407 | 0.1674 |
| DINOv3 + SigLIP2 + Qwen | H768 | 0.0425 | 0.1793 |
| SigLIP2 + Qwen | H768 | 0.0389 | 0.1656 |
| DINOv3 | H768 | 0.0380 | 0.1754 |
| Qwen3.8 tower | linear | 0.0337 | 0.1576 |
| Qwen3.8 tower | H768 | 0.0291 | 0.1438 |

### A fixed 0.40 sits above macro's optimum, and recalibration is exhausted

Global thresholds only, reading cached probability matrices, no GPU:

| thr | control macro | ASL macro | control micro | ASL micro |
|---:|---:|---:|---:|---:|
| 0.05 | 0.0330 | 0.0414 | — | — |
| 0.10 | 0.0305 | 0.0433 | — | — |
| 0.15 | 0.0290 | 0.0439 | — | — |
| 0.20 | 0.0279 | 0.0436 | 0.4503 | 0.3838 |
| 0.30 | 0.0262 | 0.0427 | 0.4437 | 0.4055 |
| 0.40 | 0.0248 | 0.0410 | 0.4372 | 0.4177 |
| 0.50 | 0.0235 | 0.0393 | 0.4304 | 0.4248 |
| 0.60 | 0.0223 | 0.0372 | 0.4232 | 0.4275 |

ASL beats the control at every threshold, so this is ranking quality and not an operating-point artefact, but part of the headline +65% at 0.40 is the operating point: at each model's own best global threshold the gap narrows to +33% (0.0439 against 0.0330). Given its best threshold on the first grid the control reaches 0.0279, still 32% below ASL's 0.0410 at the protocol threshold and 36% below ASL's own best. The control has no interior optimum, rising all the way down to 0.05 where the extended grid found 0.0330; ASL has one at ~0.15, which is what a properly shaped distribution looks like. Micro is opposite, the control peaking at 0.20 and ASL at 0.60

The tag-bin P/R split first looked like a score shift: at the 1000+ tags ASL lost 0.167 precision and gained 0.120 recall with F1 flat, a 1:1 trade, and the grid above rules that out as the source of the gain

Calibration headroom, measured on `b6e293f0_012500`, the counting pipeline first reproducing the official 0.2694 at 0.40 exactly: a per-tag oracle over a 40-bin grid gives 0.3546, 31.6% above, but fitting cuts on even rows and evaluating on odd gives 0.2640 against 0.2601, a 1.5% gain, with 96.3% of tags moving their cut between -0.40 and +0.40. The oracle fits the noise of tags with a handful of positives. The global cut curve is flat-topped around where we already are: 0.2583 at 0.40, 0.2573 at 0.45, 0.2343 at 0.30, 0.2359 at 0.60 on even rows, and fitting one global cut gives 0.2615 against 0.2601, half a percent. Our probability scale already sits at its fixed-threshold optimum within a percent, so the 2.07x macro gap to the bar cannot be closed by recalibration, per-tag bias, NTR bias or balanced threshold sampling

### The in-run subset is not a proxy for the 50k

Last in-run validation against the final 50k scoring of the same run: u0 base 512, n=10000 at step 11250, macro@0.40 0.2223 and micro 0.5464 against 0.2694 and 0.4729, ratios 1.212x and 0.865x. u1 base 512, same step and subset: 0.2276 and 0.5531 against 0.2733 and 0.4791, ratios 1.201x and 0.866x. The two agree within 1%, so an n=10000 in-run reading is about 1.20x below the 50k macro number and 1.16x above the 50k micro number

u1 base 384 read 0.1092 on n=1000 at step 11250 and scored 0.2553 at 50k, a ratio of 2.34x: the 1000-image prefix divides macro by 2.3, the 10000 by 1.2. The 384 run's in-run n=10000 said 0.1069 against 512's 0.2278, a 53% deficit, and the truth was 6.6% -- the 10k prefix has support for only a few thousand of the 12476 labels, so its macro is dominated by one- and two-positive tags, precisely the tail labels that lose their pixels at 384

The factor is phase-dependent, not a constant: when the last in-run point was mid-tail and the scored checkpoint fully annealed the ratio was 1.417x rather than 1.20x. Micro can never be read off the in-run line as a check on the bar, because the prefix flatters it by 15%

Two arms did agree: on the SigLIP2-B arm the 1k subset read 0.0438 against DINOv3's 0.0394 at step 1248, and on the mean-pool arm 0.0409 against 0.0433 at step 1040, after a day where it agreed with nothing. That is a fact about those two arms, not a licence to trust the 1k number again

The projection rule: when a curve has flattened, its 50k macro is about 1.2x the in-run 10000-image reading, so a final in-run 0.22 projects to 0.264 and 0.25 projects to 0.30. It is only valid at convergence, because the ratios were measured on arms whose curves had flattened by 11250 steps

### Throughput is governed by unfrozen blocks and the adapter's backward

Reference head, batch 48, 512px, compile(default), exact Cayley, `profile_step.py`, full step including optimizer:

| blocks | fwd ms | fwd+bwd ms | full ms | img/s |
|---:|---:|---:|---:|---:|
| 0 | 162.4 | 353.0 | 357.0 | 134.4 |
| 1 | 162.4 | 358.6 | 366.9 | 130.8 |
| 2 | 162.3 | 361.3 | 370.0 | 129.7 |
| 3 | 162.4 | 363.7 | 373.3 | 128.6 |
| 4 | 162.3 | 366.8 | 376.7 | 127.4 |
| 6 | 162.4 | 372.4 | 386.8 | 124.1 |
| 12 | 172.4 | 400.9 | 420.2 | 114.2 |

About 1.7 img/s per unfrozen block; the whole climb from 12 to 0 is worth 18%. Freezing is worth 4.25 img/s per unfrozen block at the adapter-bearing parameterisation. `opt.step()` is 0.4 ms, the model compiles to 1 graph with 0 graph breaks, and validation is not in these numbers. The fast era is on disk: `runs/_wide_head.log`, run `b2a924f1`, 249 real steps at median 160 img/s (p05 147, min 142, max 169) at 16.7 GB with `max-autotune` and no `unfreeze_blocks` key (backbone frozen, 87.0M)

max-autotune looked like +19% then and +3% now because it is a forward win: eager fwd 207.1 -> compile(default) 169.3. Frozen backbone carries it to the total, reproducing 160 from 134.4 (x1.19); with 12 blocks unfrozen backward is 224 ms of a 420 ms step and the win dilutes to +3% (401.5 against 413.6 ms)

The per-block ladder over-predicts by 25 img/s once an adapter sits inside the tower, because backward still traverses eleven frozen blocks to deliver gradients to their rotations and a flat ~28 ms compiled-optimizer dispatch runs over 73 adapter tensors: at one unfrozen block the ladder predicts 155 and the measurement is 130. The 160 img/s frozen figure applies to a tower with no adapters inside it, which is `runs/_wide_head.log` and nothing we train deliberately

Measured arms: u1 base 130 img/s at 16.8 GB, u12 109-116 at 20.4 GB, frozen second tower 106 at 17.1 GB, SigLIP2 base 138 at 15.8 GB, 384 160-171 at 9.1 GB, large batch 20 39-41 at 18.6-19.2 GB, a4674fbd 43 at 21.1 GB, main run 116-122 at 15.8 GB. A per-window 98-99 img/s is 94.7 img/s over a full epoch once the 4.4 s validation is counted, and a second CUDA context on this card costs the trainer about 45%, so a full pass over 5.32M images reads ~15.5 h at 95 img/s, not the 9.2 h the 160 number suggests

Long run: 110870 steps at 114.2 img/s = 13.5 h, at 134.4 = 11.0 h, at 160 = 9.2 h

The 384 arm bought 1.45x throughput and 40% less memory, not the 2.15x the FLOPs ratio predicts, because about 3 of the 8.8 ms each 512 image costs is tar read, jpeg decode and step overhead that resolution does not touch; at large-tower scale the compute share is bigger and the gain should approach 2x. It was scored on a 384 cache built by bilinearly downsampling the 512 one, one resample more than the training stream saw, so 0.2553 is a floor and the true cost may be under 6.6%

### Checkpoint cadence, selection and scoring artifacts cost numbers until fixed

49 checkpoints x 638.8 MB = 30.6 GB per 5000-step arm in a 41.6-minute run, written by two interleaved writers, `--ckpt-every-steps` default 200 and an unconditional epoch-end save at 208 steps, so effectively a 639 MB file every ~100 steps. Cost per write 5-15 s: 16 steps across a save measured 0.76 s/step against a 0.44 s/step baseline, 32 steps measured 1.02 s/step, total overhead ~300 s, ~8% of wall. Fixed at the source: the epoch-end save is now opt-in via `--ckpt-every-epochs N` (default 0), the cadence lives in `wdtag/config.py` as `CHECKPOINT_EVERY` 1000. Effect per arm: 49 files / 30.6 GB / ~3 min -> 5 files / 3.2 GB / a few seconds; across nine arms roughly 24 minutes of GPU time and 245 GB of writes

Disk: a checkpoint carries optimizer state, 231 MB with the linear `cls` head and 639 MB with `cls_mean_max_std`; a 5200-step arm is ~5 of them plus `best_val.pt`, so ~2-3 GB. The every-epoch save cost ~8% wall and 30.6 GB per arm. Pruning 48 intermediates from two scored arms freed 59.9 GB, leaving 388.1 GB free. The main run writes 44 files at about 248 MB, 11 GB

The two training loops, cached DataLoader and tar-streamed, were unified into one `StepHelper` (`wdtag/train.py:255`, `step` at `:285`) that owns the pause check, the first-100 debug dump, the `log_every` line, the checkpoint save, the `--max-steps` exit and validation, returning `go` / `stop` / `complete`. `train.py` went 815 -> 808 lines, +93/-100. Verified by a 40-step run of the same config before and after: identical step lines, same loss (0.0092, 0.0067), same LRs and VRAM, differing only in wall-clock img/s (98 against 107, the box idle for the second). It exposed that the cached branch printed `training done` and went straight to test evaluation, so every completed 10k arm ended with no checkpoint at its final step, up to 999 steps stale; both branches now save `ckpt_step{global_step}.pt` before evaluating

### Pause and resume are exact; an uninterrupted run is not reproducible without --deterministic

Two identical runs with the same seed, data and config do not agree: initialisation hashes identically, the first batch hashes identically, the loss of that batch is identical to twelve decimal places, and the gradient checksum differs. Flash attention's backward uses atomics (`attention_backward.cu`) and is the only non-deterministic op in the graph, so no resume can be bit-identical to an uninterrupted run because the uninterrupted run is not bit-identical to itself

With `--deterministic` on, forcing the math attention backend and disabling cudnn autotune, two processes produce 288 of 288 identical adapter tensors after 60 steps, worst absolute difference 0.0. Against a 120-step uninterrupted control in that mode: pause at step 41 and resume skipping 328 samples gives 288 of 288 adapter tensors, 80 of 80 optimiser states, identical `best` and identical logged loss at step 120; a clean `--max-steps` stop at 60 then resume to 120 also gives 288 of 288; cache mode stopping at 30 and resuming to 60 gives 288 of 288; the stream cursor gives 12 of 12 batches byte-identical across a 100-step skip, images, labels and source stems, with the resumed reader's `consumed` reading 808 for what is batch 101 at batch 8. A divergence between resumed and uninterrupted is the same size and origin as between two uninterrupted runs. The price is 23 img/s against 55 at batch 8 on a base tower, about 2.4x, so it is a flag and every number in this file comes from the fast path

Defects fixed while testing: checkpoints are written to `path.tmp` and `os.replace`d, because a job reaped during a 300 MB save left a truncated file and resume takes the highest step, so one bad write would have made a run with three good checkpoints unresumable -- proven by truncating `ckpt_step000120.pt` to half length, which now warns and falls back to step 60; resume walks newest to oldest; a `.PAUSE` present at launch exits with a message and touches nothing instead of writing a 300 MB step-0 checkpoint on every accidental relaunch, which matters because `runs/3da79e14` and `runs/f63f6f54` keep their `.PAUSE` deliberately as a guard

### Augmentation is not a lever on this label space

The trainer applies no augmentation at all and the scoring path applies no test-time augmentation, so every number here is a single forward pass on an unmodified image. Flip averaging was priced at 25 minutes and rejected: colour jitter breaks colour tags, crops break composition tags, a mirror breaks orientation tags and mirrored text, and character identity is not preserved under a mirror. 214 of 12476 labels mention an orientation or text word, a floor rather than a population because a substring count sweeps up background and backpack and misses tags like `arms_behind_back`, and 200 tags moved wrong is 1.6% of a macro average

### Two-tower concatenation is a tie, and the fusion defect was real

| arm | gamma | fusion norm | macro F1 | macro P | macro R | micro F1 |
|---|---:|---|---:|---:|---:|---:|
| u1-C-L-asl-g4.0 | 4.0 | single, one tower | 0.0632 | 0.0663 | 0.0791 | 0.3713 |
| u1-C-L-asl-g2.0 | 2.0 | single, one tower | 0.0624 | 0.0924 | 0.0591 | 0.4156 |
| arm 13, two towers | 4.0 | single, over concat | 0.0546 | 0.0693 | 0.0610 | 0.3652 |
| arm 14, two towers | 2.0 | per tower | 0.0615 | 0.0960 | 0.0570 | 0.4159 |

Two towers with per-tower norm and the better gamma land on a dead heat with one tower: -1.4% macro inside the band, +0.07% micro. Against arm 13 the change is +12.6% macro and +13.9% micro, confounded; splitting out the known gamma effect (-1.3% macro, +11.9% micro at single tower) leaves the per-tower norm worth roughly +14% macro and +2% micro

The shared `LayerNorm(1536)` was not a scale mismatch: measured on four images, DINOv3's pooled cls has mean absolute value 0.496, per-dimension standard deviation 0.029 and L2 norm 17.30, while SigLIP2's MAP output has 0.246, 0.031 and 14.00, norms differing by 1.24. It still coupled two feature manifolds through one affine, and removing that was worth more than everything the second tower contributed. Concatenation fusion is dead; anything that revives it needs a per-tag gate, cross-attention over both towers' tokens or a learned query set, 13 to 20M parameters and a real day of work

The prediction written before arm 14 ran, that per-tower norm would move the result by less than the 3% band, was wrong; the fusion defect was real and worth roughly +14% macro. The cause was not a scale mismatch but a shared affine over two manifolds, so the arm tested concat-plus-one-norm and a later arm tested proper late fusion

The frozen second tower costs 18% of throughput (106 against 130 img/s) and 0.3 GB, close to the predicted 3.5 ms per image for a `no_grad` forward, and 9.6 minutes per arm. Its checkpoint is 379.8 MB against 303.2 MB, a 76.7 MB difference that is exactly the 19.18M head in fp32, so the state-dict exclusion of the frozen tower holds on the real save path

### Loss implementation: what shipped against the papers

The first `wdtag/losses.py` did not match the papers it cited; every defect below was measured against the reference, fixed and re-verified

| function | verdict | defect |
|---|---|---|
| `bce` | matches | the control, and it defines the loss scale |
| `asl` | deviated ×3 | clip in logit space, wrong focal weight form, `/n_pos` reduction |
| `cb_ntr` | deviated | λ present once instead of twice, the whole mechanism |
| `class_balanced_weights` | formula right (Cui Eq 6) | effect inverted at our label counts |
| reducers | inconsistent | different effective optimisers under `grad_clip=1.0` |

ASL clip bounds `log(1-p)` below at `log(0.05)`; shifting the logit leaves it unbounded, so hard-negative loss grows: at `z=+3, γ⁻=4` the reference is 1.545 and the old code 2.447 (1.58x), an anti-ASL

ASL reduction `sum/n_pos` is not positives-only, it equals element-mean / ρ; at ρ=0.21% that is 480x the control, so that arm hit `grad_clip=1.0` every step while the control never did. All losses now use element-mean

NTR λ: reference `q = σ(z−v)` positive, `q = σ(λ(z−v))` negative, negative term divided by λ, so λ twice; only the outer `1/λ` is uniform down-scaling. Old code 0.476x the reference; gradient at `z=−7, λ=2` is BCE 9.0e-4, old 4.5e-4, correct 9.1e-7

NTR bias: BALMS's `v` is fixed from the prior (`v_i = −κ·logit(p_i)`, κ=0.05), not learned; it pushes rare classes down by ~0.35 logits for a 0.1% tag, and the tail gain is λ's. The learned-bias claim in our research notes was wrong

Wrong citation: NTR is DBL arXiv 2007.09654 Eq 7/9 (restated BALMS Eq 6/8); 1805.00909 is an RL/control tutorial

Double bias: a learned `class_bias` plus the classifier's own bias already at `logit(prior)`, both at `lr_head`, suppressed tails twice; the learned one was removed

After the fixes `cb_ntr` matches the reference to 0.00e+00 relative error and `asl` to the last digit (0.379707153161), and all reducers return scalars. A later extended test puts the shipped CB-NTR within 0.1% of BALMS Eq 6 rather than exact: Eq6 1.162269, corrected 1.162409, shipped 1.163065, ratio 1.001, immaterial and not a reason to distrust h9. `asl(g1)` is 0.3647 against `asl(g4)` 0.1424 on one fixed tensor, so the function responds to gamma. `asl_ntr` at λ=1 differs from `asl` by exactly 0.0

### No published loss beats ASL in this regime

A search over 2023-2026, 20+ papers, asked directly. No published loss is known to beat ASL on 10k+ labels at a fixed global threshold: every paper at our label count reports mAP or P@k (Open Images 9600 labels: ASL 86.3 micro / 92.8 macro mAP; CSL 87.34 mAP), and every paper reporting a fixed-threshold F1 does it at 20-80 labels. The intersection is empty, and our +51% at a fixed 0.40 is one of the few fixed-threshold macro-F1 measurements in this regime at all

| rank | loss | evidence | verdict |
|---|---|---|---|
| 1 | CB-NTR composed with ASL (ASL positive branch x NTR λ on negatives) | BALMS 18211 labels: macro F1 13.94 -> 18.40; tail micro 8.58 -> 24.15 | our own composition, in no paper, needs its own screening arm |
| 2 | CSL / Partial-ASL top-K unannotated-negative dropping | Open Images 9600 labels, 87.34 mAP, but mAP | only mechanism aimed at true false negatives |
| 3 | SPLC / Hill loss | COCO 80 labels, simulated missing labels, mAP | cheapest, must be top-K, never a raw tau |
| 4 | LDAM-DRW | one unreviewed 30-label report where ASL collapsed to 0.0667 | grey zone: per-class calibration wearing a loss |

Rejected outright: logit adjustment / cRT / tau-norm / Seesaw / EQL / BCL / IFL / GCL (per-class weight or offset families, evaluated at per-class or tuned thresholds, the family that gave us 0.0097); XMC ranking losses; MoE and per-label adapters. "PLC / Pearson loss" and "ASL v2" do not appear to exist

Run order was the audit's, not the literature's: `bce_pos_weight` first, then corrected `cb_ntr`, then `asl` -- measured since, and the order was wrong for pos_weight

### Gradient checkpointing trades VRAM for recompute at a measured price per unwrapped block

An OFT-only run keeps a graph through every rotation even with every weight frozen, so all 24 blocks retain activations and `grad_checkpoint_blocks` was gated on `unfreeze_blocks > 0`, silently doing nothing on the arms we actually train. Gated on `n_unfrozen > 0 or n_oft > 0` it wraps the whole encoder, and a count N wraps only the last N so the rest keep their activations and pay no recompute

Synthetic eager backward-only bench, SigLIP2-L 384, batch 48, `scratch/sophie/bench_ckpt.py`: 24 of 24 checkpointed 3.36 GB and 1615.6 ms, last 3 of 24 20.35 GB and 397.7 ms. So one retained block is about 0.81 GB and one recomputed block about 58 ms, and the two ends are 4.1x apart in step time

In the real trainer at 384 batch 48: all 24 gives 54-55 img/s at 4.0 GB (`1fee2c4f`), last 3 gives 68-70 img/s at 22.1 GB (`8854c266`), 25% faster for 18 GB of headroom. No checkpointing at all is 23.5 GB and 3 img/s: not an OOM but a collapse, so around 23.5 GB the card is unusable rather than fatal and 22.1 GB is still clean. The boundary sits in the 1.4 GB between them and has not been located

Images per update is the same lever that fixed the tail, and accumulation reaches it without any checkpointing: batch 24 x2 at 384 measured 57-60 img/s at 12.6 GB, between the two checkpointed ends, so the 4 GB arm is only worth running when VRAM is the constraint rather than the schedule

## Traps

`epochs=8` silently caps a run at 1664 steps. Check `step N/total`

`--max-steps` exits without end-of-run validation, so `best_val.pt` misses the final state

`--ckpt-every-steps 0` writes a checkpoint every step instead of disabling them

Scraping stdout of `wdtag.validate` returns the 1000-image numbers labelled as 50k

Reading `best_val.pt` as the macro peak when it was selected on micro overstates or understates by tens of percent: h1's final is 30% below its own 1k peak

`wdtag.validate` without `--ckpt` scores the final checkpoint, so every arm number here is a final-checkpoint number unless stated

A truncated checkpoint made a whole run unresumable before the atomic write and newest-to-oldest walk existed

A `.PAUSE` file left at launch used to write a 300 MB step-0 checkpoint on every relaunch

The gamma kwarg bug ran `gamma_neg=4.0` on every ASL arm while configs said 2.0, and would have made a g1.0/g3.0/g4.0 sweep three identical arms, ~2.5 h

`pos_weight` from the config never reached the loss until `_loss_kwargs` was built explicitly

The loss test imported a second copy of the losses, so it agreed with itself and not with shipped code

`summary_table.py` mapped h3/h5 and h2/h11 from `Get-ChildItem` order; three numbers that night were wrong in the labelling rather than the measurement

Scoring `f63f6f54` at step 12000 overwrote its `validation.json`; kept as `validation_step012000.json` and step 30000 rescored from cached probabilities. Use `--out`

A GPU test started beside a live arm put about 4 GB on top of its 19 GB on a 24.5 GB card; the arm survived, and no GPU test runs while a real arm holds the card

`grad_checkpoint_blocks` did nothing on an OFT-only run because its gate required `unfreeze_blocks > 0`; the adapter path keeps the graph open in every block, so the flag was a no-op on exactly the arms we run

The prefetch ring `prefetch + 2` assumed the consumer holds one batch, so with `grad_accum` 3 the producer overwrote unconsumed slots and trained on partly corrupted batches; `hold` and `ring = prefetch + hold + 1` fix it

The cached branch printed `training done` without saving a final checkpoint, so every completed 10k arm ended up to 999 steps stale

A run with validation disabled never wrote `best_val.pt` and the end-of-run evaluation loaded it unconditionally, dying on the last line

Never look a run up by re-hashing its config: adding a field moves every sha8 that omits it

A doubled-backslash tower path in a first config attempt resolved anyway but hashed differently, so `feb17953` names nothing and `2ea85651` is the file to use

The 10k in-run prefix divides macro by up to 2.3 and flatters micro by 15%; the 384 run's in-run curve said 53% worse when the 50k said 6.6%

Equal step index is not equal data across arms with different batch sizes: step 2500 is 120000 images at batch 48 and 50000 at batch 20, and reading the large tower there reversed the true order

An annealed arm is not comparable to a constant-learning-rate one at the same schedule fraction; every large-tower comparison this morning was between those two shapes

A frozen-feature probe has misled three times in a row and must not be treated as a ceiling

The per-block speed ladder over-predicts by 25 img/s when an adapter sits inside the tower

Batch 20 against 48 is a confound at equal learning rate; the fair read needs a batch-24 base arm

A 2.07x macro gap cannot be closed by recalibration: the per-tag oracle's 31.6% headroom is unlearnable and a fitted global cut buys half a percent

The retired sweep driver and its four helpers are deleted; do not rebuild them. It died 50 minutes in when the registry cleared (h2 stopped at step 620, GPU idle), a command-line self-match made the launcher see itself and exit, and PowerShell kept `$m`/`$dir` from the previous iteration so arms with no log inherited the last arm's run directory and were skipped as already scored. A driver that is scoring between arms looks identical to nothing happening

## Open decisions

The main run `3da79e14` sits paused at step 15373 of 221739 with its original cosine: resuming costs nothing and cannot be extended, while a warm relaunch under the constant schedule with a taller horizon and a tail at about 15% loses 15000 steps of momentum and buys a schedule that can be lengthened. Leaning to the relaunch

`c8f2a635` is on the card: the large tower at batch 24 x2 accumulation, 600000 images, 12500 steps, about 3.9 h. It has to beat its own predecessor's 0.2331 and base's 0.2694; a landing near or above 0.28 puts the wide tower on the main run instead of base. The decision rule for the large arm was written before the number existed: at or above 0.33 is a clear tower win, 0.277 to 0.33 is ahead but worth questioning, 0.262 to 0.277 is a tie and 4.1 more hours buy nothing a second base pass would not, and below 0.262 the tower loses on 3.4x the parameters and 39 img/s against 118

`29_u0-...-b24`, `48de8bf9`, base at batch 24, is the control that says whether tail performance moves with images per update at all; 1.7 hours at 118 img/s, single variable against `b6e293f0`'s 0.2694. Its expected signature is the support 0-5 bucket falling from 0.1687 toward 0.10 and 5-20 from 0.2508 toward 0.18, which is macro 0.227, a 16% loss, far outside the band; if base holds near 0.16 and gives up only a little, batch noise is secondary and the wide tower's capacity per rare label is the cause

The `lr_unfreeze` bracket `c75d34d4` (1e-5) and `20142533` (1e-4) is parked, probably forever: it tunes a learning rate that governs 1.4% of the result, and `lr_oft` 5e-4 is what actually matters

`b05cbca4` is resumable from `ckpt_step003597.pt` and is the weaker of the two questions

384 is priced, not dead: 1.45x throughput for 6.6% macro if a long schedule ever has to fit in half the wall clock

A second pass is worth 26 hours only if the first still adds macro when it ends; the bar's own history says 97% of its final macro arrived in its first epoch and the remaining 94 bought 2.8%

The two-tower design with OFT inside both towers and the last block of each unfrozen is not cacheable, because a cache is only valid if the tower's output cannot change during the run. Cost: each tower at u1 costs 7.7 ms per image, so the pair lands near 65 img/s at batch 48 and 55-65 at batch 24-32, which is about 15.4 ms per image; 43.1M live parameters (23930880 rotations, 19178684 head); ~70 minutes for a 10k arm and 23-27 hours for a full 5.32M pass. `block_size` 128 carries the better accuracy evidence of the two, b=128 having beaten b=32 at step 5000 on micro, macro and supported macro, though confounded with exact Cayley and the fp32 head; block 64 would halve the adapter to 2.64M per tower and halve the resolution at which information crosses dimensions, 768/64 being twelve independent groups and a hard expressiveness ceiling. It would also be the first bracket of `lr_oft` 5e-4 and `lr_unfreeze` 3e-5 on a tower that is not DINOv3, and two new knobs at once is the shape of change that hid the gamma bug. Its open question is whether a fine-tuned DINOv3 tower leaves room for SigLIP2, which the probe cannot answer and the tie at -1.4% makes unlikely at concat fusion

Anything that revives the ensemble needs real fusion machinery, 13-20M parameters of per-tag gate, cross-attention or learned query set

If the 1-9 bin is not learnable from nine examples, the remaining spend is label-text or feature initialisation (DECAF/CLIP-Decoder), not loss shaping

A ViT-L full pass is 44.8 hours, and the frozen-feature probe staged to decide capacity became the wrong instrument once SigLIP2 became the base, because it compares a tower family we are abandoning. Precomputing a frozen tower's features is worth the 5.2 hours only when a full 5.32M pass makes the per-step 30% matter; for 10k arms in-loop and precomputed are the same number

An equal-image extension of a promising arm costs 2.1 to 2.9 hours of stepping, and the large tower's 30000-step gate was 4.1 hours; both are cheap next to a 26-hour pass
