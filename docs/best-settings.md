# BEST SETTINGS — read this before configuring anything

The decisions extracted from `docs/benchmark.md`, which holds the full measurement tables. When a number here disagrees with `benchmark.md`, `benchmark.md` wins and this file is stale.

Grades: M measured on this machine, I inferred from a measurement, H hypothesis — do not quote an H as a result.

## 1. Copy this for an arm run

```jsonc
{
  "batch_size": 48,                // M  512px, ViT-B/16, 12 blocks unfrozen -> 20.6 GB
  "compile_model": true,           // M  1.25x over eager
  "compiled_optimizer": true,      // M  mandatory on the OFT/optimizer-heavy path
  "grad_checkpoint_blocks": false, // M  no win on ViT-B/16 at 512, where the activations already fit; on SigLIP2-L at 384 batch 48 it is the only way past 23.5 GB, and a count N trades VRAM for speed (benchmark.md K)
  "block_size": 128,               // M  best of {32, 64, 128}
  "cayley_terms": 0,               // M  0 = exact solve, ~9% slower than Neumann-5, numerically exact
  "head_fp32": true,               // M  BF16 on the 12,476-way head was unstable
  "compile-mode": "default",       // M  exploratory; "max-autotune" only for long runs
  "select_metric": "macro",        // M  the argparse default is still micro -- always pass this
  "ckpt_every_steps": 1000,        // M  now also the code default (section 1b)
  "ckpt_every_epochs": 0,          // M  epoch-end saves are opt-in (section 1b)
  "loss_gamma": 2.0,               // M  ASL gamma_neg: 2.0 beats 4.0 by 12% micro at equal macro (section 3b)
  "backbone_dir": "I:\\Models\\siglip2-base-patch16-512", // M  SigLIP2 beats DINOv3 by 21% macro (section 0)
  "num_workers": 0                 // H  worth <=6% at most (section 5)
}
```

Since the 2026-10-22 refactor `TrainConfig`'s defaults match this block, so a config only states deviations. `select_metric` is the exception: its default is `micro` and must be passed explicitly for any number you intend to quote.

## 0. The tower is SigLIP2, measured 2026-10-22

One recipe, one field changed, both towers at 50k and 0.40. The tower swap is the largest single-factor win in the project, and it costs nothing in throughput.

| tower | macro F1 | macro P | macro R | micro F1 | img/s | VRAM |
|---|---:|---:|---:|---:|---:|---:|
| SigLIP2-B/16-512 (`runs/66e841ba`) | 0.0755 | 0.1010 | 0.0730 | 0.4375 | 138 | 15.8 GB |
| DINOv3-B/16 (`runs/ce692296`) | 0.0624 | 0.0924 | 0.0591 | 0.4156 | 130 | 16.8 GB |

SigLIP2 wins on both metrics, runs 6% faster and uses 1 GB less, because it carries no cls token and no registers and its pooling vector comes from a small attention head. The kind is detected from the checkpoint's own `config.json`, so pointing `backbone_dir` at a siglip directory is the whole change; there is no `backbone_kind` field and no hash churn.

Consequence for the capacity question: the next size up is `siglip2-large-patch16-512`, 24 blocks and 1024 wide, not DINOv3-L. Same family, so the recipe stays constant and only the size changes.

Untested on the new tower, and both are open arms: whether its MAP attention pooler discards the small-object detail tags need, tested by `head_pool: mean`; and whether freeze depth 1 is still optimal for features that are semantic rather than structural.

## 3b. `loss_gamma` 2.0, measured 2026-10-22

The first and only bracket of the ASL gamma dial. Every arm before it ran 4.0 because the winning arm did, which is a reason to record rather than a reason to believe.

| `loss_gamma` | macro F1 | macro P | macro R | micro F1 |
|---:|---:|---:|---:|---:|
| 4.0 (`runs/b9ed94fb`) | 0.0632 | 0.0663 | 0.0791 | 0.3713 |
| 2.0 (`runs/ce692296`) | 0.0624 | 0.0924 | 0.0591 | 0.4156 |

Macro -1.3% and inside the 3% band; micro +11.9% and far outside it. Halving gamma_neg trades recall for precision at almost exactly break-even on macro F1, so the dial was never buying macro, only moving the operating point. Both 10k arms, u1 with OFT block 128 and the cls head, one seed each.

The ledger's champion is now `u1-C-L-asl-g2.0`: 0.0624 macro and 0.4156 micro, the best pair of numbers any arm has produced. 1.0 is untested and is the next loss arm; 3.0 is no longer worth a slot.

## 1b. Checkpoint cadence cost ~8% of every arm

Measured on a completed 5000-step arm (`runs/1e312196`, 41.6 min): 49 checkpoints x 638.8 MB = 30.6 GB, because the 200-step default interleaved with an unconditional epoch save (208 steps/epoch at this dataset size) — a 639 MB write roughly every 100 steps. Each write costs 5-15 s: 16 steps across a save ran at 0.76 s/step against a 0.44 s/step baseline, 32 steps at 1.02 s/step, ~300 s of overhead over the whole run.

Fix (now default): `--ckpt-every-steps 1000 --ckpt-every-epochs 0` -> 5 files / 3.2 GB / seconds. Resume is unaffected (highest `ckpt_step*.pt`); epoch-aligned points remain available via `--ckpt-every-epochs N`.

## 2. Throughput (ViT-B/16 @512, full training step)

| setting | ms/step | img/s | peak VRAM | grade |
|---|---:|---:|---:|---|
| eager | 517.4 | 92.8 | 20.97 GB | M |
| compile(default) | 412.9 | 116.3 | 20.55 GB | M |
| compile(max-autotune) | — | 150-170 | — | M, older/smaller config — re-measure |
| OFT block 32 | 392 | 122.4 | 17.0 GB | M |
| OFT block 64 | 394 | 121.8 | 17.2 GB | M |
| OFT block 128 | 388 | 123.7 | 17.3 GB | M |
| Cayley Neumann 5 | 388 | 123.7 | 17.2 GB | M |
| Cayley Neumann 15 | 423 | 113.5 | 17.8 GB | M |
| Cayley exact solve | 426 | 112.7 | 17.2 GB | M |

compile is worth 1.25x, not 1.7x — the 1.7x came from a stale config (batch 32, block 32, `cls_mean`, 8 epochs) that we never train
The block sweep above is Neumann-5, not exact solve; its block-128 row (388 ms) duplicates the Neumann-5 row. Exact solve is 426 ms / 112.7 img/s
Throughput is governed by `unfreeze_blocks`, not the compiler. The fast era was real: `runs/_wide_head.log` (run `b2a924f1`) has 249 real steps at median 160 img/s (p05 147, max 169, 16.7 GB) and its config has no `unfreeze_blocks` key — the backbone was frozen. Endpoints on the reference config: 0 blocks = 160 img/s / 16.7 GB, 12 blocks = 109-116 img/s / 20.6 GB, ~4.25 img/s per unfrozen block (N=5 ≈ 139, N=6 ≈ 135). Nothing was lost to autotune; ~30% of throughput was spent on purpose
An earlier version of this file claimed 160 img/s was never measured. That was wrong — 160 is the median of 249 samples, measured with the backbone frozen, and does not apply to 12 unfrozen blocks
max-autotune buys ~18% of the forward pass and ~3% of a step. At 12 blocks, batch 48: eager fwd 207.1 -> compile 169.3 ms, but fwd+bwd is 393.1 ms and the step 419.5 ms (max-autotune 401.5 ms = 119.6 img/s vs default 413.6 = 116.1). `opt.step()` is 0.4 ms; the model compiles to 1 graph, 0 breaks — the ceiling is the backward pass through unfrozen blocks
Plan a batch to 22.5 GB, not 24.5: 23.5 GB of allocation does not OOM, it collapses to 3 img/s, and 22.1 GB still runs at full rate. The cliff edge is measured only as lying between those two, so 22.5 is a rule of thumb with a known direction of error (benchmark.md K)
Highest credible step: 123.7 img/s at 12 blocks (Neumann-5) / 116.3 (exact, what we train); 160 at 0 blocks

## 3. Validation (forward only) — batch differs per model, never cross them

| model | batch | res | img/s | peak VRAM | 50k wall |
|---|---:|---:|---:|---:|---:|
| ViT-B/16 ours | 256 | 512 | 246 | 7.5 GB | ~3.4 min |
| ViT-B/16 ours | 384 | 512 | 258 | 11.0 GB | ~3.2 min |
| convnextv2_huge bar | 64 | 512 | — | — | — |
| ours while another arm trains | 24-96 | 512 | 145 | — | 5.8 min |

Convention: ours at batch 256, the bar at 64. A 50k pass at batch 24 took 346 s while another arm was training, so overlap is cheap. `--sizes 50000` for the fast single-subset read; the default scores 1000/5000/10000/50000 (66k images).

## 4. Load-bearing config values

| knob | value | why |
|---|---|---|
| `lr_oft` / `lr_head` / `lr_unfreeze` | 5e-4 / 1.2e-4 / 3e-5 | M — reference triple; SinkSGD_ADV is normalised so LR scales as sqrt(batch) |
| `momentum` / `weight_decay` | 0.95 / 0 | M — weight decay did not help multi-label |
| `block_size` | 128 | M — auto-adjusted per matrix by `_adjust_block_size`, mirrors OneTrainer |
| OFT params | `d*(b-1)/2` | I — from the formula; 73 OFT modules in ViT-B |
| `head_pool` | `cls_mean_max_std` | M — 3072 -> 38.34M head; other pools are arms h2/h3 |
| `epochs` | must exceed `--max-steps/batch` | M — else the run stops at an epoch boundary and looks complete (cost us a whole arm) |
| checkpoint | 639 MB each | M — with the cadence fixed, 5 per 5k-step arm |
| `optimizer_normed_momentum` / `optimizer_snr_cond` | on, and only together with `optimizer_oft_only: true` | M — measured 2026-10-24 at base-384 in four configurations. On all three trainable groups it is a loss: 0.0085 macro at step 1000 at 5e-4 and 0.0067 at matched 2.61e-4, against plain 0.0559. On the adapters only it is a small win: 0.1545 macro and 0.4338 micro on the 50k at matched displacement, 0.1555 and 0.4358 when the adapters run 1.9x hot, against plain 0.1498 and 0.4369. The head is the poison, every arm that switched it on sat at 0.008 or below. The full matrix is in `ARM-RESULTS.md`, section Optimizer switches at the three trainable groups |
| `optimizer_spectral_normalization` | off | M — measured 2026-10-24 three ways at base-384, step 1000: 0.0004 untagged, 0.0007 with the `_is_oft` tag, 0.0078 with the tag plus per-group compensating rates, against plain 0.0559. Tagged it still throttles the adapters 44x, the head 34x and the biases not at all, so no single learning rate can drive it |

## 5. `num_workers: 0` — grade H, do not over-claim

Live throughput is 109 img/s (440 ms/step) against 116.3 for the same config measured by `bench_ref.py` (412.9 ms). That 27 ms (6.2%) gap is explained by data loading + H2D, which `bench_ref.py` excludes by timing `step()` on preloaded tensors: ~10 ms load + ~4 ms H2D at batch 32 scales to ~21 ms at batch 48.

Retraction: an earlier version called `num_workers: 0` a bottleneck "costing ~1/3 of the GPU". That was wrong — it came from `nvidia-smi` spot samples (95/64/100/64), which `benchmark.md` warns are misleading. The measured data cost accounts for the whole gap, and the live distribution is 35 distinct values in 96-109 with mode 109 (196/330 samples): jittery steady state, not a throttle. Worth testing (≤6%), untested, and do not change it mid-sweep — per-worker RNG can change per-image randomness.

## 6. NOT wins (stop re-testing)

Per-class thresholds / calibration — per-tag-oracle macro ceiling is 0.3879 vs the bar's 0.5579 at a fixed 0.40. The gap is in the scores, not the cut. (M)
More passes on the long run — the bar hit 97% of final macro in epoch 1. (M)
Gradient checkpointing for ViT-B — no VRAM win worth the throughput. (M)
Neumann > 5 Cayley terms — slower, no accuracy gain. (M)
The adv_optm gold-star bundle for SinkSGD_adv — orthogonalGradient iterative, orthogonalSinkhorn, normedMomentum, snrCond, spectralNormalization, geometricWd, cautiousWd, statePrecision fp32, first written in `speedy-sdxl-vae-trainer/scratch/night3/run_grid.py:65` and mirrored in `dreamsim-dinov3` as the FULL Sink-V stack. Three matched ties in speedy and the verdict in its own journal is "no gain, THIRD confirmation"; the only per-switch ablations anywhere are dreamsim's and both regressions, normed_momentum -3.92pp test CI [-5.31,-2.49] and orthogonal_sinkhorn -3.61pp CI [-5.01,-2.16]. Our own arm 45 on base-384 with normed_momentum alone read macro 0.0079 at step 1000 against arm 1's 0.0558, a sevenfold collapse, which is the same finding on a different architecture. Every switch in that bundle changes the effective step size, so enabling one does not refine the learning-rate calibration, it voids it. (M, three sources)
bitsandbytes 4-bit — unusable on cu132 (ships only to `cuda130`). (M)
Head capacity as the differentiator — ours is 38.34M vs the bar's 35.15M and we lose macro with the bigger head. (M)

## 6b. The optimizer-switch matrix lives in ARM-RESULTS.md

`ARM-RESULTS.md` section Optimizer switches at the three trainable groups is the canonical table: switch crossed with the three groups that actually train, read at step 1000 and at 50k, with the arm number on every row. It is the source of the two rows above and it is where a new switch should be added rather than re-derived.

Two traps that table does not show. The first is the missing `_is_oft` tag: `adv_optm` asks PEFT users to label their parameters, `scale_update` branches on the attribute, and untagged it divides a (k, C(b,2)) factor by its own spectral norm, which is a 66x throttle instead of the tagged 44x. The tag now sits beside the parameter in `wdtag/oft.py`. It cannot be applied to the head, since a (12476, 768) tensor handed to the OFT path raises a shape mismatch. The second is that `normed_momentum` and `snr_cond` have no equivalent hook at all, consulting only `is_vector`, so scoping by parameter group is the only lever that exists for them.

## 7. Traps that cost real time

1. `epochs` smaller than `--max-steps` silently truncates the run; check `step N/TOTAL`.
2. `wdtag.validate` prints `at <thr>:` once per nested size — scraping stdout returns the 1000-image numbers as 50k. Read `validation.json` -> `subsets["50000"]`.
3. `validate` without `--ckpt` uses the highest-numbered checkpoint (the most overfitted one).
4. `--select-metric` defaults to micro; `macro` is what the objective needs.
5. Adding a config field changes the sha8 of every unpinned config — never look a run up by re-hashing; read `run dir` from the log or `_sha8` in `<run>/config.json`.
6. `head_fp32: true` is load-bearing: BF16 on the 12476-way head was NaN-prone.
7. `torch.compile` "ready in 1.2s" means the inductor disk cache hit; never compare a cold run to a warm one.

## 8. Configs on disk

| what | path |
|---|---|
| long run (paused at 86241) | `runs/4a13a120/config.json` |
| arm configs (head/loss sweep) | `configs/heads/*.json` |
| phase-2/3 configs | `configs/heads/phase2/`, `configs/heads/phase3/` |
| older probes | `configs/probes/` |

## 9. Not a win: augmentation, in training or at inference

No augmentation is applied in training and none in scoring, and that is deliberate rather than an omission. The label space is fact-like and dense: 12476 tags that include colour (greyscale, monochrome, sepia, the sixteen named backgrounds), orientation and composition (from_behind, from_side, looking_back, looking_to_the_side, on_back, arms_behind_back), text and marks (english_text, signature, copyright_name, logo, watermark), and character identity where a mirror is not the same character. A colour jitter breaks every colour tag, a crop breaks composition tags, and a mirror breaks orientation tags and mirrored text. The usual assumption that an augmentation is label preserving does not hold here, and a noisy label is worse than a small dataset.

Horizontal flip averaging at inference looked like a free exception because it touches no labels, but the same fact kills it: the model's legitimate prediction for a mirrored image differs for exactly those tags, so averaging blurs the orientation and text tags rather than the bulk of the distribution, and those sit in the macro average this project is scored on. Recorded as not a win on the user's reasoning, 2026-10-23, before spending the 25 GPU minutes that would have measured it.



## 6. How to choose between two arms

Adopted 2026-10-25 after an arm comparison in which a lost calibration state nearly produced the wrong decision. The rule is cheap to apply and it exists so a number that is not an arm score cannot be mistaken for one.

Primary endpoint: paired macro F1 at 0.40 on the 50k. Accept an arm only if its macro gain exceeds the 0.0093 interval and the lower bootstrap bound is above zero. Micro is not the objective; it is a guardrail with a floor of minus 0.0016. MCC must not fall, because it is the balanced check against both forms of gaming. Macro precision must not fall materially, and predicted tags per image at 0.40 must stay under about 1.25 times the dataset's 32.3, which is what stops an arm buying macro by predicting tags indiscriminately. No regression on the tail: the top thousand tags carry 80.9 percent of all positive cells, so micro can hide a tail collapse completely. Carry a threshold-free ranking number, mAP or AUPRC, so a gain cannot be a threshold artefact.

Selection uses the fixed 0.40 protocol only. Best-threshold numbers and calibration-fitted numbers are diagnostics and never select an arm.

Two hard rules earned the hard way. Any state that is not in the checkpoint does not exist: if a run applies a calibration bias or any other auxiliary tensor, that tensor goes into the checkpoint and comes back on resume, and the validator must be able to apply it. And no calibration fitted on validation images may ever decide an arm, because the validation set is what the project is scored on.

To claim the bar is beaten, both lower bounds must exceed 0.5579 macro and 0.6886 micro, not the point estimates.
