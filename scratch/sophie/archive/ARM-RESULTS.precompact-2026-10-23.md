# ARM RESULTS — every experiment, one place

Central record of arm experiments, and the only place results are written. `REPORT.md` and `wdtag/report.py` are deprecated: a generated second report drifted out of date and split the record in two, so this file is now the single destination for findings, comparisons, probes that never became runs, and anything a future reader needs to avoid re-running an arm.

## The 10k head benchmark (defined 2026-10-22)

Head/loss variants in ~45 min per arm instead of ~9 h on the full corpus.

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

Caveat that bit us once: `epochs=8` stops the run at 1664 steps, silently short of `--max-steps 5000`. Check `step N/total`: the total must exceed the intended cap. The first h1 run was invalid for this and was re-run.

Caveat when reading scores: `wdtag.validate` prints an `at 0.40:` block once per nested size (1000, 5000, 10000, 50000), so scraping stdout gives the 1000-image numbers as if they were 50k. Read `validation.json` -> `subsets["50000"]` and index `thresholds` by hand; no collector script exists.

Trainer behaviours confirmed by smoke test (`scratch/sophie/smoke/*.json`, batch 2, 2 steps, EXIT=0):

`--max-steps` exits without the end-of-run validation, so `best_val.pt` comes only from epoch-end validations; the authoritative 50k number is the separate `wdtag.validate` call

`--ckpt-every-steps 0` does not disable checkpointing: it writes one every step. Use a large number to thin them (the disk estimate below assumes they are not thinned)

Both loss paths wire up at real scale: `head 38.34M, unfrozen 85.06M in 12 block(s), 73 OFT modules` (head back to 38.34M after `class_bias` was removed), `pos_weight` computed with `n_images 10000`

## Arms

One change from the control per arm unless stated.

| arm | change | head params | trainable | status |
|---|---|---|---|---|
| h1_control_cls_mean_max_std | — (control) | 38.34M | 123.46M | run 1 invalid (1664 steps); re-run at 5k |
| h2_cls_mean | `head_pool: cls_mean` | 19.16M | — | queued |
| h3_cls | `head_pool: cls` | 9.60M | 94.71M | queued |
| h4_mlp1024 | `head_hidden: 1024` | 15.94M | 101.05M | queued |
| h5_mlp2048 | `head_hidden: 2048` | 31.86M | 116.98M | queued |
| h6_control_seed2025 | seed 2025 (noise floor) | 38.34M | — | queued |
| h7_mlp512 | `head_hidden: 512` | ~8M | — | queued |
| h8_mlp4096 | `head_hidden: 4096` | ~63M | — | queued |
| h9_cb_ntr | loss CB-NTR, β=0.9, λ=2, κ=0 (BALMS-faithful) | 38.34M | — | phase 2 |
| h10_asl | loss ASL (γ⁻=4, γ⁺=0, clip 0.05, detached focal) | 38.34M | — | phase 2 |
| h11_pos_weight | loss `pos_weight=((N−n)/n)^0.5` | 38.34M | — | phase 2, first |

Macro and micro F1 at threshold 0.40 from the 50k pass. In-run 1k numbers are trajectory only -- 1k is reliable for micro, not macro.

## Results (filled as arms complete)

`best_val` is the micro-selected checkpoint, which on h1 coincided with the macro peak (step 1040). "1k peak" is the best macro@0.40 in the arm's own 1000-image validations.

| arm | 1k peak @.40 (step) | 1k last @.40 | 50k macro@.40 (final) | 50k micro@.40 (final) | 50k macro@.40 (best_val) |
|---|---:|---:|---:|---:|---:|
| h1_control_cls_mean_max_std | 0.0298 (1040) | 0.0208 | 0.0247 | 0.4372 | 0.0272 |

h1 read-out: the final checkpoint is 30% below its own 1k peak; scoring `best_val` recovers 10% of macro on the real 50k with micro unchanged, so read the comparison on the `best_val` column and treat the final column as the trained-5x-past-optimum view. On 50k h1 shows 12454 supported labels and macro P 0.0781 / R 0.0199 at best_val: it fires almost nothing at 0.40, the tail-recall problem in its purest form.

50k at batch 24 took 346 s (145 img/s) while another arm trained, so the two scoring passes per arm add ~12 min to ~44 min of training.

## Reference points (same 50k cache, threshold 0.40)

| model | macro F1 | micro F1 |
|---|---|---|
| our full run 4a13a120 @086241 (77% of a pass) | 0.2815 | 0.6133 |
| bar `convnextv2_huge` | 0.5579 | 0.6886 |

The 10k arms are not comparable to these absolute numbers (1/50th of the validation distribution, far less training), only to each other.

## Loss implementation audit (2026-10-22)

The first `wdtag/losses.py` did not match the papers it cited. Every defect below was measured against the reference, fixed, re-verified. Full audit `scratch/sophie/loss-audit.md`, reference code `scratch/sophie/losses_corrected.py`.

| function | verdict | defect |
|---|---|---|
| `bce` | matches | it is the control, and it defines the loss scale |
| `asl` | deviated ×3 | clip applied in logit space, not probability space; wrong focal weight form; `/n_pos` reduction |
| `cb_ntr` | deviated | λ present only once instead of twice -- the whole mechanism |
| `class_balanced_weights` | formula right (Cui Eq 6) | effect inverted at our label counts |
| reducers | inconsistent | different effective optimisers under `grad_clip=1.0` |

ASL clip. It bounds `log(1-p)` below at `log(0.05)`; shifting the logit leaves it unbounded, so hard-negative loss grows: at `z=+3, γ⁻=4` reference 1.545, old code 2.447 (1.58x) -- an anti-ASL

ASL reduction. `sum/n_pos` is not positives-only, it equals element-mean / ρ; at ρ=0.21% that is 480x the control, so that arm hit `grad_clip=1.0` every step while the control never did. All losses now use element-mean

NTR λ. Reference: `q = σ(z−v)` positive, `q = σ(λ(z−v))` negative, negative term divided by λ -- λ twice. Only the outer `1/λ` is uniform down-scaling, not tolerance. Old code 0.476x the reference; gradient at `z=−7, λ=2` is BCE 9.0e-4, old 4.5e-4, correct 9.1e-7

NTR bias. BALMS's `v` is fixed from the prior (`v_i = −κ·logit(p_i)`, κ=0.05), not learned; it pushes rare classes down by ~0.35 logits for a 0.1% tag, and the tail gain is λ's. The learned-bias claim in our research notes was wrong; corrected in `docs/REPORTS/heads-tail-losses.md`, `heads-evidence-grades.md`, `heads-and-losses.md`

Wrong citation. NTR is DBL arXiv 2007.09654 Eq 7/9 (restated BALMS Eq 6/8); 1805.00909 is an RL/control tutorial

Double bias. A learned `class_bias` plus the classifier's own bias already at `logit(prior)`, both at `lr_head` -- tails suppressed twice. The learned one was removed

CB inversion. Under mean-1 normalisation the near-empty labels get the largest weights on their negatives, so the positive share falls from 0.259% (BCE) to 0.019% at β=0.999. Now β=0.9 (what BALMS used) with positive-mass normalisation

Self-inflicted #4. Operator precedence in my ASL rewrite bound `.mean()` to the focal term only, returning a tensor instead of a scalar. Caught by test, fixed

After the fixes `cb_ntr` matches the reference to 0.00e+00 relative error, `asl` to the last digit (0.379707153161), all reducers return scalars.

Recommended run order (the audit's, not the literature's): `bce_pos_weight` first -- the only variant lifting tail positives without inflating negatives of the 1686 labels that have zero positives in the 10k cache -- then corrected `cb_ntr`, then `asl`.

## How to resume this work (read this first if you are picking it up cold)

Running an arm is one command, launched as a harness background job and watched by hand:

```powershell
$env:PYTHONPATH="C:\Users\kerok\apps\wd-tagger-trainer"
C:\Python314\python.exe -m wdtag.train --config configs\heads\phase3\01_u12-C-L-asl-g4.0.json --select-metric macro
```

5200 steps on the 10k cache is ~40-45 min at 99-109 img/s. The first 40 lines are the verification: the printed run dir, `config sha8`, the checkpoint metric, the parameter grouping, and the loss kwargs it will actually use (`loss asl: gamma_pos 0.0, gamma_neg 4.0, clip 0.05, detach_focal`). Artifacts land in `runs/<sha8>/`: `config.json`, `train_log.jsonl`, `debugpics/`, `ckpt_step*.pt`, `best_val.pt`, then `validation.json` at the end.

The sweep driver that used to run this (`run_sweep.ps1`, `checkin.py`, `score_bestval.py`, `make_phase_configs.py`, `sweep_driver.lock`) is retired and deleted; do not rebuild it. It was idempotent, skipped arms whose run dir had a `validation.json`, resumed from checkpoints and ran both scoring passes, and it hid three defects exactly because nobody read its decisions. Its failure modes are worth remembering because a rewrite reintroduces them: the first queue, a harness background job, died 50 minutes in when the registry cleared (h2 stopped at step 620, GPU idle); a command-line self-match made the launcher see itself and exit, fixed with a pid file; PowerShell keeps `$m`/`$dir` from the previous iteration when `Select-String` yields nothing, so arms with no log inherited the last arm's run dir and were silently skipped as already scored.

Score the macro-peak checkpoint before pruning: `wdtag.validate --run <dir> --ckpt best_val.pt --out <dir>/validation_bestval.json`. `best_val.pt` is micro-selected unless the arm ran with `--select-metric macro`, but on h1 micro and macro peaked at the same step (1040), so it doubles as the macro peak; verify per arm from the in-run curve. Do not prune before this pass -- the ~step-1000 checkpoints are the ones worth scoring.

Both traps above have burned us once already; check them before trusting any arm.

Then, in order:

1. Fill the arm table; `h6_control_seed2025` defines the noise band -- an arm must beat the control by more than that band
2. Extend only inconclusive arms to 10000 steps: copy the arm's config by hand to `configs/heads/phase2/ext_<arm>.json`, set `epochs: 50` and `init_from: <that arm's run dir>` so it warm-starts from `best_val.pt` instead of replaying 5000 steps, then `--max-steps 10000 --select-metric macro`
3. Sweep the last N blocks with the winning head: one hand-written config per depth, as `configs/heads/phase3/05-07` already are, launched one at a time with `--max-steps 5000 --select-metric macro`. N=0 is the control (OFT only); the arms run `unfreeze_blocks: 12` at `lr_unfreeze 3e-5`, so this tests fewer unfrozen blocks, not more
4. Record each step here, one recommendation at the end

Do not resume `runs/4a13a120` (paused at 86241 with `.PAUSE`) -- the long run, not an arm.

Checkpoint selection: `--select-metric` defaults to micro but our objective is macro, so every arm's `best_val.pt` is micro-chosen, fair across arms but not macro-optimal. Use `--select-metric macro` for extension and unfreeze; choices `micro` / `macro` / `supported_macro`, and `macro_f1` is the run's own reference threshold, not exactly `macro@0.40`.

`wdtag.validate --run <dir>` without `--ckpt` uses the highest-numbered `ckpt_step*.pt`, not `best_val.pt`, so the sweep reports final-checkpoint scores uniformly. Score `best_val.pt` with `--ckpt best_val.pt --out <dir>/validation_bestval.json`.

Disk: a checkpoint carries optimizer state, so `ckpt_step*.pt` runs 231 MB with the linear `cls` head and 639 MB with `cls_mean_max_std`. At the 1000-step cadence a 5200-step arm is ~5 of them plus `best_val.pt`, so ~2-3 GB per arm. The every-epoch save is opt-in (`--ckpt-every-epochs`) because it cost ~8% wall and 30.6 GB per arm. After scoring, prune:
`Get-ChildItem runs/<sha8>/ckpt_step*.pt | Sort-Object Name | Select-Object -SkipLast 1 | Remove-Item` (keep `best_val.pt` and the last checkpoint; never touch `runs/4a13a120`).

Run dir identity: `runs/<sha8>` where sha8 hashes the whole config including defaults, so a new config field (e.g. `loss`/`cb_beta`/`ntr_kappa`) moves the sha8 of every config that does not pin it. h1 runs in `470e6fa9` while its config now hashes to `54b2ce30`; its `config.json` carries `_sha8: 470e6fa9` and everything downstream resolves from that or the log -- but never look a run up by re-hashing its config: read `run dir:` from the log or `_sha8` from `<run>/config.json`. All 11 arm configs hash distinctly, so no two arms can share a run.

## Findings that outlive any single arm

On this 10k benchmark macro@0.40 peaks early then DECLINES. h1's mid-training curve (1000-image val, `--max-steps 5000`):

  | step | 208 | 416 | 624 | 832 | 1040 | 1248 | 1456 | 1664 | 1872 | 2080 | 2288 | 2496 |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|
  | macro@0.40 | .0122 | .0234 | .0263 | .0256 | .0298 | .0278 | .0272 | .0261 | .0252 | .0235 | .0248 | .0236 |
  | micro F1 | .3263 | .3905 | .4155 | .4092 | .4313 | .4234 | .4243 | .4256 | .4238 | .4220 | .4291 | .4262 |

Macro falls 21% relative from its peak by step 2080 while micro stays flat -- the tail gets squeezed as the head overfits 10k images. Step-to-step wiggle is ±0.0015, so this is real. Consequences:

1. A 5000-step arm is ~5x past its macro optimum. Scoring the final checkpoint (what `validate` does by default) understates every arm and may rank arms by overfitting resistance rather than head quality
2. Score a mid-training checkpoint too: arms save `ckpt_step*.pt` every 1000 steps by default, so a 5200-step arm has 1000/2000/3000/4000/5000; tighten with `--ckpt-every-steps` or `--ckpt-every-epochs` when the macro curve matters, and compare final vs ~1k before any head conclusion
3. Extending inconclusive arms to 10000 steps is suspect here: more steps lower macro. Extension only makes sense measured at the macro-optimal checkpoint, or to study overfitting rather than head quality
4. This is a property of the 10k subset, not the 5.3M corpus: at 10k images a 38M-parameter head memorises fast. Do not transfer this curve to the full-corpus run

The macro deficit is tail-concentrated. Per-tag F1 vs training frequency on the 50k cache: gap to the bar 0.322 unseen, 0.319 for 10-99, 0.190 for 100-999, 0.126 for 1k-9.9k, 0.055 for >=10k, so ~97% of the total macro gap is tags with under 1000 training examples. See `docs/heads-and-losses.md`.

Our per-tag-oracle threshold ceiling is 0.3879 (bar 0.5579 at a fixed 0.40), so no calibration/threshold/rescaling trick closes the gap; it lives in the scores.

Head capacity is not the differentiator: ours (38.34M) is larger than the bar's (35.15M), whose head is just avgpool -> LayerNorm -> dropout -> Linear(2816, 12476).

## Speed — what actually governs it (measured 2026-10-22)

The lost 138 img/s was not the compiler: it was spent training the backbone.

The fast era is real and on disk: `runs/_wide_head.log` (run `b2a924f1`) holds 249 real training steps at median 160 img/s (p05 147, min 142, max 169) at 16.7 GB, with `max-autotune` and no `unfreeze_blocks` key -> backbone frozen (87.0M).

Throughput vs unfreeze_blocks (reference head, batch 48, 512px, compile(default), exact Cayley, `profile_step.py`, full step including optimizer):

| blocks | fwd ms | fwd+bwd ms | full ms | img/s |
|---:|---:|---:|---:|---:|
| 0 | 162.4 | 353.0 | 357.0 | 134.4 |
| 1 | 162.4 | 358.6 | 366.9 | 130.8 |
| 2 | 162.3 | 361.3 | 370.0 | 129.7 |
| 3 | 162.4 | 363.7 | 373.3 | 128.6 |
| 4 | 162.3 | 366.8 | 376.7 | 127.4 |
| 6 | 162.4 | 372.4 | 386.8 | 124.1 |
| 12 | 172.4 | 400.9 | 420.2 | 114.2 |

~1.7 img/s per unfrozen block; the whole climb from 12 -> 0 is worth 18%.

max-autotune looked like +19% then and +3% now because it is a forward win: eager fwd 207.1 -> compile(default) 169.3 = -18%. A forward-dominated step (frozen backbone) carries that to the total, reproducing 160 from 134.4 (x1.19); with 12 blocks unfrozen backward is 224 ms of a 420 ms step and the win dilutes to +3% (401.5 vs 413.6 ms).

Not the blocker: `opt.step()` is 0.4 ms, the model compiles to 1 graph, 0 graph breaks, and validation is not in these numbers.

Long run: 110870 steps at 114.2 img/s = 13.5 h, at 134.4 = 11.0 h, at 160 = 9.2 h. Freezing blocks is the only large speed lever left and it trades directly against backbone training.

### Check-in 02:45 — first ASL signal, and the unfreeze plan on paper

`h10_asl` at 3400/5200, in-run 1k macro@0.40 peak 0.0312 at step 1248, highest of any arm so far, against control h1's 0.0298 at step 1040 (+4.7%). Both are short of their 5000-step end and the 50k reading decides it: a lead, not a result. The peak step moved later (1248 vs 1040), what a harder tail loss should do.

The unfreeze sweep is next because at 5000 steps N=0 vs N=12 costs only ~5 min per arm: it is about the long run (110870 steps, 13.5 h at 114.2 img/s vs 9.2 h at 160). If N=6 keeps most of N=12's 50k macro it buys ~1.5 h per pass for free.

The hypothesis to falsify: backbone training is where the macro comes from, so freezing costs accuracy roughly linearly in N. Competing: OFT on the frozen backbone already does the work and N>0 only adds overfitting on 10k images, in which case N=0 or N=2 wins here and the long run should freeze more.

N=12 is measured (h1: 50k macro 0.0272 best_val / 0.0247 final), so only N=0/2/4/6 remain.

### RESULT — h10_asl beats the control on macro by +51% (verified from validation.json)

First arm to move macro materially. 50k nested cache, threshold 0.40, run `7ee12110` vs control `470e6fa9`:

| arm | macro F1 | macro P | macro R | micro F1 | micro P | micro R |
|---|---:|---:|---:|---:|---:|---:|
| h1 control (BCE) final | 0.0247 | 0.0763 | 0.0179 | 0.4372 | 0.6396 | 0.3321 |
| h10 ASL final | 0.0410 | 0.0696 | 0.0371 | 0.4177 | 0.4088 | 0.4271 |
| h1 control best_val | 0.0272 | 0.0781 | 0.0199 | 0.4385 | | |
| h10 ASL best_val | 0.0410 | 0.0696 | 0.0373 | 0.4170 | | |

+51% macro on best_val, +66% on the final checkpoint. Macro recall doubles, macro precision gives up only 11%, and micro pays 4.9%, almost entirely precision traded for recall -- the shape ASL should produce, and the first evidence that the macro gap is a tail-loss problem, not only a capacity one.

h10 shows no late decline: its peak moved from step 1040 to 4784 and the final checkpoint scores the same as best_val, while the control loses 30% from its peak. Macro-peaks-early-then-decays is a property of BCE on a long-tailed label set, not of the benchmark.

Caveat: 10k-image benchmark, 5000 steps. Absolute numbers are small; the transferable claim is the relative +51%, to be re-checked on the real corpus.

### 03:45 check-in — h11 pos_weight is failing; disk risk retired

`h11_pos_weight` at 4560/5200: 1k peak 0.0119 at step 1456, last 0.0058 (-51% from its own peak, 2.5x below the control's 0.0298). The deferred per-class `pos_weight` (0.68-100 over 10000 images) over-corrects: it protects the 1686 zero-positive labels, which was the point, but distorts the loss balance enough to lose half the signal. Expect its 50k macro well under the control -- pos_weight is not the tail fix, ASL is.

Disk: pruned 48 intermediate checkpoints from each scored arm (`470e6fa9`, `7ee12110`), keeping `ckpt_step005000.pt` and `best_val.pt`. Freed 59.9 GB, 388.1 GB free. Eight arms remain at ~32 GB each (~256 GB), so the night cannot die of a full disk.

### 04:15 check-in — h11 confirmed destroyed; the loss axis now outranks the head axis

`h11_pos_weight` final, verified from validation.json: 50k macro 0.0068, micro 0.2572 -- a 3.6x collapse on macro against both the control and ASL. pos_weight is a dead end at this label density, recorded as a negative result.

Standing after three scored arms:

| arm | 50k macro@.40 | 50k micro@.40 |
|---|---:|---:|
| h10 ASL | 0.0410 | 0.4177 |
| h1 control BCE | 0.0247 | 0.4372 |
| h11 pos_weight | 0.0068 | 0.2572 |

h2_cls_mean is at 2740/5200 and tracking the control (peak 0.0267 at step 832).

The loss axis (+51% for ASL, -72% for pos_weight) is far wider than anything the head axis has shown, so the highest-value next run is ASL crossed with the winning head, not another head under BCE, then an ASL gamma sweep {1.0, 2.0, 4.0} -- gamma=2.0 was convention, not measurement, and on a 0.21%-density label set it is the most likely remaining multiplier.

### 04:45 check-in — h2 ties the control; the head axis looks flat under BCE

`h2_cls_mean` final: 50k macro 0.0253, micro 0.4395 -- +2.4% macro and +0.5% micro. If the seed noise band is a few percent, h2 is a tie, not a win, and the 4x768 `cls_mean_max_std` pooling buys nothing over 2x768 `cls_mean` here.

Four scored arms, and the pattern is clear:

| arm | change vs control | 50k macro@.40 | 50k micro@.40 |
|---|---|---:|---:|
| h10 ASL | loss: asymmetric focal | 0.0410 | 0.4177 |
| h2 cls_mean | head: 2x768 pooling | 0.0253 | 0.4395 |
| h1 control | -- | 0.0247 | 0.4372 |
| h11 pos_weight | loss: per-class weights | 0.0068 | 0.2572 |

Loss axis +51% / -72% against head axis +2.4%: a forty-fold leverage difference between two things we swept as if comparable. h3-h8 (cls-only, MLP heads at 512/1024/2048/4096) are worth finishing for completeness, but the head is not where this is won.

### 05:15 check-in — h3 tracking control; probability caches located for the tail analysis

`h3_cls` at 4120/5200, 1k peak 0.0262 at step 1456, the band of h1 (0.0298) and h2 (0.0267). Three head variants within a few percent: the head axis stays flat.

Per-checkpoint probability dumps live at `I:\.cache\wd-tagger-trainer\probs\<run8>_<ckpt>.npy` (28 files, 1247.6 MB each = 50000 x 12476 float16), with `<run8>_005000` and `<run8>_best_val` for the scored runs, so a per-tag comparison of h1 (BCE) vs h10 (ASL) needs no inference: same 50k images, same label matrix, two probability matrices. That shows which frequency bins the +51% comes from, until now asserted from the aggregate only.

### PROVEN AT TAG LEVEL — ASL's gain is entirely in the tail, none at all in the head

`scratch/sophie/tag_bins.py` compared the cached matrices for h1 (BCE, `470e6fa9`) and h10 (ASL, `7ee12110`) over the same 50000 images at threshold 0.40, binning tags by positives in that subset. No inference was run; this is the data the 50k score came from.

| bin (val positives) | tags | BCE F1 | ASL F1 | delta |
|---|---:|---:|---:|---:|
| 1-9 | 4641 | 0.0002 | 0.0028 | +0.0027 |
| 10-49 | 5093 | 0.0034 | 0.0189 | +0.0155 (5.6x) |
| 50-199 | 1682 | 0.0413 | 0.0906 | +0.0492 (2.2x) |
| 200-999 | 779 | 0.1471 | 0.1834 | +0.0363 |
| 1000+ | 259 | 0.4100 | 0.4098 | -0.0001 |

Overall macro 0.0248 -> 0.0410. 2636 of 12454 tags improved (21.2%), essentially every bit of the gain in the 10-999 band; the 259 frequent tags, the ones that already worked, are unchanged to four decimal places.

Top movers: label 11171 (10 positives) 0.000 -> 0.667; label 9579 (29 positives) 0.000 -> 0.619; label 9736 (17) 0.000 -> 0.571. ASL is not nudging these tags, it is switching them on.

Consequence for the goal: the bar's 0.5579 macro cannot be reached by improving frequent tags, already at 0.41 and contributing almost nothing to our deficit. What is left lives in the 10-999 band, and ASL is the first thing that moved it. Two follow-ups now have evidence: push gamma up, since the 1-9 bin is still at 0.0028 and is the largest tag population; combine ASL with the best head, since the head axis is flat under BCE and untested under ASL.

### 06:15 check-in — a false "queue not running" nearly cost us a double-launch

`checkin.py` reported "queue is NOT running but arms remain -- relaunch". Wrong: the driver (pid 619676) was alive, scoring h4's 50k pass, and scoring between arms looks identical to nothing happening. A second driver would have fought for the 24 GB GPU and killed the running arm. Fixed: `checkin.py` checks `sweep_driver.lock`'s pid via `tasklist` and prints "driver alive (scoring between arms) -- nothing to launch, do NOT relaunch". Verified.

h4_mlp1024 finished training (5000/5200, 1k peak 0.0217 at step 1872) and is being scored, below every other head arm so far: the MLP family looks like a regression rather than a variant.

### 06:45 check-in — h4 lands on the control; head axis flat across four variants

`h4_mlp1024` final, verified from validation.json: 50k macro 0.0245, micro 0.4226. Standing table (50k, thr 0.40), all six scored arms:

| arm | what changed | macro | micro |
|---|---|---:|---:|
| h10 ASL | loss: asymmetric focal | 0.0410 | 0.4177 |
| h3 cls | head: cls only | 0.0258 | 0.4397 |
| h2 cls_mean | head: 2x768 | 0.0253 | 0.4395 |
| h1 control | -- | 0.0247 | 0.4372 |
| h4 mlp1024 | head: +1024 MLP | 0.0245 | 0.4226 |
| h11 pos_weight | loss: class weights | 0.0068 | 0.2572 |

Four head variants span a 5% band while the loss axis spans -72% to +51%. h4 also costs micro for no macro gain, so the MLP family is a net regression.

Prepared, not yet run: the gamma sweep, `configs/heads/phase3/gamma{1,3,4}.json`, copies of h10 with `loss_gamma` 1.0 / 3.0 / 4.0 against h10's measured 2.0. Rationale from the tag-bin proof: the 1-9 bin (4641 tags, the largest population) is still at F1 0.0028, and gamma is the only dial setting how hard ASL pushes that region. Kept in a subdirectory so the running driver's `configs\heads\*.json` glob cannot pick them up.

### 07:15 check-in — h5 sub-control; and a planning correction worth catching now

`h5_mlp2048` final: 50k macro 0.0232, micro 0.4214, the weakest head arm yet -- a 0.0232-0.0258 band, +/-5%, across five head variants, while ASL sits at 0.0410. The MLP family loses micro as it grows.

`h6_control_seed2025` is now running (360/5200), the only arm that can say whether that +/-5% band is noise or signal.

Planning correction: the queued unfreeze configs (`configs/heads/phase2/u*.json`) came from the BCE control h1, before ASL existed. With ASL our best loss by +51%, sweeping unfreeze depth under BCE answers the wrong question. Generated instead: `configs/heads/phase3/asl_u{0,4,6}.json` (ASL at 0/4/6 unfrozen blocks; 12 is h10, already scored), plus every gamma variant. None are in `configs\heads\*.json`, so the running driver cannot pick them up.

### 07:45 check-in — authoritative arm table, and a mapping error caught before it propagated

Authoritative run-dir mapping, from the `run dir:` line in each `sweep_<arm>.log`, never from directory order: h1 `470e6fa9`, h2 `c3e2744d`, h3 `8afb7e5e`, h4 `d14c3bfd`, h5 `04aa7438`, h6 `1e312196`, h10 `7ee12110`, h11 `2608eb83`.

A first `summary_table.py` had h3/h5 and h2/h11 swapped, because I filled the map from the order `Get-ChildItem` listed the run dirs; it disagreed with checkin.py, which is the only reason it was caught. `summary_table.py` now carries the mapping with a comment on its origin. Third time tonight that a number was wrong in the labelling rather than the measurement.

Corrected table, 50k nested cache, threshold 0.40, both readings:

| arm | final macro | final micro | best_val macro | best_val micro |
|---|---:|---:|---:|---:|
| h10 ASL | 0.0410 | 0.4177 | 0.0410 | 0.4170 |
| h3 cls | 0.0258 | 0.4397 | 0.0285 | 0.4450 |
| h1 control | 0.0247 | 0.4372 | 0.0272 | 0.4385 |
| h2 cls_mean | 0.0253 | 0.4395 | 0.0262 | 0.4429 |
| h4 mlp1024 | 0.0245 | 0.4226 | 0.0254 | 0.4225 |
| h5 mlp2048 | 0.0232 | 0.4214 | 0.0234 | 0.4216 |
| h11 pos_weight | 0.0068 | 0.2572 | 0.0097 | 0.2681 |

h3 (`cls` pooling only, 768-wide) is the nominal head winner on both readings, +4.8% over control on best_val; whether that means anything is what h6 says. best_val beats final on every BCE arm but is identical on ASL, consistent with ASL not decaying.

### 08:12 — h6 IS IN. The noise band is +/-3%, and it settles the whole head axis.

`h6_control_seed2025` is the control config with only `seed` changed, so h1 vs h6 measures pure seed noise:

| | final macro | best_val macro |
|---|---:|---:|
| h1 control (seed A) | 0.0247 | 0.0272 |
| h6 control (seed 2025) | 0.0252 | 0.0264 |
| spread | 2.0% | 3.0% |

The band is +/-3% on best_val. Judged against it:

| arm | best_val macro | vs control | verdict |
|---|---:|---:|---|
| h10 ASL | 0.0410 | +51% | real, 17x the band |
| h3 cls | 0.0285 | +4.8% | borderline -- above the band but barely |
| h1 control | 0.0272 | -- | reference |
| h6 seed | 0.0264 | -3.0% | noise |
| h2 cls_mean | 0.0262 | -3.7% | noise (a tie) |
| h4 mlp1024 | 0.0254 | -6.6% | slight regression |
| h5 mlp2048 | 0.0234 | -14% | real regression |
| h11 pos_weight | 0.0097 | -64% | catastrophic |

Conclusions this licenses:

1. ASL is the only unambiguous win of the night, +51% against a 3% band
2. The head axis is noise: h2 and h4 sit inside the band, h3's +4.8% is at the edge and needs a replicate, MLP heads >=2048 are a genuine regression
3. The bar's head architecture is still untested (mean-only pooling, `phase3/h12_mean_bce.json`) and unlikely to be worth a slot, but it is the one structural variant never run, so it stays queued as cheap insurance
4. Priority for the next GPU time: ASL x head, then ASL gamma, then ASL x unfreeze depth; all three configs exist in `configs/heads/phase3/`

### 13:15 check-in — h7 running; the borderline call now has a replicate queued

`h7_mlp512` at 1560/5200, 1k peak 0.0184, continuing the MLP family's downward trend. Two arms left: h8, h9.

Action taken: h3's +4.8% is the only head result above the +/-3% band and it sits barely above it, so rather than believe or dismiss it I queued `configs/heads/phase3/h3_cls_seed2025.json`, h3's config with h6's seed, so h3-vs-replicate measures the same distance h1-vs-h6 did. Surviving a second seed makes it real; collapsing to the band settles the head axis as noise.

### LOSS RESEARCH VERDICT — no published loss beats ASL in our regime, and my "1-9 bin" reasoning was wrong

A dedicated literature search (2023-2026, 20+ papers) asked the direct question:

No published loss is known to beat ASL on 10k+ labels at a fixed global threshold. Every paper at our label count reports mAP or P@k (Open Images 9600 labels: ASL 86.3 micro/92.8 macro mAP; CSL 87.34 mAP); every paper reporting a fixed-threshold F1 does it at 20-80 labels. The intersection is empty, and our +51% at a fixed 0.40 is one of the few fixed-threshold macro-F1 measurements in this regime at all.

Ranked candidates, none a clean win:

| rank | loss | evidence | verdict |
|---|---|---|---|
| 1 | CB-NTR composed WITH ASL (ASL positive branch x NTR lambda on negatives) | BALMS 18211 labels: macro-F1 13.94 -> 18.40; tail micro 8.58 -> 24.15 | our own composition, in NO paper -> needs its own screening arm |
| 2 | CSL / Partial-ASL top-K unannotated-negative dropping | Open Images 9600 labels, 87.34 mAP, but mAP | only mechanism aimed at true false negatives; mAP claim |
| 3 | SPLC / Hill loss | COCO 80 labels, simulated missing labels, mAP | cheapest; MUST be top-K, never a raw tau |
| 4 | LDAM-DRW | one unreviewed 30-label report where ASL collapsed to 0.0667 | grey zone: per-class calibration wearing a loss |

Rejected outright: logit adjustment / cRT / tau-norm / Seesaw / EQL / BCL / IFL / GCL (per-class weight or offset families, evaluated at per-class or tuned thresholds -- the family that gave us 0.0097); XMC ranking losses (P@k, dual-encoder assumptions); MoE/per-label adapters. "PLC / Pearson loss" does not appear to exist; "ASL v2" does not exist.

THE CORRECTION -- bin-mass arithmetic (`scratch/sophie/bin_mass.py`). I claimed the 1-9 bin was the priority because it holds 4641 tags. Macro is a mean over 12454 tags, so what matters is (tags/total) x F1:

| bin | tags | ASL F1 | contribution to 0.0410 macro | share |
|---|---:|---:|---:|---:|
| 1-9 | 4641 | 0.0028 | 0.00104 | 2.5% |
| 10-49 | 5093 | 0.0189 | 0.00773 | 18.9% |
| 50-199 | 1682 | 0.0906 | 0.01224 | 29.8% |
| 200-999 | 779 | 0.1834 | 0.01147 | 28.0% |
| 1000+ | 259 | 0.4098 | 0.00852 | 20.8% |

The 1-9 bin is 37% of the tags and 2.5% of the score: 5x there is worth +0.0042 total, the same +0.0042 as +5% on the 50-199 bin's 1682 tags. My aim-gamma-at-the-rarest-tags reasoning was tidy and wrong -- the real mass is 10-999, exactly where ASL already went. Judge gamma by its effect on the 10-199 bins.

The deeper point, which I cannot yet refute: with ~9 positives a tag's prior is ~1e-6, so a calibrated posterior can only cross 0.40 at a likelihood ratio >1e6. The 1-9 bin may fail because the feature is not learnable from 9 examples, in which case no loss moves it and the honest spend is label-text/feature initialisation (DECAF/CLIP-Decoder), not loss shaping.

### 08:27 — naming convention adopted, and the phase-3 ladder chained behind the current sweep

`AGENTS.md` now instructs every future agent to (a) read `sophie.md` and be Sophie and (b) name runs `u{N}-{pool}-{proj}-{loss}[+mod][seed]` instead of `h#`. `docs/arm-taxonomy.md` holds the mapping so legacy labels stay resolvable, and it is indexed in AGENTS.md's docs list.

`configs/heads/phase3/` was renamed to the notation and ordered by priority, so the sweep runs the most informative crosses first:

| order | config | what it tests |
|---|---|---|
| 01 | `u12-C-L-asl-g4.0` | ASL x best head -- the most likely remaining win |
| 02-04 | `u12-CMSS-L-asl-g{1.0,3.0,4.0}` | the gamma dial, judged by the 10-199 bins |
| 05-07 | `u{0,4,6}-CMSS-L-asl-g4.0` | ASL x freeze depth (long-run speed vs accuracy) |
| 08 | `u12-M-L-bce` | the bar's exact head (mean pooling only) |
| 09 | `u12-C-L-bce-seed2025` | is h3's +4.8% real? |

The driver that was still on `u12-CMSS-H512-bce` (h7), h8 and h9, and the detached wrapper that polled its lock file to launch phase 3 behind it, are both retired and deleted. The phase-3 ladder now runs one arm at a time, by hand, as a harness background job. Arm 01 was relaunched at 11:38 into `runs/83231ec3`.

### ASL is a REAL gain, not a threshold shift — but both models want a lower threshold than 0.40

The tag-bin P/R split looked like a score shift: at the 1000+ tags ASL lost 0.167 precision and gained 0.120 recall with F1 flat, a 1:1 trade. Tested directly over a grid of global thresholds (never per-tag), reading the cached probability matrices, no GPU:

| thr | control macro | ASL macro | control micro | ASL micro |
|---:|---:|---:|---:|---:|
| 0.20 | 0.0279 | 0.0436 | 0.4503 | 0.3838 |
| 0.30 | 0.0262 | 0.0427 | 0.4437 | 0.4055 |
| 0.40 | 0.0248 | 0.0410 | 0.4372 | 0.4177 |
| 0.50 | 0.0235 | 0.0393 | 0.4304 | 0.4248 |
| 0.60 | 0.0223 | 0.0372 | 0.4232 | 0.4275 |

Real ranking quality, not an operating-point artefact: given its own best global threshold the control reaches 0.0279, still 32% below ASL's 0.0410 at the protocol threshold and 36% below ASL's own best (0.0436), and ASL beats the control at every threshold in the grid. The head's precision-for-recall trade is a side effect of ASL scoring higher, not the source of the gain.

Second finding, about the protocol: macro still rises as the threshold falls. Both curves peak at or below 0.20, the bottom of the grid, and neither turned over. Micro is opposite -- the control's micro optimum is 0.20, ASL's 0.60, consistent with ASL shifting scores upward. Both models at a fixed 0.40 sit on the wrong side of the macro optimum; the bar is measured the same way so the comparison stays fair, but the absolute macro numbers understate what these models can do.

Open question worth one cheap pass: extend the grid below 0.20 (0.10, 0.15) to find where macro peaks. It does not change our scores but bounds how much of the gap is operating point versus score quality.

### h7 scored, and the low-threshold extension changes how I should describe the ASL win

`h7_mlp512` final: 50k macro 0.0244, micro 0.4186. The MLP family is fully flat-to-worse: H512 0.0244, H1024 0.0245, H2048 0.0232, control 0.0272. No hidden layer helps.

Extended threshold grid (control vs ASL, global thresholds only):

| thr | control macro | ASL macro |
|---:|---:|---:|
| 0.05 | 0.0330 | 0.0414 |
| 0.10 | 0.0305 | 0.0433 |
| 0.15 | 0.0290 | 0.0439 |
| 0.20 | 0.0279 | 0.0436 |
| 0.40 | 0.0248 | 0.0410 |

1. The control has no interior optimum: macro rises all the way down to 0.05, where we stopped. BCE on a 0.21%-density label set under-predicts monotonically; no threshold is well-tuned. ASL has an interior optimum at ~0.15 (0.0439), what a properly shaped score distribution looks like
2. Part of ASL's headline +65% at 0.40 is an operating-point effect: at each model's own best global threshold the gap narrows to +33% (0.0439 vs 0.0330). At the fixed protocol threshold +65%, at best-case thresholds still +33%. ASL is better either way, but the headline overstates the pure ranking improvement

The decision stands -- ASL is still the only unambiguous win and the only model whose curve turns over. The claim changes, and the claim is what goes in the report.

### 09:45 check-in — MLP family complete (monotonic decline), and a small honesty correction on CB-NTR

`h8_mlp4096` final: 50k macro 0.0219, micro 0.4197. The projection axis is fully mapped and monotonic -- more hidden width is strictly worse:

| hidden width | 50k macro |
|---|---:|
| linear (control) | 0.0272 |
| MLP 512 | 0.0244 |
| MLP 1024 | 0.0245 |
| MLP 2048 | 0.0232 |
| MLP 4096 | 0.0219 |

Four widths, monotone, all below linear: the head does not want a hidden layer at all, as the single-linear-head ablation predicted. Cleanest negative result of the sweep.

`h9_cb_ntr` (`u12-CMSS-L-ntr(l2.0)`) is now running, the last arm of the main sweep.

Verification performed rather than assumed: `scratch/sophie/test_losses.py` against the loss h9 actually trains with. ASL focal-weight detach is exact (rel diff 0.0), but CB-NTR misses the BALMS Eq6 reference to machine precision: Eq6 1.162269, corrected 1.162409, shipped-in-`wdtag/losses.py` 1.163065 -- disk/Eq6 ratio 1.001. An earlier note claimed an exact 0.00e+00 match; stale now, since the test has been extended and the shipped loss is a slightly different but equivalent-in-intent arrangement. 0.07% is immaterial and does not invalidate h9, but the record should say within 0.1%, not exact.

The diagnostic table quantifies why class weighting failed: plain BCE gives positives 0.259% of the loss mass; CB beta=0.9 (any normalisation) drops that to 0.063%, beta=0.999 to 0.019% -- a 4x and 14x cut in positive signal, precisely the 0.0097 collapse.

### Checkpoint frequency was burning ~8% of every arm (found from file timestamps)

Spotted in the creation times of `runs/1e312196`. Measured:

49 checkpoints x 638.8 MB = 30.6 GB per 5000-step arm, in a 41.6-minute run

Two writers interleaved: `--ckpt-every-steps` (default 200) and an unconditional save at every epoch boundary (208 steps at 10k/48). 200 and 208 are nearly equal, so the effective cadence was a 639 MB file every ~100 steps

Cost per write 5-15 s. Paired timestamps: 16 steps across a save measured 0.76 s/step vs a 0.44 s/step baseline; 32 steps measured 1.02 s/step. Total overhead above the compute floor ~300 s, saves the bulk -- ~8% of wall time

Fixed at the source, not papered over:

1. `wdtag/train.py` -- the epoch-end save is now opt-in via `--ckpt-every-epochs N` (default 0). Resumption is unaffected: the resume path picks the highest-numbered `ckpt_step*.pt`, and the max-steps save still writes a final checkpoint
2. `scratch/sophie/run_sweep.ps1` now passed `--ckpt-every-steps 1000 --ckpt-every-epochs 0`; the script is since deleted, so the cadence lives in `wdtag/config.py` as `CHECKPOINT_EVERY` 1000 and in the `--ckpt-every-epochs` default of 0
3. `docs/best-settings.md` gained section 1b with the numbers and the fix

Effect per arm: 49 files / 30.6 GB / ~3 min -> 5 files / 3.2 GB / a few seconds. Across the nine queued phase-3 arms, roughly 24 minutes of GPU time and 245 GB of writes recovered.

### best_val.pt was being selected on MICRO, not macro -- every arm so far

Spotted live in h9's validation line: `| best by micro_f1`. `wdtag.train --select-metric` defaults to micro and the sweep never overrode it, so `best_val.pt` for all eleven arms of the first sweep is the best-micro checkpoint, and our best_val macro column is macro at a micro-selected checkpoint, not the macro peak. That is why best_val so often equalled the final checkpoint on ASL: micro kept improving to the end.

`SELECT_METRICS` maps `macro` -> `macro_f1`, macro F1 at 0.40 (protocol threshold, not tuned), so it is exactly the objective. The phase-3 ladder is launched with `--select-metric macro`, the flag the deleted driver used to pass.

Comparability: phase-3 `best_val` numbers are macro-selected and not directly comparable to the first sweep's micro-selected ones; within phase 3 they are consistent, and the final-checkpoint column stays comparable across all arms.

### Jobs are now launched as harness background jobs, not detached processes

User directive 10:40: the sweep must run as a managed background job (job id, collectable output, visible in the job list), not a hidden `Start-Process` wrapper. Migrated at once: paused `u12-C-L-asl-g2.0` at `ckpt_step000137`, stopped the detached driver, relaunched the phase-3 ladder as job `pwsh-15d69f49d5fc4175`; the arm resumed from step 137 with no replay of samples.

Recorded in `AGENTS.md` (Training section): the old detach-or-lose-it-to-a-turn-boundary rationale is retired, because resumption from `ckpt_step*.pt` makes a reaped job cost minutes. The driver and its pid-file lock were later deleted outright, so the mutual exclusion the lock used to give is now only my own read of the job list before each launch.

### h9 CB-NTR scored: +24% macro over control, the first non-ASL loss to clear the noise band

Verified from `runs/ab446356/validation.json` (subsets["50000"], threshold index 0 = 0.40):

| arm | macro F1 | macro P | macro R | micro F1 |
|---|---:|---:|---:|---:|
| control `u12-CMSS-L-bce` | 0.0247 | 0.4523 | 0.0179 | 0.4372 |
| `u12-CMSS-L-ntr(l2.0)` | 0.0306 | 0.0756 | 0.0239 | 0.4197 |

+24% macro against a seed-noise band of +-3%; micro gives up 4%. The second real effect in the sweep, and the first that is not ASL.

The P/R split shows the mechanism: macro precision collapses 0.452 -> 0.076 while recall rises 0.018 -> 0.024. NTR's fixed bias (kappa=0.05) pushes the tail up at every step, so rare tags start firing -- the negative-side relaxation the loss research described, and why NTR+ASL is the paper's best-ranked combination (BALMS, arXiv 2109.04712, macro 13.94 -> 18.40 on 18211 labels).

So the next arm is the composition, not another one-factor probe: ASL (positive branch + focal clip) x NTR (negative-side tolerance), never run together. Queued after the phase-3 ladder.

### The ASL gamma never reached the loss — every ASL arm ran gamma_neg=4.0 while claiming 2.0

Found while wiring the ASL x NTR composition. `wdtag/train.py` built the loss but called it as `_loss_fn(logits, y)` with no kwargs for anything but `cb_ntr`, so `loss_gamma` from the config was ignored and `asl()`'s own default `gamma_neg=4.0` always ran.

Consequences, all corrected:

1. The champion h10 is really gamma_neg 4.0, not 2.0. Its config says 2.0; the log line proves what ran. ASL is still the champion, but the label was wrong
2. The phase-3 gamma sweep would have produced three identical arms: `g1.0`, `g3.0` and `g4.0` all collapsed to the default 4.0, so arms 02/03/04 would have been the same run three times (~2.5 h)
3. Every ASL arm in phase 3 was retuned to the gamma that actually won (4.0) and renamed `...-asl-g4.0` so the name never again disagrees with the run

Fixes:

`train.py` now builds `_loss_kwargs` explicitly: `gamma_neg=cfg.loss_gamma` for `asl`/`asl_ntr`, plus `ntr_lambda` for `asl_ntr`, and prints the value it will use

Verified numerically that the function responds: on one fixed tensor, `asl(g1)=0.3647` vs `asl(g4)=0.1424`

Bonus, and the reason I was in there: `asl_ntr` is now implemented (`wdtag/losses.py`), ASL's asymmetric focusing and probability clip on NTR-tolerated negatives, weighted 1/lambda on the negative side exactly as BALMS Eq 6. At `lambda=1` it reduces to `asl` with difference 0.0. The composition's rationale is now evidence, not theory: NTR alone +24% macro, ASL alone +65%, BALMS ranks NTR x positive-focusing as its best pair.

Restarted as job `pwsh-ea805eee0f644d6f` in a new run dir `9cc009da` (the gamma change moves the sha8, as it must). That job was reaped at step 624, so `9cc009da` holds three log lines and no step checkpoint; `cf852a8c`, its pre-gamma twin, stops at the same step. Both are orphaned partial dirs with no `validation.json` and nothing to resume from. The defaults change moved the sha8 a second time, so arm 01 was relaunched at 11:38 as job `pwsh-a6d9c0d8a21c43dc` into `runs/83231ec3` from step 0.

Verification still owed: that `g1.0` and `g4.0` produce visibly different loss curves in the real trainer. Arms 02 (g1.0) and 04 (g4.0) are the same config otherwise, so identical loss curves would mean the plumbing is still broken. Watch those two. Half discharged by the 11:38 relaunch, which prints `loss asl: gamma_pos 0.0, gamma_neg 4.0, clip 0.05, detach_focal` before the first step: the configured gamma reaches the loss in the live trainer, not just in a unit test.

### 10:55 - tech-debt pass: stale defaults, a dead config field, duplicated loss math, and a fake test

`wdtag/config.py` defaults contradicted every measurement in `docs/best-settings.md`. Fixed: `batch_size` 32 -> 48, `block_size` 32 -> 128, `CHECKPOINT_EVERY` 200 -> 1000, `loss_gamma` 2.0 -> 4.0 (what the winning ASL arm actually ran, once the kwargs bug is accounted for). A config now only states deviations. New default sha8 `bbf2d3ce`; this changes the hash of any config omitting those fields, so phase-3 arms start in fresh run dirs - expected, not a bug.

Removed the dead `optimizer` config field: a string nothing read (SinkSGD_ADV is wired directly in train.py). A field that cannot change behaviour is a lie in the hash.

`wdtag/losses.py`: three copies of ASL's focal expression collapsed into one `asymmetric_focal()`. `asl()` and `asl_ntr()` are the same per-element expression over different logits (plain vs NTR-tolerated), so they share it and differ only in the weighting before reduction. Verified against independent reference formulas: `asl` matches at gamma_neg 1/2/4 to 1e-6, `cb_ntr` matches BALMS Eq 6/8 to 1e-6, `asl_ntr(lambda=1)` differs from `asl` by exactly 0.0, gradients finite and non-zero.

`scratch/sophie/test_losses.py` was testing a second copy of the losses, not the shipped code -- it imported a file deleted in the scratch purge. Rewritten self-contained: the reference formulas live in the test, in plain torch, no second implementation to agree with. It passes.

Still owed: `train.py`'s two near-identical training loops (cache path vs stream path), the duplication that required a checkpoint fix twice. Being unified into one per-step helper with identical-loss verification.

### 11:05 - train.py: the two training loops now share one StepHelper, and the cached branch finally saves its last checkpoint

The debt. `main()` held two training loops -- cached DataLoader and tar-streamed -- each with its own copy of the same five per-step concerns: pause check, first-100 debug dump, the `log_every` line, the `ckpt_every` save, the `--max-steps` exit, plus validation. They had drifted: a checkpoint fix needed applying twice, and the end-of-run save existed in only one.

The fix. One `StepHelper` (`wdtag/train.py:255`, `step` at `:285`) owns all five, returning `go` / `stop` / `complete`. The resume cursor is passed in, not guessed: `epoch=None` means a streamed pass (saves carry `(0,0)`), while the cached loop passes the epoch and the batch just consumed, so a checkpoint records `(epoch, batch+1)` and a pause records `(epoch, batch)` -- the unconsumed batch, which a resume re-shows. Loop bodies are 5-6 lines each. `train.py` 815 -> 808 lines, +93/-100.

Verified, not assumed. A 40-step run of the same config before and after produced identical step lines -- same steps, same loss (0.0092, 0.0067), same LRs, same VRAM -- differing only in wall-clock img/s (98 vs 107; the box was idle for the second). The state machine was unit-checked with mocked saves: pause-stream `(17,0,0)`, pause-cache `(17,2,5)`, interval-cache `(5,3,15)`, max-steps-cache `(1,0,1)` with val correctly skipped.

The asymmetry it exposed. The streamed branch saved a final checkpoint after `training done`; the cached branch printed that line and went straight to test evaluation, while a comment claimed an end-of-run save sat below. Every completed 10k arm therefore ended with no checkpoint at its final step, the last up to 999 steps stale. Mirrored: both now save `ckpt_step{global_step}.pt` before evaluating.

Not unified on purpose: the two debug dumpers (different filename conventions and index sources), the cache-only epoch boundary (permutation, loader, epoch-end val, opt-in `--ckpt-every-epochs`), and the stream pass-chaining/skip arithmetic.

### 11:38 - phase 3 arm 01 relaunched, and what sits behind it

`u12-C-L-asl-g4.0` in `runs/83231ec3`, job `pwsh-a6d9c0d8a21c43dc`. At step 1140 of 5200 the loss is 0.0011 and the best val macro is 0.0372 at step 1040. First checkpoint at step 1000, so a reaped job resumes instead of replaying. The launcher printed `loss asl: gamma_pos 0.0, gamma_neg 4.0, clip 0.05, detach_focal`, which is the plumbing proof that gamma reaches the loss in the live trainer.

The reference for this arm is h10 (`u12-CMSS-L-asl`, which ran gamma 4.0 despite its config saying 2.0). Scored on 50k at 0.40, both its `best_val.pt` and its final checkpoint give macro 0.0410 and micro 0.4177, read from `runs/7ee12110/validation_bestval.json` and `validation.json`. The noise band is 3%.

At step 1040 arm 01's 1000-image val macro is 0.0372 against h10's 0.0306 at the same step and 0.0316 at h10's peak (step 1248), with micro 0.2988 against 0.3037. Over the first five checkpoints, 208 to 1040, macro is ahead at all five; micro is mixed, ahead at 416 by 0.027 and behind from 624 to 1040 by 0.005 to 0.011, the same trade the head axis made under BCE. That is encouraging and it is not a result: only the 50k score decides, and the in-run val is 1000 images.

A win is 0.0423 or better on the 50k of `best_val.pt`, more than 3% above h10. Inside the band, the head axis is noise under ASL as it was under BCE, and the remaining levers are gamma and freeze depth. That is the outcome I expect, because the head axis has produced nothing but ties for eleven arms.

Next in the queue was arm 04 `u12-CMSS-L-asl-g4.0`, as the champion replicate. It is now deferred behind the freeze-depth ladder, for two reasons that arrived after that plan was written.

The first is speed and an un-tuned knob. Freezing costs 4.25 img/s per unfrozen block, and `lr_unfreeze` 3e-5 has never been bracketed — `docs/hyperparameter-scaling.md` grades it U, "highest uncertainty, and the knob that matters most", with drift evidence that it may be hot. Training all 12 blocks at a learning rate nobody chose is the least defensible part of the current recipe.

The second is that the frozen-plus-adapter regime has never been priced at all. Arm 05 `u0-CMSS-L-asl-g4.0` therefore runs next: 5.32M live rotation parameters, a 38.34M head, every block frozen. Its img/s number matters as much as its macro, because an adapter inside a frozen block still needs a backward pass through that block to reach the rotation, so the 160 img/s frozen-block figure from `runs/_wide_head.log` does not apply to it. Prediction to be checked, not assumed: near 110-120 rather than 160.

That number also gates the two-tower idea the user raised on 2026-10-22. A frozen SigLIP2 tower with OFT inside it costs the same backward as this arm, so if u0 lands near 110 the ensemble buys capacity at roughly u12 prices and the decision should be made with that in view, not on the assumption that frozen means cheap.

Arm 04 and the gamma dial wait until the ladder says how much direct fine-tuning is needed at all. If u0 comes in inside 3% of u12's 50k macro, the depth question is answered in the adapter's favour and the replicate is only needed before committing to a long run.

Speed, for the record: 98-99 img/s per window, 94.7 img/s over a full epoch once the 4.4 s validation is counted. The split is measured in `docs/benchmark.md` section H, including the part that matters for planning, that a second CUDA context on this card costs the trainer about 45%. The full pass over 5.32M images therefore reads ~15.5 h at 95 img/s, not the 9.2 h the 160 img/s frozen-block number would suggest.

### 12:20 - arm 01 scored: cls pooling under ASL is +58% macro over the champion, and my prediction was wrong

`runs/83231ec3`, final checkpoint `ckpt_step005200`, 50k at 0.40, read from `validation.json`:

| arm | macro F1 | macro MCC | macro P | macro R | micro F1 | micro P | micro R |
|---|---:|---:|---:|---:|---:|---:|---:|
| arm 01 `u12-C-L-asl-g4.0` | 0.0646 | 0.0669 | 0.0764 | 0.0720 | 0.4048 | 0.3448 | 0.4899 |
| h10 `u12-CMSS-L-asl-g4.0` | 0.0410 | 0.0440 | 0.0696 | 0.0371 | 0.4177 | 0.4088 | 0.4271 |

That is +57.6% macro and -3.1% micro against a 3% band, one seed.

I called this a tie an hour ago — "the head axis has produced nothing but ties for eleven arms". Wrong, and the wrongness is the finding: under BCE the head axis was noise, because h3's cls was +4.8% and everything else sat inside the band. Under ASL at gamma 4 the same swap moves macro further than any other single change measured so far. Kept here rather than deleted, because the distinction is BCE-axis versus ASL-axis, and the next person will assume the axis is inert.

The mechanism is recall, not precision. Macro P barely moves, 0.0696 to 0.0764, while macro R doubles, 0.0371 to 0.0720, and micro P falls 0.4088 to 0.3448. The cls head at the same 0.40 fires far more positives across 12476 labels. So either the ranking genuinely improved at the protocol threshold, or this is an operating-point shift that a fixed 0.40 happens to reward — the cached-probability threshold sweep separates those two, and h10's cache (`probs/7ee12110_005000.npy`) plus this arm's (`probs/83231ec3_005200.npy`) both exist for it.

Macro rises steeply with subset size on this arm, 0.0407 at 1k, 0.0544 at 5k, 0.0604 at 10k, 0.0646 at 50k, where h10 was almost flat (0.0315 to 0.0410). More firing means the rare labels gain as they become supported, which is why the 1k in-run val understated this arm badly and why no in-run val should be used to rank arms.

Consequence for the ladder: the depth arm now uses cls too, as `u1-C-L-asl-g4.0`, so the comparison against arm 01 changes only `unfreeze_blocks`.

### 12:57 - u1 scored: one unfrozen block keeps 98% of the macro for 18% more speed

`runs/b9ed94fb`, final checkpoint, 50k at 0.40, from `validation.json`:

| arm | unfreeze | pool | img/s | VRAM | macro F1 | macro P | macro R | micro F1 |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| u1-C-L-asl-g4.0 | 1 | cls | 130 | 16.8 GB | 0.0632 | 0.0663 | 0.0791 | 0.3713 |
| arm 01 u12-C-L-asl-g4.0 | 12 | cls | 109-116 | 20.4 GB | 0.0646 | 0.0764 | 0.0720 | 0.4048 |
| h10 u12-CMSS-L-asl-g4.0 | 12 | CMSS | 109-116 | 20.6 GB | 0.0410 | 0.0696 | 0.0371 | 0.4177 |

Eleven fewer unfrozen blocks cost 2.2% macro, inside the 3% band, and 8.3% micro, outside it. In exchange it is 18% faster, 3.6 GB lighter, and a full pass drops from 13.4 h to 11.4 h. Macro is the objective and micro is not, so the depth question is answered in favour of the shallow arm: unfreezing 12 blocks was paying 11.5M parameters and 2 hours per pass for a difference this benchmark cannot see.

The same arm also prices the adapter, and the answer is uncomfortable. The old 4.25 img/s per block ladder predicts 155 img/s at one unfrozen block. Measured 130. The 25 img/s gap is the backward pass still traversing eleven frozen blocks to deliver gradients to their rotations, plus the flat ~28 ms compiled-optimizer dispatch over 73 adapter tensors. So the 160 img/s "frozen backbone" figure in `docs/benchmark.md` applies to a tower with no adapters inside it at all, which is `runs/_wide_head.log` and nothing we train deliberately. Any plan built on "freeze the tower and let OFT do the work, it is nearly free" is reading a number that does not describe that configuration.

Also visible: u1 fires harder still, macro P 0.0663 against R 0.0791 where arm 01 was 0.0764 against 0.0720. The shallower tower trades precision for recall again, which is the same mechanism as the head swap and is worth watching as more of these stack.

### 13:20 - probe: a second frozen tower is worth +32% macro over the better single tower

Frozen pooled features from each tower on the 10k train cache, the same ASL head on top (Adam 1e-3, 4000 steps, batch 48, gamma 4.0, bias initialised at the label priors), scored on the 50k nested val at 0.40. Script `scratch/sophie/siglip_probe.py`, features cached in `I:\.cache\wd-tagger-trainer\probe`.

| features | macro F1 | macro MCC | macro P | macro R | micro F1 |
|---|---:|---:|---:|---:|---:|
| DINOv3-B/16 frozen, cls, 768 | 0.0407 | 0.0533 | 0.0280 | 0.1471 | 0.1674 |
| SigLIP2-B/16-512 frozen, MAP, 768 | 0.0452 | 0.0669 | 0.0283 | 0.2144 | 0.1593 |
| concat, 1536 | 0.0597 | 0.0739 | 0.0437 | 0.1680 | 0.2065 |

The concat is +46.7% over DINOv3 alone and +32.1% over the better single tower, SigLIP2 alone, which is an order of magnitude outside the 3% band. The two towers also fail in opposite directions, SigLIP2 better on macro (+11.1%) and worse on micro (-4.8%), which is what a caption-supervised tower and a self-supervised one should do, and is the reason concatenating them pays.

Read it as a ranking, not as a score. The head here is not the arm recipe — Adam instead of SinkSGD_ADV, the 10k cache, 4000 steps — so these numbers do not belong in the arm ledger and cannot be compared to u1's 0.0632. All three are also precision-poor at the fixed threshold, macro P 0.028 to 0.044 against R 0.147 to 0.214, so part of every figure above is an operating point rather than a ranking, the same pathology the arms keep producing. What the probe does establish is the thing it was built for: SigLIP2's features carry tag information DINOv3's do not, and the pair is better than either.

It does not settle the harder question, whether a fine-tuned DINOv3 tower leaves room for SigLIP2. Frozen DINOv3 scores 0.0407 and fine-tuned u1 scores 0.0632, so the margin the second tower was filling may already be filled by the adapter. That is what the next arm measures.

One engineering note that follows from the measurement: a frozen tower with no adapter inside it needs no backward and no retained activations, so it can run in the loop under `no_grad` for 3.5 ms per image and almost no VRAM, which is what the DINOv3 forward costs at 512px. Precomputing its features is only worth the 5.2 hours when a full 5.32M-image pass makes the per-step 30% matter; for 10k exploratory arms, in-loop and precomputed are the same number and in-loop needs no sidecar files.

### 13:52 - the probe again, with a head that has capacity: the ensemble's macro edge mostly disappears

Same features, same 4000 steps, one hidden layer of 2048 with GELU between the pooled vector and the 12476 logits. A bottleneck after pooling is the one kind of adaptation a feature cache permits.

| features | linear macro | +2048 macro | linear micro | +2048 micro |
|---|---:|---:|---:|---:|
| DINOv3 frozen | 0.0407 | 0.0475 | 0.1674 | 0.2305 |
| SigLIP2 frozen | 0.0452 | 0.0577 | 0.1593 | 0.2206 |
| concat | 0.0597 | 0.0586 | 0.2065 | 0.2614 |

A bottleneck is worth +16.7% macro on DINOv3 alone and +27.7% on SigLIP2 alone, so the linear head was indeed the limitation for a single tower. On the concat it is flat on macro, -1.8% and inside the noise band, while adding +26.6% micro. Which means the +32% the concat showed over the better single tower was largely an artifact of a linear head having too few degrees of freedom to exploit either tower alone: given a bottleneck, one tower already forms the nonlinear combinations and the second one mostly buys micro.

Also worth keeping: SigLIP2-B/16-512 alone reaches 0.0577, which is 91% of the fine-tuned single-tower arm at 0.0632, with no adapter, no unfrozen block and no image pipeline in the loop at all.

So adaptation currently leads by about 7.3% macro (0.0632 against 0.0586) at 130 images per second, against a cached pipeline that runs at hundreds to thousands of images per second. That is not a decisive margin for either side, and it is the reason the next step is cheap on the cached side: our own recipe, our own optimizer, the full corpus, on vectors that already exist.

### 14:20 - arm 11 scored: the fat head loses 19.5% macro in the real recipe, and the probe was wrong about it

`u1-C-H2048-asl-g4.0`, `runs/f8ff5c91`, final checkpoint, 50k at 0.40, from `validation.json`. Identical to u1 in every field except a 2048-wide hidden layer between the pooled cls vector and the logits, head 27.14M against 9.60M.

| arm | head | macro F1 | macro P | macro R | micro F1 | micro P | micro R |
|---|---|---:|---:|---:|---:|---:|---:|
| u1-C-L | 9.60M | 0.0632 | 0.0663 | 0.0791 | 0.3713 | 0.2919 | 0.5101 |
| u1-C-H2048 | 27.14M | 0.0509 | 0.0830 | 0.0479 | 0.4166 | 0.3725 | 0.4726 |

-19.5% macro, +12.2% micro, both far outside the band, one seed each. The head axis is not monotonically good: under ASL the linear head was what carried the recall that macro F1 lives on, and a dropout-0.1 MLP on 10k images sharpens precision by 25% while halving recall.

This contradicts the cached-feature probe, which gained +17% macro from the same bottleneck on DINOv3 features. The probe was Adam, 4000 steps, no dropout, no image pipeline; the arm has dropout, SinkSGD_ADV, fp32 head and the full ASL schedule. Two differences are plausible and one is testable without a new arm: the probe's head saw only 1536 cached floats per image, so its dropout-and-capacity budget was not competing with a tower that can absorb gradient. Recorded as unresolved rather than explained.

Consequence: the head is not the cheap missing lever I hoped it was. The remaining ceiling is the representation, which is the argument for the two-tower design and for adaptation, not for wider heads.

Also worth keeping: micro 0.4166 is the highest single-tower micro we have measured, and it arrives exactly where recall falls. Micro F1 rewards firing decisions that are already precise; macro at 0.40 rewards reaching the rare labels at all. Every recent arm has moved one at the other's expense, which is the strongest evidence yet that these two numbers are measuring opposite failures.

### 14:55 - Qwen3.8 tower measured on frozen features: weakest of the three, and it dilutes the pair

Extracted with the sister project's loader from the local mmproj GGUF, strict load of 460730096 parameters, 512px, merged pooling, 5120 dimensions, 60k images in 19 minutes. Every row is the same head recipe: 10k train cache, 4000 steps, Adam 1e-3, gamma 4.0, scored on the 50k nested val at 0.40. `H` means one hidden layer of that width, no H means a linear head.

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

At equal head width, SigLIP2 0.0457 beats DINOv3 0.0380 beats Qwen 0.0291, and the pair beats any single tower, which is the result the ensemble wanted. Qwen loses at every head shape tried, including a linear head with 64M parameters, so this is not a capacity artefact: the 5120-dimensional merged features are simply less useful for tag prediction than 768 dimensions from either other tower.

Adding it is worse than not adding it. The pair drops from 0.0508 to 0.0425 macro with Qwen concatenated at H768, and SigLIP2 plus Qwen without DINOv3 collapses to 0.0389, below SigLIP2 alone. Most of the damage is dilution: 5120 of 6656 input dimensions are Qwen's, so a fixed-budget head spends most of its weights on the weakest signal. A wider head or many more steps might rescue some of it, but the direction is unambiguous and the ceiling is far below what the other two towers already provide.

This matches the sister project's result, where the same tower was statistically indistinguishable from DINOv3-B at five times the compute on a pairwise-similarity task. Two different tasks, two different heads, same verdict.

Cost avoided: 54.5 GB of cached vectors and about 25 hours of extraction at 512px, or 12 hours at 384. Qwen is closed for this project unless something changes about how its features are pooled, which is the one open question left in `docs/qwen-vision-tower.md`.

### 15:05 - two-tower support is in the trainer, and one hash changed

`wdtag/model.py` gains an optional second tower from `cfg.siglip_dir`: SigLIP2's vision model, frozen, no adapter inside, run under `no_grad` so it costs a forward pass and no retained activations, its 768 pooled dimensions concatenated to DINOv3's before the head. `wdtag/config.py` gains `siglip_dir` with default empty, so every existing config means what it meant. `wdtag/validate.py` needs no change because it builds the same class from the run's stored config.

Verified on CPU before any GPU time: head in_features 1536, head 19.18M, 31.14M trainable (4.88M OFT + 7.09M unfrozen block + 19.18M head), forward returns (B, 12476), state dict 288 MB and 269 tensors with zero `siglip.` keys, and a full state-dict round trip loads clean. The tower is excluded from checkpoints deliberately: it is re-read from disk at construction and never modified, so saving 93.5M dead weights would double every file.

One consequence to record because it looks like a bug and is not: `sha8` hashes the full config, so adding a field changes the hash of every config that omits it. `10_u1-C-L-asl-g4.0.json` was `b9ed94fb` when it ran and is now `da67aabc`, and the two-tower config is `966249d9`. Completed arms keep their recorded run directories, which are still readable because the stored config simply lacks the new key; a future re-run of the same file lands in a new directory. Same class of event as the defaults change on 2026-10-22.

### 15:26 - arm 12 scored: gamma 2.0 buys 12% micro and costs nothing on macro

`u1-C-L-asl-g2.0`, `runs/ce692296`, final checkpoint, 50k at 0.40, from `validation.json`. One field different from u1: `loss_gamma` 2.0 against 4.0. This is the first bracket the gamma dial has ever had; every ASL arm in the ledger ran 4.0, and the phase-3 plan had 1.0 and 3.0 waiting.

| arm | gamma | macro F1 | macro P | macro R | micro F1 | micro P | micro R |
|---|---:|---:|---:|---:|---:|---:|---:|
| u1-C-L-asl-g4.0 | 4.0 | 0.0632 | 0.0663 | 0.0791 | 0.3713 | 0.2919 | 0.5101 |
| u1-C-L-asl-g2.0 | 2.0 | 0.0624 | 0.0924 | 0.0591 | 0.4156 | 0.3837 | 0.4533 |
| h10 u12-CMSS-L-asl-g4.0 | 4.0 | 0.0410 | 0.0696 | 0.0371 | 0.4177 | 0.4088 | 0.4271 |

Macro -1.3%, inside the band. Micro +11.9%, far outside it. Macro precision rises 39% and macro recall falls 25%, which is the mechanism: gamma_neg 4.0 was not buying macro F1, it was buying recall and paying for it with precision, and the two nearly cancelled. Halving it keeps macro and recovers the micro.

Against the old champion it dominates: 52% more macro at micro 0.4156 against 0.4177, a 0.5% difference inside the band. So the current best single arm on both of the numbers the goal names is `u1-C-L-asl-g2.0`.

The dial now has two measured points and the objective prefers the lower one, which is an argument for testing 1.0 rather than 3.0 next on the loss axis. Recorded for whoever picks that up: it is a cheap 35-minute arm and it is the only axis with a measured win still unexplored in the other direction.

### 15:28 - arm 13 launched: the first two-tower arm, both towers adapting on the DINOv3 side only

`u1-C-L-asl-g4.0+siglipb`, `runs/966249d9`, launched as job `pwsh-12e4b8e05abe4e10`. DINOv3 at u1 with OFT block 128 as in arm 10, plus SigLIP2-B/16-512 frozen with no adapter inside, its 768 pooled dimensions concatenated to DINOv3's cls before a 1536-wide head with 19.18M parameters. 31.14M trainable.

It started at 93 img/s and 17.1 GB against arm 10's 130 img/s and 16.8 GB, which is the predicted cost of the second tower's forward: 3.5 ms per image under `no_grad`, with no backward and no retained activations, so the whole toll is the forward itself and 0.3 GB of weights. The full 5200 steps should take about 48 minutes.

Why this arm and not the user's fuller design: it differs from arm 10 in exactly one thing, the presence of the second tower, so its 50k macro against 0.0632 is attributable. The user's preference, OFT inside SigLIP as well and its last block unfrozen, is the arm after this one if the frozen version shows a gain at all. Putting an adapter inside the second tower multiplies by roughly two the thing that made arm 13 affordable, and we have no measurement yet that says the second tower is worth anything after a fine-tuned first one.

### 16:11 - arm 13 scored: a frozen SigLIP2 tower costs 13.6% macro, and the probe did not transfer

`u1-C-L-asl-g4.0+siglipb`, `runs/966249d9`, final checkpoint, 50k at 0.40, from `validation.json`. 44.1 minutes at 106 img/s and 17.1 GB, against arm 10's 35.3 minutes at 130 img/s and 16.8 GB. Checkpoint 379.8 MB against 303.2 MB, a difference of 76.7 MB which is exactly the 19.18M head in fp32, so the state-dict exclusion of the frozen tower held on the real save path.

| arm | second tower | macro F1 | macro P | macro R | micro F1 | micro P | micro R |
|---|---|---:|---:|---:|---:|---:|---:|
| u1-C-L-asl-g4.0 | none | 0.0632 | 0.0663 | 0.0791 | 0.3713 | 0.2919 | 0.5101 |
| u1-C-L-asl-g4.0+siglipb | SigLIP2-B frozen | 0.0546 | 0.0693 | 0.0610 | 0.3652 | 0.2974 | 0.4733 |

-13.6% macro, outside the band; -1.6% micro, inside it. So the second tower costs 13.6% of the objective and 9.6 minutes per arm, and it buys nothing measurable on micro either.

This is the third time the cached-feature probe has failed to transfer. Its +32% over the better single tower shrank to +1.6% once either tower got a bottleneck, and now to negative once the first tower is fine-tuned rather than frozen. The pattern is consistent: the probe measured the value of features when neither tower could adapt, and adaptation is worth more than the second tower's features.

One caveat belongs with the number because it is my implementation and not the architecture as conceived. The head normalises the concatenation with a single `LayerNorm(1536)`, so DINOv3's fine-tuned cls vector, which is already normalised by the backbone, and SigLIP2's unnormalised pooling output are forced through one shared scale. Per-tower normalisation before the concat is the standard late-fusion treatment and is a one-line change. This arm tested concat-plus-one-norm, not concat-plus-proper-fusion, and the difference is worth one more arm before the ensemble line is closed.

What it does say already: a frozen second tower is not a free win, and the user's design of adapting SigLIP with OFT would have to overcome a 13.6% deficit before its own gain starts. That raises the bar for the 23-hour two-tower design considerably.

Also recorded for the ledger: 106 img/s against 130 for the single tower, so the second tower's forward costs 18% of throughput and 0.3 GB, which is close to the predicted 3.5 ms per image and nothing else. The cost of the tower was never the problem. The value of the tower is.

### 16:20 - arm 14 launched: per-tower normalisation, and the fusion defect was mine

`model.py` now normalises each tower's vector separately before the concat instead of one `LayerNorm` over the 1536-wide concatenation, which had forced DINOv3's already-normalised cls and SigLIP2's raw pooling output through a shared scale. Verified on CPU: the single-tower path is bit-identical in shape and state (norm 768, head 768, 231 MB), and the pair is norm 768 plus norm2 768 into a 1536 head, 269 MB, with a clean state-dict round trip.

`u1-C-L-asl-g2.0+siglipb`, `runs/03612c5c`, job `pwsh-483ab714130b4361`, gamma 2.0 with per-tower norm. Two differences from arm 13, not one, which is a compromise: the config hash covers only the config fields, and arm 13's file and this one would hash identically if only the code changed, which would put two different architectures in one run directory. Using the better gamma rather than inventing a field keeps the identity honest and makes the comparison that matters the right one, against `u1-C-L-asl-g2.0` at 0.0624 with one change, the presence of the second tower.

If this still loses to 0.0624, the ensemble line closes on evidence rather than on speculation, and the remaining budget goes to the loss axis at gamma 1.0 and to the representation, which is where every real gain has come from.

Also done while arm 14 ran: `TrainConfig.loss_gamma` now defaults to 2.0 instead of 4.0, so `docs/best-settings.md` and the code agree again and a new config that forgets the field gets the measured-best value. Configs that state the field keep their files and their behaviour; the hash churn this time reaches only configs that omit it, and every phase-3 file states it. `docs/best-settings.md` gained section 3b with the two-point table, and `docs/benchmark.md` gained section I with today's four measured throughputs, including the correction that the per-block ladder over-predicts by 25 img/s because the adapter's backward traverses frozen blocks.

### 16:22 - my explanation for the two-tower loss was measured and does not hold

I blamed the shared `LayerNorm(1536)` on a scale mismatch between the two towers. Measured on four images, it is not there: DINOv3's pooled cls has mean absolute value 0.496, per-dimension standard deviation 0.029 and L2 norm 17.30, while SigLIP2's MAP output has 0.246, 0.031 and 14.00. The norms differ by a factor of 1.24, so the shared normalisation was not fighting a large scale difference and the per-tower split is a correctness tidy-up rather than the explanation.

So the 13.6% has to come from somewhere else, and the two remaining candidates are ordinary dilution and capacity: a fixed 9.60M head at 1536 inputs has to serve two unrelated feature spaces with 5200 steps on 10k images, and the mechanism would be the same one that made Qwen dilute the pair in the probe, 5120 of 6656 dimensions belonging to the weakest signal. Arm 14 will say whether the per-tower normalisation moves anything at all, but the prediction I am writing down before seeing it is that it moves the result by less than the 3% band.

The struck-out page stays: the cause was a hypothesis, cheap to state, cheap to test, and wrong.

### 17:08 - arm 14 scored: per-tower norm recovered 12.6% of the loss, and the ensemble is now exactly a tie

`u1-C-L-asl-g2.0+siglipb`, `runs/03612c5c`, final checkpoint, 50k at 0.40, from `validation.json`. 44 minutes at 106 img/s and 17.1 GB.

| arm | gamma | fusion norm | macro F1 | macro P | macro R | micro F1 |
|---|---:|---|---:|---:|---:|---:|
| u1-C-L-asl-g4.0 | 4.0 | single, over concat | 0.0632 | 0.0663 | 0.0791 | 0.3713 |
| u1-C-L-asl-g2.0 | 2.0 | single, one tower | 0.0624 | 0.0924 | 0.0591 | 0.4156 |
| arm 13, two towers | 4.0 | single, over concat | 0.0546 | 0.0693 | 0.0610 | 0.3652 |
| arm 14, two towers | 2.0 | per tower | 0.0615 | 0.0960 | 0.0570 | 0.4159 |

Against the arm it has to beat, `u1-C-L-asl-g2.0`, the two-tower model is a dead heat: macro -1.4% and inside the band, micro +0.07% and flat. So the second tower is not a loss and not a gain. It costs 18% of throughput and 0.3 GB and returns nothing measurable.

Against arm 13 it gained 12.6% macro and 13.9% micro, but that comparison is confounded: two things changed, the gamma and the norm. Splitting the known gamma effect, which at single tower was -1.3% macro and +11.9% micro, leaves the per-tower norm worth roughly +14% macro and +2% micro. So the fusion defect was real and the earlier prediction that it would move less than the band was wrong. The shared `LayerNorm` was not a scale mismatch, but it still coupled two feature manifolds through one affine, and removing that was worth more than everything the second tower contributed.

The honest summary of the ensemble line: two towers with proper late fusion, at the better gamma, land on the same numbers as one tower. Fusion that concatenates is dead. Anything that revives it has to do actual work, a per-tag gate, cross-attention over both towers' tokens, or a learned query set, because the information is apparently not in the concatenation. That is 13 to 20M parameters of fusion machinery and a real day of work, not a one-line change.

Which raises the better question this whole line should have asked first: we have never adapted SigLIP2. Its frozen features beat DINOv3's frozen features by 20 to 21% macro, and every arm in the ledger adapts DINOv3. If SigLIP2 adapts as well as DINOv3 does, its starting point is 20% higher and the whole project is built on the wrong tower.

### 17:35 - SigLIP2 is loadable as the adapted tower, verified on CPU, and both next arms are staged

`model.py` no longer assumes DINOv3. `backbone_kind(path)` reads the checkpoint's own `config.json` and returns `siglip2` for anything whose `model_type` starts with `siglip`, so the tower is chosen by which directory `cfg.backbone_dir` names and no new config field was needed, which means no third hash churn in one day. `backbone_layers()` returns `.model.layer` for DINOv3 and `.encoder.layers` for SigLIP2. The patch offset is 5 for DINOv3 and 0 for SigLIP2, because SigLIP2 has neither a cls token nor registers. The input normalisation becomes 0.5/0.5 for SigLIP2, which makes the existing `_prep` produce exactly `(x/255 - 0.5)/0.5`.

Measured before writing any of it, on the real checkpoint: `SiglipVisionModel` in transformers 5.16.1 has children `embeddings`, `encoder`, `post_layernorm`, `head`, with the blocks at `encoder.layers` and 12 of them, hidden 768, intermediate 3072, patch 16, image 512. `last_hidden_state` is `(B, 1024, 768)` with no cls token, and `pooler_output` is `(B, 768)` produced by a `SiglipMultiheadAttentionPoolingHead`, which is an attention pooler with its own attention, layernorm and two-layer MLP, about 2.3M parameters.

Smoke tested on CPU at batch 2 in three pooling modes, all producing `(2, 12476)`:

| tower | OFT params | modules | head | blocks | backbone | state |
|---|---:|---:|---:|---:|---:|---:|
| DINOv3-B/16, u1, cls | 4.88M | 73 | 9.60M | 7.09M | 91.03M | 231 MB |
| SigLIP2-B/16, u1, cls | 5.17M | 76 | 9.60M | 7.09M | 99.13M | 248 MB |

The 76 modules are 72 block Linears, the patch-embedding conv, and three inside the MAP head: two MLP Linears and the attention output projection. So under SigLIP2 the trainable adapter count rises by 0.29M and the `cls` pooling vector is not a bare LayerNorm like DINOv3's but the output of a trainable-attention pooler whose base weights stay frozen while its rotations adapt.

Staged, both with the measured-best loss and one factor away from the champion:

`15_u1-C-L-asl-g1.0.json`, `83a01eed`, running, 108 img/s and 16.8 GB, the loss axis one step lower on the champion.

`16_u1-C-L-asl-g2.0-siglip2base.json`, `66e841ba`, the champion's recipe with `backbone_dir` pointing at `I:\Models\siglip2-base-patch16-512` and nothing else changed, so its 50k macro against 0.0624 isolates the tower. No `siglip_dir`, so it is a single tower and the 18% throughput tax does not apply. This is the arm that decides whether the project is built on the right backbone, and it was never run because the ledger started with DINOv3 and stayed there.

### 17:55 - arm 15 scored: the gamma dial is bracketed at three points and 2.0 is its peak on macro

`u1-C-L-asl-g1.0`, `runs/83a01eed`, final checkpoint, 50k at 0.40, from `validation.json`. 35.1 minutes at 130 img/s and 16.8 GB.

| gamma | macro F1 | macro P | macro R | micro F1 | micro P | micro R |
|---:|---:|---:|---:|---:|---:|---:|
| 4.0 | 0.0632 | 0.0663 | 0.0791 | 0.3713 | 0.2919 | 0.5101 |
| 2.0 | 0.0624 | 0.0924 | 0.0591 | 0.4156 | 0.3837 | 0.4533 |
| 1.0 | 0.0575 | 0.1073 | 0.0474 | 0.4335 | 0.4573 | 0.4120 |

One clean monotone trade over the whole range: as gamma falls, precision and micro rise, recall and macro fall. 1.0 costs 7.8% macro, outside the band, and buys 4.3% micro. Since the bar is 0.5579 macro and 0.6886 micro, and we are about twice as far below on macro as on micro, that trade is the wrong direction and 2.0 stands as the operating point.

Extrapolating the other way, 3.0 would sit between 4.0 and 2.0 on both axes and dominate neither: roughly macro 0.0628 with micro 0.393 against 2.0's 0.0624 and 0.4156. So the gamma axis is closed. Three arms, 105 minutes, and the answer is 2.0 with no further work owed to it.

This also settles the shape of the loss: ASL's hard-negative focusing was buying recall at the fixed 0.40 threshold, and every step away from it moved probability mass toward the tail at the expense of the bulk. The bulk is what micro measures. At 12476 labels with 0.26% positives, the bulk is also what the bar wins on micro with 0.7033 precision.

### 18:00 - the capacity question, made cheap before it is made expensive

Arm 16 is running on the SigLIP2 backbone. While it does, I set up the test that should come before any 45-hour ViT-L run, because the ladder's next gate is about parameters and a training run is the wrong instrument to settle it.

`scratch/sophie/siglip_probe.py` grew a `vitl` tower, which is the same `DinoTagger` feature path with `backbone_dir` pointing at `I:\Models\dinov3-vitl16-pretrain-lvd1689m`. Verified on GPU at batch 4: 24 blocks, hidden 1024, 145 OFT modules, 1024-dimensional cls features, 0.9 GB. Frozen ViT-L features on the same 10k and 50k caches, then the same head recipe already run for ViT-B, give a direct answer to whether a bigger tower carries more tag information per image.

The comparison to beat is ViT-B frozen: 0.0380 macro and 0.1754 micro at width 768, 0.0475 and 0.2305 at a 2048 bottleneck. If frozen ViT-L clears those by more than the 3% band, capacity is the binding constraint and the main run is ViT-L at 44.8 hours per pass. If it lands within the band, the gap to the bar is about adaptation and recipe, and two days on ViT-L would buy almost nothing.

The honest caveat, since this instrument has now misled us three times in a row: a frozen-feature probe ranks representations, not recipes, so a positive result here is necessary but not sufficient. It is however exactly the right shape for a size comparison, same recipe, same head, one backbone swapped, and it costs forty minutes instead of two days.

### 18:40 - arm 16 scored: SigLIP2 is the better tower by 21% macro, and the base model changes

`u1-C-L-asl-g2.0-siglip2base`, `runs/66e841ba`, final checkpoint, 50k at 0.40, from `validation.json`. 33.9 minutes at 138 img/s and 15.8 GB, the same recipe as the champion with `backbone_dir` pointed at `I:\Models\siglip2-base-patch16-512` and nothing else touched.

| tower | macro F1 | macro MCC | macro P | macro R | micro F1 | micro P | micro R | img/s | VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DINOv3-B/16 u1 | 0.0624 | 0.0661 | 0.0924 | 0.0591 | 0.4156 | 0.3837 | 0.4533 | 130 | 16.8 GB |
| SigLIP2-B/16 u1 | 0.0755 | 0.0786 | 0.1010 | 0.0730 | 0.4375 | 0.4061 | 0.4742 | 138 | 15.8 GB |

+21.0% macro, +5.3% micro, both outside the band. The semantic tower wins on the metric the goal names and on the other one too, and it is simultaneously faster and leaner per step, because SigLIP2 has no cls token and no registers to carry and its pooler is a small attention head rather than a wide head over four pooled parts.

The champion chain, all at 50k and 0.40: h10 u12-CMSS-g4.0 0.0410, u12-C-L-g4.0 0.0646, u1-C-L-g4.0 0.0632, u1-C-L-g2.0 0.0624, and now SigLIP2-B u1-C-L-g2.0 at 0.0755 macro with the best micro any arm has produced at 0.4375. Against the original control that is +84% macro and +4.7% micro from pooling, loss, freeze depth and the tower together, with the tower alone worth +21%.

Two things to say about how much to believe it. It is one seed, so strictly it is inside the noise question, but the measured band is 2 to 3% and this is 21%. And the 1k subset predicted it, 0.0438 against DINOv3's 0.0394 at step 1248, which is the first time in the project that the 1k number has agreed with the 50k number. That is a fact about this arm, not a licence to trust the 1k number again.

What it changes. The main run is SigLIP2. The capacity question stops being about DINOv3-L: it is about `siglip2-large-patch16-512`, 24 blocks and 1024 wide, which is the same family one size up and therefore comparable at constant recipe, where DINOv3-L would have been confounded with a tower swap. The ViT-L feature probe staged at 18:00 is now the wrong instrument, since it would compare a tower family we are abandoning against the one we just adopted.

What is not yet known about the new tower. Its `cls` vector is the output of a MAP attention pooler trained for image-text alignment, which may summarise away exactly the small-object detail that booru tags need, and mean or max pooling over its 1024 patch tokens might beat it. Its last block is unfrozen and the rest adapted by OFT at block 128 on 76 modules, but whether the optimum freeze depth is still 1 for a tower whose features are this semantic is untested. Neither question existed for DINOv3 because DINOv3's pooling vector was a bare LayerNorm and its features were structural.

Staged next, hashed `c8c22df5`: `17_u1-M-L-asl-g2.0-siglip2base.json`, identical to arm 16 except `head_pool` is `mean`, which drops the attention pooler and reads the 1024 patch tokens directly, 768-wide head, same everything else. It isolates the pooler.

### 21:20 - arm 17 scored: the MAP pooler earns its place, and mean pooling loses 3.2%

`u1-M-L-asl-g2.0-siglip2base`, `runs/c8c22df5`, final checkpoint, 50k at 0.40, from `validation.json`. 34.0 minutes at 136 img/s and 15.8 GB, one field different from arm 16.

| pooling | macro F1 | macro P | macro R | micro F1 | micro P | micro R |
|---|---:|---:|---:|---:|---:|---:|
| `cls`, the MAP attention pooler | 0.0755 | 0.1010 | 0.0730 | 0.4375 | 0.4061 | 0.4742 |
| `mean` over 1024 patch tokens | 0.0731 | 0.1021 | 0.0700 | 0.4315 | 0.4075 | 0.4585 |

-3.2% macro, which is at the edge of the band, and -1.4% micro, inside it. So the hypothesis I wrote down before running it, that a pooler trained for image-text alignment would throw away the small-object detail tags need, is not supported at the resolution this benchmark can see. SigLIP2's attention pooler is at least as good as reading every token and letting the head decide, and it is cheaper, one 768-wide vector either way.

Two details worth keeping. The mean-pool arm's precision is fractionally higher and its recall lower, the same trade shape as the gamma dial, which suggests the pooler is not losing information but is choosing a different operating point. And the 1k subset called it again, 0.0409 against 0.0433 at step 1040, which is two arms in a row where the small subset agreed with the 50k, after a day where it agreed with nothing.

Decision: `head_pool: cls` stands and the main run is unchanged. The last gate before the main run is now closed, and it closed in favour of the config that was already written.

### 21:44 - the 600k gate is running, and the stream is faster than the estimate

Before committing 26 hours, the intermediate gate from `docs/main-run.md`: 600,000 images from the real tar stream in one pass, 12,500 steps, batch 48, validation on the 10000-image nested subset every 1250 steps. Two configs, one change apart, and the change is the freeze depth, which 10k images decided by 2.2% against a 3% band and therefore did not really decide at all:

`18_u1-C-L-asl-g2.0-siglip2b-600k.json`, `runs/2b6b81aa`, one unfrozen block, running now.

`19_u12-C-L-asl-g2.0-siglip2b-600k.json`, `runs/77a135b2`, twelve unfrozen blocks, queued behind it.

The question is whether being data-starved was the reason u1 looked as good as u12. At 10k images the model saturates around step 1000 and then declines, so a lighter adapter wins by overfitting more slowly rather than by being better. At 600k images, sixty times the data per step and 2.4 times the steps, underfitting is the regime instead, and twelve adapted blocks have room to earn their 18% throughput cost. If u12 wins by more than the band, the main run switches to twelve blocks and costs 32 hours instead of 26. If it does not, u1 stands and the 26-hour estimate holds.

Measured on the way in and worth recording because it corrects the plan: the stream runs this tower at 114 img/s at one unfrozen block, not the 95 to 110 that `docs/main-run.md` assumed from the paused run's 83 at twelve. So the gate is 88 minutes for u1 rather than 100, and two full passes are 26 hours rather than 28 to 32.

### 22:25 - the large tower is on disk and staged, because capacity is the question the ladder has not asked

Downloaded `google/siglip2-large-patch16-512` to `I:\Models\siglip2-large-patch16-512`: 3.53 GB of safetensors, hidden 1024, intermediate 4096, 24 blocks, 16 heads, 512px, model type siglip so `backbone_kind` detects it without any new field.

`20_u1-C-L-asl-g2.0-siglip2large.json`, hashed `2ea85651`, batch 20 because 24 blocks at 1024 wide do not fit 48 in 24.5 GB, 25 epochs at 500 steps each, which is 12,500 steps and 250000 images: the same image budget as arm 16's 5200 steps at batch 48, so the comparison isolates parameters rather than data or schedule.

Why this before the main run. The bar buys its 0.5579 macro with 692.6M parameters. We have compared an 86M tower against a 93M tower, found the second better, and never asked whether the family continues to pay. The two candidate answers are very different: if large beats base by more than 3% at equal images, the main run should be large, and a base main run would have spent 26 hours on the wrong tower. If large lands inside the band, capacity is not the lever on our data and the base main run stands on evidence rather than on assumption.

Two honest caveats before it runs. Batch 20 against 48 means noisier gradients at the same learning rate, so a loss could be the batch rather than the tower, and the fair read is against a batch-20 base arm if the number is close. And this is a 175-minute arm at roughly 33 img/s, not the 35-minute ones we have been running, so it must not be started while anything else holds the card.

Also noted and fixed while staging: my first attempt at this config wrote the tower path with doubled backslashes because PowerShell does not use backslash escapes the way JSON does. It resolved anyway, Windows collapses separator runs, but the hash differs from the correct file, so the fixed `2ea85651` is the one to use and the earlier `feb17953` is a dead hash that names nothing.

### 23:18 - the 600k gate, u1: 97% of the paused run's macro from 14% of its images, and no sign of a plateau

`18_u1-C-L-asl-g2.0-siglip2b-600k`, `runs/2b6b81aa`, one pass over 600000 images from the real tar, 12500 steps, best checkpoint at step 11250, scored on the 50k at 0.40 from `validation.json`. 93.5 minutes at 114 img/s and 15.8 GB.

| run | tower and recipe | images trained on | macro F1 | macro P | macro R | micro F1 | micro P | micro R |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| paused `4a13a120` | DINOv3 u12 CMSS asl 4.0 | 4,138,368 | 0.2815 | 0.4608 | 0.2266 | 0.6133 | 0.7068 | 0.5417 |
| gate `2b6b81aa` | SigLIP2 u1 cls asl 2.0 | 600,000 | 0.2733 | 0.2960 | 0.3213 | 0.4791 | 0.3519 | 0.7503 |

Six point nine times fewer images and one twelfth of the wall clock, for 97% of the macro. That is the single clearest statement of how much the recipe work bought: the old configuration needed 4.1M images and about thirty hours to reach a margin the new one reaches in an hour and a half.

The precision and recall columns are the other half of the story and they are not a win yet. Macro precision 0.2960 against 0.4608 and macro recall 0.3213 against 0.2266: our checkpoint is far more liberal at the fixed 0.40 threshold, predicting many more positives. Micro recall 0.7503 against 0.5417 with precision 0.3519 against 0.7068 says the same thing. This is what a model with 600000 images of training looks like at a threshold nobody tuned for it, and it is the strongest argument for the full run: with 5.3M images the probability mass should sharpen and the balance at 0.40 should move toward the bar's shape.

The internal 10k curve never bent, which is the fact the 10k benchmark could not produce:

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

Monotone, accelerating if anything, and still climbing when the schedule ended. The 10k arms peaked around step 1040 and declined for four thousand steps; at 600k images there is no peak, only a slow climb. So the entire 10k benchmark was measuring overfitting, not capability, and the freeze-depth answer it gave was measured in the wrong regime. That is what this gate existed to find and it found it before a 26-hour run, not after.

Which makes the u12 half of the gate the live question: `19_u12-C-L-asl-g2.0-siglip2b-600k.json`, `runs/77a135b2`, same 600000 images, same everything, twelve unfrozen blocks. If a lighter adapter was winning at 10k only because it overfit more slowly, twelve blocks should now win by more than the band, and the main run changes shape and cost.

### 23:45 - the u12 half is not a capacity test, and the parameterisation is why

Mid-run, at step 2920 of 12500, the twelve-block arm is far behind and the gap is not small:

| step | u1 macro | u12 macro | u1 micro | u12 micro |
|---:|---:|---:|---:|---:|
| 1250 | 0.0680 | 0.0086 | 0.4572 | 0.3383 |
| 2500 | 0.0970 | 0.0179 | 0.4702 | 0.3748 |

Five to eight times behind at the same number of images, and this is not a subtle capacity effect that might reverse. The cause is in the parameterisation, not in the tower: `unfreeze_last_blocks` deliberately freezes the OFT rotation inside every block it unfreezes, on the grounds that a rotation beside an unfrozen weight is redundant. So u12 trains 85.05M base weights at `lr_unfreeze` 3e-5 with only 0.34M live OFT parameters, while u1 trains 7.09M base weights plus all 5.17M of its rotations at 5e-4. The u12 arm is not more capacity at the same step size, it is a different optimiser path with a seventeen times smaller effective step.

At 10k the two landed 2.2% apart because the run was short and saturated early, so the difference in learning speed never had time to show. At 600000 images it shows immediately. Which means the 10k benchmark compared two things it could not distinguish, and I compared them as though it had.

The consequence for the main run is the decision it actually needed: u1 stands, and by a margin wide enough that no late catch-up changes it. The consequence for the ledger is a correction to the paused run's interpretation: `4a13a120` reached 0.2815 macro over 4.14M images at 3e-5 with no rotation in its twelve unfrozen blocks, so its learning rate was slow by construction, and part of what looked like a capacity ceiling was a step-size ceiling.

The clean version of the capacity question, if anyone wants it later: u12 with `lr_unfreeze` around 3e-4, or u12 with the rotations left trainable inside the unfrozen blocks, which is the redundancy the design avoided and the thing that makes u1 work. Neither is staged, and neither blocks the main run.

### 01:06 - the gate closes: u1 by 2.65x, and the freeze-depth axis needs a different parameterisation before it can be asked again

`19_u12-C-L-asl-g2.0-siglip2b-600k`, `runs/77a135b2`, same 600000 images and same recipe as the u1 half, final checkpoint scored on the 50k at 0.40 from `validation.json`. 101 minutes at 100 img/s and 18.5 GB, against u1's 93.5 minutes at 114 img/s and 15.8 GB.

| arm | unfrozen blocks | live OFT | frozen base | macro F1 | macro P | macro R | micro F1 | micro P | micro R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `2b6b81aa` | 1 | 5.17M | 7.09M | 0.2733 | 0.2960 | 0.3213 | 0.4791 | 0.3519 | 0.7503 |
| `77a135b2` | 12 | 0.34M | 85.05M | 0.1031 | 0.1304 | 0.1237 | 0.4054 | 0.2965 | 0.6408 |

2.65 times worse on macro and 15% worse on micro, for 8% more wall time and 2.7 GB more memory. The curve never threatened to converge: at matched steps u12 was five to eight times behind from the first validation onward, and its own end-of-run ratio to u1's midpoint was still 0.36.

So the answer the gate can give is unambiguous and the answer it cannot give matters more. More unfrozen blocks at this parameterisation are worse, decisively, at 600000 images as at 10000. But the two arms do not train at the same step size, so this is not evidence that capacity does not help; it is evidence about our optimiser. `unfreeze_last_blocks` freezes the OFT rotation in every block it unfreezes, so twelve blocks means 85.05M weights at 3e-5 with the adapter count collapsing to the patch embedding's 0.34M, against 7.09M weights at 3e-5 plus 5.17M rotations at 5e-4. Any future freeze-depth arm must first make the two paths comparable, either by raising `lr_unfreeze` to roughly 3e-4 or by leaving the rotations trainable inside unfrozen blocks, and until one of those runs, the ledger should say only what it measured: this parameterisation, at two scales, loses.

What the gate bought for the main run is the recipe confirmation at scale. One unfrozen block, SigLIP2, cls, ASL 2.0, 600000 images, 93.5 minutes, macro 0.2733, curve still climbing, 97% of the paused full-corpus run's macro from 14% of its images. That is enough to launch the full run with a defensible expectation rather than an extrapolation from ten thousand images.

### 01:10 - the main run is live: `runs/3da79e14`, two passes, 221,739 steps

Command, verbatim, because three of its four flags are load-bearing and none of them live in the config:

```
python -m wdtag.train --config configs\main\01_u1-C-L-asl-g2.0-siglip2b-main.json --select-metric macro040 --val-every-steps 10000 --ckpt-every-steps 5000
```

Confirmed at launch: sha8 `3da79e14`, `params: OFT 5.17M, head 9.60M, unfrozen 7.09M in 1 block(s), backbone 99.1M, 76 OFT modules`, `loss asl: gamma_neg 2.0`, `[stream] one pass over 10,643,426 images in 221,739 steps, batch 48, 8 decode threads`, 116-122 img/s at 15.8 GB.

Watch points. 50k validation and a `best_val.pt` selection every 10000 steps, 22 of them, about 190 seconds each. Checkpoints every 5000 steps, 44 files at about 248 MB, 11 GB. Expected wall 24 to 26 hours, so a finish somewhere around 2026-10-24 02:00 to 04:00 local, and the run is resumable from any checkpoint with the same command if it is reaped or paused.

What to read as it goes, in order of importance. Macro at 0.40 on the 50k, against 0.2733 from the 600k gate and 0.2815 from the paused full-corpus run. Micro precision, which was 0.3519 in the gate and needs to climb toward the bar's 0.7033 for the model to be calibrated at a fixed threshold rather than merely well ranked. And whether the second pass is still adding macro, because the bar's own history says 97% of its final macro arrived in its first epoch and the remaining 94 bought 2.8%, so if our second pass is still gaining hard we are in a different regime and worth continuing past it.

Nothing else will run on this card until it finishes or is paused. The next arm, if the first pass ends short, is the large tower at `2ea85651`, and if it ends near the bar, the open question becomes whether a second pass is worth another 26 hours before we spend them.

### 03:10 - main run paused on purpose at step 15373, and it is still a candidate

User instruction, not a crash. The run checkpointed at `ckpt_step015373.pt` and exited cleanly within one step of the `.PAUSE` file appearing, which is the pause path working exactly as documented. On disk: four checkpoints at 5000, 10000, 15000 and 15373, plus `best_val.pt` from step 10000 at macro@0.40 0.1775 and micro 0.5164. `.PAUSE` left in place deliberately. `runs/3da79e14/PAUSED.md` holds the reason, the exact resume command and the warning about the `.PAUSE` file, so the next person to list that directory cannot miss it.

Why it was paused instead of left alone: `lr_unfreeze` 3e-5 has never been bracketed and it is what the run's only unfrozen block has been stepping at, and the fully frozen backbone has never been priced on any tower at all. If either beats the current setting, 22 more hours at this recipe is the wrong 22 hours. If neither does, this run resumes at 15373 with its cursor intact, which is precisely why pausing beats killing.

The user's own read, recorded because it is a real prediction and not a formality: their money is on this run or on the frozen backbone.

### 03:12 - the ablation's first leg is u0, the frozen backbone, because it can cancel the other two

Three arms were staged, all at 600000 images, all one field different from the gate's `2b6b81aa`: `c75d34d4` with `lr_unfreeze` 1e-5, `20142533` with 1e-4, `b6e293f0` with `unfreeze_blocks: 0`.

`b6e293f0` runs first. The reasoning is that the LR bracket only has meaning if an unfrozen block is worth having at all: if the frozen-plus-adapter regime ties or wins, `lr_unfreeze` becomes a knob that governs nothing, the two bracket arms are unnecessary, and the whole depth ladder collapses into one configuration that is also the fastest we can build. Running the arm that can cancel two others before running the two it might cancel is the cheap order, and it is also the arm that has never existed on any tower.

### 04:38 - the frozen backbone ties the unfrozen one, so freeze depth is closed and `lr_unfreeze` governs almost nothing

`23_u0-C-L-asl-g2.0-siglip2b-600k`, `runs/b6e293f0`, zero unfrozen blocks, 5.61M live rotations plus the 9.60M head and nothing else. 75 minutes at 118 img/s and 15.8 GB. Scored on the 50k at 0.40 from `validation.json`.

| arm | unfrozen | live rotations | live base weights | macro F1 | macro P | macro R | micro F1 | micro P | micro R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `2b6b81aa` u1 | 1 | 5.17M | 7.09M | 0.2733 | 0.2960 | 0.3213 | 0.4791 | 0.3519 | 0.7503 |
| `b6e293f0` u0 | 0 | 5.61M | 0 | 0.2694 | 0.2958 | 0.3151 | 0.4729 | 0.3466 | 0.7438 |

Minus 1.4% macro and minus 1.3% micro, which is inside the band by any reading we have used. The unfrozen last block, 7.09M parameters and 12.6% of the trainable set, is worth about 1.4% of macro and nothing measurable in micro. The user predicted this outcome alongside the current configuration and the prediction held.

The internal curves explain why the margin is that small and why the early read was misleading in the opposite direction:

| step | u0 macro | u1 macro | u0 micro | u1 micro |
|---:|---:|---:|---:|---:|
| 1250 | 0.0740 | 0.0680 | 0.4617 | 0.4572 |
| 2500 | 0.1019 | 0.0970 | 0.4684 | 0.4702 |
| 3750 | 0.1211 | 0.1192 | 0.4882 | 0.4886 |
| 5000 | 0.1414 | 0.1416 | 0.5065 | 0.5099 |
| 6250 | 0.1597 | 0.1627 | 0.5096 | 0.5111 |
| 8750 | 0.1958 | 0.2001 | 0.5280 | 0.5372 |
| 11250 | 0.2223 | 0.2278 | 0.5464 | 0.5531 |

u0 is ahead for the first two thousand steps, level by five thousand, and behind by 2% from then to the end. A frozen tower with a good adapter learns faster in the first hour and plateaus very slightly lower, which is a shape worth remembering because it is the third time today an early lead pointed the wrong way.

What this closes and what it leaves open. The freeze-depth axis is done: u0 and u1 are the same number within noise, u12 in this parameterisation is 2.65 times worse, and the only reason to freeze fewer blocks was ever throughput, which u0 also wins by a hair. The `lr_unfreeze` bracket, arms `c75d34d4` and `20142533`, was staged to tune a learning rate that governs 1.4% of the result, so it drops from third in the queue to last, and probably to never. `lr_oft` 5e-4 is what actually matters in this architecture and that knob has a measured recommendation from the head sweep.

No change to the paused main run. It keeps one unfrozen block, its recipe is 1.4% better than the alternative and inside the band of everything, and restarting it to reach u0 would spend two hours to buy nothing.

### 05:52 - 384 resolution costs 6.6% macro, not the 53% the in-run curve claimed, and the subset lied again

`24_u1-C-L-asl-g2.0-siglip2b-384-600k`, `runs/2a5c9bad`, SigLIP2-B's own 384 checkpoint at 384 input, 600000 images, batch 48, 160-171 img/s at 9.1 GB. Scored on the 50k at 0.40 from `validation.json`.

| arm | resolution | img/s | VRAM | macro F1 | macro P | macro R | micro F1 | micro P | micro R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `2b6b81aa` | 512 | 114 | 15.8 GB | 0.2733 | 0.2960 | 0.3213 | 0.4791 | 0.3519 | 0.7503 |
| `2a5c9bad` | 384 | 160-171 | 9.1 GB | 0.2553 | 0.2823 | 0.2975 | 0.4723 | 0.3488 | 0.7315 |

Minus 6.6% macro and minus 1.4% micro for 1.45 times the throughput and 40% less memory. The speedup is 1.45x and not the 2.15x the FLOPs ratio predicts, because about 3 of the 8.8 ms each 512 image costs is tar read, jpeg decode and step overhead that resolution does not touch; at large-tower scale the compute share is bigger and the gain should approach 2x.

The correction worth recording, because it is the third time today: the in-run validation on the 10000-image prefix said 384 was at 0.1069 against 512's 0.2278, a 53% deficit, and it was wrong by a factor of eight. On the 50k it is 6.6%. The 10k prefix has support for only a few thousand of the 12476 labels, so macro there is dominated by labels with one or two positives, and those are precisely the tail labels that lose their pixels at 384. A subset that over-weights the tail is not a fast proxy for the 50k when the effect being measured is a tail effect.

Decision: the main line stays at 512. 6.6% macro is too much to pay while the gap to the bar is measured in tens of percent, but 384 is not dead, it is priced. If a long schedule ever has to fit in half the wall clock, 384 buys 1.45x for 6.6% and that may be the right trade on a tower whose full pass costs days.

Caveat that flatters 512 and is recorded for honesty: no 384 nested cache existed, so I built one by bilinearly downsampling the 512 cache with `scratch/sophie/build_nested_384.py`, which is one resample more than the training stream saw. The 384 arm was therefore scored on softer images than it trained on, so 0.2553 is a floor and the true cost may be slightly under 6.6%.

### 06:00 - the large tower at 512 is running: `runs/62de5705`

`26_u0-C-L-asl-g2.0-siglip2large-600k.json`, sha8 `62de5705`, SigLIP2-L/16-512, batch 20, 600000 images, 30000 steps, u0 so only the adapter and head train. Learning rates scaled for depth, width and batch: `lr_oft` 2e-4 against base's 5e-4, `lr_head` 7e-5 against 1.2e-4, `warmup_steps` 300 rather than the doc's 4000 because 4000 was 1.5% of a 266000-step run and this gate is 30000.

Confirmed at launch: `params: OFT 14.48M, head 12.79M, unfrozen 0.00M in 0 block(s), backbone 331.2M, 148 OFT modules`, 39 img/s at 19.2 GB, so about 4.3 hours. Loss was 0.0053 at step 200 with the learning rate still warming, which is the healthy shape and not the NaN the abandoned ViT-L probe produced at `lr_oft` 5e-4.

The mate for this arm is base-u0 at 0.2694 and, secondarily, base-u1 at 0.2733, so a large-tower gain can be read without unpicking the freeze-depth change.

### 02:28 - main run, first 50k validation at step 10000: macro 0.1775, micro 0.5164

480000 images seen, learning rate still at peak, 22% of the way through the first pass.

| step | images | macro | micro | macro@0.40 |
|---:|---:|---:|---:|---:|
| 10000 | 480k | 0.1792 | 0.5164 | 0.1775 |

Read this against nothing yet. It looks worse than the 600k gate's 0.2733 at 540000 images, and that comparison is invalid for a reason worth writing down before the number gets quoted: the gate's cosine schedule spanned 12500 steps, so at 90% of its run its learning rate had annealed to a seventh of peak. The main run's cosine spans 221739 steps, so at 480000 images it is still at 4.98e-4 of 5e-4. Late-schedule weights are decisively better than mid-schedule weights at equal data, and the gate was effectively a short annealed run while this is the same data volume early in a long one.

What the number does say. Micro 0.5164 is already above the gate's final 0.4791 on a sixth of the gate's data, which supports the calibration reading: the bulk of the labels sharpen early while the tail needs the schedule to anneal. Macro 0.1775 with a best threshold of 0.50 and macro@0.40 at 0.1775, so at this point the operating point is not yet distorted at all.

Also measured, because it sets the run's overhead: one 50k validation costs 284.7 seconds, not the 190 estimated. Twenty-two of them is 105 minutes, 4.4% of the wall clock, and every one of them is worth it against a 26-hour run whose checkpoint choice depends on them.

### 18:42 - the main run can now select on macro@0.40, and one smoke test is owed

`--select-metric macro040` exists. `SELECT_METRICS` maps it to `macro_f1_at_0.40`, which the in-run validation has computed and printed since the first arm but never used for selection. Two defects had to be fixed for it to work at all, and both were live bugs in the existing path:

The threshold summary's values live in `extra`, and the selection read `metrics.get(sel_key)`, so any threshold-derived metric would have resolved to NaN and selected nothing, silently, forever. It now reads the merged record.

The `best` dict only carried `val_macro_f1`, `val_micro_f1` and `val_supported_macro_f1`, so `best_value` had no key to compare a 0.40 threshold score against. It now carries `val_macro_f1_at_0.40` as well.

Verified by unit: `threshold_summary` on a hand-built 4x2 matrix returns the expected 1.0 at every threshold, and `best_value` resolves all four metrics including the new one. Not yet verified end to end, and the honesty matters because this is the switch the main run will use: the first attempt reused arm 16's config, which hashed to arm 16's existing run directory, so the resume logic saw step 5200 against `--max-steps 215` and exited before training a single step. Nothing was damaged, `best_val.pt` still holds step 1456 and the log still ends at 5200, but the smoke proved nothing. The retry, with a fresh seed so it lands in its own directory, was cut short when the card was needed, so a 220-step run with `--select-metric macro040` is owed before the main run starts, and until it passes, the main run selects by `macro` from habit rather than by the protocol.

That smoke passed at 20:15, at batch 4 so it could not take the card from anything else. `runs/671e6989`, seed 4242, 205 steps, `--val-every-steps 100 --select-metric macro040`. Both validations fired mid-epoch, which is the knob stream mode needs, and both selected: `new best val macro_f1_at_0.40 0.0023` at step 100 and `0.0044` at step 200. The 0.0023 against a best-threshold macro of 0.0033 at the same step also shows the two numbers genuinely differ, which is why the protocol needs its own metric rather than a default. The card stayed at 1.5 GB throughout and `runs/671e6989` is a smoke directory, not an arm: batch 4, 205 steps, no score worth quoting.

### 13:40 - the arm after next, decided before it is built: two towers, both adapting

The user's design, after the cache question was settled: OFT at `block_size` 128 inside both towers, and the last block of each tower unfrozen. That is `u1` applied twice, on DINOv3-B/16 and SigLIP2-B/16-512, head 1536 -> 12476.

It is not cacheable, and that is the point of writing it down. A cache is only valid if the tower's output cannot change during the run, so frozen-with-no-adapter is cacheable and frozen-with-OFT-inside is not, because the rotation changes the features after every optimizer step. `block_size` 128 also has the better accuracy evidence of the two: b=128 beat the b=32 run at step 5000 on micro, macro and supported macro, though confounded with exact Cayley and the fp32 head. Block 64 would halve the adapter to 2.64M per tower and also halve the resolution at which information crosses dimensions, 768/64 being twelve independent groups, which `docs/oft-saturation.md` calls a hard expressiveness ceiling.

The price, from measurements rather than the old ladder: each tower at u1 costs 7.7 ms per image, so a pair costs about 15.4 ms and lands near 65 img/s at batch 48, except two towers retain too much activation to fit 24.5 GB at 48, so batch 24 to 32 and 55 to 65 img/s in practice. A 10k arm is then about 70 minutes, which is affordable; a full 5.32M pass is 23 to 27 hours, which is not something to do before the recipe is settled.

Live parameters: per tower 4876800 of rotations on the eleven frozen blocks plus 7088640 in the unfrozen one, both towers 23930880, head 19178684, total about 43.1M.

What it buys over the single tower is the open question the probe could not answer, whether a fine-tuned DINOv3 tower leaves room for SigLIP2. The probe measured the concat at +32% over the better single tower with frozen features; u1 already scores 0.0632, so part of that margin may be filled.

It also carries the first-ever bracket of two learning rates on a tower that is not DINOv3: `lr_oft` 5e-4 and `lr_unfreeze` 3e-5 were chosen for ViT-B/16 with DINOv3's training and have never been tested on anything else. Two new knobs at once is the shape of change that hid the gamma bug, so if the arm underperforms the single tower, the first diagnostic is a learning-rate sweep on the SigLIP side with the DINO side held at its known-good values.

### 06:56 - pause and resume are exact, and the reason a production run still is not: flash attention's backward

Asked to verify that pausing and resuming is bit perfect before trusting it on a 39 hour run, and to test the edge cases. Three probe scripts, all kept in `scratch/sophie/` because each answers one question and none of them is a driver: `test_stream_cursor.py` compares a fresh stream reader against one that skips 100 steps worth of samples, `test_step_determinism.py` hashes initialisation, the first batch, the loss, the gradient and the weights after one step so two processes can be diffed line by line, and the trainer itself was run in matched pairs.

The first finding is the one that matters and it is not about resume. Two identical runs, same seed same data same config, do not agree: initialisation hashes identically, the first batch hashes identically, the loss of that batch is identical to twelve decimal places, and the gradient checksum differs. So the training step is not reproducible on this platform, and no resume could ever be bit identical to an uninterrupted run, because the uninterrupted run is not bit identical to itself. The cause is named by torch itself when determinism is requested: flash attention's backward pass uses atomics, `attention_backward.cu`, and it is the only non-deterministic op in our graph.

Second finding, with `--deterministic` on, which forces the math attention backend and turns off cudnn autotune: two separate processes produce 288 of 288 identical adapter tensors after 60 steps, worst absolute difference 0.0, and two identical runs therefore exist to compare against.

Third, the resume itself, measured against a 120 step uninterrupted control in deterministic mode. Pause at step 41 by touching `.PAUSE`, resume with 328 samples skipped, run to 120: adapter tensors 288 of 288 identical, optimiser states 80 of 80 identical, `best` identical, logged loss at step 120 identical. Clean stop at 60 by `--max-steps` then resume to 120: 288 of 288. Cache mode with epochs, stop at 30 then resume to 60: 288 of 288. Stream cursor, which is the mechanism that could silently replay or drop pictures: 12 of 12 batches byte identical across a 100 step skip, images and labels and source stems, and the resumed reader's `consumed` reads 808 for what is batch 101 at batch 8.

So a pause costs nothing that a second run would not cost anyway: the divergence between a resumed run and an uninterrupted one is the same size and the same origin as the divergence between two uninterrupted runs. The price of the determinism that makes the comparison possible is 23 img/s against 55 img/s at batch 8 on a base tower, about 2.4x, which is why it is a flag and not the default and why every number in this file still comes from the fast path.

Four defects fixed while testing. Checkpoints are now written to `path.tmp` and `os.replace`d into place, because a job reaped during a 300 MB save left a truncated file and resume takes the highest step number, so one bad write would have made a run with three good checkpoints beside it unresumable; proven by truncating `ckpt_step000120.pt` to half its length, which now warns and falls back to step 60. Resume walks newest to oldest instead of trusting the newest. A `.PAUSE` present at launch no longer writes a step 0 checkpoint of 300 MB on every accidental relaunch, it exits with a message and touches nothing, which matters because `runs/3da79e14` and `runs/f63f6f54` keep their `.PAUSE` deliberately as the guard against a mistaken relaunch. And the new `--deterministic` flag is the only way to make any of this testable.

Written up in `docs/pause-resume.md`, indexed in AGENTS.md. Nothing here changes any number in this file: the arms were compared on 50k macro with a 3% band that is far above anything the attention backward can do over 30000 steps, and the resume path itself is now measured rather than assumed.

### 06:58 - the large arm now runs a schedule that can be extended, `f63f6f54`, and what number would justify extending it

The user rejected cosine for the main run and for anything that might get longer: "we might extend this indefinitely, like past 1 epoch, maybe 2, 3, 4 epochs". Cosine spends the learning rate across a horizon chosen up front, so a run that keeps climbing has already thrown its own learning rate away by the time the extension is decided, and a resumed run at that point is a different experiment with a different schedule rather than a continuation.

So `TrainConfig` gained two fields. `lr_schedule` is `cosine` by default and `const` on request, and `decay_steps` gives the const schedule a trailing anneal, so the peak holds and only the last `decay_steps` steps fall to the 5% floor. Both fields are dropped from the hashed blob at their default values, the same guard `image_size` uses, and the whole tree of 55 existing run directories still reproduces its pinned `_sha8`, checked before and after. Mechanically, `lr_at(step, total, base, warmup, schedule, decay_steps)` computes `start = max(warmup, total - decay_steps)` and only enters the cosine inside that window, so raising `stream_images` slides the anneal later instead of stretching it: 30000 steps to 60000 keeps 2e-4 at step 30000 and anneals 57000 to 60000. Verified by printing the schedule at both lengths.

`configs/heads/phase3/27_u0-C-L-asl-g2.0-siglip2large-600k-const.json`, sha8 `f63f6f54`, is the large arm rebuilt on it: same tower, same batch 20, same `lr_oft` 2e-4 and `lr_head` 7e-5, same 600000 images and 30000 steps, `decay_steps` 3000 rather than 0 so the gate still ends on a properly annealed checkpoint that can be compared with base-u0's 0.2694. It supersedes `62de5705`, the cosine version, which had reached step 1535 when it was paused; its checkpoint is orphaned and that hash gets no further use.

Running now at 40 to 41 img/s and 18.6 GB, resumed from step 463 after the pause-resume verification. 600000 images at 41 img/s is 4.1 hours of stepping, plus twelve validations at the 2500 step cadence; if the in-run 50k validation turns out to cost ten minutes a piece on the wide tower rather than the three it cost on base, the cadence is the thing to cut and a restart from the checkpoint loses nothing but the queue.

The decision rule, written before the number exists so it cannot be bent afterwards. Base-u0 at 600000 images scored 0.2694 macro and base-u1 0.2733, and the noise band is 3%. Large at or above 0.33 is a clear tower win and the extension arm is next. Between 0.277 and 0.33 the tower is ahead but inside a factor worth questioning, and the extension is still worth its 4.1 hours because a larger tower on the same images was the whole hypothesis. Between 0.262 and 0.277 it ties base, the tower axis is a wash at this image budget, and spending 4.1 more hours on it buys nothing that a second pass on base would not. Below 0.262 the tower loses on 3.4x the parameters and 39 img/s against 118, and the answer is to keep the base tower and put the time into a second pass.

### 07:14 - first large-tower validation, and the step-index trap it nearly set

`f63f6f54` at step 2500: macro_f1 0.0842, macro@0.40 0.0832 on the in-run 10000 image prefix, micro 0.4753. The equivalent line for base-u0 at the same step is 0.1019 macro and 0.4684 micro, and for base-u1 0.0970 macro and 0.4702 micro, so read at the same step index the large tower looks 17% behind on macro.

That reading is wrong and the reason is worth writing down before the curve is used again. The gates run at batch 48 and the large arm at batch 20, so step 2500 is 120000 images for base and 50000 for large. Compared at equal images the order reverses: base-u0 at step 1250, which is 60000 images, scored macro@0.40 0.0743 and micro 0.4617, and base-u1 the same step scored 0.0696 and 0.4572, against the large tower's 0.0832 and 0.4753 at 50000 images. On the axis that matters, the large tower is ahead on both metrics at roughly equal data, and the macro lead is about 12% over base-u0 interpolated to the same image count.

Micro behaving the opposite way to macro is consistent with what the tail analysis already said: the large tower calibrates the head of the label distribution faster, and macro is the tail, which is what the 30000 step horizon and the extension decision are for.

Validation cost on the wide tower: 195 s for the 10000 image prefix against 38 s on base, so twelve of them across the gate are 39 minutes on top of 3.9 hours of stepping, which is affordable and no reason to cut the cadence. Expected finish around 11:25 local.

### 07:20 - which checkpoint the validator actually scores, because it is not the one the run calls best

Checked while the large arm trains, and worth having in the record because it silently decides what every arm number in this file is a number about. `wdtag/validate.py` with only `--run` takes the highest step number from `ckpt_step*.pt` and scores that. It does not read `best_val.pt` and it ignores `--select-metric`, so the in-run selection metric, `macro` for the gates and `macro040` for the main run, affects which file is called best and nothing else.

Two consequences. Arm comparisons here are comparisons of final checkpoints, which is the right artifact for an annealed schedule and the reason the const schedule ends with a decay tail at all. And if a run is stopped early, the pause checkpoint is the one that gets scored, so pausing never costs the artifact, only the steps.

If the best-validation artifact is ever the thing wanted, it has to be named: `--ckpt best_val.pt`. Nothing in this file has been scored that way, and it is now written down rather than assumed.

The doc that governs scoring, `docs/validation-pipeline.md`, had no statement about checkpoint choice at all, which is how a silent default becomes folklore. It does now.

### 07:41 - large tower at 100000 images: macro@0.40 0.1110, and it is widening its lead

Step 5000 of 30000, in-run 10000 image prefix: macro_f1 0.1121, macro@0.40 0.1110, micro 0.4949, validation 118 s. Base-u0 interpolated to the same 100000 images, between its 60000 image point at 0.0743 and its 120000 point at 0.0992, is about 0.0920 macro@0.40 and 0.4661 micro; base-u1 is lower still at both ends. So the large tower is roughly 21% ahead on macro and 6% ahead on micro at equal data, against 12% and 3% at the 50000 image point an hour ago. The lead is growing rather than settling, which is what a wider tower should do early and what makes the 30000 step horizon the question rather than the tower.

Nothing acted on and nothing launched: one arm on the card, 40 img/s, 18.6 GB, eleven validations to go at about two minutes each now that the nested cache is warm.

### 07:45 - the in-run subset overstates micro and understates macro, by a measurable amount

Done on CPU while the large arm trains, because the next decision needs a projection and a projection needs a calibration. For each arm, the last in-run validation line against the final 50k scoring of the same run.

u0 base 512, in-run n=10000 at step 11250: macro@0.40 0.2223 and micro 0.5464, against 50k macro 0.2694 and micro 0.4729, ratios 1.212x and 0.865x. u1 base 512, same step and subset: 0.2276 and 0.5531 against 0.2733 and 0.4791, ratios 1.201x and 0.866x. The two agree to within 1%, so an in-run n=10000 reading is about 1.20x below the 50k macro number and about 1.16x above the 50k micro number, in both directions for a fixed subset size.

u1 base 384 at 11250 ran its in-run validation on n=1000 rather than n=10000, quoting 0.1092, and the same run scores 0.2553 at 50k, a ratio of 2.34x. That is the third appearance of the same trap and now it has a number: the 1000 image prefix divides macro by 2.3, the 10000 image prefix divides it by 1.2, and neither is a small correction that can be waved through.

Two consequences worth carrying. When the large arm's in-run curve flattens, its 50k macro is about 1.2x the reading, so a final in-run 0.22 projects to 0.264 and 0.25 projects to 0.30; the projection is only valid at convergence, because the ratio was measured on arms whose curves had flattened by 11250 steps. And micro can never be read off the in-run line as a check on the bar, because the in-run prefix flatters it by 15%.

### 08:15 - large tower at 150000 images: 0.1247 macro, 15% ahead of base interpolated, and the lead is narrowing

Third point on the equal-image curve. Step 7500, which is 150000 images, gives macro@0.40 0.1247 and micro 0.5015. Against base-u0 interpolated to the same image count, 0.0909 at 100000 and 0.1083 at 150000, the lead goes 22.1% then 15.1%. A narrowing lead over the first 150000 images is the expected shape when the wide tower's advantage is capacity that a small image budget has not yet filled, and it is the reason the decision is being made at 600000 rather than now.

What would keep the lead narrowing to nothing: base-u0 reached 0.2223 at 540000 images in-run, and if the large tower only matches that, the tower axis is dead at this budget. What keeps it alive is the gap at the last point before the base curve flattens, so the next two validations at 200000 and 250000 images are the ones that decide whether the extension arm is worth assembling.

Timing: 8720 of 30000 steps, 38 to 41 img/s, so about 2.9 hours of stepping left plus nine validations. Still one arm, still nothing launched behind it.

### 08:45 - large tower at 200000 images: 0.1356 macro, 9% ahead, and the lead is halving every 50000 images

Fourth equal-image point. Step 10000, 200000 images: macro@0.40 0.1356, micro 0.5099. Base-u0 interpolated to the same count is 0.1244, so the lead runs 22.1%, 15.1%, 9.0% at 100000, 150000 and 200000 images. The absolute gain per 50000 images is the sharper number and it is not flattering: the large tower's macro@0.40 went 0.0832, 0.1110, 0.1247, 0.1356, so increments of 0.0278, 0.0137, 0.0109, while base-u0 over the same span increments by 0.0174 and 0.0161 and is holding its rate. If those two behaviours continue the base curve overtakes the wide tower somewhere around 250000 to 350000 images and the gate ends at or below base-u0.

I am not calling that verdict at 200000 images, for two reasons that both live in the log. First, the large arm has spent its first 10000 steps at batch 20 and `lr_oft` 2e-4 while base ran batch 48 and 5e-4, so a slower per-image climb is exactly what a too-low learning rate on a 24 block tower looks like, and this tower's learning rates were scaled by rule from ViT-B and have never been bracketed on the large model. Second, the large arm is at step 10000 of 30000 with a constant learning rate, so a curve that is still accelerating early and decelerating late is the normal shape rather than evidence.

The discriminator costs nothing and is already scheduled: where the curve sits at step 30000. If macro@0.40 is still adding more than 0.005 per 50000 images at the end, the constraint is the schedule or the budget, not capacity, and the honest follow-up is a learning rate bracket on the large tower rather than abandoning it. If it has flattened well before 30000, then the wide tower has genuinely run out of use for these 600000 images and the answer is the base tower.

Recorded so the number cannot be read two ways once it exists.

### 09:15 - the base tower has overtaken the large one: 0.1414 against 0.1379 at 250000 images

Step 12500, 250000 images: large macro@0.40 0.1379, micro 0.4925. Base-u0 interpolated to the same count is 0.1414, so the lead of 22.1% at 100000 images is now a deficit of 2.5%, and the large tower added 0.0023 over the last 50000 images while base kept adding 0.017. Micro also fell, 0.5099 to 0.4925, so this is not a macro-only wobble.

Two things stop me from declaring the tower dead on this line. The in-run 10000 image prefix has a noise floor of a few thousandths of macro, and 0.0023 is inside it, so the honest statement is that the large tower has stopped climbing rather than that it has peaked. And the comparison is unfair to the large arm at exactly this point in the schedule: base-u0's gates ran a cosine, so its late validations were taken with the learning rate already annealed, while the large arm is at constant 2e-4 and does not anneal until step 27000 of 30000. A flat curve at peak learning rate that lifts when the tail decays is the normal shape, not a paradox.

Which is why the arm finishes. It has 15000 steps, 2.1 hours, and two arguments settled that against stopping it now and handing the card to the 24 hour main run: the annealed end point is what the decision rule was written against and it is the only point comparable with base's annealed 0.2223, and the difference between stopping and finishing is 2.1 hours on a run that cannot start before the card frees up anyway. If the annealed point lands near 0.20 in-run, which is 0.24 at 50k, the tower axis is answered as a no for 600000 images and the main run starts with that answer in hand rather than with a suspicion.

### 10:00 - calibration at the fixed 0.40 threshold is already done, and that closes a cheap door worth closing

Measured on CPU from the probability cache of `b6e293f0`, `I:\.cache\wd-tagger-trainer\probs\b6e293f0_012500.npy`, 50000 by 12476 float16, by `scratch/sophie/test_calibration_headroom.py`. The counting pipeline reproduces the official number exactly, 0.2694 macro at 0.40, before it is used to claim anything.

The tempting result first: a per-tag oracle, each tag's best cut over a 40 bin grid, gives 0.3546, which is 31.6% above where we are and would have looked like a free lunch. It is not real. Fitting each tag's cut on the even rows and evaluating on the odd ones gives 0.2640 against 0.2601, a gain of 1.5%, with 96.3% of tags moving their cut somewhere between -0.40 and +0.40. The oracle is fitting the noise of the tags with a handful of positives, which is most of them, and it does not survive 25000 rows of separation, let alone 50000.

The one-knob version, which is what temperature or bias calibration amounts to at a fixed decision threshold, is also finished as an idea. The global cut curve is flat-topped around where we already are: macro 0.2583 on even rows at 0.40, 0.2573 at 0.45, 0.2343 at 0.30, 0.2359 at 0.60. Fitting a single global cut on even rows and applying it to odd yields 0.2615 against 0.2601, a gain of half a percent. Our probability scale is already sitting at its fixed-threshold optimum to within a percent.

What that rules out: the 2.07x macro gap to the bar cannot be closed by recalibration, per-tag bias, NTR bias, balanced sampling of thresholds, or any post-hoc rescaling of the probabilities. The bar's macro precision and recall, 0.6201 and 0.5333, are better separation of the tail labels at 0.40, not a better-placed decision boundary on the same ranking. Since per-tag oracle headroom of 31.6% exists but is not learnable from this much data, the remaining route is a model that ranks tail tags better, which is exactly what the training arms are for.

### 10:15 - large tower at 400000 images: 18.2% behind, and the deficit is monotone

Step 20000, 400000 images: macro@0.40 0.1521 in-run, against base-u0 interpolated at 0.1860. The lead went 22.1%, 15.1%, 9.0%, then deficits of 2.5%, 8.7%, 13.7%, 18.2%. The large tower is adding about 0.004 per 50000 images where base adds 0.017, so it is not catching up, it is being left behind at a constant relative rate.

Projection to the end, stated before the number exists so it can be scored honestly: 200000 images remain, which at 0.004 per 50000 is 0.016, giving about 0.168 in-run, and the 3000 step decay tail might add one or two hundredths to the in-run reading because base's own late points were taken annealed. Call it 0.18, which projects by the measured 1.20x factor to 0.216 at 50k, against base-u0's 0.2694. If it lands there, the tower axis is a clear no at 600000 images and the decision rule's bottom branch applies: keep the base tower and put the time into the main run.

The honest caveat is unchanged and now measurable rather than speculative. `lr_oft` 2e-4 and batch 20 were scaled by rule from ViT-B with DINOv3 and had never been tested on a 24 block tower, and the shape of the failure, a curve that climbs early then flattens well below the smaller model's, is what an under-tuned learning rate looks like as readily as what an over-large model looks like at a small image budget. Distinguishing those two would cost a bracket at 5e-4, four and a bit hours, on a tower that is 18% behind. My read is that the bracket is not worth buying now, because the base tower is the architecture we intend to ship and every hour on the large tower is an hour not spent on the main run.

### 10:45 - large tower at 450000 images: 23.2% behind, and two things we have never done

Step 22500, 450000 images: 0.1537 in-run against base interpolated at 0.2002, a deficit of 23.2% where it was 18.2% at 400000. The projection stands and slightly worsens, and the arm has 7500 steps left.

Two absences found while looking for a better use of the next GPU hour, both checked in the code rather than assumed. The trainer applies no augmentation at all: `wdtag/stream.py` shuffles tars and decodes, and nothing flips, crops or recolours, so every image is seen exactly as it sits in the tar. And the scoring path applies no test-time augmentation either: neither `wdtag/validate.py` nor `wdtag/evaluate.py` mentions a flip, so every number in this file is a single forward pass on an unmodified image.

Horizontal flip averaging at inference is the cheapest unexplored lever left. It costs two forward passes instead of one, needs no training, no new data and no schedule, and it can be applied to a checkpoint that already exists. For a fixed-threshold macro metric it usually lifts both precision and recall slightly because the errors of a flipped tagger are partly independent of the unflipped ones. The bar's recipe in `docs/bar-convnextv2-huge.md` is a plain linear head on frozen-ish convnext features and says nothing about TTA either, so this is not a trick it is using against us, it is just a free thing neither side has spent.

Priced as the first thing to try once the card frees, before any further arm: score `2b6b81aa` at 50k with and without a flip average and compare macro at 0.40. It costs about 25 minutes of GPU and could be worth more than several hours of training if it moves the number by two or three percent.

### 11:05 - augmentation is not a lever here, and the user's reason is the correct one

Raised flip averaging at inference as a free 25 minute experiment and the user answered with the reason it is not free: this label space is full of tags that augmentation changes. Colour jitter breaks the colour tags, crops break composition tags, a mirror breaks orientation tags and mirrored text, and character identity is not preserved under a mirror. The usual assumption that an augmentation preserves the label does not hold, so both training augmentation and test-time augmentation are out, and a noisy label is worse than a small dataset.

Checked the size of the affected set rather than leaving it as a hunch: 214 of 12476 labels mention an orientation or text word, which is small, but a crude substring count also sweeps up background and backpack and misses tags like `arms_behind_back`, so treat 214 as a floor rather than the population. The reason it matters out of proportion to its size is that the metric is a macro average over tags, so 200 tags moved in the wrong direction is 1.6% of the score, and the tags a mirror would blur are exactly the ones already scoring near zero.

Written into `docs/best-settings.md` as a not-a-win with the reasoning, so it does not get proposed again.

### 11:10 - correction: the large tower is not failing, it is sitting on a constant-learning-rate plateau

The user asked why the base tower is beating the large one and whether the large one is training badly. Looking at micro as well as macro changes the answer, and it corrects my own projection above.

At equal images the large tower is ahead early and then stops. Micro on the 10000 image prefix: large 0.4753 at 50000, 0.5099 at 200000, and then 0.4925, 0.5124, 0.5091, 0.4990, 0.5078, 0.5009 at 500000, so it peaked near 200000 and drifted down. Base-u0 over the same axis goes 0.4617 at 60000 monotonically to 0.5464 at 540000 and base-u1 to 0.5531. So the large tower is not undertrained in the usual sense and it is not overfitting either, because the training loss is flat across the stall, 0.0040 at 200000 images against 0.0032 at 600000 and noisier than that swing.

What it looks like is convergence to the noise floor of a constant learning rate. The large arm holds 2e-4 for 90% of its 30000 steps and only anneals in the final 3000, while every base gate ran a cosine from the first step, so base-u0's last point at 540000 images was taken at about 7% of its peak learning rate and the large arm's points so far were all taken at 100%. Every comparison in the table above is therefore an annealed model against a plateaued one, which is not a tower comparison at all.

This retires the projection I wrote at 08:45 and again at 10:15, that the arm would land near 0.21 to 0.22 at 50k. That projection assumed the plateau continues; the decay tail is precisely the part of the run that collects a constant-rate plateau, and its last point is the first point at which the large tower can be compared with base honestly. The decision rule stands but moves to that end point, and if the tail closes most of the deficit then the follow-up is not a learning rate bracket at all, it is a decay tail sized as a fraction of the run rather than a fixed 3000 steps.

One more thing worth stating because it is the user's read rather than mine: they called the large tower badly trained before I did, and they were right about the shape even though the cause is the schedule rather than the model.

### 11:15 - the anneal is already paying: macro 0.1597 to 0.1645 as the learning rate fell from 2e-4 to 1.6e-4

Step 27500, inside the decay window, with the learning rate at 1.59e-4 and still falling: macro@0.40 0.1645 against 0.1597 at step 25000, and micro 0.5323 against 0.5009. One validation window of annealing bought 3.0% macro and 6.3% micro after fourteen windows of flat learning rate bought the large tower nothing, which is the cleanest possible confirmation that the stall was the constant learning rate and not the tower.

The deficit against base-u0 at 540000 images, 0.2223 annealed, is still 26%, so the tower does not become the answer on this evidence alone. But the shape of the last hour has changed from a flat curve to a falling one at exactly the point where the schedule changed, and the honest read of the whole arm is now that it was one long unannealed run with a 3000 step tail bolted on, not a test of what the wide tower can do.

If the final point lands near 0.17 to 0.18 in-run, which is 0.20 to 0.22 at 50k, then the tower is still behind base's 0.2694 and the decision rule's bottom branch applies. What changes is the follow-up: not a learning rate bracket, a run with a decay tail long enough to matter, since 3000 steps of 30000 was 10% and the base gates annealed across their entire schedule.

### 11:45 - large tower final, 50k: macro 0.2331, micro 0.4915. It loses macro to base by 14.7% and wins micro by 2.6%

`f63f6f54` finished its 30000 steps in 276 minutes and scored 491.8 s at 102 img/s on the nested 50k, batch 128, checkpoint `ckpt_step030000.pt`. Official numbers at 0.40: macro F1 0.2331, macro MCC 0.2462, macro P 0.2701, macro R 0.2636; micro F1 0.4915, micro MCC 0.5201, micro P 0.3679, micro R 0.7400. 12454 supported labels.

Against the champion, u1 base at 0.2733 macro and 0.4791 micro, the large tower is 14.7% behind on macro and 2.6% ahead on micro. Against u0 base, 0.2694 and 0.4729, it is 13.5% behind and 4.0% ahead. The pattern in precision and recall explains it in one line: the wide tower has higher precision at both levels and lower recall at both, 0.2701 against 0.2960 macro precision and 0.2636 against 0.3213 macro recall, so it is a more conservative model that misses tail tags base catches.

The schedule correction is quantified now. The last in-run validation at step 27500 read 0.1645 at 0.40 and the run then annealed from 1.6e-4 to 5% of peak over the last 2500 steps; dividing the official 0.2331 by the 1.20 transfer factor gives an implied in-run value of about 0.194 at step 30000, so the final 2500 steps of decay were worth about 18% of macro on their own. An arm whose only anneal was its final 10% paid for that in the number.

So the verdict is the decision rule's bottom branch: at 0.2331 against base's 0.2694 the large tower does not earn 3.4x the parameters and 2.4x the wall clock per image at this image budget. It is not a verdict on the architecture, because the run was 90% unannealed and the last 10% moved it 18%, and the follow-up if anyone wants it is a 30000 step arm with the decay tail at 30% to 40% of the schedule rather than 10%, which is a schedule change and not a tower change.

One correction to the calibration note above: the 1.20x in-run to 50k factor only holds when the in-run validation and the final checkpoint sit at the same point of the schedule. Here the last in-run point was mid-tail and the scored checkpoint was fully annealed, giving 1.417x. Use the factor with the phase in mind, not as a constant.

### 12:05 - why the large tower loses: it is better on everything it sees often and worse on everything it sees rarely

The user asked what is behind the large tower's deficit before accepting it, which is the right question and had a measurable answer. Compared the two runs' probability caches tag by tag, both scored on the same 50000 images at 0.40, `b6e293f0_012500` against `f63f6f54_030000`, by `scratch/sophie/test_tag_delta.py`.

By how many of the 50000 validation images contain the tag: support under 5, 1472 labels, base 0.1687 against large 0.0900. Support 5 to 20, 6027 labels, base 0.2508 against large 0.1858. Support 20 to 50, 2257 labels, base 0.3057 against large 0.2997, a tie. Support 50 to 200, large ahead by 0.0236. Support 200 to 1000, ahead by 0.0286. Support 1000 to 5000, ahead by 0.0263. Support above 5000, ahead by 0.0102.

So the large tower beats base on every bucket from support 50 upward, ties the 20 to 50 band, and loses catastrophically on the 7499 labels below support 20, which is 60% of the label space and therefore 60% of the weight in a macro average. The large tower is better on 4319 labels and worse on 4694, which is what a swap of strengths looks like rather than a model that failed to train.

That is the whole explanation for a deficit in macro and a surplus in micro, and it is not a schedule artefact, not a learning rate artefact, and not a broken adapter. It is capacity per rare label: 24 blocks and 1024 wide on the same 600000 images means every rare tag gets the same number of examples as base had, but a much larger function has to learn it from them. The tail is exactly what macro is.

What that predicts about the experiment the user proposed, an anneal continuation at a third of the learning rate: it will recover a little on the common tags the large tower already leads, and will not close a 0.065 wide gap on 7499 rare labels, because a lower learning rate makes rare tags learn more slowly, not faster. It is still worth running if the goal is to know rather than to guess, and it costs 2.2 hours, but it is not the experiment that would fix the tower.

The experiment that would fix it is more images per rare label, which the main run buys simply by doing two passes, or a tail mechanism that shares statistics across related labels, which is a head or loss change rather than a tower change.

### 12:10 - the wide tower's features really are better, which makes the tail deficit a training-regime problem

Frozen-feature probe, the same linear head and the same 10000 training images for both towers, scored on the 50k at 0.40: SigLIP2-B 0.0452 macro and 0.1593 micro, SigLIP2-L 0.0630 macro and 0.1880 micro. The large tower is 39% ahead on macro and 18% ahead on micro before any adaptation at all. Extracted with `scratch/sophie/siglip_probe.py extract siglip_large`, both splits cached as fp16 under `I:\.cache\wd-tagger-trainer\probe`.

Read this as a controlled comparison and not as a ceiling: 10000 images is a deliberately tiny budget, and the trained base run reaches 0.2694, six times the frozen probe, because OFT and 600000 images are doing most of the work. What the probe says is narrower and still useful: with the same head and the same data, the wide tower's representation carries more tag information. It is not a worse tower.

Putting the two measurements together, the picture is consistent and it is not any of the three explanations that were on the table this morning. Not the schedule, since both endpoints are fully annealed. Not the learning rate in the sense of "the model did not train", since the training loss tracks base exactly at equal images. Not the tower's information content, which is better. The large tower is better on every label bucket with support above 50, ties 20 to 50, and loses only below support 20, where 7499 labels sit and where 60% of a macro average is decided. A larger function learning the same rare label from the same handful of examples is the plain reading, and it is the one prediction the data supports on all three fronts.

Consequence for the next decision, and it argues against the large tower for the long run: more images per rare label is the fix, and at 39 img/s a two-pass full-corpus run on the large tower is 75 hours against the base tower's 24. The base tower gets the long run, and the wide tower stays a 600k-image question until something cheaper than 75 hours exists for the tail.

### 12:15 - the user's batch-noise theory fits the evidence better than the learning rate, and it is testable without new code

Split the rare-tag deficit into precision and recall, same two caches, same 50000 images. Support 0 to 5: base P 0.2116 R 0.1682, large P 0.1155 R 0.0957, so both collapse and the large tower is about 45% worse on each. Support 5 to 20: base 0.3297 and 0.2426 against large 0.2736 and 0.1731, precision down 17% and recall down 29%. Support 20 to 50: large precision up 11% and recall down 13%. From support 50 upward large precision is up 12% to 15% and recall is level.

A collapse in both directions on rare tags is what a model that never learned the tag looks like, not a model that is being conservative. And the batch arithmetic points at a mechanism: at batch 20, a tag in 20 of 50000 images has 0.008 expected positives per update, support 50 has 0.02, support 200 has 0.08. Those tags are absent from almost every batch, and ASL normalises by the tag's positive count inside the batch, so their gradient is zero most updates and enormous on the rare update that contains one. Base at batch 48 is only 2.4x less starved, which is why base also struggles on the tail, but it is 2.4x less starved.

The trainer has no gradient accumulation: `batch_size` is the only knob and there is no micro-batch anywhere in `wdtag/train.py`, so an effective batch above what fits in 24.5 GB is not available today and would have to be built, with the caveat that accumulating micro-batches makes ASL's per-batch positive normalisation approximate rather than exact.

There is a control that tests the same mechanism with no new code at all: run the base tower at batch 20, 600000 images, everything else identical to `b6e293f0`, which scored 0.2694 and whose tail buckets are the reference above. If base at 20 also collapses to about 0.09 in the support 0 to 5 bucket, then batch size is the mechanism and the large tower's deficit is a batch artefact of our own making; if base at 20 holds near 0.16 and only loses a little, then the wide tower's capacity per rare label is the cause and no amount of accumulation will recover it. 1.7 hours at 118 img/s, single variable, known reference point.

### 12:25 - the 24 against 48 control, prepared and not launched

The user's answer to the batch-noise question is to use the headroom and test 24 against 48 directly. Prepared as `configs/heads/phase3/29_u0-C-L-asl-g2.0-siglip2b-600k-b24.json`, sha8 `48de8bf9`, which is `b6e293f0` with exactly one field changed, `batch_size` 48 to 24, verified by diffing the two configs field by field. Same tower, same 600000 images, same `lr_oft` 5e-4, same head rate, same default cosine over 25000 steps instead of 12500, so both endpoints are annealed and the only variable is images per update.

Why this and not the large tower at 24: a large tower at 24 against the large tower at 20 moves the batch by 20%, which is inside the noise, and the gap we are trying to explain is 2.4x. Base has a measured reference at 48, 0.2694, and the same bucket table, so one 1.7 hour run at 24 answers whether tail performance moves with images per update at all.

What each outcome means. If base at 24 drops the support 0 to 5 bucket from 0.1687 toward 0.10, then tail performance in this project is largely a function of images per update, the large tower's deficit is partly our own batch choice rather than its capacity, and gradient accumulation becomes worth building for every long run including the main one. If base at 24 holds near 0.16 and gives up only a little, then batch noise is a secondary effect and the wide tower's capacity per rare label is the real cause, which retires the tower without a code change.

Not launched yet: `b05cbca4`, the third learning rate continuation, is on the card until about 14:25 and only one arm runs at a time.

### 12:30 - how much a batch effect would even be visible, measured before the run exists

Macro weighs labels equally, so the bucket sizes are the sensitivity: support 0 to 5 is 1472 labels, 5 to 20 is 6027, together 7499 of the 12454 supported, which is 60.2% of the macro average. Support 20 to 50 adds 18.1%, and everything at support 50 or above is only 21.8%.

So a 0.01 fall in the low-support buckets' mean F1 moves macro by 0.006. For the question actually being asked: if base at batch 24 drops the 0 to 5 bucket from 0.1687 to 0.10 and the 5 to 20 bucket from 0.2508 to 0.18, that is a macro of 0.227 against 0.2694, a 16% loss, far outside the 3% band. If it loses a thousandth or two in each bucket, macro barely moves. The experiment is therefore sharp in exactly the direction the theory predicts, and a null result will be unambiguous rather than a shrug.

`scratch/sophie/test_tag_delta.py` now takes `--pair <runA_step> <runB_step>`, so the same bucket table can be produced for the batch-24 run the moment it is scored, without editing the file.

### 12:55 - gradient accumulation is in the trainer, and it is exact rather than approximate

The user's proposal was to emulate base's batch 48 on the large tower with 2 accumulated micro-batches of 24, so the tower that loses the tail gets the same images per update as the tower that wins it. Implemented in `wdtag/train.py` and `wdtag/config.py`.

`grad_accum` is a config field defaulting to 1 and dropped from the hashed blob at that value, so all 58 existing run directories still reproduce their pinned sha8, checked after the change. `total_steps` divides by batch times accum; the resume skip multiplies by it, so a checkpoint still names an exact sample offset; the stream loop groups accum micro-batches before calling the step function; the img/s line counts both; and with a stream length that is not a whole number of steps the tail still gets spent rather than dropped.

Why it is exact rather than approximate: our ASL reduces with an elementwise mean over the batch, and the mean of equal-sized micro-batch means is the whole-batch mean. That is not true of the BALMS reference, whose `sum/n_pos` normalisation would have divided a rare tag's positive gradient by the accumulation factor, which would have biased the experiment against exactly the labels being tested.

Verified twice. Mechanically, one batch of 12 against three of 4 on the same twelve images: loss agrees to 1.2e-5 relative and the gradient cosine is 0.999999 with an rms difference of 1.4e-3, the residual being bf16 rounding because a batch of 12 and three of 4 hit different kernel tilings. The first attempt measured a cosine of 0.947 and that was my error, not the code: `head_dropout` was 0.1, so the single batch drew one mask and the three micro-batches drew three different ones.

Two defects found while testing. The debug-picture dump indexed the first micro-batch by up to 100 images and crashed as soon as accumulation made micro-batches of 4. And a test run I started beside a live arm put about 4 GB on top of the arm's 19 GB on a 24.5 GB card; the user caught it, the arm survived, and the rule is now explicit: no GPU test while a real arm holds the card.

### 15:05 - the accumulation path was corrupting its own batches, and that is why two runs disagreed

Chasing the divergence turned up a real defect, not rounding. The streamer keeps a ring of preallocated pinned batches sized `prefetch + 2`, and that arithmetic silently assumes the consumer holds exactly one batch: at most `prefetch` sit in the queue, one is held, one is being written. Gradient accumulation makes the trainer hold the whole group, so with `grad_accum` 3 the live count is four queued plus three held plus one being written, eight against a ring of six. The producer wraps and writes into slots the trainer has not consumed yet, which overwrites part of a batch with later images and their labels.

That is why the symptom was so slippery: two runs of the same config agreed for fifty or sixty steps and then drifted apart at a random step, with the difference appearing on all seventy nine trainable tensors at once and growing smoothly. It is data corruption whose arrival depends on decoder timing. It never showed up without accumulation because the consumer holds one batch there, and it would have quietly trained the real arm on partly overwritten images.

The fix is `hold` on `TarStream`, defaulting to 1, with `ring = prefetch + hold + 1` so the ring covers the queue, the held group and the slot being written; the trainer passes `hold=accum`. At the default it is the same six slots as before, so nothing about the existing arms changes.

Ruled out along the way, each with a measurement. Stream order under bursty pulls, hashed over 300 batches three times in two processes, identical every time. Batch four on its own without accumulation, bit-identical over 120 steps. A pinned cuBLAS workspace, which did not help and moved the divergence earlier. Serialised kernel launches, which did not help either.

After the fix, two independent runs of the three-by-four config are 288/288 bit-identical at steps 20, 40, 60, 80 and 100, and again at 300. A pause at step 151, resume to 300, is 288/288 bit-identical to the uninterrupted control, resuming by skipping 1812 samples, which is 151 groups of twelve.

Edge cases, all measured after the fix. A stream length that is not a whole number of accumulated steps finishes and logs `[accum] last step is short by 4 images`. A `.PAUSE` file present before the first step exits without writing a checkpoint, zero files. Truncating the newest checkpoint to half its length makes the resume report it unreadable and fall back to the previous one, then continue from step 100.

One unrelated crash found and fixed while testing: a run whose validation is switched off never writes `best_val.pt`, and the end-of-run evaluation loaded that file unconditionally, so a finished run died at the last line. `wdtag/evaluate.py` now falls back to the highest-step checkpoint.

### 15:10 - the accumulation arm is on the card

`a4674fbd`, the large tower at batch 24 with two accumulated micro-batches, effective batch 48, 5000 optimiser steps and 240000 images, const schedule with a 1500 step tail, `lr_oft` 2e-4, cold start rather than warm. Launched 15:04, measured at 40 img/s and 3000 steps per hour once compile finished, so about 1.7 hours, finishing near 16:45. 21.1 GB of the 24.5 GB card.

This is the experiment the user asked for: the tower that loses the tail gets the same images per update as the tower that wins it, at a quarter of the images the full gate used. It is a probe rather than a champion attempt, because 240000 images cannot beat 0.2694 outright; what it can show is whether the low-support buckets behave differently when each update sees twice as many pictures.

The comparison to be careful about. `f63f6f54` ran 600000 images at batch 20 with no accumulation and ended at 0.2331 macro, with the support-under-20 buckets at 0.0900 and 0.1858. This arm reaches 240000 images and anneals its last 72000, so its macro will be far below 0.2331 regardless. The readable signal is the shape of its bucket profile at 50k against `f63f6f54`'s, not the headline number, and if the low-support buckets climb well above 0.09 at a quarter of the budget then images per update is doing what the theory says and the arm deserves a full 600000 image run.

Also parked: `b05cbca4`, the third-learning-rate continuation, stopped cleanly at step 3597 of 20000 with `ckpt_step003597.pt`. It is resumable in one command and it is the weaker of the two questions, so it waits until the card is free again.

### 15:47 - decision rule written before the result arrives

First reading from `a4674fbd`, step 1000 of 5000, which is 48000 images: macro 0.0823, micro 0.4800, 10809 labels with any positive. `f63f6f54` at 50000 images recorded 0.0832, and its micro at that point was in the same place. Level, not better, which is worth noting but not yet a verdict: 48000 images is early enough that the tail has barely moved for either arm, and the two numbers are not quite the same metric, since the in-run record gives macro at the sweep's own threshold while the trajectory I logged this morning was macro at 0.40.

Committed now so the reading cannot be retrofitted. Scored at 50k at 0.40, against `f63f6f54`'s end state of 0.0900 for support under 5 and 0.1858 for support 5 to 20, and against base's 0.1687 and 0.2508 at batch 48.

If both low-support buckets land at or above base's numbers, images per update is the lever, the tower is viable, and the next arm is the same config at 600000 images and 12500 steps to compare with `f63f6f54`'s 0.2331 at equal budget. Four and a half hours.

If they land near 0.09 and 0.186, the tail did not move at double the images per update, batch size is not the mechanism, and the wide tower is retired for good. The card then goes to the main run and the tower question closes with a measurement rather than a suspicion.

If they land in between, above `f63f6f54` but well short of base, then the honest comparison is at equal images: `f63f6f54`'s trajectory has 0.1356 at 200000 images and this arm is at 240000, so its in-run curve can be read against that, and the answer is "batch helps a little, not enough" unless the bucket gap has halved.

The other open decision, still the user's: `runs/3da79e14`, the main run, sits paused at step 15373 of 221739 with its original cosine. Resume as-is costs nothing and cannot be extended; relaunch warm under the constant schedule with a taller horizon and a tail at about 15% loses 15000 steps of momentum and buys a schedule that can be lengthened. My lean is the relaunch, on the same reasoning that made the constant schedule worth writing.

### 16:16 - like for like, effective batch 48 is ahead of batch 20 at the same images

Same metric, same code, same tower, comparing the two in-run logs at matched image counts rather than at matched steps.

`f63f6f54`, batch 20: 50000 images macro 0.0842 micro 0.4753, 100000 images 0.1121 and 0.4949, 150000 0.1264 and 0.5015, 200000 0.1371 and 0.5099.

`a4674fbd`, batch 24 accumulated twice: 48000 images macro 0.0823 micro 0.4825, 96000 images 0.1229 and 0.5047.

So at the first reading the two are level on macro and the new arm is ahead on micro, and by the second reading the new arm is 9.6% ahead on macro and 2% on micro. Twice the images per update, measured at equal budget, and the gap opens rather than closing. Still early at 96000 images, still no bucket table until the 50k score, but this is the first evidence in the whole investigation that points the way the batch theory predicted.

If that 9.6% carried to 240000 images the arm would sit near 0.154 against `f63f6f54`'s 0.140 at the same budget, which is not a champion number and is not meant to be; it is the sign of whether the tower was starved of signal per update rather than short of capacity.

### 16:46 - the gap widens: 23% ahead on macro at matched images

Step 4280 of 5000, 42 img/s, learning rate already in its tail at 9.92e-5, about fifteen minutes left.

`a4674fbd`, effective batch 48: 48000 images macro 0.0823 micro 0.4825, 96000 0.1229 and 0.5047, 144000 0.1432 and 0.5161, 192000 0.1685 and 0.5129.

`f63f6f54`, batch 20: 50000 0.0842 and 0.4753, 100000 0.1121 and 0.4949, 150000 0.1264 and 0.5015, 200000 0.1371 and 0.5099.

At 192000 images against 200000 the new arm is 22.9% ahead on macro and level on micro. At 96000 against 100000 it was 9.6% ahead. The advantage grows with images rather than settling, which is what starvation of signal per update looks like and not what a capacity limit looks like: a tower short of capacity does not pull further ahead the longer it trains at the same data.

Note the confound that is not present: both arms are cold starts on the same tower, both run the constant schedule, and at these image counts both are still at the peak learning rate, so the comparison is the batch and nothing else. The tail of the new arm starts at step 3500 and the last thousand steps anneal, so its final number will be higher again.

This does not make the arm a champion: at 240000 images against 600000 it cannot be. It does make the batch question answerable in the affirmative at the trajectory level, with the bucket table at 50k still to say whether the tail is where the gain came from.

### 17:20 - the probe answers yes, and the answer is concentrated in the tail

`a4674fbd` finished: 5000 steps, 240000 images, 108.3 minutes, 43 img/s. Scored at 50k at 0.40 it is macro 0.2197 (MCC 0.2344, P 0.2355, R 0.2781) and micro 0.4593 (MCC 0.4980, P 0.3294, R 0.7586).

The honest comparison needed the batch-20 arm at the same images, not its final checkpoint, so I scored `f63f6f54` at `ckpt_step012000.pt`, which is 240000 images: macro 0.1516, micro 0.4397. Same 50000 images, same validator, same threshold. So at matched budget, 2.4x the images per update is worth +45% macro and +4.5% micro.

The bucket table is what settles the mechanism. Support 0 to 5: 0.0423 against 0.0833, +97%. Support 5 to 20: 0.0981 against 0.1783, +82%. Support 20 to 50: 0.1993 against 0.2866, +44%. Support 50 to 200: +19.5%. Support 200 to 1000: +9.2%. Support 1000 to 5000: +3.9%. Support 5000 and above: +0.1%.

That is monotone in rarity: the rarer the tag, the larger the gain, and it vanishes entirely for the tags that already had plenty of signal. This is exactly the fingerprint the starvation theory predicted and the exact inverse of this morning's finding, where the wide tower lost only on the rare labels. More images per update buys back the rare labels specifically.

One caveat that must be carried with it, and one mistake to record. The caveat: `a4674fbd` is fully annealed while `f63f6f54` at step 12000 was still at the constant peak with 18000 steps to run, so part of the +45% is the anneal rather than the batch. My reason for thinking the tail pattern survives that is that anneal consolidates what a run has already learned, and a run that has learned less about the rare tags has less to consolidate there; if the anneal were the cause the gain would be flattest in the tail, and it is the sharpest. That is reasoning, not a measurement, and the full-length arm measured against the same arm is the way to settle it.

The mistake: scoring `f63f6f54` at step 12000 wrote over `runs/f63f6f54/validation.json`, which had held the step 30000 score. I kept the new one as `validation_step012000.json` and rescored step 30000 from its cached probabilities, so `validation.json` again reads macro 0.2331 and micro 0.4915. Nothing was lost, but the overwrite was avoidable with `--out` and it is the second time today a script of mine has touched an artifact it did not own.

Consequence. The tower is not retired: at 40% of the budget it is already within 6% of its own 600000 image result on macro, and it is 45% ahead of itself at the same images. The next arm is the same configuration at the full budget, `c8f2a635`, 600000 images, 12500 steps, and it differs from `f63f6f54` in exactly two fields, `batch_size` 20 to 24 and `grad_accum` 1 to 2, verified by diffing every field. Launched 17:22, 21.1 GB, about 3.9 hours, finishing near 21:15.

What it has to beat to matter: 0.2331 is its own predecessor, and 0.2694 is base at the same 600000 images with batch 48 in one piece. If the +45% at matched budget holds at the full budget the tower lands near or above 0.28, which would put it above base and would make the wide tower the thing to take to the main run instead of the base tower.

### 18:05 - no, four hours is not needed to know it is promising; the curve already says so

Three curves on the same 10000 image in-run cache, same code, same metric, read at matched images rather than matched steps.

Base u0, batch 48 in one piece: 60000 images 0.0740 macro and 0.4617 micro, 120000 0.1019 and 0.4684, 180000 0.1211 and 0.4882, 240000 0.1414 and 0.5065, 300000 0.1597 and 0.5096, 540000 0.2223 and 0.5464.

Large at batch 20 (`f63f6f54`): 50000 0.0842, 100000 0.1121, 150000 0.1264, 200000 0.1371, 300000 0.1452. Ahead of base early, behind by 300000 images, which is how it ended 13% below base at the full budget.

Large at effective batch 48 (`a4674fbd`): 48000 0.0823 and 0.4825, 96000 0.1229 and 0.5047, 144000 0.1432 and 0.5161, 192000 0.1685 and 0.5129.

Against base at matched images that is 11% ahead at 48000, 21% at 96000, 18% at 144000 and 19% at 192000, with micro 4% to 8% ahead throughout. Against the batch-20 arm at matched images it is 10% ahead at 100000 and 23% ahead at 200000. So the two things we can already say without finishing the run: effective batch 48 beats batch 20 at equal budget, and this configuration is tracking above base rather than below it, which no large tower run has done before.

What four more hours buys is the endpoint at 600000 images, because that is the budget at which the comparison with base's 0.2694 and the batch-20 arm's 0.2331 is fair, and because base's own curve steepens late: it gains 0.0626 macro between 240000 and 540000 images while the batch-20 tower gained 0.0081. A lead measured at 192000 images is not a lead at 600000, which is precisely what happened to the batch-20 arm.

If the reading is wanted sooner, the honest instrument is pausing at the next checkpoint boundary, scoring that checkpoint at 50k for the real metric, and resuming: about fifteen minutes of card time, and the arm keeps its schedule. Cheaper still is the in-run validation at step 2500, 120000 images, which costs nothing and lands shortly.
