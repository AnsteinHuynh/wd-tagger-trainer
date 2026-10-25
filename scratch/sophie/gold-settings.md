# The gold-star optimizer settings

## What exists, and where

They are real, they are explicitly marked with a star, and they are not a discovered optimum: they are a named bundle the user asked for once and the campaign then tried to kill three times

The three sister projects named in the brief resolve to two: `C:\Users\kerok\apps\dreamsim-dinov3` and `C:\Users\kerok\apps\speedy-sdxl-vae-trainer`. There is no `dreamsim-oft-trainer`; the full listing of `C:\Users\kerok\apps` has no other dreamsim directory, so the OFT work lives inside `dreamsim-dinov3` (its `src/dreamsim_oft/` and `scratch/sophie/rotasim/`)

The origin is a user directive, quoted in the archive at `speedy-sdxl-vae-trainer/docs/history/_archive/MEMORY-pre-cleanup-2026-10-06.md:64-65`: "want OFTv2-style, SinkSGD_ADV with our gold star settings that make sense". The bundle was assembled in `speedy-sdxl-vae-trainer`, then carried into `dreamsim-dinov3` where it is called the FULL Sink-V stack

The star marker itself: `speedy-sdxl-vae-trainer/scratch/night2/JOURNAL.md:72-73` reads `| goldstar-s1 @400 | 39.605 | ⭐ bundle |` and `| goldstar-s2 @400 | 39.460 | ⭐ bundle |`, and `docs/history/adapters.md:76` calls it "the ⭐-bundle optimizer config"

## The settings, verbatim

The canonical tuple, the eight switches as launched, at `speedy-sdxl-vae-trainer/scratch/night3/run_grid.py:65-78`:

```
# The gold-star bundle: the eight switches as they were actually launched. The
# campaign has measured this whole BUNDLE against standard three times and read
# a tie each time (banked 2-seed, dimix 39.61 vs 39.28, mc3 39.52 vs 39.30) --
# which is "unresolved", not "equal". No individual switch was ever ablated.
GOLDSTAR = (
    ("training.optimizer.orthogonalGradient", "iterative"),
    ("training.optimizer.orthogonalSinkhorn", "true"),
    ("training.optimizer.normedMomentum", "true"),
    ("training.optimizer.snrCond", "true"),
    ("training.optimizer.spectralNormalization", "true"),
    ("training.optimizer.geometricWd", "true"),
    ("training.optimizer.cautiousWd", "true"),
    ("training.optimizer.statePrecision", "fp32"),
)
```

The same eight in the CLI form that actually launched the runs, `speedy-sdxl-vae-trainer/scratch/night2/run_arms.py:27-36`:

```
GOLDSTAR = [
    "--set", "training.optimizer.orthogonalGradient=iterative",
    "--set", "training.optimizer.orthogonalSinkhorn=true",
    "--set", "training.optimizer.normedMomentum=true",
    "--set", "training.optimizer.snrCond=true",
    "--set", "training.optimizer.spectralNormalization=true",
    "--set", "training.optimizer.geometricWd=true",
    "--set", "training.optimizer.cautiousWd=true",
    "--set", "training.optimizer.statePrecision=fp32",
]
```

The eight switches are not the whole optimizer. The only file that carries the numeric arguments alongside them is `speedy-sdxl-vae-trainer/configs/oft-fullcoco.json:116-139`, the OFT run that used the bundle:

```
  "optimizer": {
   "name": "SinkSGD_adv",
   "lr": 5e-06,
   "momentum": 0.95,
   "nesterov": true,
   "weightDecay": 1e-05,
   "stochasticRounding": true,
   "orthogonalGradient": "iterative",
   "compiledOptimizer": true,
   "sinkhornIterations": 5,
   "statePrecision": "fp32",
   "trainable": [
    "encoder",
    "quant_conv",
    "post_quant_conv",
    "decoder"
   ],
   "orthogonalSinkhorn": true,
   "normedMomentum": true,
   "snrCond": true,
   "spectralNormalization": true,
   "geometricWd": true,
   "cautiousWd": true
  },
```

So the complete settings are: SinkSGD_adv, lr 5e-6, momentum 0.95, nesterov true, weight_decay 1e-5, stochastic_rounding true, compiledOptimizer true, sinkhorn_iterations 5, orthogonalGradient iterative, orthogonalSinkhorn true, normedMomentum true, snrCond true, spectralNormalization true, geometricWd true, cautiousWd true, statePrecision fp32. Note that momentum, nesterov, weight decay, the Sinkhorn count and the compile are NOT part of the starred eight; they are the surrounding recipe that happened to be running, and only the eight were ever called gold

Model context for that config, from `MEMORY-pre-cleanup-2026-10-06.md:73-77`: an SDXL VAE (83.65 M params, 211 trainable tensors on the unfrozen path) with an OFTv2 rotation adapter where "only 72 S-param tensors" train, 3010743 params in total, block 32, lr 5e-6 constant, batch 12, 400 steps, initialised from a dimix checkpoint. The earlier gold-star arms that produced the starred table rows ran the same eight switches at lr 2e-5 (`run_arms.py:52`, default `lr="2e-5"`), one at lr 2e-5 to 2e-6 linear anneal, and the marathon at 5e-6

The dreamsim mirror is the same bundle under a different name. `dreamsim-dinov3/configs/rotasim_r2_sinkv.json:108` describes it as "FULL Sink-V (Koratahiu) stack on SinkSGD_adv: normed_momentum=true (NtM), snr_cond=true (AASS variance/confidence preconditioning), orthogonal_sinkhorn=true, sinkhorn_iterations=5, orthogonal_gradient=iterative (matrix-wise OrthoGrad), state_precision=fp32 explicit, centered_wd_mode=full (INERT: centered_wd=0.0 => anchor term off; mode only names its storage format), compiled_optimizer=true". The real numbers are in `dreamsim-dinov3/release/fgresq-di-mixture-288-v1/config.json:61-78`: momentum 0.95, nesterov true, nesterov_coef null, normed_momentum true, snr_cond true, orthogonal_sinkhorn true, sinkhorn_iterations 5, cautious_wd false, geometric_wd false, centered_wd 0.0, spectral_normalization true, orthogonal_gradient iterative, state_precision fp32, compiled_optimizer true. The two weight-decay switches are false there for a stated reason: weight_decay 0.0 makes them inert, so dreamsim's "full stack" is the eight minus the two dead WD flags

## What evidence backed them

A preference first, then a measured comparison that came out a tie, then per-knob ablations that came out negative. That is the whole evidence file, and it is thinner than the name suggests

The bundle was never discovered in a sweep. It was requested by name, assembled, and then tested as one lump against the standard optimizer three separate times. `run_grid.py:66-68` states the result and the caveat in the source: measured three times, a tie each time, "which is unresolved, not equal", and "No individual switch was ever ablated". `docs/history/adapters.md:191` repeats the same warning: no individual switch, "spectral_norm solo, normed_momentum/snr_cond, centered_wd, momentum β, sinkhorn 5-vs-3-vs-1 all untested"

The three matched comparisons, all measured, all ties or small losses for the bundle:

The banked two-seed full-path comparison at 512, `speedy-sdxl-vae-trainer/docs/history/campaigns.md:57`: incumbent 24.05 and 24.77 (mean 24.41/104.44), gold-star 24.75 and 24.88 (mean 24.82/105.46). The run-to-run spread in that protocol is +-0.72 FID, so a 0.41 gap on the mean is inside noise, and the bundle is nominally worse by 0.41 not better. The same file records the single-draw "win" that started the belief: gold-star won at 256 by -0.80 FID and lost at 512 by +0.70, "so the two protocols disagree at this effect size"

The night-2 fine-tune comparison at 400 steps on the dimix stack, `scratch/night2/JOURNAL.md:72-74`: goldstar-s1 39.605, goldstar-s2 39.460, std-s1 39.282, annotated "no-⭐ marginally best of the @400 set". A fourth seed-4 A/B on the mc3 stack, `JOURNAL.md:77-78`: goldopt 39.517 vs std 39.303

The verdict line, `JOURNAL.md:82`: "⭐ optimizer bundle = no gain, THIRD confirmation (dimix seed1 + s4-mc3 + banked 2-seed tie). Std keeps the crown by parsimony"

The only per-switch numbers in existence come from the other sister, `dreamsim-dinov3`, and they are negative. `dreamsim-dinov3/MEMORY.md:479` records one-variable arms against a blk16 incumbent at lr 1.2911e-4, 300 steps, test n=6068 with cluster CIs: `orthogonal_sinkhorn` False to True gives val 71.62 / test 70.22, -3.61pp, CI [-5.01,-2.16]; `normed_momentum` False to True gives val 72.58 / test 69.91, -3.92pp, CI [-5.31,-2.49]. Both CIs exclude zero, and the entry's own words are "NONE BEATS THE INCUMBENT; TWO ARE RESOLVED HARMFUL ... do not enable either on this recipe". The same table shows `sinkhorn_iterations` 5 to 1/10/20 is a no-op (73.20 / 72.97 / 72.92, all CIs covering zero) and `orthogonal_gradient` iterative vs disabled is equivalent (+0.28pp, CI [-0.55,+1.11]). spectralNormalization and the WD switches were never ablated anywhere

The full stack as a package in dreamsim, `dreamsim-dinov3/MEMORY.md:500`: at stock lr it was actively harmful, val 71.98 at 600, "rotations plateau 10x under baseline (applied ~0.05 vs ~0.49)" and "SINK-V confidence damping crushed the ramp at stock LR". At lr x2 it reached peak val 76.13 at step 1050 and test 73.73, "EQUIVALENT" to both comparators, which took 1050 steps against the incumbent's 550 or 300. The verdict there is "a SLOWER EQUIVALENT, not a better optimizer". The same evidence is restated in `configs/rotasim_r2_sinkv.json:109`

The one item in the surrounding recipe with a real measured win is not a gold switch at all: `compiledOptimizer` took 1002 ms to 645 ms per step at batch 12, about 1.55x, `speedy-sdxl-vae-trainer/README.md:302-307` as cited in `docs/opt-report-speedy-sdxl-vae-trainer.md:9`

One nuance worth keeping straight, because it is how the myth survives: the sister project's best exported checkpoint, the champion at 38.62 mean FID, was trained with the gold-star bundle (`JOURNAL.md:89-91`, file `dimix-goldstar-lr4-003600-seed1.safetensors`). That is because almost every arm in that campaign ran the bundle, not because a matched A/B ever showed it winning. The comparison against std exists only at 400 steps, where std won by 0.32

## Whether it transfers to our run

Mostly no, and partly not even expressible in our trainer

Our plumbing passes eight arguments and nothing else, `wd-tagger-trainer/wdtag/train.py:574-584`: momentum, weight_decay, sinkhorn_iterations, orthogonal_sinkhorn, normed_momentum, snr_cond, nesterov, stochastic_rounding, compiled_optimizer. `orthogonal_gradient`, `spectral_normalization`, `geometric_wd`, `cautious_wd`, `centered_wd` and `state_precision` are not exposed at all, so four of the eight gold switches would be a code change before they were a config change. The vendor defaults we are actually running, `adv_optm/optim/SinkSGD_adv.py:57-86`, are orthogonal_gradient "disabled", spectral_normalization False, centered_wd 0.0, state_precision "auto", nesterov_coef None, geometric_wd False, cautious_wd False

The regime differs in the way that matters for these particular switches. The gold bundle was tuned on rotated weights whose gradients are wide and noisy and whose adapter is 3.0 M params against a 83.65 M model. Ours is a SigLIP2 tower with OFTv2 block 128, exact Cayley, oft_conv true, plus a 12476-way linear head at 19.18 M params that is roughly 94 percent of the trainable count. The rank gate means the head and the 73 OFT tensors are all 2-D and all enter Sinkhorn, so orthogonalSinkhorn and the iteration count cost real element work on the largest tensor in the run, and iterative OrthoGrad would project across semantically unrelated blocks of those stacked rotation parameters, exactly the objection dreamsim raised

Weight decay 0 makes geometricWd and cautiousWd no-ops, so two of the eight cannot do anything in our config. fp32 state is a memory and bandwidth cost with no measured benefit, and it partly defeats the reason we run bf16 with stochastic rounding. The Sinkhorn count is already 5 and measured a no-op, which is the one place we are accidentally already gold

The step-size problem is the load-bearing one. SinkSGD_adv is normalised, so the step is about lr regardless of gradient magnitude, and our lr_oft 5e-4 and lr_head 1.2e-4 were calibrated against exactly the plain update we run now. normed_momentum plus snr_cond is a directional damper that dreamsim measured at 10x under baseline rotation gain and 4.2 points of accuracy at matched lr; orthogonalSinkhorn rewrites the update before it is applied; iterative OrthoGrad strips the radial component. Any one of them changes the effective step, so enabling them silently invalidates the calibration rather than refining it, and the fix would be a fresh lr sweep, not a flag

## Verdict

| gold setting | gold value | ours | verdict | reason and risk |
|---|---|---|---|---|
| orthogonalGradient | iterative | disabled (vendor default) | ignore | dreamsim iterative vs disabled +0.28pp CI [-0.55,+1.11] equivalent; projects across unrelated rotation blocks; changes step direction, no measured gain anywhere |
| orthogonalSinkhorn | true | false | ignore | the only resolved-harmful switch with a tight CI: -3.61pp test, CI [-5.01,-2.16]; also the most expensive place to put it is our 12476-way head |
| normedMomentum | true | false | ignore | resolved harmful alone at -3.92pp test, CI [-5.31,-2.49]; damps the step and would invalidate the 5e-4 and 1.2e-4 calibration |
| snrCond | true | false | ignore | requires normedMomentum, so it inherits that verdict; never isolated; the 10x rotation damping was measured with this pair on |
| spectralNormalization | true | not plumbed | ignore | never ablated in either sister; adds compute on every 2-D tensor; no number behind it at all |
| geometricWd | true | not plumbed | ignore | inert at weight_decay 0, which is our measured setting; zero effect by construction |
| cautiousWd | true | not plumbed | ignore | inert at weight_decay 0; dreamsim leaves it false for exactly this reason |
| statePrecision | fp32 | auto under bf16 | ignore | no measured benefit, and fp32 optimizer state fights the bf16 plus stochastic-rounding design; only revisit if an instability appears |
| momentum | 0.95 | 0.95 | already adopted | ours matches; no action |
| nesterov | true | false | ignore for now | part of the untested surrounding recipe, not one of the starred eight; cheap future A/B, not a free win |
| weightDecay | 1e-5 | 0 | ignore | our wd 0 is graded M in `docs/best-settings.md:102`, measured not to help multi-label; the 1e-5 is the sister's regression setting, not a gold switch |
| sinkhornIterations | 5 | 5 | already adopted | ours matches; measured a no-op from 1 to 20, so do not spend a run here |
| compiledOptimizer | true | true | already adopted, keep | the one measured win in the whole neighbourhood, 1002 to 645 ms, and it is independent of the gold eight |
| lr | 5e-6 (OFT run), 2e-5 (arms), 1.2911e-4 (dreamsim) | 5e-4 oft and 1.2e-4 head | ignore | context-bound numbers on different models; our triple is measured for this model and this batch |

Net: adopt nothing new. Three settings already match by accident, one of them for a reason that is actually measured, and the other two because 5 and 0.95 are defaults that nobody had a reason to move

If the bundle is ever revisited, the cheapest honest experiment is a single 200-step A/B with normed_momentum and snr_cond both on at the current lr, scored on macro at 0.40 against the same seed, and read with the expectation from dreamsim that it will look worse before any rescue by LR. Do not run the other six as a bundle, because that experiment has already been run three times and answered the same way, and do not enable orthogonalSinkhorn on the 12476-way head to find out what a resolved-harmful switch does at 94 percent of the parameters
