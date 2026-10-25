# Rung displacement, CPU float32

Method: every config's SinkSGD_adv is built with exactly the kwargs of wdtag/train.py:574 except compiled_optimizer (CPU has no inductor here; the compiled path runs the same math), over the live u1 base-384 parameter inventory, with the three lr groups train.py builds: OFT at lr_oft, the last unfrozen block at lr_unfreeze, head and its norm at lr_head. The inventory is 91 tensors and 21901628 params, derived from the siglip2-base-patch16-384 safetensors header and the rules in wdtag/oft.py: 59 adapter tensors of (6,8128) and 12 of (24,8128), four 768x768 block weights plus the siglip2 biases, one 3072x768 and one 768x3072 block weight, the 1-D block norms and biases, two 768 norms and the 12476x768 head weight plus its 12476 bias. Every config sees the same init and the same constant random gradient per tensor, so the only variable is the optimizer. 140 steps, average of steps 60 to 140, and the weights are restored to their starting values after each measured step; the reported number is therefore the step at a fixed point in weight space with a warmed optimizer state, not the endpoint of an artificial coherent walk. Letting the weights walk does change orthograd: in that mode 48 measured 1.037x of 42 instead of 2.775x, because 140 identical steps push the parameter norm from 0.02 to about 0.07 and the orthograd projection weakens, so the fixed-weight numbers are the ones below

## Step displacement and ratios

| config | displacement | /42 | /49 | lr multiplier to restore 42 |
|---|---:|---:|---:|---:|
| 42_a1 | 1.203748 | 1.000 | 1.000 | 1.000 |
| 48_gold145 | 3.340916 | 2.775 | 2.775 | 0.360 |
| 49_plainfp32 | 1.203748 | 1.000 | 1.000 | 1.000 |
| 50_ntmsnr | 2.306204 | 1.916 | 1.916 | 0.522 |
| 51_orthograd | 2.306202 | 1.916 | 1.916 | 0.522 |
| 52_sinkhorn | 2.306203 | 1.916 | 1.916 | 0.522 |
| 53_spectral | 0.018179 | 0.0151 | 0.0151 | 66.22 |
| 54_wd1e5 | 0.018173 | 0.0151 | 0.0151 | 66.24 |
| 55_geowd | 0.018164 | 0.0151 | 0.0151 | 66.27 |
| 56_cautiouswd | 0.018170 | 0.0151 | 0.0151 | 66.25 |

The multiplier is 42's displacement divided by the config's, so it is the factor to apply to a run's lr to put its per-step movement back on the reference. 42 and 49 land on the same six digits, so optimizer_state_precision auto against fp32 is not a step-size change at fp32 parameters; the ladder baseline and the reference are the same optimizer in this probe. Rung 48 is 1.45x lr times the same 1.916 switch factor as rung 51, which is why it lands at 2.775x rather than 1.45x

## Per-shape ratios, common lr, against rung 49

Each shape is its own parameter at a common lr of 5e-4, so the 1.45x lr inside 48 is excluded and only the switch set shows. These are the per-element step ratios, which cancel the element count

| config | 1024x768 | 768x768 | 12476x768 | 1-D 12476 | OFT (6,8128) | OFT (24,8128) |
|---|---:|---:|---:|---:|---:|---:|
| 48_gold145 | 1.909 | 1.910 | 1.909 | 1.834 | 1.917 | 1.911 |
| 49_plainfp32 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 50_ntmsnr | 1.911 | 1.912 | 1.911 | 1.834 | 1.920 | 1.913 |
| 51_orthograd | 1.912 | 1.912 | 1.911 | 1.834 | 1.919 | 1.913 |
| 52_sinkhorn | 1.912 | 1.912 | 1.911 | 1.834 | 1.919 | 1.913 |
| 53_spectral | 0.0196 | 0.0183 | 0.0291 | 1.000 | 0.00030 | 0.00058 |
| 54_wd1e5 | 0.0197 | 0.0183 | 0.0291 | 1.000 | 0.00030 | 0.00058 |
| 55_geowd | 0.0196 | 0.0183 | 0.0291 | 1.000 | 0.00030 | 0.00058 |
| 56_cautiouswd | 0.0196 | 0.0182 | 0.0291 | 1.000 | 0.00030 | 0.00058 |

Two corrections to the shape premise, both material. First, no 1024x768 parameter exists in the u1 siglip2b-384 arm; the adapter parameter the optimizer actually updates is the strict-upper-triangle vector, (rank, 8128) with rank 6 for every 768-wide input and 24 for the 3072-wide MLP down projection. The 1024x768 column was measured as asked, but it is not a tensor this run contains. Second, orthograd is not shape-gated in displacement here: it is exactly neutral at fixed weights (51 and 52 are identical to 50 to seven digits), and the 1.834 to 1.920 spread across shapes in the ntmsnr rows is the only variation, about 4 percent on 1-D biases against 2-D matrices. The 0.4x drop seen for 51 in the drift mode was an artefact of the walking parameter norm, not a shape gate

Spectral is the shape gate, and it is severe. Under 53 the 1-D bias is untouched (1.000) because the spectral branch falls back to max-abs normalization on a sign update, the 12476x768 head is throttled 34x (0.0291), the 768x768 block weight 55x (0.0183), the 1024x768 matrix 51x (0.0196), and the real adapter is throttled 3333x at (6,8128) and 1724x at (24,8128). The reason is that wdtag sets no `_is_oft` attribute anywhere, so SinkSGD_adv does not take its OFT special case and treats (6,8128) as an ordinary rank-6 matrix; one power iteration estimates a spectral norm near 92 and the update is scaled to lr times sqrt(6/8128) divided by that, which is roughly 3e-4 of the unnormalized step per element. A run on rung 53 is therefore training a nearly frozen adapter while the head moves, and the measured displacement is dominated by the head: 0.01787 of the 0.01818 total

Weight decay at 1e-5 is invisible at this scale: with parameters near 0.02 the decay term is lr times 1e-5 times 0.02, about 1e-10 per step against an update near 5e-4, and 54, 55 and 56 differ from 53 by less than 0.1 percent. Geometric and cautious weight decay also only change direction, and barely that

## What is confounded

Rung 50 is a 1.916x throttle-up, uniform across shapes and groups, so comparing 50 against 42 measures a step-size change, not only a direction change. The constant gradient drives the normalized momentum buffer to the Sinkhorn fixed point, where each row's mean square is 1 and the SNR denominator 1 minus that mean sits at its clamp; the update then saturates toward a sign-like plus or minus 2 per element. That is the mechanism, and it is the reason the factor is so uniform. On real gradients the buffer rows are not exactly at the fixed point, so the real amplification may be smaller than 1.916, but the direction of the effect is the same

Rung 51 and 52 are pure direction changes: their displacement is identical to 50 to seven digits, so orthograd and orthogonal_sinkhorn alter where the step points without changing how far it goes

Rungs 53, 54, 55 and 56 are a 66x throttle-down, and essentially all of that is spectral normalization crushing the adapter, not a change of direction. The wd switches w0 to w2 move the displacement by under 0.1 percent on top of it. Any result from these rungs is confounded by step size: the adapter lr would need about 3300x to restore its reference movement, the head about 34x, and the biases 1x, so no single lr multiplier restores the reference

The one-line version: rung 50 is throttled up 1.92x, 51 and 52 are displacement-neutral, and 53 to 56 are throttled down 66x by spectral normalization on untagged adapter tensors, so 50 and 53 to 56 cannot be read as direction-only ablations

## Not measured on CPU

The compiled_optimizer=True path: inductor has no MSVC in this environment, and it would not change the arithmetic anyway. bfloat16 parameters and stochastic rounding: the task asked for float32 and the phase3 switch is about the optimizer, so no bf16 path was exercised; on GPU 42's auto precision tracks bf16 parameters, but every config here was measured at fp32 params where auto and fp32 coincide. Real loss gradients: a constant random gradient was used, as the existing verifier does, so the SNR saturation and the spectral power iteration see a stationary update rather than the moving one of training. The model itself was not instantiated: the inventory comes from the checkpoint header plus wdtag/oft.py's construction rules, which avoids loading the tower and matches the group assignment in train.py

Raw numbers: scratch/sophie/rung-displacement-freeze.json, produced by scratch/sophie/measure_rung_displacement.py --freeze
