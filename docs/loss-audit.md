# Loss implementation audit — against the authors' code, not just the papers

Date: 2026-10-22. Scope: `wdtag/losses.py` (96 lines, untracked in git) versus the reference implementations. Corrected, tested code: `scratch/sophie/losses_corrected.py`. Test harness that produced every number below: `scratch/sophie/test_losses.py`.

## 0. Read this first: the file on disk is not the file your message describes

`wdtag/losses.py` already contains a corrected ASL — probability-space clip, one-sided gamma, element-mean reduction, and a docstring that says an earlier version shifted the logit. `git status` shows the file as untracked (`?? wdtag/losses.py`), so there is no history to diff against; the version quoted in the audit request (logit shift + `/n_pos`) no longer exists on disk.

Verified numerically (`scratch/sophie/test_losses.py`): the on-disk `asl()` equals a verbatim port of `AsymmetricLoss.forward` divided by `B*C` to within 1e-6. ASL: MATCHES. Two nits remain, listed in §2. The rest of the audit therefore concentrates on `cb_ntr`, `class_balanced_weights`, and normalisation — and on the old ASL, because if any copy of it survives in a config, a log, or another branch, the numbers in §2 are what it computes.

## 1. Verdicts

| function | verdict | reason (one line) |
|---|---|---|
| `bce` | MATCHES | `F.binary_cross_entropy_with_logits` default = element-mean. It is the control and defines the loss scale. |
| `asl` (on disk) | MATCHES | equals the reference to 1e-6; only the focal detach and a cosmetic clamp differ. |
| `asl` (the version in your message) | DEVIATES | clip in logit space instead of probability space; a different focal weight; `/n_pos` instead of `/B*C`. Numbers in §2. |
| `cb_ntr` | DEVIATES | λ is missing from inside the negative sigmoid. Loses 52% of the loss magnitude vs BALMS Eq 6, and loses the entire "tolerance" mechanism. |
| `class_balanced_weights` | MATCHES the formula, DEVIATES in effect | `(1-β)/(1-β^n)` and mean-1 are both Cui et al.; but on our count distribution mean-1 reduces the positive share of the loss from 0.259% to 0.019%. |
| reducers | INCONSISTENT across arms | BCE/element-mean vs ASL-official/sum vs your `/n_pos`. With `grad_clip=1.0` those are different optimisers. §5. |

## 2. ASL — your suspicion was right about the old code, and the fix is already in

Reference: `Alibaba-MIIL/ASL/src/loss_functions/losses.py`, `AsymmetricLoss.forward`:

```python
x_sigmoid = torch.sigmoid(x)
xs_pos = x_sigmoid
xs_neg = 1 - x_sigmoid
if self.clip is not None and self.clip > 0:
    xs_neg = (xs_neg + self.clip).clamp(max=1)      # ← PROBABILITY space
los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
loss = los_pos + los_neg
pt0 = xs_pos * y
pt1 = xs_neg * (1 - y)                              # pt = p if t>0 else 1-p
pt = pt0 + pt1
one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
one_sided_w = torch.pow(1 - pt, one_sided_gamma)
loss *= one_sided_w
return -loss.sum()
```

and `train.py`: `criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)` … `loss = criterion(output, target)` — no division anywhere.

So: correct on all three of your points. `sigmoid(x - clip)` is not the same operation, `p_negγ_neg` is not the reference's weight, and the reference sums rather than dividing by positives.

Why the logit shift is worse than a cosmetic difference. The clip exists to bound `log(1-p)` from below at `log(clip)` = log 0.05, i.e. to stop hard negatives being driven to −∞. Shifting the logit instead leaves `log(1 - σ(z−clip))` unbounded, so the negative loss grows with the logit. Per-element negative losses, γ_neg=4, clip=0.05 (from `test_losses.py`):

| z | p | reference `log(1−p+clip)`, weight `(p−clip)^4` | loss | logit-shift version | loss | ratio |
|---|---|---|---|---|---|---|
| +4 | 0.982 | −3.00, 0.932 | 2.03 | −3.95, 1.000 | 3.68 | 1.81× |
| +3 | 0.953 | −2.33, 0.664 | 1.55 | −3.15, 0.951 | 2.45 | 1.58× |
| +2 | 0.881 | −1.78, 0.476 | 0.85 | −2.08, 0.876 | 1.22 | 1.45× |
| 0 | 0.500 | −0.60, 0.041 | 0.025 | −0.67, 0.053 | 0.038 | 1.54× |
| −2 | 0.119 | −0.07, 2.4e-6 | 2e-6 | −0.12, 2.0e-5 | 2e-5 | 12.5× |

The old version penalises hard negatives 1.4–1.8× more than plain BCE — it is an anti-ASL: it sharpens exactly the suppression that makes rare tags under-fire at 0.40.

And the `/n_pos` reducer is not a positives-only normaliser. `sum/n_pos = element-mean × (B·C)/n_pos = element-mean / ρ`, where ρ is the batch positive density. At ρ = 0.21% that is a ×480 global scale-up; the per-element weight of a positive relative to a negative is unchanged. So the reducer did not do what its position in the code suggests, and it made the arm's loss ~480× the control's — which, with `grad_clip=1.0`, means the arm was clipping on every step while the control never clipped. (Reference ASL is ~600000× the element-mean at B=48, C=12476. Adam absorbs that; a grad-clipped normalised optimiser does not.)

Two remaining nits in the current on-disk ASL:
1. no `.detach()` on the focal weight. Official `train.py` passes `disable_torch_grad_focal_loss=True`, so `w` is a constant w.r.t. the gradient. This is not cosmetic: measured relative L1 difference in `dL/dz` is 58% between the detached and non-detached forms on a realistic batch. If you want to reproduce reported ASL numbers, detach; if you want "focal as pure reweighting", detach. Either way, know which one ran.
2. `torch.pow((1.0 - pt).clamp_min(0.0), gamma)` — `pt <= 1` always, so the clamp is a no-op, but it would also hide a future bug. Harmless.

## 3. CB-NTR — this one is genuinely wrong, and NTR's origin is misattributed

### 3a. The citation is wrong
`NTR from 1805.00909` appears in the module docstring and in the request. Verified via the arXiv API:
1805.00909 is "Reinforcement Learning and Control as Probabilistic Inference: Tutorial and Review" (Levine), a control tutorial. LDAM is 1906.07413, and its full text contains no occurrence of "toleran" at all. DBL's references cite Cao et al. only for the margin. BALMS never cites Cao.

The canonical definition is DBL, [2007.09654](https://arxiv.org/abs/2007.09654) Eq 7/9, restated
as BALMS Eq 6/8:

```
L_NT-BCE = (1/C) Σ_i [ y_i log(1 + e^{-(z_i - v_i)}) + (1/λ)(1-y_i) log(1 + e^{λ(z_i - v_i)}) ]   (DBL 7)
b_hat_i = log(p_i/(1-p_i))  (BCE optimum, DBL 8)      v_i = -κ b_hat_i   (DBL 9)
```

### 3b. The bias is fixed, not learned — and it is not a tail-lowering term
BALMS, verbatim: "q = σ(z−v) for positive instances and q = σ(λ(z−v)) for negative ones. The v_i can be estimated by minimizing the loss function at the beginning of training with a scale factor κ and class prior p_i = n_i/N, so that b̂_i = −log(1/p_i − 1), v_i = −κ × b̂_i." Their config:
`logit_reg=dict(init_bias=0.05, neg_scale=2.0)` → κ = 0.05.

For a 0.1% tag, `logit(p) = −6.9`, so `v = −0.05 × (−6.9) = +0.345` and the loss sees `z − v`, i.e. the tag starts 0.35 logits below its BCE optimum. The fixed NTR bias pushes rare classes down, not up. Everything BALMS/DBL gain on the tail comes from λ, not from v.

This matters twice over. First, the original heads-and-losses report (now private) says NTR has "a learned class-specific bias v_i that lowers the threshold for tail classes" — that was my error in the original report, it propagated into the implementation, and it is now the project's source of truth. It should say: fixed bias from the prior, κ=0.05, no threshold-lowering effect. I did not edit the doc because you are editing it; please apply.

Second, the idea of a learned per-class bias is still the right idea for a fixed 0.40 protocol — it is just ours, not BALMS's, and BALMS's 18.40 cannot be cited as evidence for it.

### 3c. Where the on-disk version deviates
```python
p   = torch.sigmoid(z)                       # z = logits + class_bias
neg = (1-y) * torch.log(1-p) / ntr_lambda    # λ only outside
```
Missing: `q_neg = σ(λ(z−v))`. BALMS Eq 6 has λ on both sides of the negative branch, and the reference code shows it as a logit/weight rescale:

```python
logits = logits * (1 - labels) * self.neg_scale + logits * labels
weight = weight / self.neg_scale * (1 - labels) + weight * labels
```

Magnitude of the difference, at the place it matters — a rare class whose logits sit at z ≈ −7 (every negative suppressed): gradient on the negative branch is `σ(−7) = 9.0e-4` under plain BCE, `4.5e-4` under the on-disk version, and `σ(2×−7) = 9.1e-7` under NTR. NTR's point is that third number: stop pushing. Dividing by λ changes nothing about the suppression; σ(λ·) ends it. Measured on a realistic batch: on-disk `cb_ntr` is 0.476× the value of a verbatim BALMS Eq 6 (float32); the corrected version matches Eq 6 to 1.2e-11 in float64.

### 3d. Is the softmax→sigmoid port a problem? No — and that is the one thing that is fine
Your feeling was that NTR "was formulated for softmax multi-CLASS and porting it to independent sigmoids changes the meaning of the bias." Half right, and the half that is wrong is reassuring:

Wrong half: NTR was never softmax. DBL Eq 7 is explicitly a sum of independent BCE terms, one per label, and DBL is a multi-label paper (COCO-MLT / VOC-MLT). BALMS's Eq 6 is the same object. So there is no softmax→sigmoid translation step and no semantic break from that direction
Right half: the derivation of v (DBL Eq 8) assumes the bias is the only variable and that features are mean-zero, so the optimal b is the class prior log-odds. That derivation is per independent Bernoulli, so it holds per label — but it also means v is a prior-correcting term, i.e. the thing a learned bias would also discover, except that learning it costs nothing and cannot be wrong

Correct multi-label form (per label i, independent): `q_i = σ(z_i − v_i)` on positives, `σ(λ(z_i − v_i))` on negatives, negative branch × `1/λ`, per-label class weight `α_i`, element-mean reduction. That is `losses_corrected.cb_ntr`, and it is verified against Eq 6.

### 3e. One more thing to decide: the model has two per-class biases
`model.class_bias` is an `nn.Parameter(n_labels)` in the head param group, and the final `Linear` also has a bias which `init_head_bias()` sets to `logit(prior)` at full strength. Effective init = `logit(prior) + 0`. Two consequences: (i) at `lr_head` both receive a normalised step every iteration, so the per-class bias effectively moves ~2× per step relative to a single parameterisation; (ii) if you also adopt the fixed `ntr_bias_from_prior` (κ=0.05) you get `(1+κ)·logit(prior)`, i.e. doubly suppressed tails. Pick one owner for the class bias. My recommendation: keep the `Linear` bias (already prior-initialised, free) and drop `class_bias`, or zero the `Linear` bias and keep `class_bias` — not both.

## 4. `class_balanced_weights` vs Cui et al.

Form: MATCHES. Eq 6: `CB(p,y) = (1−β)/(1−β^{n_y}) · L(p,y)`;  your `(1-β)/(1-β^c)` is right
Per-class vs per-sample: you deviate from Cui's literal Eq 11, correctly. Cui's sigmoid form weights each sample by the count of its ground-truth class `n_y`. For a 26-positive multi-label row there is no single `n_y`; the usable generalisation is per-label `n_i`, which is what BALMS does (`CB_mode='by_class'`) and what you do. Keep it
Normalisation: MATCHES Cui (Σα = C, mean 1), but it is a trap for us. Cui normalise so that "the total loss is roughly in the same scale"; with 12476 labels of which 1686 have zero positives and 2191 have exactly one (measured on the 10k cache), the mean is set by the near-empty labels. After mean-1, those get the largest weights on their negatives, while the labels that actually hold positives fall below 1. Measured positive share of an element-mean weighted loss:

  | weighting | w at n≤1 | w at n≈1e3 | w at n=6840 | positive share of loss |
  |---|---|---|---|---|
  | plain BCE | 1.00 | 1.00 | 1.00 | 0.259% |
  | CB β=0.9, mean-1 | 2.00 | 0.200 | 0.200 | 0.063% |
  | CB β=0.999, mean-1 (yours) | 2.14 | 0.0034 | 0.0021 | 0.019% |
  | CB β=0.999, no norm | 1.00 | 0.0016 | 0.0010 | 0.019% |
  | power α=0.5, mass-preserved | 9.32 | 0.295 | 0.113 | 0.046% |
  | power α=0.25, mass-preserved | 3.56 | 0.633 | 0.391 | 0.097% |

Any class weighting reduces the positive share here, because the label count is dominated by the 3877 labels that can contribute almost no positives. The mean-1 version makes it 13× worse than the un-normalised one. If you want class weighting to mean anything, preserve the positive mass (`normalize="pos_mass"`) or weight only the positive term (see §6).
β: your 0.999 is not defensible at either scale. Cui et al. set `β = (N−1)/N` with N the number of "unique prototypes" and pick from {0.9, 0.99, 0.999, 0.9999} by validation (their Table 2 shows 0.9 best on CIFAR-100-LT, 0.9999 on CIFAR-10-LT). On the 10k cache n_max = 6840 → β ≈ 0.99985; on the full corpus n_max ≈ 2e6 → β ≈ 0.9999995. At β=0.999 everything above n ≈ 6000 sits at the floor weight, so the knob is saturated exactly for the head tags. BALMS used 0.9, which is a ≤3.7× tail:head ratio — i.e. their "CB" was nearly flat and most of CB-NTR's advantage over FL came from λ and the focal term

## 5. Loss normalisation — what each reference does, and what to use

| implementation | reducer | per-element ratio positive:negative |
|---|---|---|
| `F.binary_cross_entropy_with_logits` (control) | mean over B×C | 1 : 1 |
| BALMS / DBL `ResampleLoss` | `reduce_loss(..., 'mean')` | 1 : 1 (shaped only by focal/weights) |
| Cui et al. | mean, with Σα = C | 1 : 1 |
| ASL official | `-loss.sum()`, no division anywhere | 1 : 1 (a global B·C scale) |
| your old ASL | `sum / n_pos` | 1 : 1 (a global 1/ρ scale) |

Two facts that collapse most of the argument:

1. None of these reducers rebalances positives against negatives. The ratio is 1:1 in every one; only focal weights, `pos_weight`, class weights and λ change it. `sum/n_pos` sounds like "average over positives" but is exactly `element-mean / ρ`.
2. The only thing the choice changes is the global gradient scale, which matters here because `grad_clip=1.0` and a normalised (Muon-like) optimiser interact with scale in ways Adam does not. At ρ=0.2%, the spread between the options is ~500× (yours) to ~600000× (reference ASL).

Use element-mean for every arm — `mean` over B×C, which is what the control already is. Then the only difference between arms is the loss shape, which is the experiment you want to run. Set the imbalance on purpose, with a named knob: `pos_weight`, class weight, γ, or λ.

At 0.2% density, the thing that actually determines whether a rare tag crosses 0.40 is the relative gradient mass on its positives, and every reference number that exists (BALMS, DBL, ASL) came from a setting where the imbalance was attacked explicitly by such a knob — not by the reducer.

## 6. What I would run

1. `bce_pos_weight` (1 line: `F.binary_cross_entropy_with_logits(..., pos_weight=w)`), with `w_i = ((N−n_i)/n_i)^0.5`. It is the only variant that moves tail positives without inflating the negatives of the 1686 empty labels, it keeps the arm at the control's scale, and it is a clean single-knob ablation. Not from any paper — that is the point.
2. `cb_ntr` corrected (λ inside, fixed κ=0.05 bias or zero bias + keep the head's prior bias), β=0.9, weights `pos_mass`-normalised. This is the BALMS-faithful arm; label it "BALMS-faithful" and expect the λ term, not the CB term, to do the work.
3. ASL as it now stands on disk (add `detach_focal=True`). Its evidence is +0.8 macro mAP at Open Images scale and it is the cheapest negative-side reweighting.

All three need one extra branch at `train.py:438-445` for the weight construction, and all three should log the measured "positive share of loss" at step 0 so you can see whether the knob did what you think.

## 7. Is a faithful port even worth it for our metric?

Honest answer: the CB part, no; the λ part, yes; and neither is worth more than a couple of days.

Reasons, in order of weight:

1. BALMS's headline 18.40 macro came at a threshold grid-searched over ±0.10 around 0.5 on val and applied to test. Under a fixed 0.40 those absolute numbers do not transfer, and the ordering of losses at a fixed threshold is not guaranteed to be the ordering at each model's best threshold.
2. Their CB weights (β=0.9) span only 3.7×, so "CB-NTR ≈ DB" is consistent with CB contributing almost nothing. The 13.94 → 18.40 gain decomposes mostly into λ and focal, i.e. exactly the ASL family you are already going to run.
3. Their tail labels are `≤15` examples. Ours: 97% of the macro gap sits in tags with `<1000` training examples, which in a 5.3M-image corpus is a hard tag, not a data-starved one. The scarcity mechanism that makes BALMS spectacular (BCE collapsing to 0.02) is not our mechanism.
4. Our own number bounds the prize: per-tag-oracle macro is 0.3879 vs the bar's 0.5579 at a single 0.40. Loss work can capture part of that 0.106, not all of it.

So: run `bce_pos_weight` and corrected `cb_ntr` as screening arms (they are cheap and the harness already exists), but do not invest in a faithful CB reproduction beyond that, and do not expect the loss family to close a 0.17 gap on its own.
