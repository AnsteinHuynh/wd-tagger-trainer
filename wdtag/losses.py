"""Loss variants for multi-label tagging, selected by `cfg.loss`.

Corrected 2026-10-22 against the primary sources; each defect found by the audit is noted at the
function it affected, so it does not come back. Sources: ASL arXiv 2009.14119 (reference impl
Alibaba-MIIL/ASL), NTR from DBL arXiv 2007.09654 Eq 7/9 restated as BALMS Eq 6/8 -- NOT arXiv
1805.00909, which is an RL tutorial cited wrongly in earlier notes -- and CB loss arXiv 1901.05555.

Normalisation contract: every loss here reduces with an element-mean over (batch x labels), so arms
differ only in loss SHAPE, not gradient scale. An earlier version divided by the positive count,
which at 0.2% density is ~480x the control: `grad_clip=1.0` then clipped that arm on every step
while the control never did.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def bce(logits: torch.Tensor, y: torch.Tensor, pos_weight=None, **_) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)


def bce_pos_weight(logits: torch.Tensor, y: torch.Tensor, pos_weight=None, **_):
    """Re-weighted BCE with w_i = ((N - n_i)/n_i)**0.5 prepared by the caller.

    Not from a paper, on purpose: CB with mean-1 normalisation lets near-empty labels dominate the
    NEGATIVE term, and the positive share of the loss falls from 0.259% to 0.019%. This form lifts
    tail positives without inflating the negatives of labels that have no positives at all.
    """
    return F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)


def asl(logits: torch.Tensor, y: torch.Tensor, gamma_pos: float = 0.0,
        gamma_neg: float = 4.0, clip: float = 0.05, detach_focal: bool = True,
        eps: float = 1e-8, **_) -> torch.Tensor:
    """Asymmetric loss.

        p = sigmoid(x);  xs_neg = (1 - p + clip).clamp(max=1)   # clip in PROBABILITY space
        pt = p*y + xs_neg*(1-y);  gamma = gamma_pos*y + gamma_neg*(1-y)
        loss = mean(-(y*log(p) + (1-y)*log(xs_neg)) * (1-pt)**gamma)

    `detach_focal=True` matches the reference's disable_torch_grad_focal_loss: detached and
    non-detached gradients differ by 58% relative L1.

    DEVIATION FIXED: an earlier version clipped the logit, `sigmoid(x - clip)`. That leaves log(1-p)
    unbounded, so hard-negative loss grows and rare tags are suppressed harder -- an anti-ASL.
    Measured at gamma_neg=4, x=+3: reference 1.545 vs old 2.447.
    """
    return asymmetric_focal(logits, y, gamma_pos, gamma_neg, clip, detach_focal, eps).mean()


def asymmetric_focal(z: torch.Tensor, y: torch.Tensor, gamma_pos: float, gamma_neg: float,
                     clip: float, detach_focal: bool, eps: float) -> torch.Tensor:
    """ASL's per-element term, unreduced and unweighted; shared by `asl` and `asl_ntr`.

    Both losses are the same expression over different logits, so it lives here once. Callers apply
    their own weight and reduce.
    """
    p = torch.sigmoid(z)
    xs_neg = (1.0 - p + clip).clamp(max=1.0)
    los_pos = y * torch.log(p.clamp_min(eps))
    los_neg = (1.0 - y) * torch.log(xs_neg.clamp_min(eps))
    pt = p * y + xs_neg * (1.0 - y)
    gamma = gamma_pos * y + gamma_neg * (1.0 - y)
    mod = (1.0 - pt).detach() if detach_focal else (1.0 - pt)
    return -(los_pos + los_neg) * torch.pow(mod.clamp(min=0.0), gamma)


def ntr_bias_from_prior(priors: torch.Tensor, kappa: float = 0.0) -> torch.Tensor:
    """v_i = -kappa*logit(p_i), the BALMS fixed bias.

    kappa=0.0 on purpose: `init_head_bias` already sets the classifier bias to logit(prior), so
    kappa=0.05 would suppress tails twice over. Set kappa>0 only if that bias is zeroed.
    """
    b = torch.log(priors.clamp(1e-6, 1 - 1e-6)) - torch.log1p(-priors.clamp(1e-6, 1 - 1e-6))
    return -kappa * b


def cb_ntr(logits: torch.Tensor, y: torch.Tensor, ntr_bias: torch.Tensor = None,
           class_weight: torch.Tensor = None, ntr_lambda: float = 2.0,
           gamma: float = 2.0, **_) -> torch.Tensor:
    """BALMS Eq 6/8 (restating DBL Eq 7/9), with focal. LAMBDA APPEARS TWICE.

        z = logits - v;  z_eff = z for pos, lambda*z for neg
        bce = BCEWithLogits(z_eff, y);  pt = exp(-bce)
        w = class_weight * (1 for pos, 1/lambda for neg)
        loss = mean(w * (1-pt)**gamma * bce)

    Gradient at z=-7, lambda=2: BCE 9.0e-4, outer-1/lambda-only 4.5e-4, this 9.1e-7. Two measured
    defects: (1) only the outer 1/lambda, which is a uniform down-scaling and not tolerance
    (0.476x this); (2) a learned per-class bias -- BALMS's v is fixed from the prior and pushes rare
    classes DOWN by ~0.35 logits, so the tail gain comes from lambda, not from the bias.
    """
    z = logits if ntr_bias is None else logits - ntr_bias.unsqueeze(0)
    z_eff = torch.where(y > 0.5, z, z * ntr_lambda)
    bce = F.binary_cross_entropy_with_logits(z_eff, y, reduction="none")
    pt = torch.exp(-bce)
    w = torch.where(y > 0.5, torch.ones_like(y), torch.full_like(y, 1.0 / ntr_lambda))
    if class_weight is not None:
        w = w * class_weight.unsqueeze(0)
    return (w * torch.pow(1.0 - pt, gamma) * bce).mean()


def class_balanced_weights(counts: torch.Tensor, beta: float = 0.9,
                           normalize: str = "pos_mass", eps: float = 1.0) -> torch.Tensor:
    """Cui et al. Eq 6: w_i = (1-beta)/(1-beta**n_i), per class.

    beta=0.9, not 0.999: at 0.999 with our counts everything above n~6000 sits at the floor and the
    weighting stops acting. `normalize="pos_mass"` keeps total POSITIVE weight equal to unweighted;
    mean-1 inverts the intent, cutting positive share from 0.259% to 0.063% at beta=0.9 and 0.019%
    at beta=0.999 -- class weighting then REDUCES positive mass.
    """
    n = counts.clamp_min(eps).float()
    w = (1.0 - beta) / (1.0 - torch.pow(beta, n))
    if normalize == "pos_mass":
        w = w * (counts.sum() / (w * counts).sum().clamp_min(1e-12))
    elif normalize == "mean":
        w = w / w.mean()
    return w


def asl_ntr(logits: torch.Tensor, y: torch.Tensor, gamma_pos: float = 0.0,
            gamma_neg: float = 4.0, clip: float = 0.05, ntr_lambda: float = 2.0,
            ntr_bias: torch.Tensor = None, detach_focal: bool = True,
            eps: float = 1e-8, **_) -> torch.Tensor:
    """ASL x NTR: two half-losses on two branches, composed.

    ASL shapes both branches (asymmetric gamma + probability clip); NTR scales only the negative
    logits by lambda and weights them 1/lambda, so one confidently-wrong rare tag stops dominating.

        z = logits - v;  z_eff = z for pos, lambda*z for neg;  p = sigmoid(z_eff)
        w = 1 for pos, 1/lambda for neg;  loss = mean(w * asl_term(z_eff))

    Why compose rather than guess: NTR alone scored 0.0306 macro against the control's 0.0247, ASL
    alone 0.0410, and neither was ever run with the other. BALMS, the best-ranked combination in the
    research, is NTR x a positive-branch focusing loss -- i.e. exactly this. No paper runs ASL x NTR
    literally, so the arm is ours. lambda=1 with equal gammas reduces exactly to `asl`, which is the
    sanity check in `scratch/sophie/test_losses.py`.
    """
    z = logits if ntr_bias is None else logits - ntr_bias.unsqueeze(0)
    z_eff = torch.where(y > 0.5, z, z * ntr_lambda)
    w = torch.where(y > 0.5, torch.ones_like(y), torch.full_like(y, 1.0 / ntr_lambda))
    return (asymmetric_focal(z_eff, y, gamma_pos, gamma_neg, clip, detach_focal, eps) * w).mean()


def make_loss(name: str):
    return {"bce": bce, "asl": asl, "cb_ntr": cb_ntr, "asl_ntr": asl_ntr,
            "bce_pos_weight": bce_pos_weight}[name]
