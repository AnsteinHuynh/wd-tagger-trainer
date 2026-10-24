# Tail-class losses under a fixed global threshold

The real measured evidence lives here, and one paper is almost purpose-built for us: BALMS ([2109.04712](https://arxiv.org/abs/2109.04712), EMNLP 2021) compares BCE, focal (FL), class-balanced
focal (CB), rebalanced FL, negative-tolerant-regularized FL (NTR-FL), CB-NTR and distribution-balanced (DB) on an 18211-label long-tailed multi-label task (PubMed, 225k docs) and Reuters, reporting micro and macro F1 — our metric family — at one threshold chosen for micro-F1 on val.

PubMed, total micro/macro F1 (their Table 2):

| loss | total miF / maF | tail miF / maF |
|---|---|---|
| BCE | 26.17 / 0.02 | 0.00 / 0.00 |
| focal | 58.30 / 13.94 | 8.58 / 0.86 |
| class-balanced focal | 58.57 / 13.67 | 9.92 / 1.01 |
| rebalanced FL | 57.90 / 14.66 | 11.45 / 1.15 |
| NTR-FL | 60.92 / 16.99 | 15.86 / 1.82 |
| CB-NTR | 61.07 / 18.40 | 24.15 / 2.97 |
| DB | 60.63 / 19.19 | 24.15 / 2.97 |

Reuters (90 labels), same direction, stronger: BCE macro F1 47.32, tail micro 0.00 → DB macro 64.47, tail micro 48.89.

1. NTR is the dominant ingredient, not the rebalancing. FL 13.94 → NTR-FL 16.99 → CB-NTR 18.40. NTR applies λ twice on the negative branch — `q = σ(λ(z − v))` inside and `1/λ` outside — plus a fixed bias `v_i = −κ·logit(p_i)`, κ = 0.05. Corrections to an earlier version of this note (source: our own implementation audit, `docs/loss-audit.md`): the bias is not learned, and it does not lower the threshold for tail classes. For a 0.1% tag v = +0.345, so tails start ~0.35 logits below their BCE optimum. The tail gain comes from λ. A learned per-class bias is a defensible extension and is the thing that could move operating points inside the trained score, but it is our idea, not BALMS's, and BALMS's 18.40 is not evidence for it. Canonical source is DBL arXiv 2007.09654 Eq 7/9 (restated as BALMS Eq 6/8) — not arXiv 1805.00909, which is an RL/control tutorial and is cited wrongly in an earlier version.
2. CB-NTR ≈ DB. Class-frequency weights (already present for `init_head_bias`) get within ~0.8 macro of the co-occurrence-aware DB; the 12476² matrix is not required for most of the gain.
3. Caveat that limits transfer: PubMed's BCE collapses to 0.02 because labels there have ~12 examples each. Our labels average ~13k positives in 5.3M images (mean 128 per label in the 50k val; median far lower, but not zero-shot). We are not in the scarcity regime that makes DB spectacular — expect a fraction of their delta.

Supporting, at other scales:
DBL ([2007.09654](https://arxiv.org/abs/2007.09654), ECCV 2020), COCO-MLT/VOC-MLT: tail mAP 24.25 (ERM) → 51.06 (DB-Focal); total 41.27 → 53.55. Much is focal alone (ERM → focal: 24.25 → 42.14). Tens of labels: mechanism, not scale evidence
ASL ([2009.14119](https://arxiv.org/abs/2009.14119), ICCV 2021), Open Images V6 (9M images, partially annotated): CE 84.8 micro / 92.0 macro mAP, focal 84.9 / 92.2, ASL 86.3 / 92.8 — +1.5 micro but only +0.8 macro. On COCO, CF1 80.3 → 81.4 while OF1 got worse (82.3 → 81.8). Real, cheap, well-measured — smaller on macro than its reputation
Same task, same scale: wd-swinv2-tagger-v3 v2.0 added "tag frequency-based loss scaling to combat class imbalance" and macro F1 went 0.4411 → 0.4541 (+0.013) — the only same-domain, same-label-space number for loss reweighting. This is NOT a fixed-0.40 result. `heads-wrong-under-040.md` §2 records that those scores are taken at each model's P=R threshold, which overstates the value at a global 0.40 by roughly 30-40% relative. Use it as evidence that a carefully-scaled frequency weighting did not destroy the model, not as evidence of its size
ZLPR ([2208.02955](https://arxiv.org/abs/2208.02955)): 22 comparisons (11 text datasets × 2 backbones); best macro-F1 in 8/22, micro-F1 in 5/22; authors concede DL2 beats it on micro. A ranking loss, threshold-free, for uncertain positive counts (our 18–32 case) — mixed results and no calibrated 0.4 meaning → do not adopt under this protocol
Label smoothing for BCE multi-label lowers predicted probabilities and destroys calibration in the recall-reducing direction (Müller et al., [1906.02629](https://arxiv.org/abs/1906.02629)). Wrong
direction. Skip.
Plain focal: cheap, clearly non-zero (largest ingredient in DB's gain), but ASL dominates it at equal cost and its positive-side focusing moves calibration we don't need

What measurably raises macro F1 at a fixed threshold: λ-based negative tolerance (the NTR/ASL family) and class-frequency weighting, in that order — and note the gain is λ's, not the fixed bias's (see point 1). Both are ~20 lines in `train.py` (the loss is one `F.binary_cross_entropy_with_logits` call at `wdtag/train.py:572`).
