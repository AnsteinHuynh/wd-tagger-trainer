# The ranked shortlist

## 1. Replace plain BCE with class-balanced focal + negative-tolerant regularization (CB-NTR)

Mechanism. (a) down-weight easy negatives so positive gradients survive 12451-way negatives per image; (b) multiply each class's loss by a frequency weight (1/√freq or effective-number); (c) give every class a learned bias v_i and scale negatives by 1/λ, moving each class's 0.40 crossing point during training, not at eval — the one family both legitimate under a fixed threshold and unbounded by the 0.3879 ceiling.

Evidence. BALMS at 18211 labels, DBL, ASL and the same-domain +0.013 all point the same way; the numbers are in [tail losses](heads-tail-losses.md): macro F1 13.94 → 18.40 → 19.19, tail micro-F1 8.58 → 24.15.

Cost. `train.py:572` is a one-line loss call; loss + config knobs ≈ 2–3 h, no co-occurrence matrix (CB-NTR ≈ DB). Screening arm = existing 5–10k-step probe (~2 h).

Risk. It trades the axis we lead: micro precision 0.7068 vs the bar's 0.7033. λ, γ and the CB exponent need a small sweep or micro P collapses. Label noise interacts with negative down-weighting: ASL's hard clip can amplify a spurious "positive" pattern. Watch macro P (0.4608 today).

## 2. Post-hoc class-prior logit adjustment, τ=1, fitted on train priors only

Mechanism. z_i ← z_i − log(π_i/(1−π_i)); a per-class vertical shift removing the frequency bias. One global 0.40 threshold still applies to every class.

Evidence. Menon et al. and Decoupling: post-hoc LA beats ERM, dominates weight normalization, moves few-shot from 61.7 to 69.3 — measured numbers in [bias fixes](heads-bias-fixes.md).

Cost. 30–60 min from the existing probability cache — the cheapest experiment here, and it measures how much of our deficit is prior bias. Then 1–2 h to wire into eval.

Risk. Inside the per-class-offset family, so its ceiling is the 0.3879 oracle and realistically much less. Must be fitted on training priors (val fitting = threshold tuning by another name), and a reviewer can reasonably call it per-class thresholds. Run it as a diagnostic; ship only if >0.02 macro.

## 3. Label-text classifier initialization / low-rank label-feature term (SigLIP2 or MetaCLIP-2)

Mechanism. Build the 12476×D text matrix once; initialize Linear rows from a learned image→text projection, or add a shared low-rank term score_i = w_i·x + (u·x)·t_i so every label has a usable direction on day one. "Extra head capacity" of the only kind that should help: shared across labels and semantically tied, so it does not need 20 positives to fit.

Evidence. DECAF's +2–6% and CLIP-Decoder's +3.9 zero-shot mAP — both in [label semantics](heads-label-semantics.md); gains concentrate in the data-scarce tail.

Cost. 4–6 h; text encoding is minutes (models on disk); the projection trains with the head.

Risk. Highest "wrong regime" risk: both citations are about labels with few/zero examples, and our median tag has thousands. Risks hurting head classes by constraining weights to a text subspace. Prototype as an additive term (not a hard init) so it can be ablated.

## 4. Frequency-binned error analysis — the 20-minute experiment that decides 1–3

Mechanism. Existing probs cache + `valid_nested/labels.npz`: per-tag F1 vs training frequency in 5 bins, ours and the bar's. Not a modelling change — it tells you whether the deficit is uniform (→ features/backbone) or concentrated in rare bins (→ items 1–3).

Evidence. BALMS, DBL and DECAF all report this split and all find "concentrated in the tail" — on their data. Ours is 5.3M images with ~13k positives per label, a different regime, and the bar's macro-P 0.6201 hints its rare-tag scores are broadly good rather than merely re-thresholded.

Cost. ~30 min, reusing `wdtag/validate.py` machinery.

Risk. None. Doing this after a 9 h loss run instead is the real risk.

## 5. ML-Decoder-style query head with group decoding (only if 1–3 fail)

Mechanism. 100–400 non-learnable queries cross-attend the 1029 patch tokens, one shared output projection per label group → per-label logit from a pooled, label-conditioned feature instead of one global 3072-d vector.

Evidence. [Single linear head](heads-single-linear-head.md) has the ablation: 86.0 → 86.7/86.8 mAP at 9600 labels for +0.5–1.8 GFLOPs, 87.0 → 88.1 at 80 labels. Flat in label count.

Cost. 6–10 h; group decoding needs care; +10–30% head FLOPs (our head is 0.02% of step FLOPs).

Risk. Expected macro gain at 0.40 in the +0.01 range against a 0.17 score-quality gap. Best-evidenced head architecture change and still the worst use of a week.
