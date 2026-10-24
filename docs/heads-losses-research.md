# Classification heads & losses for 12476-label multi-label tagging — a ranked, evidence-grounded shortlist

Scope: heads and losses only. Backbone/OFT/schedule changes are out of scope except where they explain why head work cannot close the gap.

Grading rule. Paper headline numbers are usually mAP/AP, which is threshold-free. Our protocol is macro/micro F1/MCC/P/R at one global 0.40. A method that raises mAP by ranking tail labels slightly better can leave F1@0.40 unchanged, and a method that only shifts a class's probability can raise F1@0.40 with no ranking gain. Every claim below is tagged with which of the two it measured.

## 0. The two local measurements that reframe everything

From `runs/_sweep_last3.json` and `runs/4a13a120/validation.json` (checkpoint 086241, 50k, our data, no paper involved):

| operating point | macro F1 | note |
|---|---|---|
| threshold 0.40 (protocol) | 0.2814 | the number we report |
| threshold 0.05 (best in grid) | 0.3575 | still rising at the bottom of the grid |
| per-tag best threshold (oracle, fit on this val set) | 0.3879 | ceiling for any per-class scheme |
| the bar at 0.40 | 0.5579 | its macro at 0.20 is 0.5709, still rising downward |

1. The ceiling of every calibration/threshold/prior-offset/weight-norm trick is ~0.388, oracle-fitted tag-by-tag on the same 50k. The bar beats it at a single global 0.40 by 0.17 macro points; no post-hoc re-scoring of this checkpoint can reach it.
2. The deficit is not "we need a lower threshold". A global recalibration is worth at most +0.076 (0.2814 → 0.3575, i.e. sliding the operating point to 0.05), exactly the threshold-moving the protocol forbids. Per-class offsets buy at most +0.106, oracle-fitted. Everything below is judged on "does it change the scores, not just the cut".

Corollary: rare-tag scores are worse than the bar's in a way more head capacity, thresholds or calibration cannot fix. Any recommendation must add information or inductive bias. Items 3–5 are the only ones that even claim to.

## 1. Is a single Linear head still competitive at 10k+ labels?

GAP 86.0 vs 86.7–86.8 mAP at 9600 labels; +0.8 at 9.6k, +1.1 at 80; every public 12k tagger is linear:
[single linear head](../../docs/REPORTS/heads-single-linear-head.md).

## 2. Query / attention decoders

Small, mAP-shaped, label-count-flat gain, ~+0.01 macro for 6–10 h:
[query decoders](../../docs/REPORTS/heads-query-decoders.md).

## 3. Tail-class losses under a fixed global threshold

BALMS@18211: NTR's learned class bias dominant, CB-NTR ≈ DB, same-domain +0.013:
[tail losses](../../docs/REPORTS/heads-tail-losses.md).

## 4. Classifier bias / weight-imbalance fixes

Rebalancing +7.6 tail / −4.6 head; every post-hoc per-class rescale is capped by 0.3879:
[bias fixes](../../docs/REPORTS/heads-bias-fixes.md).

## 5. Label-semantics heads

DECAF/CLIP gains are for scarce labels; GCN heads are dominated: [label semantics](../../docs/REPORTS/heads-label-semantics.md).

## 6. Anything newer (2024–2026)

MoE, per-label adapters, any-label heads: unproven at 12k; backbones and data moved the needle:
[newer work](../../docs/REPORTS/heads-newer-2024-2026.md).

## The ranked shortlist

CB-NTR, prior logit adjustment, label-text init, frequency-binned analysis, query head last:
[ranked shortlist](../../docs/REPORTS/heads-ranked-shortlist.md).

## Evidence grades

Well-evidenced vs plausible-but-unproven vs hype: [evidence grades](../../docs/REPORTS/heads-evidence-grades.md).

## Things actively WRONG under a fixed global 0.40

Per-class thresholds, P=R selection, ranking losses, mAP-only claims, smoothing, hard resampling:
[what is wrong](../../docs/REPORTS/heads-wrong-under-040.md).

## The single change I would try first

CB-NTR, gated behind the prior-offset probe and the frequency-binned analysis, ~2 days:
[first change](../../docs/REPORTS/heads-first-change.md).

## The strongest counter-evidence

The smaller-headed bar wins 0.5579 vs 0.2815; the ~0.17 gap is in the scores:
[counter-evidence](../../docs/REPORTS/heads-counter-evidence.md).

## Limitations of this review

No study matches our regime; BALMS' threshold differs; a 1.6% micro-F1 discrepancy is unexplained:
[limitations](../../docs/REPORTS/heads-limitations.md).
