# Heads and losses for a 12476-label tagger — index

Research on whether a better head or loss can move our macro gap. Full text is split by section under `docs/REPORTS/heads-*.md`; this file is the entry point.

The one-paragraph version. A single linear head is still competitive at ~10k labels (ML-Decoder is worth ~+1 mAP and is flat in label count), and head capacity is provably not our problem: our head is 38.34M against the bar's 35.15M, and the bar wins macro with a smaller, simpler head. Our measured per-tag-oracle threshold ceiling is 0.3879 against the bar's 0.5579 at a fixed global 0.40, so no calibration, threshold or rescaling trick closes the gap — it lives in the scores. What does have tail-specific evidence is the loss:
class-balanced focal with λ applied twice on the negative branch (CB-NTR). Read point 1 of [tail losses](REPORTS/heads-tail-losses.md) before assuming the bias is doing any work: BALMS's bias is fixed from the prior and pushes tails slightly down; λ is the mechanism. `wdtag/losses.py` is implemented from that corrected reading, not the popular summary.

## Sections

| section | what it covers |
|---|---|
| [Single linear head](REPORTS/heads-single-linear-head.md) | is a Linear still competitive at 10k+ labels |
| [Query decoders](REPORTS/heads-query-decoders.md) | ML-Decoder / Query2Label and their real gains |
| [Tail losses](REPORTS/heads-tail-losses.md) | ASL, CB, NTR, BALMS — the evidence that matters to us |
| [Bias fixes](REPORTS/heads-bias-fixes.md) | logit adjustment, cRT, cosine classifiers, weight decay |
| [Label semantics](REPORTS/heads-label-semantics.md) | text-embedding init, label graphs, hierarchies |
| [Newer work](REPORTS/heads-newer-2024-2026.md) | 2024–26: MoE heads, any-label / zero-shot |
| [Ranked shortlist](REPORTS/heads-ranked-shortlist.md) | the five changes worth time, ranked |
| [Evidence grades](REPORTS/heads-evidence-grades.md) | (a) well-evidenced / (b) plausible / (c) hype |
| [Wrong under fixed 0.40](REPORTS/heads-wrong-under-040.md) | things that would actively hurt this protocol |
| [First change to try](REPORTS/heads-first-change.md) | CB-NTR, gated behind two 30-min probes |
| [Counter-evidence](REPORTS/heads-counter-evidence.md) | why head capacity is not the differentiator |
| [Limitations](REPORTS/heads-limitations.md) | caveats, and the unresolved 0.6133 vs 0.6235 micro-F1 discrepancy |

## How this connects to the code

`wdtag/losses.py` — `bce`, `asl`, `cb_ntr`, `class_balanced_weights`; selected by `cfg.loss`
`wdtag/model.py` — `head_pool` (any of `cls/mean/max/std`), `head_hidden` (0 = linear), and `class_bias` created only for `cb_ntr`
`ARM-RESULTS.md` — the arms that test these, with results and the noise-floor arm

## Provenance

Full-text source before the split: `scratch/sophie/heads-losses-research.md` (itself now an index into the same section files). Token cost of the split, measured with the DeepSeek tokenizer: 36170 → 35179 tokens total (−2.74%), section bodies −5.8%, and — the point of splitting — a single section now costs 231–2435 tokens to load instead of 4753–8375.
