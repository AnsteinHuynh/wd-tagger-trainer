# Glossary: metrics, thresholds, and compute

Scope — 12476 tags (4 rating + 9225 general + 3247 character) on a fixed 1000-image test split; ~32 positive tags/image -> 32 / 12476 = 0.26% positive rate. Independent per-tag sigmoid probabilities, each cut by a threshold. 4697 tags present in test, 7779 (62.35%) absent

Provenance — `[M]` measured here (`runs/*/eval_test.json`, `train_log.jsonl`, `docs/`); `[C]` bar's card/checkpoint, not re-measured; `[E]` arithmetic from these; `[I]` inferred

## 1. Precision, recall, F1

TP / FP / FN / TN — per tag: predicted yes&yes / yes&no / no&yes / no&no; `TP+FN` = images with the tag

```
P  = TP / (TP + FP)      "of the images we tagged, how many really have it"
R  = TP / (TP + FN)      "of the images that have it, how many we found"
F1 = 2PR / (P + R) = 2TP / (2TP + FP + FN)
```

Zero rule — `0/0 -> 0` (`wdtag/metrics.py`); never NaN, never 1

F1 — harmonic mean of P,R (at most their arithmetic mean, equal only if P = R; P 0.60/R 1.00 -> 0.75, not 0.80). Micro `fe92f4df` `[M]`: P 0.6554, R 0.3028, arithmetic mean 0.4791, F1 = 2(0.6554)(0.3028)/0.9582 = 0.4142

F1 ignores TN — at 0.26% prevalence, `k` positives with one missed caps F1 at `2(k-1)/(2k-1)`: k 2 -> 0.667, 3 -> 0.800, 10 -> 0.947; k 1 -> 0 or 1. Most tags have 1-3 positives/1000: coarse, high-variance, usually 0, and macro averages those

## 2. MCC (Matthews correlation coefficient)

```
MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
```

Range — -1..+1 (+1 perfect, 0 chance, -1 perfectly anti-correlated); a zero margin is undefined and assigned 0 (`0/0 -> 0`)

Examples — 3 positives/1000: always "no" -> 99.7% accuracy, F1 0, MCC 0; always "yes" -> 0.3% accuracy, F1 0.006, MCC 0; 100 fired with 3 right (97 FPs) -> 90.3% accuracy, F1 0.058, MCC 0.165 `[E]`

Chance — MCC puts "no information" at exactly 0 at any prevalence, so values compare across tags and splits; F1 ≈ the prevalence (~0.0026), so 0.05 = 20x chance or noise

`sqrt(P*R)` — rare-tag limit: geometric mean vs F1's harmonic (GM >= HM); `MCC >= F1`; bar macro 0.1964 vs 0.1938, micro 0.6811 vs 0.6800 `[M]`

Both — F1 = field headline (bar's card quotes F1) and per-tag tuning target; MCC = prevalence-normalised cross-check. Noisy at 1-3 positives; MCC 0 by fiat for degenerate tags

## 3. Micro averaging

Micro — pool TP/FP/FN over 1000 x 12476 = 12.5 M pairs -> one P/R/F1; micro F1 = harmonic mean of micro P, micro R. Bar at 0.5, squashed: P 0.7349, R 0.6328 -> F1 = 2(0.7349)(0.6328)/1.3677 = 0.6800 `[M]`

Hides — ~32000 positive pairs, mostly head; a 2-positive tag adds at most 2 to the numerator, a 2000-positive tag adds 2000; tail invisible

## 4. Macro averaging

Macro — metric per tag, then unweighted mean over all 12476

62.35% floor — `TP = 0` -> F1 0 (absent from test / never predicted / predicted wrongly). 7779 of 12476 = 62.35% structurally 0 for every model `[M]`; macro = `0.3765 x (mean over 4697 present)` = 12476 / 4697 = 2.656 dilution `[E]`; bar 0.1938 x 2.656 = 0.5147, ours 0.0223 x 2.656 = 0.0591 `[M]`

Not F1(macro P, macro R) — bar macro P 0.2100, R 0.1917 -> 0.2004 vs reported 0.1938 (0.0066, averaging order `[E]`); card mean P 0.612, R 0.630 -> 0.6209 vs Macro@Best 0.611 `[C]`. Never rebuild a macro F1 this way

## 5. Why micro >> macro in this project

| test, threshold 0.5 | bar, padded | bar, squash | ours `fe92f4df` |
|---|---|---|---|
| micro F1 | 0.6935 `[M]` | 0.6800 `[M]` | 0.4142 `[M]` |
| macro F1, supported (4697) | -- | 0.5147 `[M]` | 0.0591 `[M]` |
| macro F1, full vocabulary | 0.2017 `[M]` | 0.1938 `[M]` | 0.0223 `[M]` |
| micro / full macro | 3.44 `[E]` | 3.51 `[E]` | 18.6 `[E]` |
| micro / supported macro | -- | 1.32 `[E]` | 7.00 `[E]` |
| full / supported (dilution) | 2.656 | 2.656 | 2.656 |

Decomposition — `3.51 = 1.32 x 2.656`: head-vs-present-tail difficulty (0.6800 vs 0.5147) x dilution (62% of tags unable to score). Ours: 2.656 x 7.00 = 18.6. Diagnostic: micro up with macro flat = frequent tags; macro up with micro stalled = rare tags

## 6. "Supported macro"

Supported macro — macro over labels with `TP + FN > 0` (at least one test positive; `wdtag/metrics.py`). `n_supported_labels`: 4697 test, 4621 validation (a split property, not training data). Sibling: training-side minimum-examples cut; `REPORT.md` tabulates >=10 training images (2882)

Why — excludes no-chance tags (absent from 1000 images, or never in 10000 training images) that plain macro counts as failures, conflating bad-at-tag with tag-not-in-sample

| subset (REPORT decomposition) | labels | bar macro F1 | ours macro F1 |
|---|---|---|---|
| every label in the vocabulary | 12476 | 0.1938 | 0.0226 |
| labels present in the test split | 4697 | 0.5147 | 0.0601 |
| labels with >=10 training images | 2882 | 0.4546 | 0.0958 |
| labels with 1-9 training images | 7908 | 0.1260 | 0.0008 |
| labels never in train, present in test | 217 | 0.5091 | 0.0000 |

Reading — all-label 0.0226 is a different run from `fe92f4df` (0.0223); last three rows are the honest reading: bar 4.7x on learnable tags (>=10 images), essentially infinite at 1-9 (we score 0.0008). Does not fix small-sample noise (1 test positive -> 0 or 1); unweighted; split-size dependent: card 0.580 on a ~296k-image test (all 12476 present), not comparable to 0.1938 or 0.5147

## 7. Thresholding: `@0.5` vs tuned

0.5 is not F1-optimal — 0.5 = Bayes cutoff for 0-1 accuracy at equal error costs; at 0.26% base rate a calibrated model outputs ~0.0026, so the F1-optimal cutoff sits well below 0.5. Fixed 0.5 over-predicts "no" on the tail: micro P 0.6554 vs R 0.3028 `[M]`, high precision and low recall

Tuned (`threshold_tuned`) — per tag sweep 0.02, 0.04, ..., 0.98 on the 1000-image validation split, keep best-F1, freeze, apply to test (`wdtag/metrics.py`); no validation positive -> 0.5. 49 thresholds x ~4700 present tags on 1000 images ≈ 0.2 validation images per tag

| measured `[M]` | macro@0.5 | tuned macro | change | micro@0.5 | tuned micro | change |
|---|---|---|---|---|---|---|
| bar, squash | 0.1938 | 0.2008 | +3.6% | 0.6800 | 0.6655 | -2.1% |
| ours `fe92f4df` | 0.0223 | 0.0331 | +48.5% | 0.4142 | 0.4020 | -2.9% |

Calibration, not capability — +48.5% vs +3.6%: ours mis-calibrated (P >> R), bar near its operating point. Per-tag tuning maximises per-tag F1, not pooled F1 -> micro falls for both (our micro P 0.6554 -> 0.3581, R 0.3028 -> 0.4584). All 4697 thresholds fitted on the same 1000 images (a 2-positive tag chooses on two coin flips) -> optimistic for another 1000. Bar's card: category thresholds 0.38 (general), 0.51 (character), 0.24 (rating) plus per-tag `best_threshold` `[C]`; "0.5" is our yardstick, not its deployment default

## 8. The `@` notation: `Macro@0.40`, `Micro@0.40`, `Macro@Best`

This repo (verified) — grep `README.md`, `REPORT.md`, `docs/`, `scratch/sophie//*.md` for `@0`, `@Best`, `Macro@`, `Micro@`: only `@0.5` (`micro F1 @0.5`, README line 12; `macro F1 @0.5`, REPORT lines 15-16, 147) and `tuned` (`threshold_tuned` in `eval_test.json`); `@0.40`, `@Best`, `Macro@`, `Micro@` appear nowhere

Bar's card, verbatim (verified) —

| token | card definition | meaning |
|---|---|---|
| `Macro@0.40` | "the metrics on the threshold 0.40" | full-vocabulary macro with every tag cut at 0.40; card column is F1/MCC/P/R |
| `Micro@0.40` | "the metrics on the threshold 0.40" | pooled micro at 0.40 |
| `Macro@Best` | "the mean metrics on the tag-level thresholds on each tags, which should have the best F1 scores" | per-tag best-F1 thresholds (`best_threshold` in `selected_tags.csv`, 12476 rows), then unweighted mean over tags; card column is F1/P/R, no MCC |
| `Micro@Thr` | "the metrics on the category-level suggested thresholds" | category cutoffs 0.38/0.51/0.24 from `thresholds.csv` |

On the baseline's card, not in this repo, where `Macro@Best` is spelled `tuned macro F1`. Card values: Macro@0.40 0.580, Micro@0.40 0.697, Macro@Best 0.611/0.612/0.630 (F1/P/R) `[C]`

Ambiguity — `Macro@0.40` states neither denominator nor split: card 0.580 = all 12476 tags on a ~296k-image test; our full-vocabulary macro at 0.5 = 0.2017 (padded), supported 0.5147 `[M]` — same words, ~2.9x apart. Always write `Macro@0.40 (all 12476 labels, 1k test)`. Inferred `[I]`: a `Macro@0.40` with no denominator almost certainly means fixed 0.40 over the full vocabulary. Not verified: which split `best_threshold` was selected on

## 9. FLOPs and MACs

FLOP / MAC — FLOP = one floating-point op (add, multiply, sqrt); MAC = multiply+add = 2 FLOPs; 1 TFLOP = 10^12 FLOP. Papers quote MACs/GMACs, cards often FLOPs (sometimes counting MACs): silent 2x

Bar — `FLOPs / MACs: 1.2T / 600.4G` `[C]`, self-consistent (600.4 G x 2 = 1.2008 T); forward at 512x512, card-reported; order-of-magnitude arithmetic ~1.2 TFLOP `[E]`. Use 1.2 TFLOP forward / 600.4 GMACs, not "~2 TFLOP"

Ours — 505 GFLOP/image forward+backward `[M]` (`docs/hdd-image-pipeline.md`); 87.0 M frozen backbone + 1.30 M OFTv2 + 19.18 M head = 107.5 M parameters `[M]`; forward ~1/3, ~170 GFLOP `[E]`. Never compare 505 (fwd+bwd) with 1.2 T (forward): like-for-like the bar is ~7x our compute (692.6 M vs 107.5 M parameters)

What they buy — step time: 505 / 75 TFLOPS = 6.7 ms/image `[M]`, x 32 = 215 ms/step vs measured 190-211 ms compiled `[M]`; utilisation: 75 / 165 TFLOPS dense BF16 peak (RTX 4090; 330 sparse `[C]`) = ~45% `[E]` — the sparse 330 reads 23% and wrongly implies a half-idle GPU. Eager optimizer update ~88 ms/step launch overhead -> FLOPs-based time is a floor

## 10. How to read a results table

| pattern | most likely reading | check first |
|---|---|---|
| micro up, macro flat | head improved, tail unchanged | supported macro, >=10 / 1-9 buckets |
| both up, MCC flat | P/R became more balanced (F1 is more imbalance-sensitive) or gain is in high-prevalence tags | macro MCC, per-tag deltas |
| macro up only after tuning | calibration / decision rule, not learning | fixed-@0.5 column, size of tuned gain |
| micro down after tuning | expected: tuning maximises per-tag F1, not pooled F1 | both columns, micro P/R |
| everything up by <0.005 | possibly noise | matched-seed spread 0.0044 micro `[M]` |

Nuisance sizes — same 1000 images need same geometry, threshold policy, macro denominator. Geometry: bar 0.6935 / 0.2017 (padded, native) -> 0.6800 / 0.1938 (squashed) = 0.0135 micro, 0.0079 macro `[M]`; threshold policy 0.1938 -> 0.2008 tuned `[M]`; full vs supported 2.656x `[E]`; seed noise 0.0044 micro `[M]`. Any is as large as the effect compared

Checklist — same 1000 test images; same preprocessing (pad vs squash, interpolation, crop); same thresholds (fixed / per-category / per-tag, tuned on what); same macro denominator (all 12476 or supported, with `n_supported_labels`); same 12476-label order from the bar's `config.json`; all four of F1/MCC/P/R
