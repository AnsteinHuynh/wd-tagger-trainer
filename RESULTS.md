# Results

Minimal public record. Only the best-trained run ships with this repo (provenance JSONs in `runs/8f6de35e/`); everything else lives in the author's private experiment log.

## Headline: parameter efficiency

How close can one epoch of adapter training get to a released full-size tagger?

| model | trainable params | total params | epochs | macro F1 (50k @0.40) | micro F1 |
|---|---:|---:|---:|---:|---:|
| ensemble, runs/8f6de35e | 31,144,124 | 110.2M live + 99.1M frozen (cached) | 1 | 0.4162 | 0.5154 |
| SigLIP2 solo, 2 epochs | 21,852,860 | 99.1M | 2 | 0.4066 | 0.5131 |
| DINOv3 solo, long run | - | 91.0M | ~0.77 of one pass | 0.2815 | 0.6133 |
| reference bar `convnextv2_huge` (released, not trained here) | - | ~692.6M | - | 0.5579 | 0.6886 |

Parameter counts are counted from the checkpoints, not estimated. The ensemble's 31.1M trainable parameters are: 4.88M OFTv2 rotation factors across all 12 DINOv3-B blocks (73 modules, block size 128, exact Cayley), 7.09M weights of the one unfrozen transformer block, and 19.18M for the Linear(1536, 12476) fp32 head plus its two LayerNorms. Everything else was frozen. The frozen second tower (SigLIP2-B, 99.1M) was served as precomputed fp32 cls features and was never trained in this run.

The reference bar is the released `convnextv2_huge` WD-style tagger: its ConvNeXt V2 Huge backbone is about 655M parameters (ConvNeXt V2 paper, Woo et al., CVPR 2023) and the released tagger totals 692.6M with its 12476-label head.

So the headline: after a single epoch, training 31.1M parameters, 4.5 percent of the bar's count, the ensemble reaches 74.6 percent of the bar's macro F1 (0.4162 vs 0.5579) on the 50k validation instrument. The best single tower reaches 72.9 percent of the bar's macro with 21.9M trainable parameters, 3.2 percent of the bar's count. The SigLIP2 solo run needed two epochs for its number; the ensemble needed one.

## Scoring protocol

Comparable numbers are macro F1 and micro F1 at a fixed global threshold of 0.40 on a 50000-image nested validation subset, read from `validation.json` -> `subsets["50000"]`. Macro is the objective. No per-tag threshold tuning.

## Held-out test split

The ensemble was evaluated once on the same 1000-image test split and protocol as the baseline (threshold 0.5, plus a threshold tuned on the valid split), final checkpoint 13859. At threshold 0.5: macro F1 0.1694, micro F1 0.6018 (baseline: 0.194 / 0.680). Tuned threshold: macro F1 0.1618, micro F1 0.3525. For context: the ensemble beats the SigLIP2 solo tower on the validation instrument; the side-by-side numbers against the bar are tabulated above. This is the official result; no checkpoint or threshold was selected after seeing it. The remaining gap is macro, concentrated in rare tags, and the in-run validation curve was still rising when the epoch ended.

The author's view on what that gap means: Danbooru labels come from humans who omit far more often than they add wrong tags, so macro F1 against human labels measures agreement with noisy annotators, not correctness. Part of the rare-tag gap is plausibly missing human labels rather than model failure, and training past a point does not just overfit the images, it starts fitting the annotators' omission patterns. The one-epoch budget above is partly a deliberate choice in that light.

## Experiments around the training method

- OFTv2 (orthogonal fine-tuning, block size 128, exact Cayley) in every backbone block with one unfrozen transformer block was the champion recipe. Freeze depth was measured: unfreezing 12 blocks instead of 1 was 2.65x worse at this parameterisation, because rotations inside unfrozen blocks are frozen, a step-size confound recorded in the log.
- The production optimizer was SinkSGD_ADV, a Riemannian Sinkhorn-geometry update (normed momentum plus an arcsine variance preconditioner) scoped to the adapter tensors and the unfrozen block only. Measured honestly: on the adapters it is worth +0.0047 macro at matched displacement over the plain update, and +0.0145 macro with the block included; applied to the classifier head it collapses macro (0.0085 vs 0.0559 at step 1000), because it erases the tag-frequency prior the head carries. The scope is a config field, `optimizer_switch_groups`.
- Loss shaping mattered more than head architecture: ASL (gamma 2.0) over BCE was the largest single-change effect measured at small scale, and class weighting / pos_weight were dead ends at 12476 labels and 0.26 percent positives.

## Compute lesson

The practical discovery of this study is what the bar actually demands. Its 692.6M parameters and 12476-label head (35.15M parameters, larger than this repo's entire trainable budget) sit an order of magnitude above what a one-epoch 31M-parameter adapter run on a single 24 GB GPU costs. The bar's class is reachable to within a quarter of its macro by adapter training; the last quarter is where the full-size model earns its keep, and the measured gap is concentrated in rare and relational tags rather than in frequent-tag bulk.
