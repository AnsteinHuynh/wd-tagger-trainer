# The bar: `convnextv2_huge.dbv4-full`

The model we must beat. Everything here was reverse-engineered from its local files (`C:\MyApps\ComfyUIRoot\convnextv2_huge.dbv4-full`) and from scoring it ourselves, not
taken on trust.

## Identity

| property | value |
|---|---|
| architecture | `convnextv2_huge` — a single model, not an ensemble |
| params | 692.6M (one 2.77 GB fp32 `model.safetensors`) |
| classes | 12476 — identical label space to ours |
| timm base | `convnextv2_huge.fcmae_ft_in22k_in1k_512` |
| dataset | `animetimm/danbooru-wdtagger-v4-w640-ws-full` — the same corpus we use |
| preprocessing | squash to 512x512, ImageNet mean/std — the same pipeline as ours |

Because the corpus and preprocessing match, the comparison is apples-to-apples: same pixels, same labels, same metric definitions.

## Published scores (its model card)

| split | Macro@0.40 (F1/MCC/P/R) | Micro@0.40 (F1/MCC/P/R) | Macro@Best (F1/P/R) |
|---|---|---|---|
| Validation | 0.580 / 0.584 / 0.626 / 0.556 | 0.697 / 0.696 / 0.692 / 0.701 | — |
| Test | 0.580 / 0.584 / 0.627 / 0.556 | 0.697 / 0.696 / 0.693 / 0.702 | 0.611 / 0.612 / 0.630 |

Per-category thresholds: general 0.38, character 0.51, rating 0.24. Per-tag tuned thresholds reach macro 0.611.

## Scored on our own 50k validation cache (all eight numbers measured)

| metric | value |
|---|---|
| Macro F1 | 0.5579 |
| Macro MCC / P / R | 0.5658 / 0.6201 / 0.5333 |
| Micro F1 | 0.6886 |
| Micro MCC / P / R | 0.6880 / 0.7033 / 0.6746 |

Within 2% of its published 0.580 / 0.697, so our 50k subset is representative. Its probabilities are cached (`cache/wd-tagger-trainer\probs\bar_convnextv2_huge.npy`),
so re-scoring, or ensembling it with ours, takes ~2 s.

To re-score: `python -m wdtag.validate --bar --sizes 1000,5000,10000,50000` (batch 64).

## Its classifier head, exactly (measured 2026-10-22 from the timm module)

Loaded through `timm.create_model("convnextv2_huge", num_classes=12476, drop_path_rate=0.4, act_layer="gelu_tanh", drop_rate=0.2)`, the head is timm's `NormMlpClassifierHead`, 35.15M params:

```
SelectAdaptivePool2d(pool_type=avg)   -> one global-average-pooled 2816-d vector
LayerNorm2d(2816)                     -> LayerNorm on that vector
Flatten
Dropout(0.2)
Linear(2816 -> 12476)                 -> 35.14M
```

No hidden layer, no projection, no multi-statistic pooling (no max, no std, no CLS concatenation). `pre_logits` is `Identity()`.

Why this matters for us: our head is `LayerNorm(3072) -> Dropout(0.1) -> Linear(3072, 12476)` = 38.33M, i.e. larger than the bar's, with more expressive pooling (CLS of 4 layers + mean + max + std). So head capacity and head sophistication are NOT what separates us: the bar wins macro with a simpler and smaller head on a bigger backbone. Any head-side proposal has to explain why it helps when capacity demonstrably is not the differentiator.

## How it was trained (from its own `meta.json` and tfevents)
| knob | value |
|---|---|
| optimizer | AdamW, lr 2e-4, betas (0.9, 0.999), wd 1e-3 |
| batch | 32 |
| epochs | 100 (max), shipped weights = epoch 85 |
| mixup | 0.6 |
| drop_path | 0.4, drop_rate 0.2 |
| eval threshold | 0.40, key metric macro_f1 |
| wall clock | ~10 days (event files Sept 30 → Oct 10) |

## The fact that reframes the whole comparison

It reaches 97% of its final macro in epoch 1.

| | epoch 1 | epoch 96 |
|---|---|---|
| macro | 0.5634 | 0.5792 |
| micro | 0.6959 | 0.6964 |

95 further epochs bought +0.0158 macro (+2.8%). Eval loss was lowest at epoch 2–3 (0.0756) and rose to 0.0859 while train macro climbed 0.488 → 0.661 — it overfits for 94 epochs to gain almost nothing.

Consequence: the 10-day training time is not the handicap it looks like. On one pass over the same 5.32M images the bar reaches macro ~0.56; we reach 0.28 on one pass. The gap is per-sample efficiency at equal budget — capacity (692.6M vs our 123.5M trainable) and/or architecture — not schedule length. More ViT-B steps cannot halve it.

## Our standing against it @ 0.40 (checkpoint 086241, 50k)

| metric | ours | bar | ours/bar |
|---|---|---|---|
| Macro F1 | 0.2815 | 0.5579 | 50.4% |
| Macro MCC / P / R | 0.3062 / 0.4608 / 0.2266 | 0.5658 / 0.6201 / 0.5333 | 54 / 74 / 43% |
| Micro F1 | 0.6133 | 0.6886 | 89.1% |
| Micro MCC / P / R | 0.6179 / 0.7068 / 0.5417 | 0.6880 / 0.7033 / 0.6746 | 90 / 100.5% / 80% |

We already lead on micro precision. Every other axis is a recall deficit: at 0.40 we miss ~46% of micro positives and ~77% of rare-tag (macro) positives.

Holding our precision at 0.7068, micro F1 reaches the bar at recall > 0.6713 (the bar has 0.6746). So matching its recall edges ahead, 0.6903 vs 0.6886 — a real but narrow win. Macro is far away: recall must more than double, 0.2266 → 0.5333.

## Caveat on geometry

Squashing also distorts the bar's input. On its own native padded geometry it scores macro F1 0.2017 / micro F1 0.6935 at 0.5, so treat all bar numbers on our cache as a lower bound on what it could do.
