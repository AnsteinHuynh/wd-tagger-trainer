# Calibration at a fixed threshold

Read this before optimizing anything else. On 2026-10-25 the model's calibration turned out to be worth more macro than every optimizer switch in `optimizer-gold-settings.md` combined, and it is nearly free.

## The measurement

Production checkpoint 8083 on the 50k, threshold curve, macro and micro at fixed cutoffs:

| threshold | macro | precision | recall | micro | predicted tags per image |
| --- | --- | --- | --- | --- | --- |
| 0.30 | 0.2864 | 0.2183 | 0.6073 | 0.3474 | 131.3 |
| 0.35 | 0.3271 | 0.2686 | 0.5533 | 0.4262 | 95.4 |
| 0.40 | 0.3569 | 0.3171 | 0.5045 | 0.4929 | 72.9 |
| 0.45 | 0.3747 | 0.3636 | 0.4593 | 0.5457 | 57.7 |
| 0.50 | 0.3824 | 0.4068 | 0.4188 | 0.5837 | 46.9 |
| 0.60 | 0.3676 | 0.4705 | 0.3446 | 0.6184 | 32.1 |

The validation set has 32.3 labels per image. At the scored threshold the model emits 72.9. That excess is not a threshold artifact to be argued away, it is the precision deficit in one number: we predict 2.3 times as many tags as exist, so half our positives cannot be right no matter how good the features are.

Macro peaks at 0.50 and micro keeps climbing to 0.60, so the ranking is better than the calibration at every cutoff above 0.40. The binding constraint is where the probabilities sit, not what the features know.

## Why this happens

ASL with `gamma_neg` 2.0 down-weights the negative term on easy cells. That is what makes rare tags trainable, and it is also what makes the loss's minimizer uncalibrated: the loss no longer pays for a confident false positive the way log loss would, so the model drifts toward emitting positives. The head bias is initialised at logit(prior) and then the loss pushes it away from the prior.

## Three remedies, ranked by cost

One, a single temperature on the logits at inference. Monotone, changes no ranking, moves the point where 0.40 falls. Fitted on the first 25000 validation images and judged on the other 25000: 0.3465 uncalibrated, 0.3686 at T 0.25, so +0.0221. The grid chose its edge, meaning the optimum is effectively a hard decision boundary at logit zero, which is the same thing as the 0.50 peak in the table above.

Two, temperature plus a per-tag bias, grid searched per tag over [-3, 3] and shrunk toward zero below 200 positives: 0.3737 on the held-out half, so +0.0272 against none. The fit-half oracle reached 0.4671, and the 0.09 gap between oracle and held-out is the overfit that a table of 12476 biases earns.

Three, fix it in training so no post-hoc step is needed. The metric threshold is fixed, so the cheapest honest route is to keep the model's predicted count near the observed label count during training, by adapting a bias in logit space toward matching the batch's predicted positive rate to its label rate. This needs no external data, no validation fit and no inference-time table, and it moves 0.40 to where the model's own optimum already is.

## The honesty rule

Any calibration fitted on validation images cannot be quoted as a final score on those same images. Fitting on half the 50k and reporting the other half is a legitimate experiment and not a legitimate result. The shippable versions are remedy three, or remedy one or two fitted on the training split or a disjoint slice that is never scored.

## What to check first on any future run

Print the predicted tags per image at 0.40 next to the label rate, 32.3 for this dataset. A ratio near 1.0 means calibration is fine and any remaining gap is capacity or features. A ratio near 2.0 means the run is reporting a precision deficit that a scalar could mostly remove, and no amount of optimizer study will show it.

## Matched count per image is the wrong fix, measured

Forcing each image to emit exactly as many tags as it truly has, taking the top k by probability, scores 0.3193 macro and 0.5310 micro. Four fewer reads 0.3109 and 0.5328, four more reads 0.3244 and 0.5235. Every one of them loses to the plain 0.50 threshold's 0.3824 macro while beating its 0.4929 micro.

The reason is what macro averages over. A per-image count budget is spent first on the high-probability common tags, so the tail gets fewer predictions and the tail is precisely what macro punishes per tag. Micro improves because the count is right on average; macro falls because the count is wrong for the tags that matter most to the metric.

So the count has to be controlled per tag, not per image. This is why the per-tag bias is worth more than a temperature alone, and it is the target the training-time remedy has to hit: a shrunk per-tag bias adaptation, not a global count constraint.

## The controller that should ship, measured at 04:20

A running per-tag bias update was simulated on the fit half and scored on the held-out half. For each batch of 384 images the controller compares each tag's predicted positive rate with its label rate and nudges that tag's logit bias, `bias += eta * (label_rate - predicted_rate) * 8`, clipped to plus or minus 6.

| rule | held-out macro |
| --- | --- |
| temperature 0.25, no bias | 0.3686 |
| grid-searched per-tag bias, shrunk below 200 positives | 0.3737 |
| online, eta 0.5, four passes | 0.3744 |
| online, eta 2.0, two passes | 0.3758 |
| online, eta 2.0, four passes | 0.3781 |

The online rule beats the offline grid search, which is the whole argument: it needs no fit split, no 12476-entry table to store or version, and it keeps working as the model drifts, because it measures the drift every batch. At eta 2.0 over four passes the bias ends with median minus 0.28 and spread 0.93, so it is doing real per-tag work rather than a global shift.

Caveat on this experiment: it was simulated on validation images because that is where the probabilities are cached, so the gain is measured under a fit-on-validation regime. In training the same rule would consume training batches, which is strictly more honest and is the reason this is the version to implement.

## Not yet implemented, on purpose

The trainer change is small but it touches the head and the validation path, and it cannot be smoke tested while production holds 21.2 GB of a 24.5 GB card. It was 04:20 and production was mid-anneal, so the honest move was to measure and write the proposal rather than patch a live trainer blind. When the card frees, the change is: a `bias_match_eta` config field defaulting to 0, an extra per-tag bias tensor applied to logits before the loss and before validation, updated with no_grad after each optimizer step from the accumulated batch's predicted and label rates, and a print of the ratio of predicted to true positives every thousand steps so the fix is visible while it runs.
