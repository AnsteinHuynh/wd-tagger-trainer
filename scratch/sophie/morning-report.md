# Morning report, 2026-10-25 02:35

## Where production is

Run 459e2a02 resumed at 02:03 from checkpoint 10526 and is at about step 10700 of 27718, loss 0.0027, 93 to 117 img/s on 21.2 GB. It is unchanged: no loss change, no scope change, cosine anchored at 9400 with adapter 1.77e-4 falling to zero by 27718. About 17000 steps remain, roughly fifteen hours, so it finishes late this afternoon rather than overnight.

## The omission experiment, and its answer

Your premise was that human annotators leave out tags that belong on the image, and that a loss punishing every unannotated tag as a hard negative teaches the model to suppress correct answers. Measured on 20000 training images with the bar as a diagnostic only, the omission was real: of the cells the bar scores at 0.9 or above, 14.7 percent are unannotated, and at 0.5 it is 40.3 percent.

The bar's predictions never entered training, per your rule. That killed the direct prior and also killed the co-occurrence substitute, which came back with one usable partner per tag because omission depresses the very co-occurrence counts that would reveal it.

So the arm tested the hypothesis using only the model's own confidence: any negative the model already scored above 0.5 had its loss halved. Two arms at production batch, 500000 images, effective 384, micro 96 times accum 4.

| arm | 50k macro | MCC | precision | recall | micro |
| --- | --- | --- | --- | --- | --- |
| 86 control | 0.2221 | 0.2364 | 0.2351 | 0.2815 | 0.4255 |
| 87 relaxation 0.5 | 0.2196 | 0.2366 | 0.2227 | 0.3016 | 0.3990 |

Falsified. Recall rose 0.0201 as the mechanism predicted, but precision fell 0.0124 instead of rising, so the model's confident negatives were mostly genuine absences rather than omissions. Macro down 0.0025, inside the 0.0093 interval; micro down 0.0265, far outside its 0.0016 interval. The full-relaxation arm was skipped because it pushes the same trade further in the direction that costs micro, and production resumed unchanged.

The useful lesson: our precision deficit of 0.235 against the bar's 0.620 is not caused by being too harsh about tags that belong on the image. It is what a 20M adaptation on a frozen backbone does against 692M fully fine-tuned.

## The cheapest macro on the board

The model prefers a threshold of 0.50 while the protocol scores at 0.40. A single temperature on the logits is monotone and legal, and fitted on the first 25000 validation images and judged on the other 25000 it reads:

| temperature | macro on the held-out half |
| --- | --- |
| 1.00, unchanged | 0.3465 |
| 0.45 | 0.3652 |
| 0.25 | 0.3686 |

Up to +0.0221 for one scalar. The gain is monotone to the edge of the grid, which means it is not really a soft temperature: it is the 0.50 decision boundary expressed as a rescale, which is the optimum the run's own validation had been reporting since step 2000.

Caveat that must be respected: the fit used validation images, and validation is what we are scored on. Fitting on half and quoting the other half is honest as an experiment, not as a final result. Two legitimate routes: fit the temperature on the training split or a disjoint slice, or bake it into training so the model's own 0.40 lands at its optimum with no post-hoc step. I recommend the second, and it is a small change to the head bias schedule rather than a new pipeline.

## Throughput

At 384 the micro-batch is free speed and we had been leaving it on the floor. Micro 48 used 9.1 GB and ran 121 to 152 img/s; micro 96 uses 17.9 GB and runs 168 to 187 img/s at the same effective 384. Also recorded in docs/benchmark.md, with the related trap: the bar at batch 64 ran 10 to 12 img/s while batch 16 ran 46 img/s, because memory pressure near the ceiling costs more than the batch buys.

## What I would do next, in order

First, nothing until production finishes, since it is the run that matters and every arm competes with it for the card.

Then the calibration work, because +0.022 is larger than anything the optimizer study produced and it is nearly free.

Then one memory-capacity or unfreeze-depth arm at B scale, since u1 is what production does and it is the only capacity lever never bracketed at production scale.

The bar is still out of reach this pass: production at 8083 was 0.3568 macro against 0.5579, and even with recalibration the projection is 0.44 to 0.48 by the end.
