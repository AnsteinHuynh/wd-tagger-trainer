# Normed momentum and SNR conditioning, and when they eat your model

A guide for other projects, written after a night of getting it wrong and then measuring why. It concerns the pair `normed_momentum` and `snr_cond` in `adv_optm`'s SinkSGD_ADV, but the failure mode is not specific to that optimizer. Any update rule that normalises a matrix row by row has the same cliff in it.

## What the pair actually does

`normed_momentum` keeps the momentum buffer and renormalises it so that every row of a two-dimensional parameter has equal root-mean-square magnitude before the update is formed. `snr_cond` then gates the update on the ratio of coherent signal to noise in that buffer, so a row whose momentum points the same way every step gets moved and a row whose momentum is thrashing does not. The intent is a trust region: step where the evidence is consistent, sit still where it is noise.

`snr_cond` cannot be switched on without `normed_momentum`. The optimizer raises rather than silently ignoring it, which is the one courtesy it extends you.

## The assumption nobody writes down

Both halves assume the rows of the matrix are the same kind of object. Equal RMS is a sensible constraint when row three and row five are interchangeable — six blocks of the same projection, say, or the rows of a hidden layer where nothing distinguishes one unit from another. Under that assumption the normalisation is close to free: there is no information in the relative magnitudes to destroy, and you get a step that cannot be dominated by one lucky row.

That assumption is false the moment your rows carry identity. A classification head with one row per tag is the clearest case. The row for a tag seen two million times and the row for a tag seen nine times are not interchangeable, and their norm difference is the trained model's entire frequency prior.

## The cliff, measured

We ran the pair three ways on the same model, same data, same 200000-image schedule, and read every arm at the same step 1000 checkpoint. The task was multi-label tagging with 12476 output rows, so the head is the case where rows definitely carry identity.

| where the pair was applied | macro F1 | micro F1 |
| no pair, plain SinkSGD | 0.0559 | 0.4386 |
| every trainable tensor | 0.0085 | 0.4277 |
| every trainable tensor, learning rate matched to plain | 0.0067 | 0.4003 |
| the 70 adapter tensors only | 0.0631 | 0.4601 |
| the 70 adapter tensors only, learning rate matched | 0.1545 (full protocol) | 0.4338 |

Read the two middle rows with the last two. Applied everywhere it is a catastrophe that no learning-rate correction rescues. Applied to tensors whose rows are interchangeable it is a small win. The difference between those two outcomes is not the switch, it is the parameter group it was allowed to touch.

The metric shape is the tell. Everywhere, micro fell two and a half percent while macro fell eighty five percent. A switch that damages the model uniformly drags both down together; this one leaves the bulk of the distribution intact and saws off the tail. If you see that signature — micro roughly intact, macro or rare-class F1 collapsed — suspect row normalisation before you suspect the learning rate.

## The canary, which costs one CPU command

Before you trust any row-normalising switch on a matrix whose rows have meaning, compare the spread of row norms with and without it. The coefficient of variation of the row norms on our head was 0.255 in the plain arm and 0.021 in the arm with the pair applied to the head. Not flattened a little. Flattened to nearly a single number, across 12476 rows with wildly different evidence counts.

That single statistic is the cheapest possible warning, it needs no GPU, and it decides the question before you spend an hour on training. If the switch collapses the spread of row norms on a matrix whose rows carry identity, stop.

## The scoping recipe

Apply the pair only to parameters whose rows are structurally interchangeable, and use a per-group flag rather than a global one. In our case that meant the seventy OFTv2 adapter factors, which are stored as six rows of 8128 coefficients — one row per diagonal block, all six describing rotations of different channel chunks of the same projection. Forcing those six to equal RMS costs nothing and buys a little.

Leave the head plain. Always. It is the one part of the model whose rows are guaranteed to carry identity, and it is the part that decides the metric you care about.

## What it is really doing, and the experiment you owe it

A constant-gradient probe said the pair steps 1.9x further than plain. Real training says the opposite. After the same 4000 steps the total L2 norm of the adapter tensors was 218.76 in the plain arm and 34.41 with the pair — six and a half times less travel — because real gradients are incoherent most of the time and the SNR gate refuses to move.

So if you are measuring a gated optimizer with a synthetic steady-state probe, you are measuring the one case where the gate is fully open. Do not quote that number as a step size.

That also means the obvious control is missing in most reports, ours included: if the gate is really just damping, then a plain optimizer at a proportionally lower learning rate should match it without any switch at all. Until you have run that, you do not know whether you have a mechanism or an expensive learning-rate reduction.

## The short version

The pair is a real idea and it is safe where rows are interchangeable and dimensions are large, which is why adapter factors and hidden layers respond to it well.

It is a model-deleting idea on a matrix whose rows are classes, tags, tasks or users, because equalising rows deletes the frequency prior and the rare classes are what most metrics reward.

Scope it with parameter groups, check the row-norm spread before and after, and do not trust a constant-gradient benchmark to tell you how far it will actually move.

Source arms, all base-384, micro batch 48, 200000-image cosine: plain 49, pair everywhere 46 and 59, pair scoped hot 57, pair scoped matched 71, spectral 64, 64b and 70. The full switch-by-group matrix is in the private experiment log under Optimizer switches at the three trainable groups, and the sister-project evidence the bundle came from is in `docs/optimizer-gold-settings.md`.
