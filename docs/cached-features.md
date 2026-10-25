# Cached features: how to run twenty experiments in the time of one

The production model spends almost all of its step time in the backbone. Once a checkpoint is chosen and its features are cached, the backbone is a fixed function and every head, loss, calibration and ensembling experiment becomes a memmap read. This is the tool that turns a fourteen-hour run into an afternoon of experiments.

## The two commands

Cache one checkpoint over one split:

    python -m wdtag.cache_features --run runs/459e2a02 --split valid --out I:/.cache/wd-tagger-trainer/feat/459e2a02-val

    python -m wdtag.cache_features --run runs/459e2a02 --split train --limit 5321713 --out I:/.cache/wd-tagger-trainer/feat/459e2a02-train

Train and score a head on those caches:

    python -m wdtag.train_cached --train-feats <train dir> --val-feats <val dir> --out runs/cached-linear --head linear --epochs 3

## What is cached and why it is safe to reuse

`forward_features` output before the head, 768 dimensions for the production `head_pool = cls`, stored float16. Rows are written in the streamer's deterministic order and labels go beside them in the same CSR format as the validation cache, so row i of `feats.npy` always belongs to row i of `labels.npz`. A partial cache resumes by skipping the rows already written, which is exact because the order is a pure function of seed and pass index.

Sizes: train is 8.2 GB at 768 dimensions for 5.3 million images, validation is 77 MB for 50 thousand. The forward pass is the cost, roughly four hours for the full corpus at 90 to 120 img/s.

## What it cannot do

It cannot improve the backbone. The ceiling is the representation in the checkpoint that was cached, and our own measurements say that ceiling is real: OFT rotation plateaued at step 9457, and per-tag F1 is flat from five positives to five hundred. What caching buys is the ability to test whether the head, the loss, the calibration or a second tower's features can reach that ceiling, instead of guessing at one configuration per night.

## The head trainer's outputs

`report()` prints the protocol number at the fixed 0.40 threshold plus the predicted tags per image, which is the calibration diagnostic: 32.3 is the label rate and anything near twice that means precision is being thrown away. `best_head.pt` and `metrics.json` land in the output directory, selected on macro by default.

The trainer refuses a dimension mismatch between the train and validation caches, because concatenated towers are how the ensemble experiment works and a silent mismatch there would score a model that never existed.
