# Cached features: how to run twenty experiments in the time of one

The production model spends almost all of its step time in the backbone. Once a checkpoint is chosen and its features are cached, the backbone is a fixed function and every head, loss, calibration and ensembling experiment becomes a memmap read. This is the tool that turns a fourteen-hour run into an afternoon of experiments.

## The two commands

Cache one checkpoint over one split:

    python -m wdtag.cache_features --run runs/459e2a02 --split valid --out ./cache/wd-tagger-trainer/feat/459e2a02-val

    python -m wdtag.cache_features --run runs/459e2a02 --split train --limit 5321713 --out ./cache/wd-tagger-trainer/feat/459e2a02-train

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

## bf16 against fp32, measured 2026-10-25

The question was whether the pass should compute in fp32 for accuracy. Two identical heads, same seed, same hyperparameters, same 20000 training images in the same order, differing only in how the features were produced, scored on the 50k at 0.40:

| features | macro | micro | precision | recall | MCC |
| --- | --- | --- | --- | --- | --- |
| bf16 compute, fp16 store | 0.0062475 | 0.2978656 | 0.2160629 | 0.4793505 | 0.3192977 |
| fp32 compute, fp32 store | 0.0062529 | 0.2978951 | 0.2160894 | 0.4793728 | 0.3193252 |

The gain is 0.0000055 macro and 0.00003 micro, which is five orders of magnitude inside the 0.0093 bootstrap interval, so it is unmeasurable rather than small. The cost is real: fp32 computes at 91 images per second against bf16 at 170 through the same tar stream, so the corpus pass takes 1.9 times as long.

The features themselves do differ, and by more than the storage question suggests: over 2000 validation images the median relative difference is 0.57 percent and the largest absolute difference is 7.9 on a block whose largest value is 34. That is accumulated bf16 error through twelve layers, not the fp16 container, and it is why the storage precision argument and the compute precision argument are separate. Despite the difference, the two heads converge to the same point, with identical weight norms to three decimals, because the perturbation is coherent noise that 20000 samples average over.

Conclusion: cache in bf16 arithmetic and fp16 storage. fp32 is a 1.9x slower pass for a difference no experiment at our scale can resolve.

## How the cache is read, and the bug that made it unreadable, fixed 2026-10-25

The cache is a memory-mapped file, never loaded into RAM. The operating system pages in only the rows the trainer touches, so a 32.7 GB cache is fine on a 24 GB card and 32 GB of RAM.

The first version of the trainer gathered a batch by slicing from the smallest to the largest selected row. With a randomly shuffled batch of 1024 out of 5.3 million, that span is almost the entire file, so every step would have read 32.7 GB and the experiment would have been impossible. It went unnoticed because the pilot ran on a 20000-row cache, where the span is 123 MB and the mistake is cheap. Fixing the indentation of the optimiser block that the same edit displaced was the second half of the repair, and it is why the pilot first reported two training steps per epoch instead of thirty-eight.

The reader now takes contiguous chunks of 50000 rows by default, one seek each, and shuffles inside the chunk. Measured on the CMSS cache at 3072 dimensions, batch 512, one epoch of 40000 rows: 10330 rows per second, against 931 for the span reader. A full 5321713-row epoch is therefore 8.6 minutes of GPU and 32.7 GB of sequential disk read, not a night.

## Crash safety, tested by killing it four times on 2026-10-25

A cache pass is six to eight hours, so the resume path is the feature. Every 200 rows the writer flushes the features, writes a labels file named for its row count, and only then advances meta.json. A crash therefore loses at most 200 rows, about one second, and resume skips by reading tar headers rather than decoding images.

Testing that claim found three defects that would each have cost hours, which is why the claim is now measured rather than asserted. First, an atomic rename onto labels.npz raised WinError 5 at row 418560, a transient Windows denial after a 55 MB write, and killed the run. Renaming over a file another process may hold open is not a safe hot-path design. Second, the resume restored the row count from meta.json but not the pointer to which labels file it belonged to, so it silently loaded the previous generation, 512 rows short. Third, that mismatch then made the offsets array the wrong length and killed the run again with a cumsum error.

The fixes are structural: labels are written to a fresh name each time so there is nothing to overwrite, the pointer travels in meta.json with the row count, a labels file whose row count does not match stops the run with a message instead of corrupting it, and the features are flushed before the counter moves so a power cut cannot leave the counter ahead of the pages. Killing the process at 418048 and at 438528 both resumed at exactly the recorded row.

## Why throughput fell, and the quadratic checkpoint, fixed 2026-10-25

The rate fell from 195 to 147 images per second over two hours, and the cause was the checkpoint rather than the drive. Every checkpoint rewrote the entire cumulative label array as a new npz: 198 MB at 1.5 million rows, every four seconds, and growing linearly with the cache. Total label I/O is therefore quadratic in the corpus size, which is invisible for the first hour and fatal by the last.

Labels are now append-only raw arrays, `labels_indices.i32` and `labels_offsets.i64`, one int64 cumulative offset per row. A checkpoint appends the delta and flushes, so total label I/O equals the labels themselves, about 190 MB for 1.5 million rows and 680 MB for the whole corpus. The rate returned to 199 images per second immediately after the change.

Two consequences worth knowing. A crashed run truncates both files back to the recorded row count on resume, so a half-appended tail cannot desynchronise labels from features. And because the arrays are read whole by whoever needs them, a npz reader must be closed explicitly; `np.load` on an npz is lazy and held a 199 MB file open for the life of the process until this was noticed.

## Sharding by tar, 2026-10-25

The train cache is cut into 213 shards, one per tar, in `feat/siglip2-cmss-shards`. Each is self-contained, so it can be read, trained on or discarded alone:

```
tar_00019.npy    25000 x 3072 fp16, memmap-able, 154 MB
tar_00019.npz    indices int32, offsets int64 relative to the shard
shards.json      every shard with rows, source row range, label count and blake2b
```

`shards.json` is rewritten after every shard, so a half-finished split is usable and resumable. Every shard is hashed against its source slice as it is written, and `--check` re-hashes the lot: 213 shards, 5321713 rows, 0 mismatched. The labels were compared separately against the source offsets, 11 sampled shards, 0 mismatched, and the label totals agree at 171947311. Only then was the monolithic copy deleted.

The point is that the order becomes explicit and free. The rows are fixed; a seed or a tar list is a permutation of 213 files:

```
--tars 00022,00015,00012     those three tars, in that order
--shard-seed 2               a different permutation of all 213
(no flag)                    the order the cache was written, seed 1 pass 0
```

`wdtag.shard_cache` derives those boundaries from the manifest and the seed, never from the tars, and never by shuffling the manifest's key order, which is a different permutation and was a real bug caught before any shard was written.

### The off-by-one that the label totals caught

`offsets[i]` is the cumulative end of row i, so the start of a row range is the previous end, and row 0 needs a leading zero. Slicing with `offsets[start]` shifted every shard's labels by one row: the features were byte-identical, the shard was internally consistent, and it trained without complaint while attributing every image's tags to its neighbour. The tell was a label count of 802969 where the earlier verified split said 802950. The splitter now refuses to finish unless the shard label totals equal the source total.

### The tail that a pass never saw

The streamer emitted a batch only when it filled, so the final partial batch was dropped: a corpus pass covered `floor(5321713 / 512) * 512 = 5321472` images and the cache ended 241 rows short of its own declared row count, with zeros in the rows it claimed. The tail is now emitted and the cache was completed to 5321713 by resuming at the checkpoint.
