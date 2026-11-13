# The tar manifest

`runs/_manifests/manifest_<split>.json` is the durable record of the corpus: which tars exist, how many image+annotation pairs each holds, and the file identity of each tar. It is mirrored to the cache root for older readers. The durable copy is authoritative because the cache root is disposable and losing the table costs a header scan of 213 tars, about an hour at ten seconds each.

Schema 2, and schema 1 readers keep working because `split`, `tars` and `total` are still at the top level:

```
schema      2
split       train
root        datasets/train
count_rule  complete image+json pair, matched by stem, headers only
generated   2026-10-25T13:44:11
total       5321713
tars        { "00001.tar": 25000, ... }      count per tar
files       { "00001.tar": { bytes, mtime } } identity per tar
unverified  tars with no recorded count yet
```

## The ordering trap

Order is `random.Random(seed * 1000003 + pass_index)` shuffled over the sorted directory listing. It is not shuffled over the table's key order, and the difference is not cosmetic: the table is written in the order counts were learned, which is itself a shuffle, so shuffling it again produces a different permutation while still looking plausible. Caching with the wrong permutation is invisible, but a shard labelled with the wrong tar name is silently mislabelled data that still trains and still scores.

`wdtag.manifest.order` is the single definition. `wdtag.stream._order` and `wdtag.shard_cache.tar_order` both call it, and a test in this file's history confirmed all three agreed for train and valid. Never write the shuffle again in a new file.

## Commands

```
python -m wdtag.manifest --split all --build          stat every tar, record identity
python -m wdtag.manifest --split all --check          compare the table against the filesystem
python -m wdtag.manifest --split train --verify 3     walk three spread-out tars and recount
python -m wdtag.manifest --split train --verify 0     recount every tar, about 35 minutes for train
python -m wdtag.manifest --split train --ranges       row ranges for seed 1, pass 0
```

`--build` never recounts, because recounting costs a header walk per tar; it carries counts forward and records identity. That is why `unverified` exists and why `--verify` is a separate, deliberate act. Verified so far on 2026-10-25: valid 00001 and 00007, test 00001 and 00007, all four counted 25000 against 25000 recorded, at 11 to 18 seconds per tar.

`--check` is cheap and compares names, totals and identity against disk. It catches a truncated tar, a replaced tar, a missing tar and a tar with no recorded count. It cannot catch a wrong count, which is what `--verify` is for.

## Row ranges

`ranges(split, seed, pass_index)` returns `[(tar, first_row, rows)]`, which is what makes a cache addressable by tar. Any seed and any explicit tar order are free: the cache rows are fixed, the order is a permutation over them, so tar 22 then 15 then 12 is a list of ranges to read rather than a new six-hour pass.
