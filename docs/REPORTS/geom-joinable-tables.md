# Joinable tables

| file | rows (excl. header) | order | columns |
|---|---|---|---|
| `scratch/sophie/test_aspect.csv` | 1000 | `CacheDataset` index order = prediction order | `row,w,h,aspect,key,shard` |
| `scratch/sophie/train_sample_aspect.csv` | 10000 | same, for the training set the trainer sees | `row,w,h,aspect,key,shard` |
| `scratch/sophie/train_broad_aspect.csv` | 40000 | broad sample index | `row,w,h,aspect,key,shard` |

Layout matches `build_plan` exactly: test = 84 consecutive images from each of `00001.tar` … `00011.tar` (rows 0–923), then 76 from `00012.tar` (rows 924–999); train = 47 consecutive images from each of the 213 train tars (rows 0–9963), then 36 from `00213.tar` (rows 9964–9999). `row` equals the 0-based line index; all rows are contiguous per shard; 0 duplicate keys.

`test_aspect.csv`'s `row` is the index `wdtag/infer.py:predict_probs` writes into `test_probs.npy` (iterating `range(0, n, batch_size)` over `CacheDataset.index`, shard-major — shard 0 rows 0…624, shard 1 rows 0…374, …), so `test_probs.npy[i]` ↔ `test_aspect.csv` row `i`. The order was not read back from `I:\.cache` (forbidden); it is derived from the same code path the cache builder uses and
corroborated at 800/800 images against the real train cache via run manifests.
