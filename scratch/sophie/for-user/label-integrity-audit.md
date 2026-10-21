# Label-integrity audit — WDTagger v4 training sample

Scope: the **10,000 train / 1,000 valid / 1,000 test** images that `wdtag.cache.build_plan` actually selects — 213 train tars x `ceil(10000/213)=47` = 10,011 read, first 10,000 kept; 12 x `ceil(1000/12)=84` = 1,008 per eval split, first 1,000 kept. Reads are contiguous from each tar's start (no seeking), in tar-name order, using the repo's own `_read_samples`.

## Headline

1. **No leakage of any kind.** Train/test share **0 keys, 0 ids, 0 byte-identical images, 0 pixel-identical 512x512 tensors**. An exhaustive search over all **10,000,000 train x test pairs** (cosine of a normalised 32x32 grayscale thumbnail) gives a **maximum similarity of 0.869**, against 0.9999 for the same picture re-encoded at JPEG q=20 and 0.9999 for a downscale-and-reupload. The 13 train/test pairs at or above 0.80 were rendered and inspected (`similar-pairs-train-test.png`): all are different pictures, none is a crop or a re-encode. Train/valid max 0.889; test/valid max 0.806.
2. **Zero usable labels: 0 train / 0 valid / 0 test** (0.00%). The minimum is 10 positive labels in train, so every one of the 10,000 images contributes signal; none are 'all-negative' filler.
3. **Out-of-vocabulary rate is exactly 0.00%** — 0 of 322,944 train tag occurrences are outside the 12,476-name space (same for valid/test). This is structural, not luck: the annotation JSONs only carry the dataset's *selected* 9,225 general + 3,247 character + 4 rating names, and our label space is that same selected list. Nothing is dropped when we build targets.
4. **Positive labels per image: mean 32.29 in train**, 32.67 valid, 33.03 test; train median 30, p5 15, p95 58, max 275. The '~33 per image' figure is right for the mean but the median is 30.
5. **No repeats anywhere**: 0 duplicate tags within annotations, 0 duplicate keys, 0 duplicate image bytes, 0 duplicate target vectors (10,000/10,000 distinct). Every image has exactly one valid `rating`.

### Method / verification of the sample

* `wdtag/cache.py` + `wdtag/labels.py` read first; annotation schema on every image is `{id, width, height, rating, general_tags, character_tags}`.
* The selected keys were asserted to equal the trainer's own cache keys **in the same order** for all 12,000 images, and each image's recomputed positive count was asserted equal to its cache CSR row length (`ptr[j+1]-ptr[j]`), i.e. this audit measures the trainer's actual targets, not a re-derivation.
* OOV detection was self-tested by injecting a fake general and a fake character tag; `tag_indices` returned both as unknown and mapped `sensitive`->1, `1girl`->4, so a genuine OOV would have been counted.
* 12,027 annotation JSONs parsed; 0 `json.loads` failures (a failure would have aborted the scan).

## A. Missing labels

| # | Metric | train (n=10,000) | valid (n=1,000) | test (n=1,000) |
|---|---|---|---|---|
| A1 | images with zero tags of any kind | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| A2 | images whose 12,476-dim multi-hot target is all zeros | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| A3 | images with >= 1 OOV tag occurrence | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| A5 | images with an empty `general_tags` | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| A5 | images with an empty `character_tags` | 4,211 (42.11%) | 414 (41.40%) | 426 (42.60%) |

### A3. Out-of-vocabulary tags

| Metric | train | valid | test |
|---|---|---|---|
| all raw tag occurrences | 322,944 | 32,674 | 33,035 |
| OOV occurrences | 0 | 0 | 0 |
| OOV per image | 0.00 | 0.00 | 0.00 |
| OOV share of occurrences | 0.00% | 0.00% | 0.00% |

| category (train) | occurrences | OOV | OOV share of that category |
|---|---|---|---|
| general (9,225 names) | 303,285 | 0 | 0.00% |
| character (3,247 names) | 9,659 | 0 | 0.00% |
| rating (4 names) | 10,000 | 0 | 0.00% |

Top-20 OOV offenders: **none — the list is empty** (0 distinct OOV names across all three splits). Signal discarded at label-loading time: 0 occurrences.

Caveat on what this does *not* measure: the tar metadata already contains only Danbooru's *selected* tags (README: 9,225 general / 3,247 character / 4 rating selected). Any tag that was filtered out before the webdataset was built is absent from the JSON and therefore invisible here; that loss happened upstream of this repo, not in `tag_indices`.

### A4. Rating field

| rating tags per image | train | valid | test |
|---|---|---|---|
| 1 | 10,000 | 1,000 | 1,000 |

| rating value | train | valid | test |
|---|---|---|---|
| general | 2,961 (29.61%) | 315 (31.50%) | 295 (29.50%) |
| sensitive | 5,009 (50.09%) | 490 (49.00%) | 501 (50.10%) |
| questionable | 1,153 (11.53%) | 114 (11.40%) | 125 (12.50%) |
| explicit | 877 (8.77%) | 81 (8.10%) | 79 (7.90%) |

Malformed rating (not exactly one tag, or a value outside the 4-name set): train 0, valid 0, test 0. `rating` is a JSON list of exactly one string on every image, so the 'exactly one per image' expectation holds at 100%.

### A5. Annotation schema / parse health

| check | train | valid | test |
|---|---|---|---|
| `json.loads` failures | 0 | 0 | 0 |
| all three tag fields are JSON lists | 10,000 | 1,000 | 1,000 |
| unexpected extra top-level keys | 0 | 0 | 0 |
| bad `id` (missing / non-int / != filename stem) | 0 | 0 | 0 |
| zero-byte image blob | 0 | 0 | 0 |

Every one of the 12,000 annotations has exactly the six expected keys, correct types, and a matching integer `id`. **0 parse or schema failures.**

## B. Repeated labels

| # | Metric | train | valid | test |
|---|---|---|---|---|
| B6 | images with a tag repeated inside one annotation | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| B6b | extra occurrences from within-annotation repeats | 0 | 0 | 0 |
| B7 | duplicate image keys | 0 | 0 | 0 |
| B7b | distinct keys | 10,000 | 1,000 | 1,000 |
| B8 | redundant byte-identical images (sha1 of tar blob) | 0 | 0 | 0 |
| B8b | distinct raw-blob hashes | 10,000 | 1,000 | 1,000 |
| B8c | redundant pixel-identical 512x512 tensors | 0 | 0 | 0 |
| B9 | distinct in-vocab tag sets (target vectors) | 10,000 | 1,000 | 1,000 |
| B9b | images sharing a target vector with another image | 0 | 0 | 0 |

There is no duplicate-tag inflation to report: since B6 is 0, raw occurrence counts and unique-tag counts are identical. `tag_indices` dedupes through a `set` anyway, so even if repeats existed they could not inflate the multi-hot target — only the raw count statistics.

B9 needs no 'largest groups' table either: all 10,000 training images have a distinct target vector. With ~32 labels drawn from 12,476 names, collisions are essentially impossible, which is what we see.

## C. Cross-split leakage (highest-stakes section)

| comparison | train vs test | train vs valid | test vs valid |
|---|---|---|---|
| shared image **keys** | 0 | 0 | 0 |
| shared annotation `id` | 0 | 0 | 0 |
| shared **raw image bytes** (sha1) | 0 | 0 | 0 |
| shared **512x512 pixels** (sha1) | 0 | 0 | 0 |

### C11. Content leakage, searched exhaustively rather than by threshold

For each cross-split pair of sets, every pair of images was compared using cosine similarity of a per-image z-scored 32x32 grayscale thumbnail. First the descriptor was calibrated on 5 real training images, degraded the way a repost is degraded:

| variant of the *same* picture | cosine (min / median / max) |
|---|---|
| JPEG q=90 | 1.0000 / 1.0000 / 1.0000 |
| JPEG q=50 | 0.9999 / 0.9999 / 1.0000 |
| JPEG q=20 | 0.9998 / 0.9999 / 0.9999 |
| downscale to 256px + JPEG q=85 | 0.9999 / 0.9999 / 1.0000 |
| 90% centre crop | 0.6836 / 0.8241 / 0.8663 |
| horizontal mirror | 0.0130 / 0.5005 / 0.5930 |

So re-encoding, resizing and re-uploading all stay >= 0.9998 (they cannot hide), while a 90% centre crop drops into the 0.68-0.87 range — meaning the band from 0.80 up had to be checked by eye, and it was.

| pair | pairs compared | max cosine | median | >= 0.99 | >= 0.98 | >= 0.95 | >= 0.90 | >= 0.85 | >= 0.80 |
|---|---|---|---|---|---|---|---|---|---|
| train vs test | 10,000,000 | 0.8691 | 0.032 | 0 | 0 | 0 | 0 | 1 | 13 |
| train vs valid | 10,000,000 | 0.8885 | 0.035 | 0 | 0 | 0 | 0 | 4 | 20 |
| test vs valid | 1,000,000 | 0.8059 | 0.031 | 0 | 0 | 0 | 0 | 0 | 1 |

| pair | same-split max cosine (all pairs) |
|---|---|
| train (49,995,000 pairs) | 0.8910 (0 pairs >= 0.95) |
| valid (499,500 pairs) | 0.8713 (0 pairs >= 0.95) |
| test (499,500 pairs) | 0.7829 (0 pairs >= 0.95) |

The closest train/test pair overall:

| rank | train key | train tar | test key | test tar | cosine |
|---|---|---|---|---|---|
| 1 | `1565129` | 00051.tar | `3807947` | 00004.tar | 0.8691 |
| 2 | `3271209` | 00126.tar | `8099907` | 00007.tar | 0.8410 |
| 3 | `6511274` | 00014.tar | `3512922` | 00002.tar | 0.8379 |
| 4 | `6356109` | 00053.tar | `6805901` | 00004.tar | 0.8346 |
| 5 | `5403020` | 00097.tar | `1806929` | 00010.tar | 0.8210 |

Every pair at cosine >= 0.80 was rendered side by side and visually inspected: `similar-pairs-train-test.png` (13 pairs), `similar-pairs-train-valid.png` (20 pairs), `similar-pairs-test-valid.png` (1 pair), plus the closest six in `closest-train-test-pairs.png`. **All are different characters, compositions and backgrounds** — shared 'bright sky / white background / 1girl portrait' composition, not the same picture and not a crop of one. As a second, independent check, the dHash candidate pairs at Hamming <= 6 were compared pixel by pixel on the 512x512 tensors: minimum mean-absolute-error 11.37/255, and 0 pairs below MAE 5. Identical content would be ~0.

Limits of this test, stated plainly: (i) a mirrored repost is not detectable this way, because mirroring moves the descriptor to 0.50 — but a mirror is a different pixel array, not a duplicate; (ii) 'same artwork, different crop' is the only shape of leakage the descriptor could miss, and the exhaustive 0.80+ band inspection above is what rules it out for this sample.

**Conclusion: zero leakage**, by key, by id, by byte hash, by pixel hash, by exhaustive nearest neighbour, and by visual inspection of the entire high-similarity band. The 1,000-image test set is disjoint from train.

## D. Practical impact

| # | Metric | train | valid | test |
|---|---|---|---|---|
| D13 | images contributing **zero positive labels** | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| D13b | images with <= 1 positive label | 0 | 0 | 0 |
| D14 | mean positives / image | 32.29 | 32.67 | 33.03 |
| D14 | p5 / median / p95 | 15 / 30 / 58 | 16 / 30 / 56 | 15 / 30 / 59 |
| D14 | min / max | 10 / 275 | 11 / 141 | 11 / 143 |
| D14 | mean raw general / character / rating | 30.33 / 0.97 / 1.00 | 30.72 / 0.95 / 1.00 | 31.09 / 0.94 / 1.00 |

Positive-label histogram (train):

| positives | images |
|---|---|
| 10-19 | 1,621 |
| 20-29 | 3,273 |
| 30-39 | 2,689 |
| 40-49 | 1,361 |
| 50-59 | 614 |
| 60-69 | 249 |
| 70-79 | 114 |
| 80-89 | 31 |
| 90-99 | 23 |
| 100-109 | 13 |
| 110-119 | 5 |
| 120-129 | 2 |
| 130-139 | 1 |
| 140-149 | 3 |
| 270-279 | 1 |

The 275-label maximum is a single 10-character group illustration (`8368367`, `00123.tar`: 264 general + 10 character + 1 rating), not a parsing artefact.

### D14b. Label coverage (the one number worth acting on)

| | labels |
|---|---|
| label space | 12,476 |
| labels with >= 1 positive in train | 10,790 (86.49%) |
| labels with **zero positives in train** | 1,686 (13.51%) |
| labels appearing in < 10 train images | 7,908 (63.39%) |
| labels with >= 10 train images | 2,882 |
| labels that appear in **test but never in train** | 217 |
|  ... their share of test positives | 237 of 33,035 (0.72%) |
| labels that appear in **valid but never in train** | 198 (214 of 32,674 valid positives, 0.65%) |
| labels absent from all three sets | 1,296 |


## What this means

* **Label integrity is not a risk here, and cannot explain any gap to the baseline.** Missing labels, OOV loss, duplicate tags/keys/images, schema drift and cross-split leakage are all exactly zero. Any difference against `convnextv2_huge` will come from the model, the adapter, the optimiser or the 10k budget — not from dirty labels or a contaminated test set.
* **The test score will be honest.** 1,000 images, disjoint from train by every test available, each with 11-143 positive labels. Reported Macro/Micro F1/P/R/MCC on it can be compared to the baseline without a leakage asterisk.
* **All 10,000 training images are usable.** No zero-label images, no all-negative filler, no duplicate pictures: effective training-set size equals nominal size. The thing that limits you is not wasted images, it is that 10,000 images only cover 10,790 of 12,476 labels at all.
* **The real weakness is the long tail, and it interacts with the metric you choose.** Only 2,882 labels appear in >= 10 training images; 7,908 appear in only 1-9. Of the 4,697 labels that actually occur in test, 217 (4.62%) have **no training example at all** (237 of 33,035 test positives, 0.72%), so on Macro-F1 those columns are a guaranteed 0 for a model trained on this 10k sample, while the full-corpus baseline has seen them. Micro metrics are barely touched; Macro is where this sample will lose points.
* **Practical expectations confirmed rather than corrected**: ~32.3 positive labels per train image (median 30, not 33), exactly one rating each (50.1% `sensitive`, 29.6% `general` in train), and 42% of images with no character tag at all — expected for scenery/object art and multi-character scenes, and it means the character head must tolerate 'no character' as a valid answer.

## Exact commands run

```powershell
$env:PYTHONPATH="C:\Users\kerok\apps\wd-tagger-trainer"
cd C:\Users\kerok\apps\wd-tagger-trainer
# stage 1: one contiguous pass over every plan entry of all three tars (writes records_*.pkl)
C:\Python314\python.exe scratch\sophie\audit_scan.py test valid train
# stage 2: hash the trainer's own 512x512 cache tensors + assert cache key order == tar order
C:\Python314\python.exe scratch\sophie\audit_pixels.py
# stage 3: all A-D statistics -> scratch/sophie/summary.json
C:\Python314\python.exe scratch\sophie\audit_report.py
# extra verification: OOV self-test, pHash, pixel MAE on near-dup candidates, within-split pairs
C:\Python314\python.exe scratch\sophie\audit_verify.py
# exhaustive all-pairs similarity (10M train x test, 10M train x valid, 1M test x valid)
C:\Python314\python.exe scratch\sophie\audit_similarity.py
# calibrate the descriptor on degraded copies (JPEG q=90/50/20, downscale, 90% crop, mirror)
C:\Python314\python.exe scratch\sophie\audit_calib.py
# render every cross-split pair with cosine >= 0.80 for visual inspection
C:\Python314\python.exe scratch\sophie\audit_pairs.py
# contact sheet of the closest pairs + label-coverage table
C:\Python314\python.exe scratch\sophie\audit_extra.py
```

Nothing outside `scratch/sophie/` was written; `runs/`, `wdtag/`, `configs/`, the tar dataset and `I:\.cache` were only read. No GPU was used and no training was launched.
