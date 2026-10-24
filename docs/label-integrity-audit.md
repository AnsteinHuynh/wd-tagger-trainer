# Label-integrity audit — WDTagger v4 training sample

Scope: the 10000 train / 1000 valid / 1000 test images `wdtag.cache.build_plan` selects — 213 train tars × `ceil(10000/213)=47` = 10011 read, first 10000 kept; 12 × `ceil(1000/12)=84` = 1008 per eval split, first 1000 kept; contiguous, tar-name order, `_read_samples`

Verdict: 0 leakage, 0 unusable labels, 0 OOV loss, 0 duplicates; all 10000 training images are usable, the one real weakness being long-tail coverage (D14b)

### Method

`wdtag/cache.py` + `wdtag/labels.py`; schema `{id, width, height, rating, general_tags, character_tags}`; 12027 JSONs, 0 `json.loads` failures; keys and positives asserted equal to the trainer's cache (same order, CSR row lengths `ptr[j+1]-ptr[j]`); OOV detection self-tested with injected tags

## A. Missing labels

|#|Metric|train (n=10000)|valid (n=1000)|test (n=1000)|
|---|---|---|---|---|
|A1|images with zero tags of any kind|0 (0.00%)|0 (0.00%)|0 (0.00%)|
|A2|images whose 12476-dim multi-hot target is all zeros|0 (0.00%)|0 (0.00%)|0 (0.00%)|
|A3|images with >= 1 OOV tag occurrence|0 (0.00%)|0 (0.00%)|0 (0.00%)|
|A5|images with an empty `general_tags`|0 (0.00%)|0 (0.00%)|0 (0.00%)|
|A5|images with an empty `character_tags`|4211 (42.11%)|414 (41.40%)|426 (42.60%)|

### A3. OOV tags

|Metric|train|valid|test|
|---|---|---|---|
|all raw tag occurrences|322944|32674|33035|
|OOV occurrences|0|0|0|
|OOV per image|0.00|0.00|0.00|
|OOV share of occurrences|0.00%|0.00%|0.00%|

|category (train)|occurrences|OOV|OOV share|
|---|---|---|---|
|general (9225 names)|303285|0|0.00%|
|character (3247 names)|9659|0|0.00%|
|rating (4 names)|10000|0|0.00%|

Top-20 OOV offenders: none — empty list; 0 distinct OOV names, 0 signal discarded

Not measured: the tar metadata holds only Danbooru's selected tags, so anything filtered out before the webdataset was built is invisible — upstream of this repo

### A4. Rating

Exactly one rating tag per image in every split; malformed 0/0/0

|rating value|train|valid|test|
|---|---|---|---|
|general|2961 (29.61%)|315 (31.50%)|295 (29.50%)|
|sensitive|5009 (50.09%)|490 (49.00%)|501 (50.10%)|
|questionable|1153 (11.53%)|114 (11.40%)|125 (12.50%)|
|explicit|877 (8.77%)|81 (8.10%)|79 (7.90%)|

### A5. Schema / parse health

|check|train|valid|test|
|---|---|---|---|
|`json.loads` failures|0|0|0|
|all three tag fields are JSON lists|10000|1000|1000|
|unexpected extra top-level keys|0|0|0|
|bad `id` (missing / non-int / != filename stem)|0|0|0|
|zero-byte image blob|0|0|0|

All 12000 annotations have the six expected keys, correct types and a matching integer `id` — 0 parse or schema failures

## B. Repeated labels

|#|Metric|train|valid|test|
|---|---|---|---|---|
|B6|images with a tag repeated inside one annotation|0 (0.00%)|0 (0.00%)|0 (0.00%)|
|B6b|extra occurrences from within-annotation repeats|0|0|0|
|B7|duplicate image keys|0|0|0|
|B7b|distinct keys|10000|1000|1000|
|B8|redundant byte-identical images (sha1 of tar blob)|0|0|0|
|B8b|distinct raw-blob hashes|10000|1000|1000|
|B8c|redundant pixel-identical 512×512 tensors|0|0|0|
|B9|distinct in-vocab tag sets (target vectors)|10000|1000|1000|
|B9b|images sharing a target vector with another image|0|0|0|

B6 is 0, so raw counts equal unique-tag counts and `tag_indices` dedupes through a `set` regardless; ~32 labels from 12476 names makes collisions essentially impossible

## C. Cross-split leakage

|comparison|train vs test|train vs valid|test vs valid|
|---|---|---|---|
|shared image keys|0|0|0|
|shared annotation `id`|0|0|0|
|shared raw image bytes (sha1)|0|0|0|
|shared 512×512 pixels (sha1)|0|0|0|

### C11. Content leakage

Calibrated first on 5 real training images degraded like a repost:

|variant of the same picture|cosine (min / median / max)|
|---|---|
|JPEG q=90|1.0000 / 1.0000 / 1.0000|
|JPEG q=50|0.9999 / 0.9999 / 1.0000|
|JPEG q=20|0.9998 / 0.9999 / 0.9999|
|downscale to 256px + JPEG q=85|0.9999 / 0.9999 / 1.0000|
|90% centre crop|0.6836 / 0.8241 / 0.8663|
|horizontal mirror|0.0130 / 0.5005 / 0.5930|

Re-encodes cannot hide (≥0.9998); a 90% crop drops to 0.68–0.87, hence the visual check

|pair|pairs compared|max cosine|median|>= 0.99|>= 0.98|>= 0.95|>= 0.90|>= 0.85|>= 0.80|
|---|---|---|---|---|---|---|---|---|---|
|train vs test|10000000|0.8691|0.032|0|0|0|0|1|13|
|train vs valid|10000000|0.8885|0.035|0|0|0|0|4|20|
|test vs valid|1000000|0.8059|0.031|0|0|0|0|0|1|

Every cross-split pair compared by cosine of a per-image z-scored 32×32 grayscale thumbnail. Same-split max cosine: train 0.8910 (49995000 pairs), valid 0.8713 (499500 pairs), test 0.7829 (499500 pairs); 0 pairs ≥0.95 anywhere

Closest train/test pair overall:

|rank|train key|train tar|test key|test tar|cosine|
|---|---|---|---|---|
|1|`1565129`|00051.tar|`3807947`|00004.tar|0.8691|
|2|`3271209`|00126.tar|`8099907`|00007.tar|0.8410|
|3|`6511274`|00014.tar|`3512922`|00002.tar|0.8379|
|4|`6356109`|00053.tar|`6805901`|00004.tar|0.8346|
|5|`5403020`|00097.tar|`1806929`|00010.tar|0.8210|

Every pair ≥0.80 rendered and inspected — `similar-pairs-train-test.png` (13), `similar-pairs-train-valid.png` (20), `similar-pairs-test-valid.png` (1), closest six in `closest-train-test-pairs.png` — all different characters, compositions and backgrounds. dHash at Hamming ≤6: minimum MAE 11.37/255, 0 pairs below 5

Limits: a mirrored repost is undetectable (mirroring moves the descriptor to 0.50) but is a different pixel array, not a duplicate; "same artwork, different crop" is the only missable shape, ruled out by the 0.80+ inspection

Conclusion: zero leakage by key, id, byte hash, pixel hash, exhaustive nearest neighbour and visual inspection; the 1000-image test set is disjoint from train

## D. Practical impact

|#|Metric|train|valid|test|
|---|---|---|---|---|
|D13|images contributing zero positive labels|0 (0.00%)|0 (0.00%)|0 (0.00%)|
|D13b|images with <= 1 positive label|0|0|0|
|D14|mean positives / image|32.29|32.67|33.03|
|D14|p5 / median / p95|15 / 30 / 58|16 / 30 / 56|15 / 30 / 59|
|D14|min / max|10 / 275|11 / 141|11 / 143|
|D14|mean raw general / character / rating|30.33 / 0.97 / 1.00|30.72 / 0.95 / 1.00|31.09 / 0.94 / 1.00|

Positive-label histogram (train): 10–19 1621; 20–29 3273; 30–39 2689; 40–49 1361; 50–59 614; 60–69 249; 70–79 114; 80–89 31; 90–99 23; 100–109 13; 110–119 5; 120–129 2; 130–139 1; 140–149 3; 270–279 1

The 275-label maximum is one 10-character group illustration (`8368367`, `00123.tar`: 264 general + 10 character + 1 rating), not a parse artefact

### D14b. Coverage

||labels|
|---|---|
|label space|12476|
|labels with >= 1 positive in train|10790 (86.49%)|
|labels with zero positives in train|1686 (13.51%)|
|labels appearing in < 10 train images|7908 (63.39%)|
|labels with >= 10 train images|2882|
|labels that appear in test but never in train|217|
|... their share of test positives|237 of 33035 (0.72%)|
|labels that appear in valid but never in train|198 (214 of 32674 valid positives, 0.65%)|
|labels absent from all three sets|1296|

## What this means

Label integrity cannot explain any gap to the baseline — every check above is zero, so any gap against `convnextv2_huge` comes from the model, adapter, optimiser or the 10k budget
The test score will be honest — 1000 disjoint images, 11–143 positives each, no leakage asterisk on Macro/Micro F1/P/R/MCC
All 10000 images are usable — no zero-label, filler or duplicate; coverage, not volume, is the limit
The long tail is the weakness, and it lands on Macro — 2882 labels appear in ≥10 images, 7908 in 1–9; 217 of 4697 test labels (4.62%; 237 of 33035 positives, 0.72%) have no training example at all, so those Macro-F1 columns score 0 while the full-corpus baseline has seen them; Micro is barely touched
Expectations confirmed — ~32.3 positives per image (median 30, not 33), one rating each (50.1% `sensitive`, 29.6% `general`), 42% no character tag, so the head must accept "no character"

## Commands

```powershell
$env:PYTHONPATH="C:\Users\kerok\apps\wd-tagger-trainer"
cd C:\Users\kerok\apps\wd-tagger-trainer
C:\Python314\python.exe scratch\sophie\audit_scan.py test valid train
C:\Python314\python.exe scratch\sophie\audit_pixels.py
C:\Python314\python.exe scratch\sophie\audit_report.py
C:\Python314\python.exe scratch\sophie\audit_verify.py
C:\Python314\python.exe scratch\sophie\audit_similarity.py
C:\Python314\python.exe scratch\sophie\audit_calib.py
C:\Python314\python.exe scratch\sophie\audit_pairs.py
C:\Python314\python.exe scratch\sophie\audit_extra.py
```

Nothing outside `scratch/sophie/` was written; `runs/`, `wdtag/`, `configs/`, the tar dataset and `I:\.cache` were read only. No GPU, no training
