# Tag forensics — are distortion and padding annotated?

Two independent measurements. Vocabulary scan: the dataset's `tags.csv`, all 12476 labels with corpus-wide counts over 5914596 images — a zero here is a strong negative, not a sampling artifact. Sample scan: 40000 contiguous train images; `exact` = whole tag equals the pattern, `token` = whole underscore-token(s) inside the tag, `sub` = substring anywhere (noise-prone: `clothing` contains `thin`). Prevalence is images (multi-label) out of 40000.

## Stretch / distortion patterns

| pattern | class | matched tag(s) | corpus count | images / 40k | prevalence |
|---|---|---|---|---|---|
| `deformed` | exact | `deformed` | 5834 | 28 | 0.070% |
| `bad proportions` | exact | `bad_proportions` | 2250 | 6 | 0.015% |
| `bad anatomy` | exact | `bad_anatomy` | 16592 | 49 | 0.122% |
| `distort` | sub | `distortion` | 1055 | 8 | 0.020% |
| `stretch` | token | `cross-body_stretch` | 768 | 3 | 0.008% |
| `stretch` | sub | `outstretched_arms` 71812 · `outstretched_arm` 70317 · `outstretched_hand` 25745 · `stretching` 17214 · `outstretched_leg` 1786 | — | 820 | 2.050% |
| `thin` | sub | `clothing_cutout` 697 · `clothing_aside` 165 · `heavy_breathing` 163 · `shrug_(clothing)` 116 · `shared_bathing` 85 · …17 tags | — | 1364 | 3.410% |

The 2.05% "stretch" rate is entirely pose (`outstretched_arms`, `stretching`); the 3.41% "thin" rate is substring noise (`clothing`, `breathing`, `bathing`, `thinking`). Neither is image distortion. Only `deformed` (0.07%), `bad_proportions` (0.015%), `bad_anatomy` (0.12%) and `distortion` (0.02%) describe deformed output, all ordinary art-criticism tags below 1%.

## Padding / letterbox / bar patterns

| pattern | class | matched tag(s) | corpus count | images / 40k | prevalence |
|---|---|---|---|---|---|
| `letterbox` | sub | `letterboxed` | 59279 (1.00%) | 268 | 0.670% |
| `pillarbox` | sub | `pillarboxed` | 6152 (0.10%) | 33 | 0.083% |
| `border` | exact | `border` | 155761 (2.63%) | 757 | 1.893% |
| `border` | token | `white_border` | 99060 (1.68%) | 449 | 1.123% |
| `border` | token | `outside_border` | 61780 (1.05%) | 315 | 0.788% |
| `border` | token | `black_border` 17664 · `inset_border` 2605 · `ornate_border` 2232 · `grey_border` 3795 · `pink_border` 2668 · +5 | — | 83 / 12 / 16 / 27 / 8 (per tag; nearly all co-occur with `border`) | ≤0.21% each |
| `border` | any | any `*border` tag (848 distinct images; only 1 lacks one of `border`/`white_border`/`outside_border`) | — | 848 | 2.120% |
| `panel` | sub | `borderless_panels` | 1294 | 3 | 0.008% |
| `padding` | token | `breast_padding` (garment, not image) | 1373 | 11 | 0.028% |
| `white background` | exact | `white_background` | 1650067 (27.9%) | 7000 | 17.500% |
| `simple background` | exact | `simple_background` | 1993448 (33.7%) | 8461 | 21.152% |
| `comic` | exact | `comic` | 572373 (9.68%) | 2302 | 5.755% |
| `comic` | token | `silent_comic` (87) · `segmented_comic` (6) · `right-to-left_comic` (6) | — | 87 / 6 / 6 | 0.218% / 0.015% / 0.015% |
| `comic` | any | any `*comic` tag | — | 2308 | 5.770% |
| `4koma` | exact | `4koma` (+`multiple_4koma` 26, `square_4koma` 11) | 102478 (1.73%) | 417 (454 family) | 1.043% |
| `2koma` | exact | `2koma` | 31895 (0.54%) | 137 | 0.343% |
| `multiple views` | exact | `multiple_views` | 183377 (3.10%) | 868 | 2.170% |
| `split screen` | exact | `split_screen` | 1958 (0.03%) | 12 | 0.030% |
| `out of frame` | exact | `out_of_frame` | 35274 (0.60%) | 168 | 0.420% |
| `out of frame` | token | `feet_out_of_frame` 723 · `foot_out_of_frame` 157 · `head_out_of_frame` 131 · `knees_out_of_frame` 5 | — | 996 | 2.490% |
| `cropped` | token | `cropped_jacket` 254 · `cropped_legs` 227 · `cropped_torso` 207 · `cropped_shirt` 123 · +6 | — | 865 | 2.163% |
| `two side` | token | `two_side_up` (hairstyle — noise) | — | 770 | 1.925% |
| `manga` | token | `manga_cover` 15 · `manga_(object)` 14 · `left-to-right_manga` 10 | — | 39 | 0.098% |

`white_background` / `simple_background` are listed for completeness only: a white canvas is not a white bar.

## The user's terms that do not exist at all

Zero hits in the sample and zero in the entire 12476-tag vocabulary (the string occurs in no tag name): `aspect ratio`, `bad aspect ratio`, `disproportion`, `white bar`, `side panel`, `white background panel`, `squashed`, `squished`, `elongat…`, `fattened`, `messed up`, `side by side`.

Nearest real tags: `letterboxed` / `pillarboxed` (exact intent) and `border` / `white_border` (frame present) — exist and are measurable.

## Direct pixel check — do the bars exist?

4000 decoded train images. Flat strips matching the edge-row colour (tol ≤ 6/255 per channel) were required to be symmetric (equal on opposite sides within 2%) — the signature of real padding, not a plain background.

| measurement | top+bottom | left+right |
|---|---|---|
| flat strip ≥ 5 px | 15.90% | 21.52% |
| flat strip ≥ 2% of the dimension | 8.97% | 16.12% |
| symmetric flat strip ≥ 2% | 5.55% | 8.65% |
| symmetric flat strip ≥ 2%, strip white (>235) | 4.08% | 7.10% |
| symmetric flat strip ≥ 5% | 1.50% | 4.25% |

Cross-tab against tags (base rate for "symmetric strip ≥2%": 11.97%):

| tag | n in 4k | has flat strip | has symmetric strip | vs base |
|---|---|---|---|---|
| `letterboxed` | 19 | 52.6% | 47.4% | 4.0× |
| `pillarboxed` | 1 | 100% | 100% | — |
| `white_border` | 43 | 53.5% | 41.9% | 3.5× |
| `border` | 76 | 38.2% | 31.6% | 2.6× |
| `4koma` | 25 | 56.0% | 56.0% | 4.7× |
| `white_background` | 718 | 38.9% | 20.5% | 1.7× |
| `simple_background` | 855 | 37.0% | 19.4% | 1.6× |

Bars are real (the `letterboxed` tag is enriched 4× over base rate, so tag and detector agree):
1.87% of images (746/40000) carry a `letterboxed` | `pillarboxed` | `white_border` tag, 2.12% carry some `*border` tag, and 11.97% have a symmetric flat strip ≥2% of at least one dimension — 5.55% top/bottom, 8.65% left/right, of which 4.08% / 7.10% are white. Pixels cannot tell a flat background from a bar, which is why `white_background` sits at only 1.7× base rate — the false-positive floor, not tampering. This padding is in the source artwork (scanlation borders, screenshot bars, framed illustrations); the w640 resize only rescales, adding or removing nothing.
