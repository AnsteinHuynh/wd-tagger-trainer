# scratch/sophie

Scratch space for the DINOv3-B/16 + OFTv2 tagger: six one-purpose tools, the data they read, and the raw logs behind `ARM-RESULTS.md`. Nothing here launches training — arms go one at a time, by hand, as harness background jobs.

## Tools

`test_losses.py` checks the shipped losses in `wdtag/losses.py` against reference formulas written out in plain torch, and prints ALL OK. Self-contained on purpose: it used to import a second copy of the losses, so it could agree with the copy instead of the shipped code

`tag_bins.py` per-tag F1 binned by positive count, BCE control `470e6fa9` against ASL `7ee12110` on the 50k cache at 0.40, read from cached probability matrices, no GPU

`bin_mass.py` arithmetic answer to whether the 1-9 bin matters to macro, given the tag counts and the F1s `tag_bins.py` prints

`prob_threshold_sweep.py` macro and micro across global thresholds from the cached probabilities, testing whether ASL's win is ranking quality or an operating-point shift

`val_curves.py` one val point per epoch for every run in `runs/`, train loss included, for spotting overfitting

`repair_ckpt.py` repairs checkpoints written before the frozen-OFT save bug, which dropped rotations an earlier stage had trained

## Data, do not regenerate casually

`test_aspect.csv`, `train_sample_aspect.csv`, `train_broad_aspect.csv` width, height and aspect for the 1000 test images, the 10000 training images the trainer sees, and a 40000-image broad sample; roughly 55000 tar reads to produce, and row order matches the probability dumps

`records_*.pkl`, `phash_*.pkl`, `pixels_*.pkl`, `neardup_*.pkl`, `desc_*.npy`, `pairs_ge80.json`, `similarity.json`, `calibration.json`, `coverage.json`, `train_counts_160k.npy` label-integrity and geometry audit inputs: per-image records, perceptual hashes, pixel statistics, near-duplicate pairs, descriptor matrices

`asl.txt`, `balms.txt`, `cb.txt`, `dbl.txt`, `decouple.txt`, `logitadj.txt`, `ml_decoder.txt` text extracts of the loss and decoder papers, the basis of `docs/loss-audit.md`

## Records

`sweep_*.log` plus their `.val50k` and `.valbest` companions, `sweep_driver.log`, `phase3_wait.log` console output of the retired sweep driver and the early arms. The driver that wrote them is deleted; the logs stay as the raw record behind the numbers in `ARM-RESULTS.md`

`smoke/` three configs from the pre-notation smoke test that `ARM-RESULTS.md` cites

`NIGHT_PLAN.md` the plan the phase-3 ladder came from, kept as history

`_num_before.json`, `summary.json`, `probe_validation.json`, `verify.log`, `profile_default.log`, `bench_autotune.log` one-off outputs quoted in the docs

## archive/

60 one-off diagnostics: benchmarks, probes, the geometry-audit pipeline, rendered dumps. Kept for provenance and not expected to run again. Paths and command examples inside them, and in `docs/dataset-geometry-audit.md`, refer to their old top-level names, so prepend `archive/` to re-run one
