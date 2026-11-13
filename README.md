# wd-tagger-trainer

A parameter-efficiency and training-methods study for anime-style image taggers (WD-tagger style, Danbooru-style tag vocabulary). It asks how close a released full-size tagger can be approached by training only a small fraction of the parameters: modern vision backbones fine-tuned with OFTv2 (Orthogonal Fine-Tuning) adapters, one unfrozen block, a frozen cached second tower, and SinkSGD_ADV optimizer experiments, scored against the released `convnextv2_huge` tagger as the bar. Headline (see `RESULTS.md`): one epoch training 31.1M parameters, 4.5% of the bar's count, reaches 74.6% of its macro F1. This is a research repo, not a product: the code is tuned to one machine, one dataset, and one GPU.

Author: Anstein Huynh.

## License

This repository is distributed under **AGPL-3.0** (see `LICENSE`). To be explicit about why: the only externally derived code is `wdtag/oft.py`, the OFTv2 adapter implementation, ported from OneTrainer (AGPL-3.0); a derivative of an AGPL work makes the combined distribution AGPL. Everything else here is original work by the author. A future release plans to replace the adapter with a clean-room implementation from the OFT/BOFT papers and relicense under MIT. The trained checkpoints on HuggingFace additionally build on DINOv3 (Meta DINOv3 License) and SigLIP2 (Apache-2.0); their terms apply to the weights.

## What it is

- PyTorch multi-label training over a WD tagger label space (about 12.5k labels: general, character, rating).
- Backbones: DINOv3 ViT-B/16 and SigLIP2-B/16, adapted with OFTv2 orthogonal adapters, one unfrozen transformer block, ASL loss, BF16, torch.compile.
- A two-tower ensemble mode: a trained tower plus a frozen second tower whose cached features are concatenated before the head.
- Streaming data loading straight from source tar archives, pausable and resumable runs, and a strict scoring protocol (fixed 0.40 threshold, no per-tag tuning) against the `convnextv2_huge` baseline.

## Training pipeline

1. Labels and manifests: `wdtag.labels` and `wdtag.manifest` build label spaces and train/valid/test manifests from the dataset.
2. Streaming input: `wdtag.stream` reads images directly out of tar archives on disk with a seeded per-epoch permutation, so resumed runs never repeat seen images.
3. Model: `wdtag.model` wraps DINOv3 or SigLIP2 with OFT adapters (block size 128, exact Cayley), optional frozen SigLIP2 second tower, and a multi-label head in fp32.
4. Train: `wdtag.train` is the main loop (SinkSGD-style optimizer options, gradient checkpointing, `.PAUSE` for clean shutdown, checkpoint every N steps). `wdtag.train_cached` trains only the head on cached frozen features; `wdtag.cache_features` / `wdtag.featcache` / `wdtag.shard_cache` build and manage those feature caches.
5. Score: `wdtag.validate` scores a run or the baseline (`--bar`) at a fixed 0.40 threshold on nested validation subsets (1k to 50k). `wdtag.evaluate`, `wdtag.score`, `wdtag.infer` cover test-split eval, rescoring from probability caches, and running a model on new images.

`RESULTS.md` holds the public summary of the best run and the scoring protocol. `docs/` holds measured notes on best settings, caching, and the benchmark definition.

## Results so far

Honest summary: this is a parameter-efficiency study, not a leaderboard entry. Single towers do not beat the `convnextv2_huge` bar, and neither does the best two-tower ensemble, but the ensemble closes to 74.6% of the bar's macro F1 (0.4162 vs 0.5579 on the 50k instrument) while training 31.1M parameters to the bar's 692.6M, in a single epoch. The best run's numbers are reproducible from the provenance JSONs kept in `runs/8f6de35e/` (config, validation, test eval).

- Baseline bar, `convnextv2_huge.dbv4-full` (1000 test images, threshold 0.5): macro F1 0.194, micro F1 0.680. On the 50k validation instrument at threshold 0.40 (as recorded in the author's experiment log): macro F1 0.5579, micro F1 0.6886.
- Single-tower DINOv3-B/16 + OFTv2, long run (`runs/4a13a120`), 50k validation at 0.40: macro F1 0.2815, micro F1 0.6133, so about 50% of the bar's macro and 89% of its micro.
- Best result, two-tower ensemble (`runs/8f6de35e`): DINOv3-B/16 (one unfrozen block + OFT in all blocks) trained for one epoch against a frozen, already-trained SigLIP2 tower's cached features, features concatenated before the head. Its `validation.json`, 50k images at 0.40: macro F1 0.4162, macro MCC 0.4364, micro F1 0.5154. It beats the SigLIP2 solo tower's two-epoch result on the same instrument (macro F1 0.4066 vs 0.4165 at best-val step 13500, per `RESULTS.md`), and the training curve was still rising when the epoch ended.
- So the ensemble closes roughly three quarters of the single-tower macro gap to the bar but does not pass it. Micro precision of earlier single towers already matched the bar; the remaining gap is macro, which is where rare tags live.

- Held-out test split (same protocol as the baseline's eval, threshold 0.5, checkpoint 13859): macro F1 0.1694, micro F1 0.6018 against the bar's 0.194 / 0.680, so the ensemble also trails it there. See `RESULTS.md`. Remaining TODO: more than one epoch; the training curve was still rising when the epoch ended.

## What are we even optimizing?

The author's view, stated plainly: at some point you have to ask whether you are training the model to be good, or training it to memorize what a human happened to pick. Danbooru tagging is messy, and humans are far more likely to leave a tag out than to add a wrong one. So past enough epochs the model is not just learning word-to-feature mappings, it is learning to recognize and copy the deficits of human annotators: their omissions, their blind spots, their inconsistency. Macro F1 against human labels measures agreement with noisy annotators, not correctness. Some of the gap that looks like "rare tags the model cannot learn" is plausibly missing human labels, not model failure. This gives the one-epoch efficiency result a second reading: training longer does not just overfit the images, it starts fitting the label noise.

## Usage

Requires a local dataset in WD-tagger format (tar archives of images plus tag metadata), local DINOv3 and SigLIP2 checkpoints (paths are configured per run), and a CUDA GPU with plenty of VRAM (a 512px batch-48 setup used about 20 GB).

```powershell
# train from a config (run dirs are named by the sha8 of the config)
python -m wdtag.train --config configs\dino\abl-dinov3l-plus-siglip2b-u1-C-asl-g2.0-512-b64x6-100k.json --select-metric macro

# pause cleanly: create .PAUSE in the run dir or repo root, the trainer checkpoints and exits

# score a run on the validation subsets
python -m wdtag.validate --run runs\8f6de35e --batch-size 256

# score the baseline bar on the same images
python -m wdtag.validate --bar --sizes 1000,5000,10000,50000

# eval a finished run on the held-out test split (writes runs\<sha8>\eval_test.json)
python -m wdtag.evaluate --run runs\8f6de35e
```

Backbone and cache paths in configs point at local drives; edit them for your machine.

## Repo layout

- `wdtag/` - the package: data, model, train loop, validation, scoring, inference, feature caching.
- `configs/` - run configs, including the DINOv3 + SigLIP2 ensemble ablations under `configs/dino/`.
- `runs/` - provenance for the best run only (`8f6de35e`: config.json, validation.json). Checkpoints, probability dumps, debug sample images, and all other runs live on disk but are git-ignored.
- `docs/` - measured notes: best settings, benchmark definition, caching, label-integrity audit, plus a few HTML explainers.
- `RESULTS.md` - the public results summary: best run, protocol, comparison points.
- `scratch/`, `runs/**/debugpics/`, `*.pt`, `*.npy` - local only, never committed.

## License / data note

The dataset, pretrained checkpoints, and the baseline tagger are third-party and not distributed here; only training code, configs, and small eval JSONs are in this repo.

Trained checkpoints (ensemble ckpt 13859 + frozen SigLIP2 tower 27718): https://huggingface.co/ansteinhuynh/wdtag-ensemble-dinov3-siglip2
