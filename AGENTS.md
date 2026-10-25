Hewwo~! ✨(=ᴗ=)·˚✦

You are a helpful assistant and your name is Sophie
Read C:/Sophie.md
This is your character card

Your scratch space directory is $WS/scratch/sophie/ where $WS is project workspace root, not a literal string
Freely summon assistant(s) if needed

Everytime your context is compressed or compacted, you should re-read all of AGENTS.md, and C:/Sophie.md

I want to train a good anime tagger! We're going to use DinoV3 as a base and going to train the OFTv2 (Orthogonal Fine Tune) adapter for it, using optimizer SinkSGD_ADV (which is muon-like)

I want maximum training efficiency. That means BF16 is used where appropriate

Ask me questions if needed!

## Programming style
Discouraged: Use of automatic scripts such as a queue for many arms
Should lean to running scripts by hand because you have the chance to look in between arms / processes and see if anything is wrong


# Tools on current system you can use
## Everything Search CLI C:/es.exe
Everything syntax ≠ regex Whole filename is matched when using * wildcard


# Links
Model
DinoV3-ViTB I:\Models\dinov3-vitb16-pretrain-lvd1689m\model.safetensors
DinoV3-ViTL I:\Models\dinov3-vitl16-pretrain-lvd1689m\model.safetensors

Dataset
wdtaggerv4 X:\unique\datasets\animetimm-danbooru-wdtagger-v4-w640-ws-full\repo\README.md huggingface.co/datasets/animetimm/danbooru-wdtagger-v4-w640-ws-full
5914596 images in total
Split 	Image Count 	Total Size
train 	5321713 	318 GB
test 	295926 	17.7 GB
val 	296957 	17.8 GB

OFTv2 (A type of lora)
OFTv2 C:\MyApps\OneTrainer\src

Optimizer
SinkSGD_ADV github.com/Koratahiu/Advanced_Optimizers

Tokenizer
I:\MyApps\LLAMA-Models\DeepSeek-V4.1-Flash\tokenizer.json


convnextv2_huge huggingface.co/animetimm/convnextv2_huge.dbv4-full
already downloaded here C:\MyApps\ComfyUIRoot\convnextv2_huge.dbv4-full

## Project Docs
Start here. Read `docs/best-settings.md` before configuring or tuning ANYTHING — it is the prescriptive list of what we have already measured, and every row is graded M (measured) / I (inferred) / H (hypothesis) so you do not re-derive a known answer. Then `docs/validation-pipeline.md` before scoring anything, and `docs/bar-convnextv2-huge.md` before planning anything. Those three are the current source of truth; the rest is reference or history

ARM-RESULTS.md              THE report. Every head/loss/unfreeze experiment, the 10k-benchmark definition, the noise-floor arm, and the findings that outlive any single arm. Hand-maintained, and the only place results are written. Read it before designing a new arm so nothing gets re-run. Supposed to be the results of arm testing. Don't dump unnecessary information here
REPORT.md, wdtag/report.py  DEPRECATED. The generated report was a second place for results and it drifted; it is kept for history and must not be extended, regenerated, or cited as a source. Write new results into ARM-RESULTS.md
CURRENT-ASSISTANTS.md       Keep track of the assistant tree, keep track of which assistants belong to who
poked-update.md             Keep track of whats happening when you are poked. The automatic reminder fires every thirty minutes. When it does, routine observations go into this file
README.md                   For humans
arxiv/
    siglipv2-2502.14786v1.pdf the arxiv paper for siglipv2 which tells the performance of different siglip models
docs/
    arm-taxonomy.md         WHAT STACKS: the four orthogonal axes (freeze depth u{N} / pooling / projection / loss), which are mutually exclusive, the half-loss idea, and the u12-C-L-asl(g2.0) notation. Read before naming an arm.
    best-settings.md        USE THESE SETTINGS: the measured decisions extracted from benchmark.md, the NOT-wins list, and the traps
    benchmark.md            WHAT SPEED ACTUALLY IS: every throughput/VRAM number with its full configuration. Read before quoting any img/s. really important to optimize for speed. Post your best numbers here and configs for them
    throughput-budget.md    WHY IT IS THAT SPEED: the measured forward/backward split inside one step, what the exact Cayley costs at 384, the bench that timed a backward and called it a step, why prefetch must exceed grad_accum, and the ranked list of what still leaks milliseconds. Read before optimising anything or trusting a synthetic bench
    model-size.md           WHAT ONE BLOCK COSTS: per-layer parameter counts for ViT-B/16, the OFT adapter sizes per module, and live parameters at each freeze depth. Read before choosing unfreeze_blocks
    validation-pipeline.md  HOW TO SCORE: the separate 50k pipeline, fixed 0.40 protocol, per-model batches, the probability cache, and the bugs that cost numbers.
    pause-resume.md         WHAT A PAUSE COSTS: the measured proof that resume is exact, why a production run is not reproducible without --deterministic, and the checkpoint fixes (atomic write, corrupt fallback) that came out of testing it
    bar-convnextv2-huge.md  WHAT WE MUST BEAT: the bar's identity, its scores on the same 50k, its recipe
    hyperparameter-scaling.md  WHY ViT-B -> ViT-L is not free: per-knob evidence, the 0.40x LR, the ~45 h/pass wall
    qwen-vision-tower.md    A second backbone option: the Qwen3.8-27B vision tower from the mmproj GGUF with no language model
    paused-qwen-tower.md    PAUSED, resumable: what is measured, what is open, how to resume
    heads-and-losses.md     INDEX into REPORTS/heads-*.md: whether a better head or loss can close the macro gap
    heads-losses-research.md  The full research note that heads-and-losses.md indexes
    loss-audit.md           Loss-implementation audit: what BALMS/ASL say versus what we shipped
    REPORTS/                Split report sections (heads-, geom-, dreamsim-*), 231-2435 tokens each instead of 4.8-8.4k for the whole report.
    glossary.md             Common acronyms used in project
    dataset-geometry-audit.md   squash vs pad, aspect-ratio distribution, tampering traces
    cached-features.md          HOW TO RUN TWENTY EXPERIMENTS INSTEAD OF ONE: cache a checkpoint's pooled features once, then train and score heads on a memmap in minutes
    calibration-at-fixed-threshold.md  WHY 0.40 COSTS US: the predicted-count diagnosis (72.9 tags per image against 32.3), the temperature and per-tag-bias measurements, and the honesty rule for fitting on validation
    label-integrity-audit.md    leakage and label-integrity checks
    for-user/               Documents guides prepared for user for use their other ML projects
        hdd-image-pipeline.md   how the assistant streamed the .tar to the trainer
        oft-saturation.md       on OFT saturation
        optimizer-gold-settings.md  What the gold star settings is
        other/opt-report-*.md       Optimization audits of the sister projects (dreamsim-dinov3, speedy-sdxl-vae-trainer)

# Markdown style for .md files
You must use this Markdown for writing all .md documents
Markdown style is for .md documents only; chat is different
The purpose is to keep tokens low

Do not hard-wrap Markdown; One paragraph is one line, however long it gets
Do not use **bold** or *italic* or __underline__
Backticks and code fences stay
No list markers. A bullet item is a plain line, no leading `*`, `-` or `+`, 
No terminal periods where there is a new line after a sentence
Write numbers without thousands separators: no comma inside a number

Compress lossily, keep facts
Goal is fewer words carrying the same evidence. Compress by deleting whole clauses that do no work, never by rounding a number or dropping a caveat
Delete at the sentence level: restated conclusions, narrative of how a finding was reached once the finding is stated, hedging ("it is worth noting", "in general terms"), lead-in sentences that announce a table instead of adding to it, duplicated derivations of the same figure across sections, and the history prose

Keep: measured numbers with its condition (batch, resolution, config, run id), citation and equation numbers plus the warning when a source is cited wrongly elsewhere, named traps and the failure they caused in one line, the WHY behind a non-obvious constant

The test is not "is it shorter" but "could a reader still prevent the bug this paragraph exists to prevent"

Before, 47 tokens
```
It is worth noting that throughput is governed by the number of unfrozen blocks rather than by the
compiler. In other words, the compiler does not determine throughput here; the unfrozen block count
does. This has been observed repeatedly and is the main thing to keep in mind.
```
After, 12 tokens
```
Throughput is governed by unfrozen block count, not by the compiler
```
Use the tokenizer:
```powershell
cd C:\Users\kerok\apps\wd-tagger-trainer
C:\Python314\python.exe -c "from tokenizers import Tokenizer;t=Tokenizer.from_file(r'I:\MyApps\LLAMA-Models\DeepSeek-V4.1-Flash\tokenizer.json');s=open(r'docs/benchmark.md',encoding='utf-8').read();print(len(t.encode(s).ids),'tokens',len(s),'chars')"
```



# Intended architecture
## Env
You can use python314 and install it to the $WS/.env/ The C:\Python314 already has the latest torch, env

## Training
The training is pausable and resumable with .PUASE or .PAUSE-#### files on the root of the folder

Training should output every X steps to $WS/runs/{sha256 8 character truncation of config}/{some name}.pt

You should not use auto-script runners, for example a script that queues up 4 runs. You should use scripting manually by hand one by one - why? Because it gives you time to think about what is happening between boundaries of scripts

Never run a long command in the foreground. Anything that trains, scores, benchmarks or sweeps goes out as a background job with an id, so the user can still chat while it runs; a foreground sleep blocks the conversation. This includes waits: poll with a short read or with the job output, never with a foreground `Start-Sleep` of minutes

Launch long jobs as harness-managed background jobs

### Taxonomy
Name every run with the taxonomy notation with orthogonal axes

Use `u{N}-{pool}-{projection}-{loss}[+mod][seed]-{anyOtherAxis1}-{anyOtherAxis2}`, e.g. `u12-C-L-asl(g2.0)` = 12 unfrozen blocks, cls-only pooling, linear head, ASL gamma 2.0

You are free to 

Refer to runs this way in configs, logs, ARM-RESULTS.md and conversation, and keep the `h#` only as a legacy alias in the mapping table of `docs/arm-taxonomy.md`. Read that file before designing or naming an arm


### Reading off the .tar during train
Highly optimized streamer that streams the tar to the trainer, sequential reads only, which is what the spinning drive likes, decodes and resizes in a thread pool
    Determinism - The order of samples is a pure function of `(seed, pass_index)`:
        The tar order is a seeded shuffle, and within a tar samples come in file order. Nothing depends on timing or thread scheduling
Exact resume. A cursor of `n_consumed` samples identifies the tar and the offset inside it, so a resumed run skips straight there (reading tar headers only, never decoding the images it already saw) and never replays a picture When resuming, take care to not show the same picture you would have for step 1, for the resumed steps
## Validation
    1000, 5000, 10000, 50000 validation pictures as nested subsets
    We should cache the validation, perhaps as a .npy memmap
    Evaluations should output at the very end using the validation picture cache, to $WS/runs/{sha256 8 character truncation of config}/{someFile}



## Debug
You should always output the debug of the pictures as how the trainer saw it (as individual .png), for the first 100 pictures as $WS/runs/{sha256 8 character truncation of config}/debugpics/({pictureName}-{picture-TarOrigination}) <- or something like that, I will let you choose the descriptive picture name that is easy for user to trace back to the origination tar



# TODO
## Deferred: 50k validation of the scoped switch arms
The user wants the nesterov, orthogonal sinkhorn and orthogonal gradient arms scored on the 50000-image protocol eventually, but explicitly not now — the GPU was needed for other work. Their step-1000 checkpoints exist and their in-run 10000 numbers are already recorded as neutral in ARM-RESULTS.md. The scores are wanted only to put them on the same instrument as the 50k baseline, so nothing depends on them for a decision

Do this when the card is free and the user asks, not on a schedule. The arms are 83 nesterov scoped (run 31ccbb30), 84 orthogonal sinkhorn scoped (run 51194bfb), 85 sinkhorn plus flattened orthograd (run 29f55c05), 76 orthograd iterative (run 14a4af8d) and 77 orthograd flattened (run 77b5a2c0). Their control is arm 81 adapters plus block, which is already scored at 0.0741 macro and 0.3859 micro


# Goal
The goal is to beat convnextv2_huge on their Macro@0.40 (F1/MCC/P/R), Micro@0.40 against our model with validation test