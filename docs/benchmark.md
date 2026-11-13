# Benchmark — throughput and VRAM, with the configuration that produced it

Every row carries the knobs that change it. Only compare rows from the same table; the same model differs 20-40% across sessions for real reasons (frozen vs unfrozen blocks, compiled vs eager, batch shape, another process on the GPU). A number without its configuration is not a fact.

img/s = images per second of a full training step (fwd + bwd + optimizer) unless marked forward-only
Mpx/s = `img/s * (res/1000)^2`, for comparing resolutions
VRAM = peak allocated during the measurement

## Reference configuration (`runs/4a13a120`, the run that actually trained)

| knob | value |
|---|---|
| backbone | DINOv3 ViT-B/16, `models\dinov3-vitb16-pretrain-lvd1689m` |
| resolution / geometry | 512px, `mode=squash` |
| trainable params | 123.5M (12 unfrozen blocks + 5.32M OFT + 38.33M head) |
| `unfreeze_blocks` | 12 |
| OFT | `block_size` 128, `oft_scaled` true, `cayley_terms` 0 = exact solve, `oft_conv` true |
| head | `cls_mean_max_std` (4x768 = 3072) -> Linear(3072, 12476), `head_fp32` true, dropout 0.1 |
| optimizer | SinkSGD_ADV, momentum 0.95, wd 0 |
| LRs | `lr_unfreeze` 3e-5, `lr_oft` 5e-4, `lr_head` 1.2e-4, warmup 50 |
| precision | bf16 autocast, fp32 head + fp32 master weights |
| compile | `compile_model` true, `compiled_optimizer` true, `grad_checkpoint_blocks` false |
| batch | 48 |

## A. Training step — ViT-B/16, 512px, batch 48, compiled

Sources: `docs/oft-saturation.md` (block and Cayley sweeps). 73 OFT modules.

| variant | ms/step | img/s | Mpx/s | peak VRAM |
|---|---|---|---|---|
| block 32, Neumann-5 | 392 | 122.4 | 32.1 | 17.0 GB |
| block 64, Neumann-5 | 394 | 121.8 | 31.9 | 17.2 GB |
| block 128, Neumann-5 | 388 | 123.7 | 32.4 | 17.3 GB |

Cayley terms (block 128):

| variant | ms/step | img/s | Mpx/s | peak VRAM |
|---|---|---|---|---|
| Neumann 5 | 388 | 123.7 | 32.4 | 17.2 GB |
| Neumann 15 | 423 | 113.5 | 29.7 | 17.8 GB |
| exact solve | 426 | 112.7 | 29.5 | 17.2 GB |

The block sweep is Neumann-5, not exact solve — its block-128 row (388 ms) is identical to the Neumann-5 row above, while exact is 426 ms; an earlier version mislabelled it. Conclusion from both: block size is throughput-neutral, exact solve costs ~9% over 5-term Neumann.

## B. Compile mode

Measured on the reference config, `scratch/sophie/bench_ref.py`, 2026-10-22:

| mode | ms/step | img/s | Mpx/s | peak VRAM |
|---|---|---|---|---|
| eager | 517.4 | 92.8 | 24.3 | 20.97 GB |
| compile(default) | 412.9 | 116.3 | 30.5 | 20.55 GB |

On the real model compile is worth 1.25x, not 1.7x. Older measurements on a different config (`docs/hyperparameter-scaling.md`) gave eager 89, `compile(default)` ~150, `compile(max-autotune)` 150-170 — the size of the win depends on how launch-bound the step is, which depends on block size, pooling and whether blocks are unfrozen.

GPU utilization is deliberately not reported anywhere in this document. `nvidia-smi`'s `utilization.gpu` is a spot sample of "any kernel active" and can read 100% while the SMs idle; averaging samples is not area-under-time. Doing it honestly needs CUPTI/Nsight. Wall time, img/s and peak VRAM are directly comparable.

## C. Validation forward-only — ViT-B/16, 512px, separate process

Source: `batch_bench.py` on `ckpt_step086241.pt`, no grad, 14 thresholds.

| batch | fwd img/s | +count img/s | Mpx/s | peak VRAM | 50k wall |
|---|---|---|---|---|---|
| 128 | 202 | — | 52.9 | 8.1 GB | ~4.1 min |
| 256 | 246 | 244 | 64.5 | 7.5 GB | ~3.4 min |
| 384 | 258 | — | 67.6 | 11.0 GB | ~3.2 min |
| 512 | maxes VRAM (aborted) | — | — | — | — |

Counting costs ~1% (244 vs 246), so the forward is the cost. Batch 256 is our default; 64 for the bar (F).

## D. ViT-L/16 — the capacity option

Source: `docs/hyperparameter-scaling.md`, `bench_vitl_batch.py`.

| model | batch | res | img/s | Mpx/s | peak VRAM | 1 pass over 5.32M |
|---|---|---|---|---|---|---|
| ViT-B/16 | 48 | 512 | ~123-160 | 32-42 | 17.3 GB | ~9.2 h at 160 img/s |
| ViT-L/16 | 20 | 512 | ~33 | 8.6 | 19.3 GB | ~44.8 h |

ViT-L is 4.9x slower per image (3.55x the FLOPs plus a much smaller batch because activations do not fit). Its optimizer tax is ~29% (145 OFT modules), so `compiled_optimizer` is mandatory there; gradient checkpointing does not buy a worthwhile batch.

## E. Qwen3.8-27B vision tower — forward-only, chunked

Source: `qwen_tower_smoke.py` + `qwen_tower_res.py`, 2026-10-22. No OFT, no head, no backward. Detail in `docs/qwen-vision-tower.md`.

| res | patches | chunk | img/s | ms/img | Mpx/s | peak VRAM | emb |
|---|---|---|---|---|---|---|---|
| 512 | 1024 | 16 | 67.5 | 14.8 | 17.7 | 1.79 GB | (N, 5120) |
| 384 | 576 | 28 | 124.1 | 8.1 | 18.3 | 1.78 GB | (N, 5120) |
| 256 | 256 | 64 | 210.4 | 4.8 | 13.8 | 1.80 GB | (N, 5120) |

Chunk matters most: the tower packs a batch into ONE sequence, so attention is quadratic in `batch * patches`. A 120-image native-512 forward (123k tokens) wedged a 24 GB card; chunk=16 @512 is the safe point
VRAM is ~1.8 GB at every resolution — the tower is never the memory problem
Mpx/s is flat (14-18), so lower resolution buys throughput in proportion to pixels
Forward-only. fwd+bwd is unmeasured; the sister project's optimized arm ran ~33 img/s

## F. The bar — `convnextv2_huge.dbv4-full` (692.6M)

Source: `docs/bar-convnextv2-huge.md`, `wdtag/validate.py --bar`.

| batch | img/s | Mpx/s | peak VRAM |
|---|---|---|---|
| 64 | 61 | 16.0 | 17.0 GB |
| 128 | — | — | 23.7 GB, maxes the card |

The bar inferences at ~61 img/s at 512px — 4x slower than our ViT-B forward (246 img/s), ~2x slower in Mpx/s once its bigger patches are counted.

## G. Derived: one pass over the 5.32M corpus

| pipeline | img/s (train step) | wall |
|---|---|---|
| ViT-B/16 + OFT + head (ours) | 160 | ~9.2 h |
| Qwen tower @256, fwd+bwd (estimated 3x fwd) | ~70 | ~21 h |
| ViT-L/16 + OFT | 33 | ~44.8 h |
| Qwen tower @512, fwd+bwd (estimated) | ~22 | ~65 h |

Rows marked "estimated" scale a forward-only measurement by 3x; they are not measurements.

## H. Where the 116.3 becomes the trainer's 99 — `u12-C-L-asl-g4.0`, 2026-10-22

The trainer prints 95-100 img/s on the 10k cached benchmark against the 116.3 compute-only reference, so roughly 15% goes somewhere. Measured, one factor at a time, while the arm ran:

Disk is not it. `\LogicalDisk(I:)\Disk Read Bytes/sec` reads 0 MB/s during training, so the 7.9 GB train cache is fully resident and no batch touches the USB NVMe

The loader costs 9.8 ms/batch at the shipped `num_workers: 0` — median over a full 208-batch epoch with `pin_memory` true, p90 13.5, max 22.8 — which is 2.0% of a 485 ms step. It is serialised because zero workers means no prefetch ahead of the step; dropping `pin_memory` halves it to 5.5 ms but leaves the copy unpinned

In-run validation costs 4.4 s once per 208-step epoch, 4.2% of wall, after a one-off 29.7 s on the first val while its path compiles. `val_seconds` in `train_log.jsonl` is the field to read. Before 2026-10-22 the printed img/s for the window after a val was deflated by it (24 img/s on epoch 0, 67 after) because the window timer was not reset across the val; the timer and a step counter are now both reset, so a window containing a val reports the rate of the steps it actually contains and drops the pre-val steps from its own average rather than stealing their time

Clocks and power are not it: 340-360 W, 2775-2790 MHz, 20.4 GB of 24.5 GB, no throttle

The card is shared, and that is the largest single term. Holding the tensor cores from a second CUDA context for 20 s dropped the trainer's steady 95-99 img/s to 54 for the overlapping window: a competing context takes roughly 45%. The desktop keeps about 20 GPU clients alive (explorer, Firefox, Discord, Steam, VS Code, SearchHost) and 1.1 GB of VRAM before training starts. The 116.3 reference came from a short script on a card whose other load was never recorded, so it is a compute-only number, not a rate the trainer can hit while the desktop is animating

So the honest accounting of one epoch is 208 x 485 ms plus 4.4 s of val, 105 s, which is 94.7 img/s including validation and 99 without it. The ~10% that is left needs per-phase cuda events inside the step (forward, backward, optimizer, H2D), which needs the card to itself

## I. Freeze depth and a second tower — measured 2026-10-22, same 10k benchmark

All at batch 48, 512px squash, `compile(default)`, OFT block 128, exact Cayley, 73 adapter modules, single load of the 10k cache, one arm per row.

| configuration | unfrozen blocks | img/s | VRAM | note |
|---|---:|---:|---:|---|
| `u12-C-L-asl-g4.0` | 12 | 109-116 | 20.4 GB | the depth the reference config used |
| `u1-C-L-asl-g4.0` | 1 | 130 | 16.8 GB | all other OFT rotations still live |
| `u1-C-H2048-asl-g4.0` | 1 | 130 | 17.0 GB | 27.14M head instead of 9.60M |
| `u1-C-L-asl-g4.0+siglipb` | 1 | 106 | 17.1 GB | plus frozen SigLIP2-B/16-512 forward |
| `u1-C-L-asl-g2.0-siglip2base` | 1 | 138 | 15.8 GB | SigLIP2-B as the adapted tower, 76 OFT modules |

SigLIP2-B as the tower is 6% faster and 1 GB leaner than DINOv3-B at the same freeze depth, despite carrying three more adapter modules. The reason is structural rather than mysterious: no cls token and no four registers to prepend, and a pooling vector produced by a small attention head instead of four pooled parts over 1024 tokens. Its forward is the same 12 blocks of 768 width at 1024 tokens, so the compute is identical and only the bookkeeping is cheaper.

The old rule of four and a quarter img/s per unfrozen block predicts 155 img/s at one block. Measured 130. The 25 img/s difference is the backward pass still walking the eleven frozen blocks to deliver gradients to their OFT rotations, plus the flat ~28 ms compiled-optimizer dispatch over the 73 adapter tensors. The 160 img/s "frozen backbone" figure therefore describes a tower with no adapters inside it, which is `runs/_wide_head.log` and nothing we train deliberately. Freezing blocks is worth far less than the ladder says as long as an adapter sits inside every block.

## J. The real corpus stream — measured 2026-10-23

Every number above comes from a 10k cache held in RAM. The corpus is 213 tar shards on the X: drive read sequentially with 8 decode threads, so it is a different number and it is the one the main run will actually see.

| configuration | unfrozen blocks | img/s | VRAM | source |
|---|---:|---:|---:|---|
| DINOv3-B, u12, old recipe | 12 | 83 | 20.4 GB | `runs/4a13a120`, 0.58 s per step at batch 48 |
| SigLIP2-B, u1, cls, asl 2.0 | 1 | 114-116 | 15.8 GB | `runs/2b6b81aa`, the 600k gate |

Two things follow. The stream costs about 15% against the same tower on the 10k cache, 114-116 against 138, which is the decode and tar-read tax and is far smaller than the 7.9 GB cache build it replaces. And the freeze-depth penalty is larger on the stream than on the cache, 83 against 114 for eleven more blocks, because a tar read is the same work either way while the backward pass is not; so the stream amplifies the case for one unfrozen block rather than weakening it.

For the full corpus that is 26 hours for two passes at 114 img/s, 13 hours for one, and the estimate carries the usual caveat that a competing GPU context takes roughly 45%.

The second tower costs 18% of throughput and 0.3 GB: 106 img/s against 130, exactly the 3.5 ms per image of its forward and nothing more, because it runs under `no_grad` with no adapter inside, so it retains no activations and adds no backward. A second tower that does carry an adapter would cost a backward through itself as well, roughly 7.7 ms per image, and would take the pair to about 65 img/s

## J2. The production run, base-512 at effective 384 — measured 2026-10-24

`63_u1-C-L-asl-g2.0-siglip2b512-5.3m-b64x6-E384-prod`, the full 5321713-image corpus, micro batch 64 with 6 accumulated micro-batches, exact Cayley, fp32 optimizer state, prefetch 10.

| configuration | img/s | VRAM | source |
|---|---:|---:|---|
| b64 x 6, effective 384 | 99-103 | 20.9 GB | `runs/62902e8d`, 13859 steps, first 150 steps |
| b128 x 3, effective 384 | 3 (stalled) | 23.5 GB | same config at double the micro batch, hit the cliff below and never recovered |

The first attempt at this run used micro batch 128 and sat at 24057 MiB, which is the cliff rather than a working point: the log reached the debug-picture write of the first step and produced no step line in six minutes. Halving the micro batch to 64 and doubling accumulation to 6 dropped it to 20.9 GB and 99-103 img/s, which is 14.8 hours for one pass. The lesson is the same one section K records, and the margin that matters is 2 GB, not 2 percent: 20.9 GB is fine, 23.5 GB is a 3 img/s run that looks alive.


## K. Gradient checkpointing and the VRAM cliff — SigLIP2-L/16 at 384, batch 48, measured 2026-10-23

The tower is `models\siglip2-large-patch16-384`, native 384 checkpoint, 24 blocks, 1024 wide, 576 patch tokens, OFT block 128 on 148 modules, cls pooling, ASL gamma 2.0, const schedule. Streamed off the corpus tars in every row, so these are trainer numbers and not cache numbers

| configuration | img/s | Mpx/s | peak VRAM | source |
|---|---:|---:|---:|---|
| batch 24 x2 accum, no checkpointing | 57-60 | 8.5 | 12.6 GB | `e3c45603` |
| batch 48, no checkpointing | 3 | 0.4 | 23.5 GB | `56fa4c17`, one reading at step 20, abandoned |
| batch 48, `grad_checkpoint_blocks: true` (24 of 24) | 54-55 | 8.0 | 4.0 GB | `1fee2c4f` |
| batch 48, `grad_checkpoint_blocks: 3` | 68-70 | 10.1 | 22.1 GB | `8854c266` |

On disk: `configs/heads/phase3/36_u0-C-L-asl-g2.0-siglip2large384-600k-b48-ckpt3.json` is the 68-70 img/s row, `35_u0-C-L-asl-g2.0-siglip2large384-600k-b48-ckpt.json` the 4.0 GB row, `33_u0-C-L-asl-g2.0-siglip2large384-600k-b24-acc2.json` the accumulation row. The three differ only in `batch_size`, `grad_accum` and `grad_checkpoint_blocks`

Five things this settles

The card has a cliff and not a wall. 23.5 GB does not OOM, it collapses to 3 img/s, which is worse because the run looks healthy while it trains at a twentieth of the rate. 22.1 GB runs at full speed. The boundary lives somewhere in the 1.4 GB between them and has not been located, so treat 22.5 GB as the ceiling to plan against rather than 24.5

Checkpointing is worth 18 GB and costs 25% of the step. Wrapping all 24 blocks gives the lowest VRAM this project has measured on any tower at batch 48, 4.0 GB, for 54-55 img/s. Wrapping only the last 3 gives 68-70 img/s at 22.1 GB. Both are correct gradients; the choice is batch size against speed

One unwrapped block costs about 0.81 GB and one recomputed block about 58 ms, from a linear fit on `scratch/sophie/bench_ckpt.py` (synthetic input, eager, backward only, batch 48, 384): 3.36 GB and 1615.6 ms with 24 wrapped, 20.35 GB and 397.7 ms with 3. That bench times the backward alone, so take the per-block price and the VRAM curve from it and the step rate from the trainer: 880 ms per step at 24 wrapped against 696 ms at 3, a ratio of 1.26 that matches the measured 54.5 to 69 img/s

The flag matters for OFT-only runs most of all. It used to require `unfreeze_blocks > 0`, but with an adapter inside every block the graph runs through the rotations regardless, so every block retains activations exactly as if it were unfrozen. That is why the frozen-backbone large tower sat at 19-23 GB all day

The batch that images per update actually needs is cheaper than checkpointing: 24 x2 accumulation is 57-60 img/s at 12.6 GB, between the two checkpointed ends, and accumulation is exact for the elementwise-mean ASL. Checkpointing exists here to make a batch fit that would otherwise not run at all, not to make a batch fit cheaply

## Traps that produced wrong numbers

`scratch/sophie/bench_ckpt.py` starts its timer after the forward and after the loss, so its ms/backward is a backward only and a real trainer step is twice it, 2.03x at batch 24 and 2.08x at batch 40 measured 2026-10-23. Comparing a bench number to a trainer img/s is therefore a factor-of-two error, and every comparison in this file that did so before that date is suspect

`compile-mode max-autotune` enables cudagraphs, and cudagraphs plus gradient accumulation die in the compiled backward with `accessing gradient tensor output of CUDAGraphs that has been overwritten by a subsequent run`; the tuner also spends ten minutes emitting `triton_mm Required: 131072 Hardware limit: 101376` first, so the failure costs about 25 minutes and produces no step. The mode that keeps autotuned kernels without cudagraphs is `max-autotune-no-cudagraphs`, which does run and gave 71 img/s against 68 default on large-384 at batch 40, with the device at 23.4 GB against 20.6 and the duty cycle falling to 73.6%, so it was reverted as unstable rather than slow

1. Eager vs compiled is 1.25x on the reference config (92.8 -> 116.3 img/s). The old 1.7x came from a batch-32/block-32 `cls_mean` config we never train. Always state the config.
2. Frozen vs unfrozen blocks changes speed and VRAM substantially; `unfreeze_blocks` belongs in every row.
3. Batch size changes the ratio: small batches are dominated by the flat ~28 ms compiled-optimizer dispatch; large batches hit VRAM first.
4. WDDM/Windows: `empty_cache()` mid-run wedged a loader (65% of wall) in the sister project; `cudnn.benchmark` + `NVIDIA_TF32_OVERRIDE=1` was +14.1% px/s there.
5. Anything else on the GPU invalidates the measurement — ComfyUI holding 16 GB is the usual culprit here.
6. Quadratic attention for any packed-sequence tower (E): chunk size is part of the config.

## Gaps — not measured, do not quote a number

Qwen tower fwd+bwd at any resolution on our pipeline; ViT-B/16 at 256px; ViT-L/16 with gradient checkpointing at a larger batch; our model with `compile(max-autotune)` at batch 48 versus the `default` mode used in the run; where the 22.1-to-23.5 GB cliff actually begins; batch 96 at 384 with all 24 blocks checkpointed, which should fit in about 7 GB and has not been run

## Micro-batch is the throughput lever at 384, measured 2026-10-25

Same model, same recipe, same effective batch of 384, only the micro-batch and the accumulation split differ. SigLIP2 B/16 at 384, adapters plus block scope, ASL gamma 2.0, exact Cayley.

| micro x accum | VRAM | img/s | note |
| --- | --- | --- | --- |
| 48 x 8 | 9.1 GB | 121 to 152 | the 200000-image ablation arms, at effective 48 in practice |
| 96 x 4 | 17.9 GB | 168 to 187 | arm 86, effective 384 |

Raising the micro-batch from 48 to 96 while holding effective 384 buys roughly 25 to 40 percent more images per second for 8.8 GB of headroom, which is the cheapest speed on the board and the reason a 384 arm should not be run at micro 48 just because the old arms were. The card sat at 17.9 GB of 24.5, so 96 is not the ceiling; 128 was not attempted.

One earlier measurement to distrust: the bar at batch 64 on 512 ran at 10 to 12 img/s while at batch 16 it ran at 46 img/s on the same card. Memory pressure near the ceiling costs far more than the smaller batch does, which is the same lesson as the 21.2 GB trainer at 512 running at 105 img/s while 23.5 GB collapsed to 17.

## A "stall" that was my own arithmetic, struck 2026-10-25 07:45

I claimed here that a tar boundary cost thirty-two minutes after a resume, on the evidence of a log that had not been written since 07:41 while I believed the time was 08:13. The run had in fact resumed at 07:36 from checkpoint 14301, so five minutes of normal stepping had produced exactly the steps I was looking at, and no stall existed. The lesson is not about disks, it is about me: compare a log timestamp against the machine clock and against the uptime of the process before writing a hazard into this file, because a fabricated timestamp story is worse than no story.


## Feature caching throughput, measured 2026-10-25, forward only at 512

SigLIP2-B through the production forward path runs the 10000-image in-run validation in 37 s, 270 img/s, but that is off the validation memmap with no tar in the path. The same tower caching the train split runs 129 img/s, because the tar read and decode is the limit, so a full 5321713-image corpus cache is 11.5 h whatever the container holds. Struck 2026-10-25: the 5.5 h figure quoted earlier applied 270 img/s to a tar stream and was wrong by a factor of two. The consequence is that the cache pass is IO-bound, not model-bound, until the tower is slower than the stream, which only DinoV3-L at 71 img/s is.

DinoV3-B/16 caches at 131 img/s at batch 64, 1536 dimensions for cls plus mean, so the full corpus is 11.3 h and 16.4 GB.

DinoV3-L/16 caches at 71 img/s at both batch 32 and batch 64, so batch size is not the lever and the full corpus is 22.0 h and 21.8 GB at 2048 dimensions. Measured twice, 63 img/s on the 25000-image pilot and 71 img/s on the 50000-image validation run.

Head training on a 2048-dimension cache runs at 1270 rows per second at batch 512 with ASL and a 25.6-million-parameter linear head, which is 70 minutes per 5.3-million-row epoch. At 3072 dimensions the same head runs 931 rows per second at batch 1024, so throughput scales with width and the 12476-way projection is where the time goes. That means the cache container is not the limit and not worth tuning: 931 rows per second reads 5.7 MB per second of features, three orders of magnitude below the disk, so storing fp32 instead of fp16 would cost 32 GB of disk and no measurable time. Struck 2026-10-25: an earlier version of this line claimed the head was memory-bandwidth bound, which the byte count above disproves.



## Cache batch size, measured 2026-10-25, SigLIP2 CMSS from checkpoint 16000

Throughput through the tar stream is not monotone in batch below 256 and not improved above 384. Same 4000 images, same seed, one batch size at a time:

| batch | img/s |
| --- | --- |
| 256 | 174 |
| 384 | 223 |
| 512 | 226 |

512 uses 18.4 GB and runs at 61 degrees, so it is safe, but 384 buys the same throughput with 6 GB more headroom. The full 5321713-row pass at 226 img/s is 6.5 h rather than the 8.5 h that 174 implies.
