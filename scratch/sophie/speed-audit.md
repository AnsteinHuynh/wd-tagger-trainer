# Speed audit: where the wall clock goes inside one optimizer step

Applies to the streamed large-384 configs, which is where the throughput question lives: `41_u0-C-L-asl-g2.0-siglip2large384-5.3m-b40x20` (d8eb6994 / 2066dc7e, 800 images per step, 65-69 img/s, 588 ms per micro-batch) and `33_u0-...-b24-acc2` (e3c45603, 48 images per step, 57-60 img/s, 410 ms per micro-batch). Every line reference is train.py unless stated otherwise

## What the trainer actually does per optimizer step

The stream loop gathers the whole group before it touches the card: `group_x.append(x)` and `if len(group_x) < h.accum: continue` (848-853), so at accum 20 the consumer holds 20 ring slots, then calls `h.step` once. One micro-batch is one preallocated pinned uint8 ring slot of `[B,3,SIZE,SIZE]` plus one float32 label block of `[B,12476]` (stream.py:230-233)

`StepHelper.step` (283) is: pause probe over 2 directories and 6 filenames (289, 12 `os.path.exists` calls), debug dump while under 100 pictures (300-312, inert after step 3), `step_fn` on the whole group (314), log block (317-327), checkpoint (329), optional validation (339)

`step_once` (763) is: lr per group (772-775), `zero_grad(set_to_none=True)` (779), then `accum` iterations of: pinned x copy and pageable y copy on the current stream (786-787), one call into the compiled model (788), `logits.float()` then ASL in fp32 over `[B,12476]` (789-798), backward (799), detached accumulate (800); then one `clip_grad_norm_` over 152 tensors (803-805) and one `opt.step()` (806)

The compiled object is `torch.compile(model)` (530-533) and it is invoked once per micro-batch, 20 times per step at b40. `[compile] ready in 1.4s` is not a compile: `torch.compile` is lazy, so the first micro-batch pays the graph build and the first validation pays a second one

The OFT path is the part the step description usually gets wrong. `apply_oft_to_backbone` wraps every `nn.Linear` and, because `oft_conv` is true, the patch-embedding `nn.Conv2d` (model.py:53-73), which is 148 modules at large-384 and 76 at base-384/512. Exactly one of them is a Conv2d and therefore exactly one calls `rotate_matrix` (oft.py:113 via 195); the other 147 call `rotate_vector` (oft.py:97 via 153). Both rebuild the rotation from the parameter on every call: `_cayley_batch` (oft.py:70-95) builds the skew in fp32 (74) and then, because the 384 configs set `cayley_terms: 0`, takes the exact branch `torch.linalg.solve(eye - q_skew, eye + q_skew)` (82) instead of the five-term Neumann series. That rebuild is per module, per forward, and again inside the backward, and it is independent of the image count

The streamer's ring is `prefetch + hold + 1` slots with `hold = accum` (stream.py:141, 229) and the producer-to-consumer queue is `queue.Queue(maxsize=prefetch)` (stream.py:275). x slots are pinned (230); y slots are plain `np.zeros` (233), so the `non_blocking=True` at 787 cannot be asynchronous for labels

## The budget

Measured anchors used below. Full-step times are the trainer's own img/s lines divided into `batch x accum`, so they include data, optimizer and logging

| config | images/step | ms/step | ms/image | Mpx/s |
|---|---:|---:|---:|---:|
| large-384 b40 x20 no ckpt | 800 | 11765 | 14.7 | 10.0 |
| large-384 b24 x2 no ckpt | 48 | 827 | 17.2 | 8.5 |
| large-384 b48 ckpt3 | 48 | 696 | 14.5 | 10.2 |
| large-384 b48 ckpt24 | 48 | 881 | 18.4 | 8.0 |
| large-384 b336 ckpt24 | 336 | 4541 | 13.5 | 10.9 |
| large-512 b224 x6 ckpt12 | 1344 | 12800 | 9.5 | 27.5 |
| base-384 b48 no ckpt | 48 | 298 | 6.2 | 23.8 |
| base-384 b48 ckpt12 | 48 | 338 | 7.0 | 20.9 |
| base-512 b48 u1 | 48 | 348 | 7.3 | 36.2 |

The only backward-only measurement in the project is bench_ckpt.py, and it starts its timer after the forward and after the loss (56-66), so it never times the forward. At large-384 it gives 202 ms at b24, 252 at b32, 282 at b40, all with checkpointing off (the 18.52 GB at b40 says it: all-24 checkpointing is a 4 GB configuration). A forward+bwd split of 1:2 is the usual transformer ratio, so a full micro-batch is 1.5x the bench number; below I use that and show what it leaves unaccounted

Batch 40, 20 micro-batches, per micro-batch

| line | ms | source |
|---|---:|---|
| backward, measured | 282 | bench_ckpt b40 |
| forward, inferred at 0.5x backward | 141 | ratio, no measurement exists |
| exact Cayley rebuild, forward share | 68 | derived below |
| `logits.float()` + ASL forward + uint8 prep + H2D | 9 | elementwise traffic, 35 MB per op |
| subtotal per micro-batch | 500 | |
| 20 micro-batches | 10000 | |
| clip 152 tensors + 148-tensor compiled optimizer, per step | 45 | docs/benchmark.md 149 scaled |
| burst gather, 16 batches at about 66 ms each | 1056 | stream.py:275, fitted |
| total | 11101 | measured 11765, slack 664 (6%) |

Batch 24, 2 micro-batches, per micro-batch

| line | ms | source |
|---|---:|---|
| backward, measured | 202 | bench_ckpt b24 |
| forward, inferred at 0.5x backward | 101 | ratio |
| exact Cayley rebuild, forward share | 68 | derived below |
| loss + prep + H2D | 7 | |
| subtotal per micro-batch | 378 | |
| 2 micro-batches | 756 | |
| per-step clip + optimizer, undiluted by accumulation | 45 | |
| total | 801 | measured 827, slack 26 (3%) |

The Cayley figure, shown because it is the least obvious line. docs/benchmark.md table A measures, at base-512 batch 48 with 73 modules, Neumann-5 at 388 ms/step and exact solve at 426 ms/step, so the exact branch costs 38 ms more over 73 modules, or 0.52 ms per module per step for forward plus backward. Scaling by module count 148/73 and by mean rotation rank from 8.6 to 10.6 gives 0.52 x 148 x 1.23 = 95 ms per micro-batch as the cost of preferring the solve over Neumann-5. The same table's 5-term Neumann is 5 x (35/10) = 17.5 ms at 73 modules, which scales to 44 ms at 148, so the exact Cayley in total is about 95 + 44 = 139 ms per micro-batch, of which the forward share is roughly half, 68 ms. That is 12% of a b40 micro-batch and 17% of a b24 one, and it is a per-module constant, not a per-image one

Two parameters in the tables are inferred rather than measured: the forward/backward ratio, and the 66 ms per batch decode rate inside the gather term. The first has never been measured anywhere in this project, the second is fitted to close the b40 step. The b24 table contains no gather term because accum 2 is smaller than prefetch 4

## Ranked suspects

### 1. The exact Cayley is rebuilt 148 times per forward and again per backward, at 384

oft.py:70-95, decision at oft.py:76-82, config `cayley_terms: 0` in d8eb6994, 8854c266, 1fee2c4f and e3c45603. Predicted cost 139 ms per micro-batch total, of which about 68 ms is the forward that the bench never times, and predicted saving of 95 ms per micro-batch from setting `cayley_terms: 5`, which is 1900 ms per step at accum 20 and 190 ms per step at accum 2. Worth adding, and now measured: a CPU profile of `torch.linalg.solve` on this torch (2.16.0.dev20261006+cu132) shows it decomposes to `aten::linalg_solve_ex`, `aten::linalg_lu_factor_ex`, `aten::linalg_lu_solve` and `aten::_linalg_check_errors`. The last of those exists only to raise, has no tensor output, is registered effectful and ordered (torch/_higher_order_ops/effects.py:64-66) and is an inductor fallback (torch/_inductor/lowering.py:3894). On CUDA, deciding whether to raise requires reading the `info` tensor on the host, which is one D2H and one stream drain per module per forward, 148 per micro-batch. I could not confirm the drain on this exact build without a GPU; if it is there, this suspect is larger than 139 ms, not smaller. Cheapest test: two 60-step runs or two 40-iteration bench runs identical except `cayley_terms` 5 against 0; the delta is the answer and it needs no profiler. Second cheapest, a one-line change in oft.py:82: use `torch.linalg.solve_ex(..., check_errors=False)` and keep the exact rotation while removing the check, which is the cleanest way to separate the solve's arithmetic from its host sync

### 2. The consumer's burst gather when accum exceeds prefetch

stream.py:275 (`maxsize=self.prefetch`) against 848-853 (gather `accum` batches, then step). The producer can never be more than 4 batches ahead, so after any point where the pipeline drains, the consumer must wait for `accum - prefetch` fresh decodes with the GPU idle. At accum 20 that is 16 batches; at accum 2 it is zero. Predicted 1000-1300 ms per step at b40 and 0 at b24, and the utilisation trace agrees: 11% of 11765 ms is 1294 ms, which is the low end of the quoted 89 to 95% mean utilisation. The quoted dip length of about one micro-batch of decode time understates it if decode runs at 600 to 900 img/s, which the close-out arithmetic implies. It only bites when a per-step sync drains the pipeline, and with `--log-every 1` (the config in d8eb6994) there is exactly such a sync at 325. Cheapest test, and it needs no GPU: instantiate `TarStream("train", batch_size=40, workers=12, prefetch=4)` and replay the consumer rhythm, pull 20 batches then sleep 12 s, ten times, printing the wall time of each 20-batch pull. If the pull after each sleep is about a second, the bubble is real and independent of the card. Caveat: it reads the tars and would compete with the live arm for the X: drive, so keep it to ten bursts or copy one tar to a local disk first

### 3. Per-step host work: fp32 clipping over 152 tensors, then 148 separate compiled optimizer calls

803-805 and SinkSGD_adv.py:397-399, with the per-call work at SinkSGD_adv.py:221-242 and param_update.py:124-146. Each step makes 152 Python calls into the optimizer, each with a dynamo guard, a fresh `torch.as_tensor(step_size)`, a `randint_like` over the parameter for stochastic rounding, and a `p.float()` plus `update.float()` upcast. Across 27.3M parameters that is about 330 MB of per-step temporaries and 150 small launches, plus a `_foreach_norm` sweep over the same 152 grads for the clip. docs/benchmark.md:149 measures 28 ms of compiled-optimizer dispatch for 73 adapter tensors, which scales to about 55 ms here. Predicted 30 to 60 ms per step, which is 0.3 to 0.5% at accum 20 and 3.6 to 7% at accum 2. Cheapest test: wrap `opt.step()` and the clip in two `torch.cuda.Event`s in a 30-step run and print both, or, cruder and free, compare two 200-step windows with `grad_clip: 1` and `grad_clip: 0`; the second isolates the clip

### 4. A host sync every step from `--log-every 1`

325-326. `loss.item()` is a full device sync; `max_memory_allocated` is not, it reads allocator counters. The comment at 449-451 prices this at about 30 ms per step, but the real damage is that it empties the pipeline and makes suspect 2 possible: with `log_every 20` the CPU runs 25 batches ahead of the GPU and the gather is free, with `log_every 1` every step starts from an empty queue. Predicted 20 to 30 ms plus the bubble in suspect 2. Cheapest test: same config, same 300 steps, `--log-every 1` against `--log-every 20`. If the 20 run is more than 2% faster, the sync is the cause and the per-step line is costing more than the information in it

### 5. Labels are copied from pageable memory, so the copy is not asynchronous

stream.py:233 allocates `by` with `np.zeros`, while the image slots at 230 are pinned, and 787 asks for `non_blocking=True`. A pageable source makes that a synchronous copy, 2.0 MB per micro-batch at b40, 20 copies per step, on the same stream as the forward. Predicted 10 to 20 ms per step at b40, 2 to 4 ms at b24, and it also inserts a host stall between micro-batches that the image copy does not. Cheapest test: pin the label ring (`pin_memory` on a torch tensor, or `torch.from_numpy` into a pinned buffer) and compare 200-step windows; it is a five-line change and it cannot make anything slower

### 6. The validation forward is 2-3x less efficient than the training forward, and it pays a second graph

108-117, called from 726 with `cfg.batch_size`. Run 36's first 10000-image validation took 130.4 s, which is 13.0 ms/image; its training step is 14.5 ms/image for forward plus backward plus optimizer. A forward-only pass at 90% of a full step is not possible unless the eval graph is much worse than the train graph. Part of it is the one-off eval compile (docs/benchmark.md:127 measured 29.7 s on the base tower), the rest is the per-image Python at 115 (`torch.stack([ds[i][0] for i in idx])` over a memmap) plus a per-batch D2H at 117. Same shape as training, different `self.training` and different `requires_grad`, so it is a second graph and one recompile, which is correct but not free. Predicted 40 to 52 ms per step amortised at val-every-2500, so 6 to 7% of wall on run 36. Cheapest test: time `evaluate` with the 10k images pre-stacked into one uint8 tensor against the current per-image path, and read `val_seconds` from train_log.jsonl before and after

### 7. Not a millisecond term, but the b48 configs sit 1.4 GB from a cliff

8854c266 peaks at 22.1 GB; 56fa4c17 at 23.5 GB collapses to 3 img/s (docs/benchmark.md K). Run 36's window readings spread 65 to 74 img/s, 14%, which is the shape allocator pressure makes. Predicted effect is variance and a tail risk, not a mean. Cheapest test: none, it is a monitoring question; the safe move is to plan against 21 GB, not 24.5

## Where the record contradicts itself

The two bench checkpoint rows and the trainer cannot all be right. bench_ckpt at b48 large-384 gives 1615.6 ms of backward with 24 of 24 blocks wrapped and 397.7 ms with 3 (docs/benchmark.md K), while the same script at b40 with no checkpointing gives 282 ms. Wrapping all 24 blocks therefore costs 5.7x in the bench, but the trainer moves only 1.26x between the same two configurations, 881 ms against 696 ms. The bench's checkpoint numbers are eager, and the trainer's recompute is compiled; the bench is not transferable for this question

docs/benchmark.md K also states "one recomputed block about 58 ms" from that bench. The trainer's own A/B says 3.3 ms: base-384 batch 48 runs at 298 ms with no checkpointing (2a5c9bad) and 338 ms with all 12 blocks wrapped (6644279b), a 40 ms step for 12 blocks. That is a factor of 17, and it is why the checkpoint-cost intuition in the docs is wrong

The stated anomaly is not an anomaly. Going b24 to b40 moved the trainer 59 to 68 img/s, a ratio of 1.15, and the bench 119 to 142, a ratio of 1.19. Those are the same number within run-to-run noise, so batch scaling is not where the gap lives; the gap is the constant factor of about 2.0 between a full micro-batch and a backward-only bench, of which 1.5x is simply the forward the bench never timed

The base-512 row proves the bench numbers and the trainer numbers are not on the same footing at all. bench_ckpt reports 142 img/s backward-only at base-512 batch 256, while the trainer's full base-512 step at batch 48 with one unfrozen block runs at 138 img/s (16_siglip2_base). A backward alone cannot be as fast as a full step at the same configuration, so one of those two measurements is a different model or a different card state, and every cross-comparison built on it inherits the error

Large-512 at b224 reaches 27.5 Mpx/s while large-384 at b40 reaches 10.0 Mpx/s. A 2.75x per-pixel difference between the same tower at two resolutions is far more than the 1.78x pixel ratio explains. The candidates are batch 224 against 40 and 1024 tokens against 576; both favour the 512 run, and nothing in the repo separates them

The utilisation trace (89 to 95% mean, one dip per optimizer step) and the arithmetic of suspect 2 agree only if the sample came from a run with `log_every 20`. If it came from d8eb6994, which is `log_every 1`, the trace should show a dip of about 1.3 s per step, three or four consecutive samples, not one. The run the sample belongs to is not recorded anywhere I can read

## What I could not determine without touching the GPU

The forward/backward split. Nothing in the repo has ever timed a forward alone at these configs, and every "bench vs trainer" statement in the project rests on an assumed 1:2 ratio that the data cannot confirm

Whether the `_linalg_check_errors` call inside `torch.linalg.solve` is a full stream drain on CUDA on this build, and how much of the Cayley cost is the solve itself versus the fp32 skew construction, the `torch.eye(...).repeat(b,1,1)` allocation at oft.py:85 and the bf16 cast at oft.py:99. That needs one profiler trace or one A/B with `cayley_terms 5`

The true decode rate of the stream at 384 with 12 workers, and therefore the exact size of the burst bubble. A CPU-only probe can measure it, but not without competing with the live run for the X: drive

Whether the 6% slack in the b40 budget sits in the gather, the loss, or the forward, and whether the real forward/backward ratio is 0.5 or closer to 0.7. Only a 30-step instrumented run with `perf_counter` around the gather and CUDA events around forward, loss, backward, clip and optimizer can settle it

The clock and card-load history of the bench runs. bench_ckpt takes `min` of four warm iterations on a card that may have been quieter than the trainer's; docs/benchmark.md H measured a competing context taking 45%. A mean over 100 iterations, run beside nothing, would tell us how much of the 1.35x per-image residual is measurement and how much is real
