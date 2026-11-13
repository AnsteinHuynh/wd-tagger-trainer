# Throughput budget: where the wall clock actually goes inside one optimizer step

`docs/benchmark.md` records what the trainer does in img/s. This file records why. It was opened on 2026-10-23 after a night of numbers that would not reconcile: the trainer ran at 68 img/s where a synthetic bench promised 142, batch 24 to 40 moved the trainer 15% and the bench 19%, and the GPU duty cycle dipped to 4% roughly once per optimizer step. The audit found three separate explanations and one large measurement error of mine.

## The measurement error that started it

`scratch/sophie/bench_ckpt.py` records its start event after the forward and after the loss (`bench_ckpt.py:56-59`), so its ms/backward is a backward only. A real training step is 2.03x that at batch 24 and 2.08x at batch 40, constant across batches. Every comparison in this project that set a bench number beside a trainer img/s is therefore a factor-of-two error, and the ones made before this file existed should be re-read with that in mind. `scratch/sophie/bench_step.py` fixes it: it records before the forward, so its total is comparable to the trainer's step.

The second half of the anomaly is not an anomaly. Large at 384 really does run at 57 to 69 img/s where base at 384 runs at 132 to 160, and the ratio is almost exactly the block count, twelve against twenty-four at the same token count. The 160 to 171 attributed to 384 in the private experiment log was base all along.

## Measured step budget, base B/16 at 384, batch 48, eager, card otherwise idle

`bench_step.py --config 42_...b48-a1.json --batch 48`, ten steps, minimum of ten:

| branch | forward | backward | step | img/s | peak |
|---|---:|---:|---:|---:|---:|
| exact Cayley (`cayley_terms` 0) | 132.2 ms | 133.0 ms | 265.2 ms | 181 | 8.51 GB |
| Neumann-5 (`cayley_terms` 5) | 88.0 ms | 127.2 ms | 215.2 ms | 223 | 8.55 GB |

So the exact Cayley costs 44.2 ms of the forward and 5.8 ms of the backward per micro-batch of 48, 50 ms of 265, 19% of the compute. That is the single largest identified term after the two big ones, and it is the reason the 384 configurations feel expensive per image.

The trainer's own rate for the same shape is 140 to 149 img/s, which is 331 ms per step against the bench's 265. The 66 ms difference is what the bench does not do: the pinned copy of 48 images at 0.44 MB each, the label copy, the fp32 global-norm clip, the compiled optimizer step over 152 tensors, and the python loop. That closes the budget to a few percent.

## Neumann is not a free 19%

The tempting move is `cayley_terms: 5` for 223 img/s. It is a trap, and `docs/oft-saturation.md` already contains the proof: five terms gives a median error of 7.7e-4 against the exact solve but a maximum error of 35%, with 179 blocks above 1%, 53 above 5% and 3 above 20%, and the series diverges entirely once the rotation leaves the radius of convergence at `||Q|| = 1`. Rotations grow during a long run. Exact solve stays.

The safe way to attack the same 50 ms is `torch.linalg.solve_ex(..., check_errors=False)`, which removes the host-side status read without changing the arithmetic, at the cost of no longer surfacing a singular matrix. Unmeasured; predicted 5 to 10% of the forward.

`wdtag/config.py` used to default `cayley_terms` to 5, so a config that simply omitted the key would have run the truncated series while looking exact. It defaults to 0 as of 2026-10-24. Every 384 config on disk already set it explicitly, so nothing that ran was affected, but the default was a trap waiting for the next config.

## Accumulation costs about 23 percent at 512

Measured on the production run, 2026-10-24, base-512, effective batch 384: at micro 64 with 6 accumulated micro-batches the trainer holds 87 to 103 img/s and drifts down over the first hour, where micro 48 with no accumulation on the same tower and resolution held 114 to 116. GPU utilisation sits at 44 percent and 129 W at 52 degrees during the drift, and no physical disk is above 1 percent, so it is not thermal and it is not the tar.

The likely cause is that six micro-batches per optimizer step means six smaller forward and backward passes, each with worse kernel occupancy than one at 224, plus the per-step host work amortised over the same images rather than more. The lesson for the next long run is that effective batch is not free at 512: reaching 384 by accumulating 64s costs roughly a fifth of throughput against reaching it with fewer, larger micro-batches, and the cliff at 23.5 GB is what forces the small micro-batch in the first place.


## The burst gather, and why prefetch must exceed accum

`stream.py:275` caps the queue at `prefetch`, and the consumer gathers every micro-batch of a step before it steps, so when `accum` exceeds `prefetch` the consumer waits for `accum - prefetch` decodes with the GPU idle. At accum 20 against prefetch 4 that is 16 decodes, 1000 to 1300 ms of a 11765 ms step, and it matches the 11% idle implied by a 89% duty cycle. At accum 1 to 4 it is zero, which is why the 2026-10-23 accumulation arms are immune and the full-corpus runs would not have been.

The rule: keep `--stream-prefetch` above `grad_accum`, not merely at its default.

## The rest of the list, ranked by what they would save

Labels are copied from pageable memory. `stream.py:233` builds `by` with `np.zeros` while the image slots two lines above are pinned, and `stream.py:787` then asks for `non_blocking=True`. A pageable source makes that a synchronous 2.0 MB copy per micro-batch on the same stream as the forward, and it inserts a host stall between micro-batches. Predicted 10 to 20 ms per step at batch 40. The fix is to pin the label ring, five lines, and it cannot make anything slower.

The optimizer is called once per tensor. 152 separate compiled updates, each carrying a dynamo guard, a fresh `as_tensor(lr)`, a `randint_like` and an fp32 upcast, plus an fp32 global-norm clip over the same 152 tensors: 30 to 60 ms per step. That is 0.3 to 0.5% at accum 20 but 3.6 to 7% at accum 2, so it quietly penalises exactly the small-batch arms we are about to compare.

Validation is 2 to 3 times less efficient per image than training. It runs at `cfg.batch_size` in a second compiled graph with `requires_grad` off, and `evaluate` at `train.py:108-117` stacks images one at a time out of the memmap. On run 36 that was 130.4 s for 10000 images, 13.0 ms per image against a 14.5 ms full training step. On base at 384 tonight it was 61.8 s for 10000, 6.2 ms per image. At `val-every-steps` 700 on a 22 hour run this is 1% of wall, not the problem.

`--log-every 1` costs one host sync per step, 20 to 30 ms, which is 0.2 to 0.25% at a 12 s step and worth it while diagnosing. It also creates the pipeline drain that the burst gather feeds on, so do not combine per-step logging with `accum > prefetch`.

## What is still unmeasured

The forward and backward split inside a real trainer step, as opposed to a synthetic one. Whether the `linalg` error check is a full stream drain on this build as opposed to a cheap read. The true decode rate for the 384 pipeline, which the arg help claims as 250 img/s at eight threads and which nothing has measured since. How much of the remaining 1.35x gap between bench and trainer is the bench taking the minimum of ten warm iterations on a quiet card.
