# fp32 feature-cache throughput: where the 84 img/s goes and what actually moves it

The pass is `python -m wdtag.cache_fp32_cls --tars 00001 --compute-dtype fp32 --store-dtype fp32`, forward only, no grad, one tar at a time. Output is the 768-d cls vector this pass has always written: fp32 through the trunk, PIL BILINEAR squash to 512, same rows in the same order.

Card RTX 4090, 24564 MiB, driver 617.42, torch 2.16.0.dev20261006+cu132, cuda 13.2. Model `runs/459e2a02/ckpt_step016000.pt`: SigLIP2-B/16-512, OFTv2 on 76 modules, block 128, exact Cayley, cls pool, `oft_scaled`, `oft_conv`. Tars 00001 and 00002 of `train`. Every timing run was 2000-2048 images with the card idle (326-600 MiB used, 4-5% util, 49-58 C, 67-130 W between runs). Real-script rows use `--limit 2000`. Instrumented replica and profiles: `scratch/sophie/bench_cache.py`, `scratch/sophie/prof_fwd.py`, `scratch/sophie/bench_decode.py`.

## Where one batch goes

Batch 192, 2048 images, tar already in RAM so the disk cannot hide in the numbers, workers 12, serial pipeline as shipped:

| phase | s | % of wall | ms/img |
|---|---:|---:|---:|
| tar read | 0.00 | 0.0 | 0.00 |
| PIL decode + resize (12 threads) | 4.77 | 21.8 | 2.57 |
| np.stack into the batch | 0.28 | 1.3 | 0.15 |
| H2D copy (pageable) | 0.16 | 0.7 | 0.08 |
| GPU forward (cuda events) | 16.49 | 75.3 | 8.88 |
| D2H fp32 + .numpy | 0.00 | 0.0 | 0.00 |
| total | 21.9 | | 11.7 |

84.8 img/s. The forward is three quarters of the wall; the decode is the other fifth and, in the shipped code, it runs before the forward instead of under it. That is the whole opportunity: 8.88 ms/img of GPU work against 2.57 ms/img of CPU work that can overlap it.

The 12-thread decode delivers 2.57 ms/img aggregate against 13.92 ms/img single-threaded, so it scales about 5.4x on 8 cores and never will reach 12x; making it faster buys nothing once it is hidden, which is why the recommendation overlaps it rather than replacing it.

D2H is 0.00 ms because the host is already waiting on the GPU: 192x768 fp32 is 590 KB and the `.cpu()` is just the join. Host-side launch is 90 ms per batch, 0.47 ms/img, 5% of the forward, so the step is device-bound and not launch-bound.

Counted from the shapes (1024 tokens, 768 wide, 12 blocks, 3072 MLP): 174 GFLOP/img for the frozen linears, 14.5 for the OFT rotations, 38.6 for attention, 227 in total. 8.88 ms/img is therefore 26 TFLOPS against the card's measured 52.4 TFLOPS fp32. The profile says the sgemm kernels carry 54.7% of the forward at 38 TFLOPS (73% of peak) and the fp32 attention 21.5% at 20 TFLOPS, with elementwise, layer norm, copies and the Cayley rebuild making up the rest. The linears have almost nothing left in them without changing precision, which is exactly why TF32 is the only lever pointed at them.

## The disk is not the limiter

Whole tar preloaded into RAM, then the same decode+forward loop, versus streaming the tar from X: with the same batch 64 and 2000 images: 79.7 against 78.5 img/s, a 1.5% difference. On a genuinely cold tar (00002, never read this session, 4000 images, batch 64) the read is 0.23 ms/img, 1.8% of wall. The tar is 1.6 GB over 25000 images, 64 KB per image; a consumer that needs one 151 MB batch every 1.7 s never outruns the readahead, so there is nothing to win by staging tars or by adding read threads. Preloading would also cost 1.6 GB of RAM per tar for 1.5%.

## Batch size

Serial pipeline, tar in RAM, 2048 images, workers 12, fp32. `img/s` is steady state, the first batch dropped from the timer; `fwd ms/img` is the cuda-event forward.

| batch | img/s | fwd ms/img | peak VRAM | verdict |
|---:|---:|---:|---:|---|
| 64 | 80.7 | 9.40 | 3.92 GB | worse forward |
| 96 | 83.4 | 9.09 | 5.66 GB | plateau begins |
| 128 | 84.2 | 8.96 | 7.39 GB | |
| 192 | 84.8 | 8.88 | 10.87 GB | best at a sane VRAM |
| 256 | 85.2 | 8.82 | 14.34 GB | ties 192, 3.5 GB more |
| 384 | 17.3 | 53.90 | 21.28 GB | VRAM cliff, 5x slower |
| 512, 768 | not run | | | 384 is already 21.3 GB; these cannot fit |

The forward saturates by 192: 8.88 against 8.82 ms/img at 256 is 0.7%, inside the run-to-run spread. 384 does not OOM, it collapses (21.28 GB, 53.9 ms/img), which is the documented cliff in `docs/benchmark.md` and the reason the sweep stops there. Splitting the same images into smaller forwards is this same curve read leftwards and cannot help: 64 and 96 are 80.7 and 83.4 img/s against 84.8 at 192, so micro-batching only buys VRAM.

## Overlap: decode under the forward

Pinned double buffer, one producer thread doing tar read + decode + word into a pinned slot, the main thread doing H2D + forward + D2H. Two slots is enough because the producer needs 2.6 ms/img and the consumer 8.9. Same rows, same order, same batch boundaries. Bench, 2048 images, steady state, stream source (bench configs isolate the knob; the real-script rows are below):

| config | img/s | peak VRAM | note |
|---|---:|---:|---|
| overlap, batch 128, workers 6 | 103.7 | 7.39 GB | forward 9.57 ms/img |
| overlap, batch 192, workers 12 | 107.3 | 10.87 GB | waiting on decode 0 ms in every batch |
| overlap, batch 192, workers 6 | 107.3 | 10.87 GB | workers do not matter while decode is hidden |
| overlap, batch 256, workers 6 | 107.4 | 14.34 GB | ties 192 |
| overlap + cached rotations, batch 192, workers 6 | 110.8 | 10.87 GB | best measured |

`wait_decode` is 0 ms on every batch of every overlap run: decode is fully hidden, so its thread count and its speed stop mattering. The producer thread and the pinned slot add about 0.3 ms/img to the forward itself (9.2 against 8.9), which is the CPU contention of 6-12 decode threads against the thread that launches the GPU work.

## The OFT rotation rebuild

`_cayley_batch()` is recomputed on every forward and every call rebuilds the same six 128x128 Cayley transforms per module from frozen weights: 76 modules, so 456 small `getrf`/`trsm` solves plus their skew and eye construction per batch. Profiled per batch-192 forward: 15 ms of `getrf_panel` and 9.5 ms of `trsm` on top of the skew and eye elementwise kernels. Building each rotation once at load and returning the cached tensor is bit-identical here (weights frozen, `eval()`, `no_grad`) and measured 1770 -> 1705 ms per batch, 65 ms, 3.7% of the forward. In the real script 82 -> 84 img/s on the same 2000 images with the same blake2b. Exposed as `--cache-rot`, off by default because it is only valid while the weights cannot change.

## torch.compile

Batch 192, forward only, compiled `model.forward_features`, `dynamic=False`:

| mode | fwd ms/img | first-call cost | verdict |
|---|---:|---:|---|
| eager | 8.88 | 0.2 s | |
| default | 8.65 | 42 s | 2.6% faster forward |
| max-autotune | 8.44 | 274 s | 5% faster forward, 6.76 GB peak |

End to end it loses. With overlap + cached rotations, max-autotune measured 108.0 img/s against 110.8 for eager, because the graph replay removed the 90 ms of CPU launch time that was not the bottleneck anyway, and the compiled forward still waits on the same bytes. Worse, `dynamic=False` recompiles for every new batch shape: the 128-image tail of a 25000-image tar cost 28.7 s, so a full pass would pay about 1.7 h of recompiles over 213 tars unless the tail is padded to a full batch. Not used. `max-autotune` also spends its tuner on triton_mm variants that exceed shared memory and are discarded ("out of resource: triton_mm Required: 131072 Hardware limit: 101376"), which is most of the 274 s.

## Decode path alternatives

Same 300 images, single thread, decode + resize to 512 squash, compared against the shipped PIL path byte for byte:

| path | ms/img | img/s | byte-identical | max abs diff | elements differing |
|---|---:|---:|---:|---:|---:|
| PIL BILINEAR (shipped) | 13.92 | 71.8 | reference | | |
| cv2 INTER_LINEAR | 5.84 | 171.4 | 0/300 | 169 | 30.88% |
| cv2 INTER_AREA | 6.49 | 154.1 | 0/300 | 187 | 24.86% |
| torchvision decode + resize, antialias=False | 8.75 | 114.2 | 6/300 | 169 | 29.40% |
| torchvision decode + resize, antialias=True | 8.84 | 113.1 | 6/300 | 2 | 1.16% |

cv2 is 2.4x faster and unusable: it does not antialias the 640 -> 512 downscale, so 31% of the bytes change and the features change with them. torchvision with antialiasing is within 2 counts on 1.16% of elements and still not byte-identical, and 300-image sampling cannot certify exactness over 5.3M. turbojpeg is not installed and would not help: it decodes jpeg only and the corpus is webp. Nothing here matters for throughput either way, because decode is already fully hidden by the overlap.

## SDPA backend and memory format

The model is built with `attn_implementation="sdpa"` and fp32, and torch picks memory-efficient attention: measured on one batch 192 forward, default 2.206 s (first call, cold), explicit `EFFICIENT_ATTENTION` 2.058 s, identical outputs to 8 decimal places. `MATH` is 53.0 s and 27.17 GB, 24x slower, because it materialises the 192x12x1024x1024 score matrix. `FLASH_ATTENTION` needs fp16/bf16 and `CUDNN_ATTENTION` raises "No available kernel" for fp32. There is nothing to choose: the default is already the only usable fp32 backend.

`channels_last` on the model and on the prepped input: 8.90 against 8.88 ms/img, no effect, as expected for a ViT whose only convolution is a 16x16 patch embed. Not used.

## TF32, measured separately, not IEEE fp32

`torch.backends.cuda.matmul.allow_tf32 = True` plus `float32_matmul_precision("high")`, everything else identical (overlap + cached rotations, batch 192): 144.1 img/s, forward 6.87 ms/img against 8.88, peak 10.92 GB. That is +30% over the best IEEE configuration and turns 14.0 h into 10.3 h.

It is not fp32. TF32 keeps a 10-bit mantissa in the multiply, so every one of the 76 rotated linears and every attention matmul is a different arithmetic from the IEEE path; the output is a different function of the same pixels, not a rounding-order difference. It is reported here as a measured option that needs an explicit decision, and it is not part of the recommendation below. The shipped pipeline already runs its one convolution (the patch embed) through cuDNN with `cudnn.allow_tf32=True`, which is torch's default and was not changed here; it is 1.2 GFLOP/img of the 227 counted above, and it is the only place the shipped path itself is not strict IEEE fp32.

How big the arithmetic change is, counted and not measured: TF32 has an 11-bit significand against fp32's 24, so u is 2^-11 = 4.88e-4 against 2^-24 = 5.96e-8, a factor of 8192. One 768-long dot product therefore carries about 1e-3 relative error under TF32 against 1.2e-7 under fp32, because the input rounding dominates and the fp32 accumulation does not. End to end the prediction is a 1e-3 to 1e-2 relative L2 move in the cls vector (cosine 1 - 5e-7 to 1 - 5e-5), roughly 1000 to 10000 times the 9.4e-7 that a pure fp32 reassociation costs, with the layer norms capping the growth. TF32 is fp16's mantissa with fp32's exponent range, and 4x more precise than bf16. The 384-image comparison that would turn this prediction into a number was not run.

## Numerics: what the two speed options change

`--cache-rot` is bit-identical. Two 2000-row shards from tar 00001, one serial and one with the flag, have the same blake2b `cf71f5d5d39d` and 0 of 1536000 elements differ.

`--overlap` is not bit-identical, and that is worth knowing before pointing it at an existing cache. The same 2000 images give 1493357 of 1536000 elements different (97.2%), max abs 7.75e-6, mean abs 3.8e-7 over the elements that differ, and the stored blake2b differs (`3c10e670d39d`). Measured in float64 on a 384-row dump of the same images, because a float32 cosine sits on its own noise floor: relative L2 difference 9.4e-7, rms 4.8e-7, per-row cosine min 1 - 2.6e-12, against feature rms 0.505. This is not a precision change and not a race in the copy: the uint8 batch fed to `_prep` hashes identically in both pipelines (85734ee3c2e0c6f6 and a3dfb1ea64155e5d for the two batches) and so does the prepped float tensor (90ecd82c380d98a1, 99c2dde9f7bd773c), the four serial runs are byte-identical to each other, and the overlap variants (pinned non-blocking, pinned blocking, pageable, 1 or 6 or 12 workers) are byte-identical to each other. A held 302 MB pinned allocation, an idle background thread, and the pageable copy path each fail to reproduce it. So it is one deterministic reassociation inside the fp32 forward that switches with the pipeline, at the 1e-6 relative level, not a change of arithmetic: no bf16 and no TF32 anywhere in it. Shards from the overlap pipeline are interchangeable for training with shards from the serial one but are not byte-equal to them, so a cache built with `--overlap` should not be mixed into a directory of shards built without it if the blake2b records are meant to match.

## Recommendation

```
python -m wdtag.cache_fp32_cls --tars all --compute-dtype fp32 --store-dtype fp32 \
    --batch 192 --workers 12 --overlap --cache-rot --out <dir>
```

Measured with the real script, tar 00001, 2000 images, one invocation:

| configuration | img/s | note |
|---|---:|---|
| shipped (`--batch 192`, no flags) | 82 | 10.87 GB peak |
| `--cache-rot` | 84 | bit-identical shard |
| `--overlap --cache-rot`, workers 12 | 106.6 | 10.9 GB peak |
| `--overlap --cache-rot`, workers 6 | 106.0 | decode already hidden at 12 |

5321713 images, fp32, at the measured 106 img/s: 50205 s, 14.0 h. At the bench steady state of 110.8 (the 2000-image run pays one 2.2 s warmup batch that a 25000-image tar does not) it is 13.3 h. The shipped 82 img/s is 18.0 h, so the two flags are worth 4 h of a one-off pass. For reference, TF32 at 144.1 img/s would be 10.3 h and bf16 at 167 img/s, the number this pipeline was already measured at, 8.9 h; both change the arithmetic and are not this cache.

Add about 0.3 GB of host RAM for the two pinned slots (not VRAM) and 1.6 GB more only if a future variant preloads tars, which measured 1.5% and is not worth it. VRAM stays at 10.9 GB of 24.5, so the pass is nowhere near the 22.5 GB cliff that bites batch 384.

## Tried, did not help

Whole tar in RAM: 79.7 against 78.5 img/s at batch 64, +1.5%. The disk is 1.8% of wall on a cold tar
Batch above 256: 384 is the cliff, 21.28 GB and 17.3 img/s against 85.2, so 512 and 768 were not attempted
torch.compile default: 2.6% faster forward, no end-to-end gain, and a recompile for every new batch shape
torch.compile max-autotune: 5% faster forward, 108.0 against 110.8 img/s end to end, 274 s to build, 28.7 s per recompile on a tar tail
cv2 decode, INTER_LINEAR or INTER_AREA: 2.2-2.4x faster decode, 25-31% of pixels changed
torchvision decode + antialiased resize: only 1.16% of pixels differ by at most 2 counts, still not byte-identical
Decode worker count: 1, 6 and 12 are identical once decode is hidden
channels_last: 8.90 against 8.88 ms/img
SDPA math backend: 24x slower and 27 GB
SDPA flash and cudnn backends: unavailable for fp32
Pinned H2D copy and non-blocking: part of the overlap, worth nothing on its own (the copy is 0.08 ms/img)
Reducing host-side work (compile, cached rotations in the launch path): the host is 5% of the forward, so it cannot be the lever; `--cache-rot` only pays because it removes GPU kernels, not launches
