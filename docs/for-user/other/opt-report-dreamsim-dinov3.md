# DreamSim-DINOv3 — optimisation audit

Repo: `C:\Users\kerok\apps\dreamsim-dinov3` (read-only). Production trainer
`src/dreamsim_oft/train.py`; the faster successor loop is `scratch/sophie/rotasim/train_rotasim.py` — they differ on exactly the things that matter.

Not a throughput exemplar: best measured ~0.69 optimizer steps/s at 180 img/step × 288² (≈10–14 Mpx/s, `runs/20261003-183132-.../steps.jsonl`) vs your 23 Mpx/s — it pays for 3 images/sample, 2–3 towers, fp32 tower weights, fp32 OFT einsums. What transfers is five specific mechanisms plus a well-measured list of Windows-only traps, not its raw speed.

## TL;DR

1. Compile forward only + dynamo-disable the cache getter (`oft.py:263-269`; `train_rotasim.py:371-384`, `suppress_errors=True`): solves your `FailOnRecompileLimitHit` class of failure.
2. Rotation cache: build Cayley rotations once per microbatch, invalidate after backward (`oft.py:184-196`, `272-291`; calls `train.py:1418`, `1601-1602`) — part of measured 60 s/step → 5.5 s/step (11×) (`MEMORY.md:56-66`).
3. Embed a shared tensor once (`model.py:597-621`); their old loop ran 3 passes per 2 unique images (`train.py:1581-1584`). You have no shared ref; keep the rule.
4. fp32 island + exact Cayley: autocast off around solve/einsum (`oft.py:243-260`), `oft_weight_dtype=float32` (`train.py:182-185`); Neumann diverged (drift 1264 vs bound 2, `‖RᵀR−I‖`→0.156, `train.py:186-195`). Blocked einsum only (`oft.py:254-260`).
5. uint8 on the bus, dequant on GPU (`train_rotasim.py:502-506`); loader-side dequant measured slower, 30 s vs 24/25 s (`pixel_cache.py:67-74`).
6. Their flag A/B (`docs/startupflags-costs.md:4-11`): TF32 override + `cudnn.benchmark` +14.1%; benchmark alone +1.1%; `expandable_segments` no-ops on Windows; `cudaMallocAsync` −3.8%.
7. Windows/WDDM: no mid-run `empty_cache()` (`train.py:718-722`, `1255-1264`; wedged 65% of wall); 2 GiB-guarded `anti_creep` (`1266-1276`); `num_workers=4` (`MEMORY.md:622`); `persistent_workers`/`prefetch_factor=1` (`data.py:907-913`).
8. Absent, do not hunt: compile in the production trainer, CUDA graphs, `foreach`/`fused` optimizer, TF32/cudnn flags in code, GradScaler, async checkpointing, dual-stream prefetch, classification head (§7).

## 1. Data pipeline

`num_workers=4` (8 wedges WDDM), `pin_memory` + `non_blocking`, `persistent_workers` as mitigation, gated decode caches, deterministic custom `batch_sampler`, offset-indexed shards:
[data pipeline](../../../docs/REPORTS/dreamsim-data-pipeline.md).

## 2. Precision

bf16 autocast without GradScaler, the fp32 OFT island, fp32 loss, an fp32 frozen tower (bf16 untried), and no TF32 flags despite a +14.1% A/B:
[precision](../../../docs/REPORTS/dreamsim-precision.md).

## 3. Compilation and kernels

No compile in production; the good pattern compiles only `forward_pair`, disables dynamo on the mutable cache getter, and uses a blocked einsum:
[compilation](../../../docs/REPORTS/dreamsim-compilation.md).

## 4. Memory and scheduling

Gradient checkpointing 1214 → 750 MiB but ~30% compute, the VRAM ladder, micro-batching over checkpointing, and the WDDM `empty_cache` trap:
[memory and scheduling](../../../docs/REPORTS/dreamsim-memory-scheduling.md).

## 5. Optimizer

Same `SinkSGD_adv` family and knobs, `compiled_optimizer=False` reproducing your crash, and their param-tagging census that plausibly explains your 88 ms:
[optimizer](../../../docs/REPORTS/dreamsim-optimizer.md).

## 6. Loop-level details

Three `.item()` syncs per step, a cheap non-finite guard, bounded previews, no streams, and sampler-level resume without re-decoding:
[loop details](../../../docs/REPORTS/dreamsim-loop-details.md).

## 7. DINOv3-specific notes

Frozen tower re-pinned across `train()`, CLS readout with register accounting, no position surgery, stochastic RoPE jitter, one grid per batch, no classification head:
[DINOv3 notes](../../../docs/REPORTS/dreamsim-dinov3-notes.md).

## 8. Measured numbers found in the repo

The 14 measured numbers, each traced to a file and line:
[measured numbers](../../../docs/REPORTS/dreamsim-measured-numbers.md).

## 9. Ranked transfer list

16 techniques ranked by expected gain, with effort, risk and a concrete change:
[transfer list](../../../docs/REPORTS/dreamsim-transfer-list.md).
