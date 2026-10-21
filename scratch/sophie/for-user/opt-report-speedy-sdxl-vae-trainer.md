# Speedy SDXL VAE trainer — optimisation audit

Read-only audit of `C:\Users\kerok\apps\speedy-sdxl-vae-trainer`; all paths below are relative to that
repo root. Tuned for the same box we use: Windows 11, RTX 4090 sm_89, torch 2.13.0+cu132, BF16, TF32
(`logs/nanwatch-mainline.log:1`). Calibration: 83.65M-param SDXL VAE (211 trainable tensors on the
unfrozen path, `scratch/fuseval/std.log:5`), 288×288, batch 12, Sinkhorn-SGD. So: compute-light,
**optimizer-heavy, data-trivial** — the tagger's regime exactly.

## TL;DR (max 8 bullets, the things most worth stealing)

1. **Their per-step accounting is the most reusable artifact**: CPU-submission spans + one deferred
   `torch.cuda.synchronize()`, plus a GPU-idle-share profiler (`sdxlvae/train.py:803-941`,
   `scratch/sophie/profile_step.py:98-104`). It answers "is our 88 ms dispatch or arithmetic?" Steal first.
2. **`compiledOptimizer` is their biggest measured win**: step-wide 1002 → 645 ms at batch 12, i.e.
   ~1.55× on its own (`README.md:302-307`). Mechanism: `torch.compile(self._step_parameter,
   fullgraph=True, dynamic=False)` cached per parameter shape
   (`vendor/Advanced_Optimizers/adv_optm/optim/SinkSGD_adv.py:203-224`) + raised Dynamo limits
   (`sdxlvae/optimizer.py:68-83`, `sdxlvae/runtime.py:26-33`).
3. **…but their pre-bound "fused step" plan bought nothing** (opt 275–278 ms vs 277–287 ms,
   `scratch/fuseval/fused.log` vs `std.log`) — in their regime SinkSGD cost is GPU/element work. They pay
   **3.3 ns/param/step**; we pay **4.3 ns/param**. Same regime: do not assume Python.
4. **`torch.compile` only on a narrow wrapper module**, `dynamic=False`, `fullgraph=False`, inside
   autocast (`sdxlvae/train.py:111-138`), plus Dynamo health counters every 50 steps so a silent eager
   fallback cannot hide (`sdxlvae/train.py:77-94,982-987`). Directly counters our 19.2M-param head.
5. **One D2H sync per step, deferred past the optimizer**: losses accumulate as device tensors and are
   converted once after optimizer kernels are enqueued (`sdxlvae/train.py:870-882,897-901`). No `.item()`.
6. **uint8 cache + GPU-side dequant/normalise**, never a CPU float image (`sdxlvae/losses.py:331-333`,
   `sdxlvae/cache.py:186`): 4× less H2D than fp32, one GPU op instead of a CPU pass.
7. **Batch, not accumulation, because optimizer cost is batch-independent** — their words
   (`README.md:309-317`); they ladder 12/16/18 instead of accumulating. Same lever for our 88 ms.
8. **Do not copy**: their ordering/sampler machinery, VSR builders, FAMO, gradient checkpointing defaults.
   Fetch is 0.1 ms/step for them (`scratch/fuseval/std.log` `fetch=0.1`) — data is not the problem.

## 1. Data pipeline

- **No `torch.utils.data.DataLoader` anywhere** (`grep DataLoader sdxlvae/*.py` → 0 hits). Batches come
  from `_batch_iterator` (`sdxlvae/train.py:626-644`) + a prefetch `threading.Thread`
  (`:604-624`) writing into `queue.Queue(maxsize=3)`, 8 batches per thread spawn (`chunk_batches = 8`,
  `:630`). No workers, no collate, no per-sample objects. Measured 0.1 ms/step. **Transfers: no.**
- **Pinned-memory double buffering, correctly done**: prefetch thread puts
  `latents.pin_memory(), frames.pin_memory()` on the queue (`:613-622`); loop does
  `.to(device, non_blocking=True)` (`:812-813`). No side stream, no `torch.cuda.Stream` anywhere in the
  repo. The gather itself is a per-image Python loop (`:617-621`) — fine, it is in the prefetch thread.
- **mmap'd sharded cache**: latent cache `torch.load(..., mmap=True)` with a hard failure above 8 GiB if
  unmappable (`sdxlvae/cache.py:288-310`, `:36`); pixel cache = manifest + ~1 GiB shards, mmap'd lazily
  (`sdxlvae/image_cache.py:1081-1092,1159-1160`).
- **Shard-size measurement worth knowing**: one 16 GiB pack ≈ 160 Mpx/s random-read; 1 GiB shards ≈ 615;
  512 MiB ≈ 629; 256 MiB ≈ 643; per-image files ≈ 157 (`sdxlvae/image_cache.py:23-30`). If our uint8
  cache is one big file, this is ~4× free on reads — but at 6 ms/step we barely care.
- **Windows mmap trap, measured**: `torch.load(mmap=True)` is not file-backed here — ~1.02 GB of
  *private commit* per resident 1 GiB shard, capped at 48 live mappings
  (`sdxlvae/image_cache.py:1095-1107`); >100 live sections gave an access violation in `torch_cpu`
  (`docs/history/infra.md:83`). **Transfers: as a warning.**
- **Ordering is precomputed, not sampled in-step**: vectorised splitmix64 over ordinals
  (`sdxlvae/ordering.py:73-86`), materialised in 262144-draw chunks (`:178-204`, `configs/default.json`
  `chunkImages`); `draw()` is a slice + `cat` (`:293-309`). Exists for resume-safety, costs 0 ms/step.
- **`training.numWorkers` is a dead key** — in the schema (`sdxlvae/config.py:48`) and every config, read
  by nothing. `README.md:158` ("Cache reader threads") is **unverified/incorrect**.

## 2. Precision

- **BF16 autocast for the model, fp32 pinned per loss term**: `torch.autocast("cuda",
  dtype=torch.bfloat16)` (`sdxlvae/runtime.py:66-73`); each term declares `precision="ambient"|"fp32"`,
  fp32 terms run under `torch.autocast(enabled=False)` (`sdxlvae/losses.py:287-304,316-320`), values
  always cast to fp32 for accounting (`:303`).
- **Trainable params are stored in BF16 — no fp32 master weights**: `param.data =
  param.data.to(torch.bfloat16)` for every `requires_grad` tensor (`sdxlvae/train.py:334-339`, same at
  `sdxlvae/train_unfrozen.py:331-336,675-677`). BF16 is compensated by *stochastic rounding in the
  optimizer* instead of a master copy; there is no `GradScaler` anywhere. Halves param/state traffic and
  removes a per-step cast; SR keeps the update unbiased (`docs/history/optimizer.md:48`).
  **Transfers: partially** — only 20.5M of our params train, so state size is not our problem; the
  validated pattern is "BF16 params + SR, no scaler".
- **TF32 on, cuDNN benchmark off**: `allow_tf32 = True` for matmul and cudnn,
  `set_float32_matmul_precision("high")`, `cudnn.benchmark = bool(cfg.get("cudnnBenchmark", False))` —
  default **False**, and no config sets it (`sdxlvae/runtime.py:16-24`). For a fixed-shape conv net that
  is an untried free win. **No `channels_last`/`memory_format` anywhere** (0 grep hits) — a missed
  10-20% for a conv VAE, **irrelevant for us** (ViT, GEMM + SDPA).
- **uint8 → float on GPU in the loss path**: `frames_u8.to(device, dtype=torch.float32,
  non_blocking=True).permute(0,3,1,2).div_(255.0)` (`sdxlvae/losses.py:331-333`), cached dtype uint8
  (`sdxlvae/cache.py:134,185-186`). Note it sits *outside* the compiled region → 2 eager kernels/batch.
  **Transfers: yes**, and better for us: `uint8 → BF16 /255` fused into the compiled forward.

## 3. Compilation and kernels

- **Compile the wrapper, not the model root**: `VaeTrainPath` compiles only `post_quant_conv + decoder`
  so the frozen encoder never enters the graph — "keeps torch.compile focused on the trained subgraph"
  (`sdxlvae/train.py:111-138`); `dynamic=False`, `fullgraph=False`, `mode` from config (always
  `"default"`; no config in the repo uses another mode). Started in `setup()`
  (`sdxlvae/train.py:346-351`).
- **Not compiled**: the loss stack, `frames_to_float`, the H2D copies, the optimizer's outer loop (only
  the per-tensor kernel is), and the step as a whole. **No CUDA graphs anywhere** (0 grep hits for
  `cuda.graph|CUDAGraph`), so `mode="reduce-overhead"` and `max-autotune` are unexplored — a gap we can
  beat them on, not steal.
- **Per-shape compiled optimizer kernel is the real compile trick**:
  `self._compiled_step_fns[(p.shape, state['factored'])] = torch.compile(self._step_parameter,
  fullgraph=True, dynamic=False)` (`vendor/Advanced_Optimizers/adv_optm/optim/SinkSGD_adv.py:203-224`),
  with `lr` passed as a **0-dim tensor** so LR changes do not recompile (`:204`; rationale at
  `sdxlvae/config.py:51-58`).
- **Dynamo config is part of the trick**: `force_parameter_static_shapes = False` so one graph covers all
  ranks/shapes, plus raised cache/recompile limits (`sdxlvae/runtime.py:26-33`), raised again to
  1024/1024/4096 when `compiledOptimizer` is on — >64 distinct shapes otherwise hard-fail at
  `fullgraph=True` (`sdxlvae/optimizer.py:68-83`). **Transfers verbatim**: a 1536→12476 head plus 48
  OFT-wrapped Linears is exactly that many-shape case.
- **Dynamo health instrumented** so silent eager fallback cannot hide: `dynamo_health()` reads
  `torch._dynamo.utils.counters` (`sdxlvae/train.py:77-94`), logged every 50 steps as
  `frames ok=N/M graph_breaks=K` (`:982-987`).
- **No Triton, no custom CUDA, no SDPA/FA settings** in `sdxlvae/` (0 grep hits). The vendored
  `vendor/AFINE/**/ops/*.cu` are third-party inference files, unused by the trainer.

## 4. Memory and scheduling

- **Gradient checkpointing ON by default** for the whole VAE (`sdxlvae/train.py:341-343`,
  `configs/default.json`), non-reentrant under compile (`:41-44`). This is how 140 tensors of conv net fit
  batch 12-18 in 24 GiB.
- **Batch chosen by an automated VRAM probe**: start 4, +4 until nvidia-smi reports 20.5-22.9 GiB
  (`sdxlvae/probe.py:158-210`), with a persistent `nvidia-smi --loop-ms=100` sampler so the *backward
  peak* is captured (`sdxlvae/probe.py:21-74`). Eval uses a fixed batch because forward-only never
  reaches the band (`README.md:119-121`). **Transfers: yes** — ~200 lines, exactly our batch-48-vs-64
  question.
- **Explicit large-batch argument**: "optimizer cost is batch-independent … larger batches are nearly
  free in throughput terms" (`README.md:309-317`), backed by 12→16→18 = 18.6→20.4→22.2 img/s
  (`README.md:105-111`). Maps 1:1 onto our fixed 88 ms.
- **Gradient accumulation implemented, never used**: loss divided by `accum_steps`
  (`sdxlvae/train.py:750,809-833`), means divided at `:945`; every config sets it to 1. They chose batch.
- **Exactly one device sync per step**, deliberately after optimizer kernels are enqueued
  (`sdxlvae/train.py:897-901`), GPU wait measured separately (`:930-941`). Everything else is async
  submission. **No streams/events**, so saves and H2D are never overlapped.
- **No allocator tuning**, with a reason: "expandable_segments is not supported on Windows"
  (`tools/probe_batch.py:25-26`). Our batch-64 plan must come from real memory savings, not allocator flags.

## 5. Optimizer

- `build_optimizer` maps config → vendor kwargs (`compiled_optimizer`, `sinkhorn_iterations`,
  `state_precision`, `orthogonal_gradient`, SR, +7 pass-throughs) (`sdxlvae/optimizer.py:25-89`); single
  param group (`sdxlvae/train.py:353-354`).
- **`compiled_optimizer=True` ≈ 1.55× on the whole step** (1002 → 645 ms, batch 12, `README.md:302-307`).
  Mechanism: `step()` is a Python `for group / for p` loop over `step_parameter`
  (`SinkSGD_adv.py:374-383`, `:190-224`) which re-runs the branch tree, does `torch.as_tensor(lr)`,
  draws SR randoms, then calls a per-shape compiled kernel — compiling the kernel removes the eager
  op-by-op launch chain per tensor.
- **The `fusedStep` pre-bound plan is implemented, bit-exact, and worth ~nothing.** `_attach_fused_step`
  pre-binds `(param, state, compiled fn, SR branch)` once, then per step only draws RNG and calls the
  kernel (`sdxlvae/optimizer.py:92-170`); its docstring claims "~1.8 ms of dispatch per tensor, ~450 ms
  at 248 tensors, almost all pure Python" (`:96-105`). There is a bit-exactness unit test
  (`scratch/fuseval/unit_test.py:1-15`) and three full logs: `opt=` **275.2/277.0/278.x ms (fused)** vs
  **276.4/279.9/284.0 (std)** and **277.2/281.0/287.0 (std2)** — ≤1-3%, run noise
  (`scratch/fuseval/{fused,std,std2}.log`). **Dispatch hypothesis falsified in their regime**: 277 ms /
  83.65M params = **3.3 ns/param/step**, and we pay 88 ms / 20.5M = **4.3 ns/param** — the same
  per-element constant. Our 88 ms is very likely intrinsic per-element GPU work, not Python.
- **Where that per-element work goes** (per 2-D+ tensor, all fp32): Sinkhorn does `update.float()` then 5
  iterations × 2 L2-renormalisations (+ optional ortho-projection)
  (`vendor/Advanced_Optimizers/adv_optm/util/sinkhorn.py:15,48-63`) — ~20 full-tensor passes plus two
  dtype conversions; `orthogonalGradient="flattened"` adds `p.float()`, two dots, a norm pair and a
  scaled subtract (`.../util/OrthoGrad.py:16-32`). For our 1536×12476 head that is a ~77 MB fp32 tensor
  traversed ~25× per step.
- **Rank gate**: 1-D tensors never enter Sinkhorn — they get `grad.sign_()`
  (`SinkSGD_adv.py:250-256`), 100 of 140 trainable VAE tensors (`AGENTS.md:47`). "More parameters" is
  cheap only for vectors; matrices are the expensive ones. Their full no-op map (which switch makes which
  feature inert) is `AGENTS.md:43-68` — a day of flag-poking saved.
- **State-precision tricks exist, unused**: `statePrecision: "auto"` everywhere; `bf16_sr`/`int8_sr`/
  `factored` (SMMF, ~2× smaller state) branches available (`AGENTS.md:52-54`; one config uses `"fp32"`,
  `configs/oft-fullcoco.json:126`). No `foreach`/fused torch optimizers, no shape grouping, no clipping
  (`sdxlvae/train.py:884-888`).
- **Gradient-coverage guard**: raises after the first backward if any declared-trainable tensor got no
  grad, since the optimizer silently skips `grad is None` (`sdxlvae/train.py:648-670`). Cheap insurance
  for our OFT/head plumbing.

## 6. Loop-level details

- **Device-side loss accumulation, one conversion at the end**:
  `terms_total[k] = terms_total.get(k,0.0) + value.detach().to(torch.float32)` in the micro loop
  (`sdxlvae/train.py:870-881`), `{k: float(v) ...}` only after `optimizer.step()` (`:897-899`);
  comment: "the float() D2H sync is deferred … so it cannot drain the GPU queue mid-step". No `.item()`.
- **`zero_grad(set_to_none=True)`** every step (`:891`); `del` of all step tensors at micro-step end
  (`:882`); batch tensors freshly allocated each step (no reuse buffers).
- **Checkpointing is synchronous and blocking**: every 200 steps, whole state dict → fp32 CPU then
  `torch.save` (`sdxlvae/train.py:678-692`) + hardlink alias (`:180-197`). No async writer, no sharded
  save — a real gap (they had 20 consecutive 2-44 s steps around a save window,
  `docs/history/infra.md:39`, attributed to GPU contention but the pattern is "sync save in the loop").
- **Per-step metadata written once at setup**, wrapped in `try/except` so bookkeeping can never kill a
  run (`sdxlvae/train.py:469-509`, `sdxlvae/train_unfrozen.py:406-522`).
- **Nothing but the objective in the step**, and they measured the cost of violating that: "Measuring
  untrained reference terms per step cost ~35% throughput (820k vs 1.3M px/s)"
  (`docs/history/optimizer.md:32`). A `.pause` flag is polled per step (`sdxlvae/train.py:1008-1011`).
- **Precomputed frozen-target embeddings (sidecar)**: the target's pixels never change, so its backbone
  embedding is precomputed into `.sinkosaur3.pt` and only the prediction is embedded per step — "same
  numbers, roughly half the step cost of the term" (`sdxlvae/config.py:100-104`,
  `sdxlvae/sinkosaur3.py:47-50`). **Do NOT copy that for DINOv3 features**: our backbone weights move
  every step through the OFT rotations, so cached features go stale after step 1 — the repo makes exactly
  this argument for latents (`sdxlvae/image_cache.py:18-21`).
- **Offline/eval patterns**: sidecar build 94 → 219 img/s (2.3×) via a double-buffered producer
  (`docs/history/infra.md:114`); eval ~2× from not rewriting unchanged target PNGs + 8-thread PNG encode
  (`docs/history/infra.md:89-91`). Relevant to our eval harness, not to training.

## 7. Measured numbers found in the repo

Recorded measurements, not README aspiration; authors state ±15% run-to-run (`README.md:318`).

| Number | Context | Source |
|---|---|---|
| 645–669 ms/step, 1.49–1.54 Mpx/s | compile ON + compiledOptimizer ON, batch 12, decode-only | `README.md:302` |
| 805 ms/step, 1.24 Mpx/s | compile OFF + compiledOptimizer ON | `README.md:303` |
| 1002 ms/step, 0.99 Mpx/s | compile ON + compiledOptimizer OFF (⇒ ~1.55× from the optimizer) | `README.md:304-307` |
| optimizer ~490 ms, backward ~146 ms, forward ~14 ms | where the step goes, batch 12 | `README.md:309-315` |
| ~250 W at ~100% utilisation | launch/memory-bound, not compute-bound | `README.md:314-315` |
| 1.488 s/step eager (b8, 0.45 Mpx/s) → 0.810 s/step compile (b12, 1.23 Mpx/s) = "2.7× eager" | VRAM ladder, genuine unfrozen | `docs/history/adapters.md:48` |
| batch 16 compile = 1.028 s/step (1.29 Mpx/s) | 5% faster, at the OOM edge | `docs/history/adapters.md:48` |
| ~835–940 ms/step, ~1.19 Mpx/s, 16.9–18.0 GiB, `framesOk 1 / graph_breaks 0` | real 400-step run | `docs/history/adapters.md:48`, `docs/history/campaigns.md:57` |
| opt 275–278 ms (fused) vs 277–287 ms (vendor loop) | fusedStep A/B, 83.65M params, 211 tensors, b12 | `scratch/fuseval/{fused,std,std2}.log` |
| fwd 337–342 ms, bwd 453–460 ms, fetch 0.1–0.2 ms, `gpuutil=100%`, `gpu=0.1 ms` | same runs: submission/GPU saturated, data free | `scratch/fuseval/std.log` |
| −35% throughput from measuring a weight-0 term per step (820k vs 1.3M px/s) | per-step machinery ban | `docs/history/optimizer.md:32` |
| sinkosaur3 (DINOv3 tower) costs ~12%; 3-term 762k vs 2-term 866k px/s | loss-term cost | `MEMORY.md:127-128` |
| 16 GiB pack 160 Mpx/s → 1 GiB 615 → 256 MiB 643 → per-image 157 | cache layout A/B | `sdxlvae/image_cache.py:23-30` |
| ~1.02 GB private commit per resident 1 GiB mmap shard; AV above ~100 live mappings | mmap budget, Windows | `sdxlvae/image_cache.py:1099-1107`, `docs/history/infra.md:83` |
| sidecar build 94 → 219 img/s (2.3×) via double buffering | offline producer | `docs/history/infra.md:114` |
| eval ~2× faster (skip unchanged target PNGs + 8-thread encode) | eval harness | `docs/history/infra.md:89-91` |
| 3.3 ns per parameter per optimizer step | derived: 277 ms / 83.65M params | `scratch/fuseval/std.log` |

Unverified / claim-only: `README.md:158` `numWorkers` = "cache reader threads" (key read by no code);
"~1.8 ms dispatch per tensor, ~450 ms/step" (`sdxlvae/optimizer.py:96-105`), a code comment contradicted
by the repo's own `scratch/fuseval` A/B; "compiledOptimizer is worth ~1.5× on its own"
(`README.md:307`) — supported only by the 3-row table above (fair, but n=1).

## 8. Ranked transfer list for the DINOv3 tagger

Assumes 290 ms/step compiled, 88 ms SinkSGD, 6 ms data, BF16, batch 32 (11-12 GB), batch 64 = 22 GB,
20.5M trainable params.

| # | technique | gain for us | effort | risk | concrete change |
|---|---|---|---|---|---|
| 1 | Per-step phase accounting + GPU-idle-share profile before touching anything | info (decides #2 vs #3) | low | none | Port `sdxlvae/train.py:803-941` marks and `scratch/sophie/profile_step.py:98-104`: wall vs Σ CUDA self-time, one sync at step end, per-phase CPU ms. Settles dispatch-vs-GPU for the 88 ms. |
| 2 | Per-shape compiled optimizer kernel, `fullgraph=True`, `dynamic=False`, cached by shape | med-high (~1.55× step-wide in their measurement) | low | low | `SinkSGD_adv.py:203-224` pattern: compile the per-tensor update, key cache on `(p.shape, factored)`, pass `lr` as a 0-dim tensor. |
| 3 | Pre-bound fused plan (only if #1 says CPU-bound) | low-med (≤1-3% in their A/B) | med | med (bit-exactness contract) | `sdxlvae/optimizer.py:92-170` is the reference; expect ~nothing per their own logs. |
| 4 | Raise Dynamo limits + `force_parameter_static_shapes=False` | med (avoids hard failure / silent eager) | low | low | `sdxlvae/optimizer.py:68-83`, `sdxlvae/runtime.py:26-33`; then log `dynamoFramesOk/Total` + graph breaks every 50 steps (`sdxlvae/train.py:77-94`). |
| 5 | Compile only the trainable subgraph, `dynamic=False`, `fullgraph=False` | low-med | low | low | `VaeTrainPath` wrapper (`sdxlvae/train.py:111-138`): compile OFT-wrapped blocks + head, keep frozen patch/pos-embed outside. |
| 6 | Deferred single D2H sync; metrics accumulated on device | low (few ms), free | low | none | `sdxlvae/train.py:870-899`. For our per-tag F1/precision accounting: accumulate on GPU, convert once per step, drop every `.item()`. |
| 7 | uint8 cache + `uint8 → BF16 /255` **inside** the compiled forward | low-med | low | low | `sdxlvae/losses.py:331-333` but moved into the graph so Inductor fuses dequant+normalise+patchify; 4× less H2D than fp32. |
| 8 | Bigger batch instead of accumulation (optimizer cost is batch-independent) | med-high (88 ms over 48-64 images) | med | med (VRAM 11-12 GB @32 → 22 @64) | `README.md:309-317` argument + their probe (`sdxlvae/probe.py:158-210`, band 20.5-22.9 GiB) to find the true max; try 40/48 before 64. |
| 9 | Cut optimizer *element* work: sinkhorn iters, ortho mode, state precision | med-high, and the only lever if #1 shows GPU-bound | med | med (changes the algorithm) | At ~3-4 ns/param/step with fp32 Sinkhorn (5 iters × 2 renorms) + flattened OrthoGrad (`sinkhorn.py:15,48-63`, `OrthoGrad.py:16-32`), and our head is 94% of the params: try `sinkhorn_iterations` 5→2, `orthogonal_gradient=disabled`, `state_precision="bf16_sr"` one at a time; mind the rank gate (`AGENTS.md:43-68`). |
| 10 | 1 GiB sharded mmap cache instead of one big file (if ours is one file) | low (data is 6 ms/step) | low | low (>100 mappings bug) | `sdxlvae/image_cache.py:23-30,1081-1092,1095-1107`. Only if #1 shows fetch growing. |
| 11 | Grad-coverage guard after the first backward | 0 (correctness) | low | none | `sdxlvae/train.py:648-670` — proves every OFT/head tensor really gets a gradient. |
| 12 | Async / sharded checkpoint save | ~0 throughput; kills the rare 2-44 s stall | med | low | Not in this repo (theirs is sync, `sdxlvae/train.py:678-721`) — a gap we can beat them on. |

**Absent from this repo (cannot steal, should own)**: CUDA graphs / `mode="reduce-overhead"` /
`max-autotune` (only `"default"` was ever used, `README.md:155`), SDPA/FA backend selection (no attention
config anywhere), `channels_last` (irrelevant for ViT), `cudnn.benchmark=True` (available, defaulted off,
`sdxlvae/runtime.py:23`), overlapped checkpoint I/O.
