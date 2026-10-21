# DreamSim-DINOv3 — optimisation audit

Repo: `C:\Users\kerok\apps\dreamsim-dinov3` (read-only). Production trainer
`src/dreamsim_oft/train.py`; the *faster* successor loop is
`scratch/sophie/rotasim/train_rotasim.py` — they differ on exactly the things that matter.

**Framing, so nothing below is oversold:** this repo is not a throughput exemplar. Best
measured point ~0.69 optimizer steps/s at 180 img/step × 288² (≈10–14 Mpx/s,
`runs/20261003-183132-.../steps.jsonl`) vs your 23 Mpx/s — it pays for 3 images/sample,
2–3 towers, fp32 tower weights, fp32 OFT einsums. **What transfers is five specific
mechanisms plus a well-measured list of Windows-only traps, not its raw speed.**

## TL;DR (max 8 bullets, the things most worth stealing)

1. **Compile the forward only, and `@torch._dynamo.disable` the mutable-cache fetch.**
   `oft.py:263-269` disables dynamo around `_cached_rotation` *"so the cache branch is not
   baked into the graph or recompiled every step"*; `train_rotasim.py:371-384` compiles
   `model.forward_pair` **only** (state_dict/resume stay on the uncompiled module), with
   `torch._dynamo.config.suppress_errors = True`. This is your `FailOnRecompileLimitHit`
   class of failure, solved.
2. **OFTv2 rotation cache:** build every Cayley rotation **once per microbatch, outside any
   checkpoint region**, invalidate after each backward (`oft.py:184-196`, `272-291`; called
   `train.py:1418`, `1601-1602`). Motivation quoted in code: *"forward() is called once per
   image per chunked batch (~80×/step), each call rebuilding the whole Cayley pipeline for
   all 108 modules (~10k redundant small-kernel chains/step = the GPU-idle signature)"*.
   Part of a measured **60 s/step → 5.5 s/step (11×)** (`MEMORY.md:56-66`).
3. **Embed a shared tensor once** (`model.py:597-621`). Their old loop ran `model(ref,left)`
   then `model(ref,right)` — the ref went through the tower **twice, 3 passes per 2 unique
   images** (`train.py:1581-1584`). You have no shared ref; keep the rule anyway.
4. **fp32 island + exact Cayley for OFTv2:** autocast *suspended* around solve and rotation
   einsum (`oft.py:243-260`), `oft_weight_dtype=float32` (`train.py:182-185`), truncated
   Neumann diverged in practice (`train.py:186-195`: drift 1264 where Cayley is bounded by
   2, `‖RᵀR−I‖`→0.156). 108 × 32×32 exact solves/step is negligible. Apply the rotation as
   a **blocked einsum** `"...bk,bkc->...bc"` (`oft.py:254-260`), never a dense (d,d) matrix.
5. **uint8 on the bus, dequant on the GPU:** `refs.to(device, non_blocking=True).float().div_(255.0)`
   (`train_rotasim.py:502-506`, *"u8 corpora: ship bytes to GPU, convert there (cheaper
   bus)"*). Their other measurement shows the opposite choice is **slower**: dequantising
   in the loader = 30 s vs 24/25 s fp32 for the same 15 steps (`pixel_cache.py:67-74`).
6. **Their own flag A/B** (`docs/startupflags-costs.md:4-11`): `NVIDIA_TF32_OVERRIDE=1` +
   `cudnn.benchmark` = **+14.1% px/s**; `cudnn.benchmark` alone +1.1%; `expandable_segments`
   **silently no-ops on Windows**; `cudaMallocAsync` −3.8%.
7. **Windows/WDDM discipline:** no `empty_cache()` mid-run (`train.py:718-722`, `1255-1264`
   — it wedged the loader for the rest of the run, 65% of wall time); 2 GiB-guarded
   `anti_creep` instead (`1266-1276`); `num_workers=4` because *"8 workers wedged a run"*
   (`MEMORY.md:622`); `persistent_workers` + `prefetch_factor=1` (`data.py:907-913`).
8. **Absent here — do not hunt for them:** no `torch.compile` in the production trainer, no
   CUDA graphs, no `foreach`/`fused` optimizer, no TF32/cudnn flags in code, no GradScaler,
   no async checkpointing, no dual-stream H2D prefetch, and **no classification head**
   (section 7).

## 1. Data pipeline

- **`num_workers=4` default; 8 is a known-bad value on this box** (`train.py:97`;
  `MEMORY.md:622`). WDDM/host-commit pressure, not CPU throughput. → Low for you (6 ms/step)
  but it names the failure mode to expect if you raise workers to hide a stall.
- **`pin_memory` + `non_blocking=True` everywhere** (`data.py:891-905`: *"Measured 9→13 GB/s
  … non_blocking silently degrades to synchronous on pageable memory"*; every H2D at
  `train.py:1430-1431`, `1559-1560`, `train_rotasim.py:504-506`). → **Medium-high**: the
  one data-side change that can matter if any of your H2D is currently pageable.
- **`persistent_workers` + `prefetch_factor=1` are WDDM mitigation, not throughput**
  (`data.py:907-913`, defaults off at `train.py:260-261`): teardown/re-priming costs
  16 batches of host commit. → Low, but keep it if you ever rebuild a per-epoch loader.
- **Epoch-boundary hygiene / loader pre-spawn:** `cuda_settle()` = `synchronize()`+`gc`, no
  `empty_cache()` (`train.py:1258-1264`), and epoch *e+1*'s loader is built in a background
  thread during epoch *e* (`1306-1328`, `1395-1408`) so the ~16 s respawn is off the
  critical path. → Low gain, good shape.
- **Pre-decoded caches, gated:** per-image `.pt` decode cache (`data.py:233-259`); post-resize
  288 cache with a **runtime bit-identity check** against a live decode+resize
  (`pixel_cache.py:193-224`) and a directory-name method token (`118-124`). → The *gate*
  transfers, the cache does not.
- **fp32 vs uint8 cache, measured** (`pixel_cache.py:67-74`): *"training 24 s vs 25 s (inside
  noise, GPU-bound), eval wall 21.2 vs 24.4 s … u8 is 4× smaller but SLOWER on the training
  path (30 s — dequantising costs more than the decode+resize it saves)"*. Confound: that u8
  path dequantises on the **CPU** (`187-190`). → Keep uint8 (you need the shrink), convert
  after H2D.
- **No `sampler` for shuffling — custom `batch_sampler`s** (`AspectBatchSampler`,
  `data.py:831-888`): deterministic in `seed+epoch`, plan materialised in `__iter__` before
  workers spawn, `skip_batches` slices the consumed prefix so a resume decodes nothing
  (`841-844`). → **Medium** for your resume-without-re-showing requirement; the plan is a
  pure function of `(seed, epoch)`, so slicing reproduces the continuation exactly.
- **Bucketing beats accumulation, stated** (`data.py:834-836`): *"bucketing or per-sample
  forwards with accumulation (strictly worse: forfeits batched attention)"*. → Confirms
  batch-32 over grad-accum.
- **The successor loop drops `DataLoader` entirely:** offset-indexed shards
  (`shardio.py:26-44`: 32 B sha + u64 offset + u32 length), fd cache with LRU-8
  (`rotasim_data.py:33-60`), synchronous `pread` + `torch.load(BytesIO)` + `stack` in the
  main process (`131-157`). Only viable at seconds-per-step. → Low for you, but that
  manifest is exactly what your `shard#####-N.pt` cache wants — and store **raw uint8
  blobs**, not torch-serialised tensors, so you can `np.memmap` slices and skip
  `torch.load` per record.

## 2. Precision

- **bf16 autocast, no GradScaler, by design** (`train.py:688-702`: *"its exponent range
  matches fp32, so it needs no GradScaler"*; default `amp="off"` at `241`, real runs pass
  `bf16`). → Same as yours; nothing to steal.
- **The fp32 island is the interesting part** (`oft.py:232-239`, `243-260`): autocast
  suspended around the Cayley solve **and** the rotation einsum, because *"casting the
  tensors alone is not enough, the autocast policy would downcast the matmuls right back"*.
  `oft_weight_dtype="float32"` justified as *"only ~1.3M params … costs essentially
  nothing"* (`train.py:182-185`) — the same 1.3M you have. → **High, and it is a
  correctness+speed pair.** The einsum left in fp32 is the *apply* — lowest arithmetic
  intensity, least to gain from bf16, most to lose.
- **Loss kept fp32 on purpose:** forward under autocast, then `loss_sum.float()` before the
  mean/backward so *"the scale of the update never depends on the autocast dtype"*
  (`train.py:1585-1588`). Low, free.
- **Frozen tower loaded in `float32` and cast per-op by autocast** (`train.py:944`, `1095`;
  `backbones.py:521`). They never tried loading the frozen tower *in* bf16. → **Gap, not a
  trick.** For you a frozen 87M backbone in bf16 removes a per-op weight cast and halves
  resident weights; unmeasured here, worth one A/B.
- **No TF32/cudnn flags anywhere in `src/`** (zero grep hits) — yet their A/B says that is
  the biggest flag lever (+14.1%, `docs/startupflags-costs.md:10-11`), precisely *because*
  they leave fp32 matmuls on the table. → **Medium** for you: helps only your fp32 islands
  (optimizer math, Cayley/rotation apply, any fp32 loss path); one env var, one A/B.
- **Channel-last: never used, never measured.** Correct for a ViT (one patch-embed conv).

## 3. Compilation and kernels

- **Production trainer: no `torch.compile` at all.** The only dynamo reference in `src/` is
  the disable decorator (`oft.py:263-269`). → Your 359→290 ms compile win already beats this
  repo's production loop.
- **The good pattern is in `train_rotasim.py:371-384`:** bind `fwd = model.forward_pair` once,
  and if enabled `torch._dynamo.config.suppress_errors = True; fwd = torch.compile(...)`.
  Three load-bearing details from their comments: (a) forward only — *"All checkpointing /
  state_dict / named_parameters access stays on the uncompiled module, so saved keys never
  gain a `_orig_mod.` prefix"*; (b) `suppress_errors` makes inductor fall back to eager for
  CPU-side graphs (`376-380`: `InvalidCxxCompiler`, no MSVC on the box); (c) bound once, not
  re-wrapped per step. → **High**: this is the shape that cannot trip a recompile limit,
  because the optimizer and the mutable cache never enter a graph.
- **`@torch._dynamo.disable(recursive=True)` on the cache getter** (`oft.py:263-269`):
  *"Inlining this branch into dynamo would either bake the stale path in or force a recompile
  every step; a disabled region keeps the graph guards stable."* → **High**: a Python-side
  mutable tensor handed to a compiled region looks like a new input every step.
- **No SDPA/flash pinning, no `attn_implementation`, no cudnn.benchmark, no CUDA graphs, no
  Triton kernels, no `foreach`** (all confirmed absent). They use HF's default sdpa. → Gap;
  at 1024 tokens your bottleneck is GEMMs, not attention, but it is untested ground here.
- **Blocked einsum for the OFT apply** (`oft.py:254-260`): reshape to
  `(..., n_blocks, block_size)`, `einsum("...bk,bkc->...bc")`, `repeat` only when
  `block_share` → cost linear in block_size² per block instead of a dense per-layer matrix.

## 4. Memory and scheduling

- **Gradient checkpointing: measured, off by default, with the trap documented.**
  `backbones.py:304-310`, `443-461`: HF checkpointing only engages in `train()` mode, so the
  frozen tower must *not* be pinned to eval, and *"measured: eval-mode = no checkpointing,
  1214 MiB; train-mode = checkpointed, 750 MiB"*; cost *"~30% compute … 0.4 GiB/triplet at
  the 544 budget"* (`train.py:278-281`), `use_reentrant=False` so grads still reach adapters
  (`backbones.py:405-408`). → **High relevance to your 22 GB batch-64, but it HURTS img/s**:
  you are compute-bound at 359 ms. Use it only to make a memory-bound config fit.
- **VRAM ladder, same tower** (`MEMORY.md:629-632`, DINOv3-B @288², ckpt on, all-six
  adapters): b12p 4316 / b16p 5639 / b20p 6959 MiB; *"b32p k8 (288 img/step) and b28p k8 OOM
  at 288; b24p k8 (216 img) fits"*. Their lever at the wall was **micro-batching with grad
  accumulation** (`train_rotasim.py:551-557`: *"the loss is a SUM over slots, so
  forward+backward per group and one opt.step() … is MATHEMATICALLY IDENTICAL to one big
  batch — but autograd only holds ONE group's activations"*), not checkpointing. →
  **High**: batch-64-equivalent optimisation at batch-32 memory with no compute tax.
- **`anti_creep`: `empty_cache()` only when reserved−peak > 2 GiB, every 25 steps**
  (`train.py:1266-1276`); `cuda_settle()` does `synchronize()`+`gc` and deliberately **not**
  `empty_cache()` (`1258-1264`). Reason in code: on WDDM it *"releases on the driver side
  outside torch's accounting and the ensuing host-commit churn wedged the input pipeline for
  the rest of the run"* (`718-722`). → **High (Windows-only)**: if your loop calls
  `empty_cache()`, delete it.
- **No `set_per_process_memory_fraction`, no in-code allocator tweaks** — they A/B'd them
  externally and rejected all (`docs/startupflags-costs.md:6-9`). → Don't spend time here.
- **No CUDA graphs, no dual-stream prefetch, no async checkpointing.** `save_checkpoint` is
  a blocking `torch.save` plus a blocking `shutil.copy2` of the whole file each boundary
  (`train.py:1699-1712`) — cheap at ~5 MB (`model.py:635-637`: *"~5 MB instead of ~1 GB"*),
  but your 19.2M head + optimizer state will not be. → **Gap on your side.**
- **Logging is bounded:** one JSON line per `log_every=10` (`train.py:1642-1690`) with
  `host_mem_sample()` (ctypes `GlobalMemoryStatusEx`, `771-803`) and `adapter_health()`
  (`@torch.no_grad()` walk, `806-812`). → `host_commit_pct` is a cheap early wedge detector.

## 5. Optimizer

- **Same family as yours — `adv_optm.SinkSGD_adv`** (`train.py:419`, `479-501`), knobs
  mirrored 1:1 from the library signature (`train.py:302-320`: momentum, nesterov,
  normed_momentum, snr_cond, orthogonal_sinkhorn, `sinkhorn_iterations=5`, cautious_wd,
  `spectral_normalization=False`, `orthogonal_gradient="disabled"`,
  `state_precision="auto"`, `stochastic_rounding=True`, `compiled_optimizer=False`). →
  **Directly comparable, and it confirms your negative results**: they also leave
  `compiled_optimizer` off, and `AGENTS.md:816-817` documents the same crash class —
  *"`compiled_optimizer=true` needs MSVC (`cl`) on the CPU path … it works on CUDA via
  Triton."* Their working answer is **not to compile the optimizer at all**; compile the
  forward (`train_rotasim.py:382`).
- **Param grouping: two groups, OFT isolated for a per-group LR multiplier**
  (`train.py:435-465`). No `foreach`/`fused`/state-dtype work of their own. → No fix for your
  88 ms here.
- **`adv_optm` dispatches per-Parameter by underscore attributes** — `tagging.py:1-14` lists
  exactly which (`_is_oft`, `_oft_scale_factor`, `_is_lora_A/B`, `_is_dora_scale`, read in
  `scaled_optm.py`, `centered_decay.py`, `Kourkoutas.py`); `tag_peft_parameters()` sets them
  in **one** pass at build time with a census print so a mis-tag cannot pass silently
  (`tagging.py:54-124`). → **Medium, and it plausibly explains your 88 ms**: attribute-
  dispatched per-tensor control flow is exactly what yields N small kernels + Python work
  per step, and why `stochastic_rounding`/`sinkhorn_iterations` change nothing (the cost is
  *launch count*, not arithmetic). Two escapes: (a) verify the census — an untagged 19.2M
  head silently takes the generic per-tensor path; (b) keep the head as **one 2-D tensor** so
  the muon-like path does one matrix update instead of looping.
- **`zero_grad(set_to_none=True)`** every step (`train.py:1639`, `train_rotasim.py:483`);
  **one `opt.step()` per accumulation group** with LR written into `g["lr"]` per group
  (`train.py:1630-1635`). → Free, already yours.
- **`clip_grad_norm_` is off by default** (`train.py:146`, `1636-1637`) — clipping 1.3M
  adapter params is a full extra pass, and SinkSGD's sign-like steps arguably don't need it.

## 6. Loop-level details

- **Three `.item()` per step — a real (small) sync tax:** `loss_sum.item()`
  (`train.py:1608`), `correct.item()` / `scored.item()` (`1618-1619`), plus a `.cpu()` per
  preview image (`1443-1452`, `1566-1580`). Each drains the stream inside the step, blocking
  overlap of the next H2D and the optimizer. → **Low-medium** (~1–2 ms of your 359 ms):
  accumulate on device, read once per log tick.
- **Non-finite guard is a per-step scalar reduction** (`torch.isfinite(loss_sum)`,
  `train.py:1593`), with the expensive per-input diagnostic only on failure (`1594-1599`). →
  Copy the shape: fail loudly, name the input, keep the last checkpoint usable.
- **Previews are budget-bounded and read once per step, not per image**
  (`train.py:1566-1580`, `preview_images=10`). Maps directly onto your first-100-PNG
  requirement — keep a countdown and batch the `.cpu()` copies.
- **No stream usage anywhere; H2D is plain `non_blocking` into freshly allocated tensors**
  (`train.py:1430-1431`, `1559-1560`). → Gap: double-buffered `next_batch` prefetch on a copy
  stream is the textbook next step and nobody here has done it.
- **Resume without re-decoding** at sampler level (`data.py:841-844`) or by pre-slicing
  indices (`train.py:1380-1386`: *"iterating N pre-sliced batches (consumed prefix never
  decoded)"*). → Copy for your pause/resume.
- **Hot-loop bookkeeping is plain ints/floats** — no dict lookups, no `time.time()` outside
  the log branch; this is why *their* per-step Python is ~0 (contrast section 5).

## 7. DINOv3-specific notes

- **Frozen, and `train()` overridden to keep it frozen** — unless checkpointing is on:
  `_freeze()` = `requires_grad_(False)` + `.eval()` (`backbones.py:438-441`); `Branch.train()`
  re-pins `self.tower.eval()` (`443-461`); `PerceptualModel.train()` re-applies it
  (`model.py:627-632`) so a stray `model.train()` cannot unfreeze. → **Directly yours** (same
  tower, same requirement), including the trap that grad-ckpt demands train mode (`450-455`).
- **Readout is CLS token 0, registers explicitly accounted for** (`backbones.py:552-554`;
  `595-625` reads `num_register_tokens` off the tower config — *"4 for dinov3-vitb16 … so a
  different checkpoint cannot silently shift the slice"*). Multi-depth readout comes from
  **one** pass with `output_hidden_states=True` (`567-593`). → **Medium**: if you ever
  concatenate layers, one pass gives you several, for free.
- **Resolution: DINOv3 needs nothing** (`backbones.py:362-394`, `263-265`): no position-table
  resize, no `interpolate_pos`, no baked-pos cache for `kind=="cls"` — *"DINOv3 returns False
  here (RoPE) … accepts any size"*. All that machinery is for SigLIP2/MetaCLIP2. →
  **Confirms your 512² needs zero position surgery; do not port `baked_pos`.**
- **`pos_embed_rescale` (stochastic RoPE coord jitter) is a first-class knob**
  (`train.py:282-287`; `backbones.py:396-402`): *"True = keep it ON (positional regularizer);
  the towers are then stochastic during TRAINING ONLY"*. → **Medium**: free regularizer, and
  your 10k-image regime is where overfitting binds. Worth one A/B.
- **One grid per batch, always** — the whole 544 class exists to keep token counts uniform
  (`data.py:834-836`). → Keep 512² square, never mix shapes.
- **"Fewer, larger forwards always beat chunking."** Their Qwen path chunked to dodge
  quadratic attention (`backbones.py:529-540`: *"a 120-image native-512 forward = 123k tokens
  wedged a 24 GB card"*), then found the per-image split was pure waste: **chunk 1→60 gave
  14 s → 5.5 s/step and 15.24 → 12.86 GB** (`MEMORY.md:56-60`). → **High signal for you**:
  chunk only at a memory wall, and you are 11–12 GB into 24 GB.
- **No classification head exists in this repo.** The model is a cosine metric: concat branch
  embeddings → L2-normalise → `1 − cos` (`model.py:519-565`, `597-621`); `readout_layers` /
  `readout_mode` / `attention_pool` are depth and pooling selectors, not logits. →
  **Plainly: nothing to steal for a 1536→12,476 head** — no sampled softmax, no logit
  chunking, no label-sparsity tricks. Your head is 10× their entire trainable count. The only
  transferable advice is section 5's (one dense tensor; verify the adv_optm tags).

## 8. Measured numbers found in the repo

| # | number | what it measures | source |
|---|--------|------------------|--------|
| 1 | **60 → 5.5 s/step (11×)**, peak 15.24 → 12.86 GB | packed batch forward + OFT rotation cache + `forward_pair` | `MEMORY.md:56-66` |
| 2 | 14 s/step → ~5.5 s/step | chunk 1→60 packing alone (Qwen tower) | `MEMORY.md:57-60` |
| 3 | ~80 calls/step × 108 modules ≈ **10k redundant kernel chains/step** | what the rotation cache removed ("the GPU-idle signature") | `oft.py:184-196` |
| 4 | **0.43 → 1.14 Mpx/s**, peak 19375 → 15240 MiB | A/B: `compile(model)` vs `compile(model.forward_pair)`+cache, same step 74→75, same run | `scratch/sophie/rotasim/abit_phaseA.log` vs `abit_phaseB2.log` |
| 5 | 0.687 steps/s, ~14.0 Mpx/s, peak 5056 MiB | steady state, b60 × 288², blk32 | `runs/20261003-183132-rotasim-fgresq60-288-blk32-qkvoupdown-seed1234/steps.jsonl` |
| 6 | 1.276→1.456 Mpx/s (**+14.1%**) / 1.290 (+1.1%) / 1.228 (−3.8%) / no-op | `NVIDIA_TF32_OVERRIDE=1`+benchmark / benchmark alone / `cudaMallocAsync` / `expandable_segments`, batch 8 | `docs/startupflags-costs.md:4-11` |
| 7 | 9 → 13 GB/s | H2D with `pin_memory`; `non_blocking` degrades to sync on pageable memory | `data.py:903-905` |
| 8 | 24/25 s vs **30 s** (u8); eval 21.2 vs 24.4 s | fp32 288 cache vs uint8 cache vs none, 15 train steps, identical accuracy | `pixel_cache.py:67-74` |
| 9 | 1214 → 750 MiB per tower | grad-ckpt silently OFF in eval mode vs engaging in train mode | `backbones.py:450-455` |
| 10 | 4316 / 5639 / 6959 MiB (b12p/b16p/b20p @288²) | DINOv3-B VRAM ladder; b24p fits, b28p/b32p OOM | `MEMORY.md:629-632` |
| 11 | ~30% compute, 0.4 GiB/triplet @544 | price and saving of tower gradient checkpointing | `train.py:278-281`, `backbones.py:304` |
| 12 | drift **1264** (bound 2), `‖RᵀR−I‖` **0.156** | truncated Neumann diverging vs exact Cayley for OFTv2 | `train.py:186-195` |
| 13 | 11.06× / 7.7460× | SOFT's hidden effective-LR division (blk32 / blk16) and the compensation applied | `train.py:389-405`, `MEMORY.md:621-622` |
| 14 | ~16 s worker respawn; "16 batches of host commit" | loader teardown that wedged a run (~30 s/batch after, 65% of wall) | `train.py:1310-1318`, `data.py:907-913` |

Items 1–2, 4–6, 9–11, 14 are **quotations from the repo's own logs/docs**; each was traced
to a file and line and the code path confirmed, but I ran no benchmarks (read-only audit).

## 9. Ranked transfer list for the DINOv3 tagger

| # | technique | expected gain for us | effort | risk | concrete change |
|---|-----------|----------------------|--------|------|-----------------|
| 1 | **Compile forward only; dynamo-disable the mutable cache fetch** | **High** — attacks the 69 ms compile gap *and* the recompile-limit crash | S (1–2 h) | Low | `fwd = torch.compile(model.forward)` bound once; `torch._dynamo.config.suppress_errors = True`; `@torch._dynamo.disable(recursive=True)` on any getter returning a cached/versioned tensor (rotation R, EMA) |
| 2 | **OFTv2 rotation cache: precompute R once/microbatch, invalidate after backward** | **High** — kills ~108 × (Cayley + einsum) redundant chains/step; part of their 11× | S | Low (bit-identical loss if invalidated right) | port `precompute_rotations`/`invalidate_rotation_caches` (`oft.py:272-291`); call before the forward group and after `loss.backward()`; guard with `weight._version` |
| 3 | **Keep the optimizer eager — stop compiling SinkSGD_ADV** | **High** — their answer to your exact crash; removes 88 ms from the compile-critical path | XS | Low | leave `compiled_optimizer=False`; compile the forward instead; if you must, `suppress_errors=True` + `dynamic=True` |
| 4 | **fp32 island for the OFT apply/solve; fp32 adapter weights; blocked einsum** | **Medium-high** — numerics insurance + no bf16 on the lowest-intensity op | S | Low | autocast `enabled=False` around Cayley + `"...bk,bkc->...bc"` einsum (`oft.py:243-260`); `oft_weight_dtype=float32` |
| 5 | **Micro-batch + grad accumulation over the loss SUM instead of checkpointing** | **Medium-high** — batch-64-equivalent optimisation at b32 memory with no compute tax | S | Low | slice groups, sum the loss, one `opt.step()` (`train_rotasim.py:551-557`); you already have `grad_accum` |
| 6 | **uint8 over PCIe, dequant on GPU** | **Medium** — 4× fewer H2D bytes, no worker float pass | XS | Low | `x.to(dev, non_blocking=True).float().div_(255.0)` post-H2D; never dequantise in workers |
| 7 | **Audit adv_optm per-Parameter tagging; keep the 19.2M head as one 2-D tensor** | **Medium** — best honest explanation of 88 ms/step that ignores its own knobs | S | Low | copy the `tag_peft_parameters` census (`tagging.py:54-124`); assert every trainable tensor is tagged or deliberately generic |
| 8 | **Windows discipline: no in-loop `empty_cache()`, guarded `anti_creep`, nw=4, persistent workers** | **Medium** — avoids a catastrophic stall rather than raising steady-state rate | XS | Low | delete in-loop `empty_cache()`; trim only when reserved−peak > 2 GiB; log `host_commit_pct` (`train.py:771-803`) |
| 9 | **`NVIDIA_TF32_OVERRIDE=1` + `cudnn.benchmark=True`** | **Low-medium** — +14.1% only where fp32 GEMMs remain (optimizer math, fp32 island) | XS | Very low (A/B loss on fp32 paths) | set env + flag, then A/B |
| 10 | **Sampler-level resume (deterministic plan, slice the consumed prefix)** | **Low-medium** — makes your pause/resume free and exactly reproducible | S | Low | materialise the epoch plan from `(seed, epoch)` before workers spawn; `skip_batches` (`data.py:841-844`) |
| 11 | **Load the frozen backbone in bf16 instead of fp32+autocast** | **Low-medium**, unmeasured here | S | Medium (A/B accuracy; frozen tower only) | `from_pretrained(..., dtype=torch.bfloat16)` for the tower; adapters stay fp32 |
| 12 | **Single-pass multi-layer readout (`output_hidden_states=True`)** | **Low** now, **medium** if you add layer concat | XS | Low | one forward, index `hidden_states[l][:, 0]` (`backbones.py:567-593`) |
| 13 | **Stochastic RoPE jitter (`pos_embed_rescale`) as a regularizer** | **Low-medium** in a 10k-image regime | XS | Medium (recipe change; not bit-reproducible in train mode) | `tower.config.pos_embed_rescale = 2.0` in training, `None` for eval (`backbones.py:396-402`) |
| 14 | **Remove per-step `.item()` syncs** | **Low** (~1–2 ms/359 ms) | XS | Low | accumulate on device, read once per log tick |
| 15 | **Offset-indexed shard manifest; raw uint8 blobs + memmap** | **Low** — you are already at 6 ms/step | S | Low | `key → (shard, offset, len)` manifest + fd cache (`shardio.py:26-44`); skip `torch.load` per record |
| 16 | **Async checkpoint write** | **Low-medium** — removes a periodic stall; your artifact is far bigger than their 5 MB | S | Low | write to a temp name on a background thread, rename on completion; drop the second `copy2` |
