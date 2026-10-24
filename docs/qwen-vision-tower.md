# Qwen3.8-27B vision tower as a backbone (mmproj route, no language model)

Vision encoder only, no LLM, no token generation, OFT on the tower like any backbone; `dreamsim-dinov3` did this and their numbers are the best prior. Found 2026-10-22; sister project `C:\Users\kerok\apps\dreamsim-dinov3`

## The mmproj is the tower, not a quantized VLM

`I:\MyApps\LLAMA-Models\qwen38-27b\mmproj-Qwen3.8-27B-bf16.gguf` (888 MiB) is the vision tower + merger and nothing else:

> 334 GGUF tensors -> 333 HF params, 460730096 params, 0 unmapped / missing / extra / shape-mismatched, `load_state_dict(strict=True)`. Every mismatch RAISES, deliberately: a silently mis-mapped tower still produces plausible embeddings and never crashes.

The 55.6 GB / 18-shard `X:\unique\models\Qwen3.8-27B` repo is not needed for the image path: no tokenizer, no `<image>` tokens, no decode, no llama.cpp. The full VLM (`llama-server` + mmproj, ~3.85 s/image, 19-20 GB VRAM) is a slower tool

## Loading it

```python
import sys; sys.path.insert(0, r"C:\Users\kerok\apps\dreamsim-dinov3\src")
from dreamsim_oft.qwen_vision import load_vision_tower, qwen_vision_embed, POOL_DIMS

tower, info = load_vision_tower()                     # bf16; dtype=float32 => fp32 math
emb = qwen_vision_embed(tower, x01, pool="merged")    # (B, 5120);  "premerge" -> 1152
```

`dreamsim-dinov3\src\dreamsim_oft\qwen_vision.py`, loading into `transformers.Qwen3_5VisionModel` (`C:\Python314` has 5.16.1, in `CONFIG_MAPPING`; the OneTrainer venv has 5.5.4). `QWEN_TOWER_PATH` takes the mmproj `.gguf` or an HF dir (`_load_vision_from_hf_dir`, strips `model.visual.`, 1:1). Config: 27 blocks, hidden 1152, 16 heads, intermediate 4304, patch 16, spatial-merge 2, out 5120, `num_position_embeddings 2304` (48x48, bilinear per grid + 2D RoPE => no position surgery). `POOL_DIMS = {"merged": 5120, "premerge": 1152}`; frozen 288px is a dead heat (70.43% vs 70.51%), so pooling is a knob, not a bug; `merged_tokens` 32px / `premerge_tokens` 16px for spatial readouts

## Parameter breakdown (2026-10-22, `scratch/sophie/param_breakdown.py`)

Total 460.7M:

|module|params|
|---|---|
|`patch_embed` (Conv3d 3->1152, 2 frames x 16x16)|1.771M|
|`pos_embed` (48x48 learned table)|2.654M|
|`rotary_pos_emb` (precomputed freqs, no params)|~0|
|`blocks`, 27 x 15.240M|411.467M|
|`merger` (4608 -> 5120)|44.839M|

Per block (15.240M):

|submodule|shape|params|
|---|---|---|
|`norm1`, `norm2`|1152 each|0.004M combined|
|`attn.qkv`|1152 -> 3456, fused|3.985M|
|`attn.proj`|1152 -> 1152|1.328M|
|`mlp.linear_fc1`|1152 -> 4304|4.963M|
|`mlp.linear_fc2`|4304 -> 1152|4.959M|

### Block-size constraint and adapter count

OFT rotates `in_features`; `_adjust_block_size` (mirrors OneTrainer) silently falls back to a divisor. `block_size=32`:

|module|in_features|block used|OFT params|
|---|---|---|---|
|`attn.qkv`|1152|32|17856|
|`attn.proj`|1152|32|17856|
|`mlp.linear_fc1`|1152|32|17856|
|`mlp.linear_fc2`|4304|16 (auto)|32280|
|per block|||85848|
|x 27 blocks|||2317896 (~2.32M)|

~2.32M trainable of 460.7M. `4304 = 2^4 x 269`, so 32 does not divide it; nearest divisor 16 (269 blocks of 16). The sister project's 1571400 does not reconstruct — their wrap surface differed; compute, do not copy

## Attaching OFT

Same `oft_targets` suffix mechanism as DINOv3; only the list and block size change:

|item|DINOv3-B|Qwen tower|
|---|---|---|
|OFT target suffixes|`q_proj,k_proj,v_proj` (default)|`qkv,proj,fc1,fc2`|
|`block_size`|16|32 (`attn.qkv` is FUSED: 1152->3456 = 108x32)|
|OFT params, one arm|~1.3M|1571400|

`attn.qkv` is fused: add that suffix to `inject_oft` or attention gets no adapter while the MLP silently does

## Three silent-failure traps

All produce plausible output rather than an error:

Patch-embed `Conv3d` is stored as TWO temporal-frame tensors -> `torch.stack(dim=2)`. Order-invariant for static images (the processor duplicates the frame), so "frames bit-identical=False" in the log is EXPECTED

Patch order is MERGED-BLOCK `(block_row, block_col, intra_row, intra_col)`; raster order is wrong by up to 2.0 on [-1,1] pixels and never crashes. Validated against the official Qwen2-VL processor: max error 5.9e-08 at 288/544/224

Non-multiple-of-32 sides must resample UP before building `grid_thw`, or the tower dies on length mismatch

## Measured result on the sister project's task (FGResQ 2AFC, n=2509)

|arm|frozen val|trained val / test|wall|VRAM|
|---|---|---|---|---|
|DINOv3-B + OFT|71.18%|73.54 / 72.76|253.7 s|6961 MiB|
|Qwen3.8 tower (460.7M) + OFT|72.14% (+0.96pp, CI crosses zero)|71.86 / 71.16|794.8 s (3.13x)|15452 MiB (2.2x)|

Frozen: dead heat. Trained: worse on both splits: "the 460.7M-param vision encoder is statistically indistinguishable from the 86M-param DINOv3-B, at ~5x the per-image compute." Closed for that project

## Cost: WITH PACKING, ABOUT THE SAME AS ViT-L

Ignore the superseded projection (80 ms/img, ~12 img/s); a measured 11x engineering win replaced it:

> QWEN SPEED/VRAM LAW (2026-10-04): HF `Qwen3_5VisionAttention` splits per image via `cu_seqlens` in every code path, so a packed full-batch forward never leaks across images; `QWEN_FWD_CHUNK=1` was pure waste. chunk 1->60 (one packed 60-image call): 14 s/step -> ~5.5 s/step, peak 15.24 -> 12.86 GB; 60 s/step -> 5.5 s/step (11x) with the OFT rotation cache and `forward_pair`.

Both sister arms ran 180 img/step, so the Qwen arm is 180 / 5.5 = ~33 img/s, same as our ViT-L (33 img/s @ batch 20), at 12.86 GB peak

|option|params|img/s|one pass over 5.32M|
|---|---|---|---|
|DINOv3-B + OFT (current)|86M|~160 (ours, single-image)|~9.2 h|
|ViT-L + OFT|303M|33|~44.8 h|
|Qwen tower + OFT, packed|460M|~33|~44.8 h|

ViT-L-cost at 1.5x the parameters; a 5k-step probe is about an hour

Smoke test 2026-10-22 (`scratch/sophie/qwen_tower_smoke.py`, 500 val images, chunk 16, bf16, 512px):

|measurement|value|
|---|---|
|tower load from the mmproj GGUF|30.6 s|
|strict load|460730096 params, 334 -> 333 tensors, 0 unmappable|
|forward @512px|60.1 img/s, 16.6 ms/img|
|peak VRAM, tower only|1.79 GB (1.27 GiB at 8 images, as banked)|
|embeddings|(500, 5120), norm 66.8, std 0.137|
|centroid AUC on 6 frequent tags|0.648 - 0.820|

Ran on `C:\Python314` (transformers 5.16.1) though the sister project used 5.5.4, so `Qwen3_5VisionModel` is stable across that range; forward-only 60 img/s implies low-20s img/s fwd+bwd without their wins: ~70 h naive, ~45 h optimized. At 1.79 GB, OFT + head + optimizer states are the VRAM limit

The centroid AUC is not evidence: zero-parameter readout, no control arm; it needs DINOv3-B on the same 500 images first

### Resolution scaling (2026-10-22, 200 images, constant ~16k packed tokens)

|res|patches|chunk|img/s|ms/img|VRAM|emb norm|
|---|---|---|---|---|---|---|
|512|1024|16|67.5|14.8|1.79 GB|67.3|
|384|576|28|124.1|8.1|1.78 GB|71.7|
|256|256|64|210.4|4.8|1.80 GB|76.4|

256px is 3.1x faster than 512px (4x fewer patches, larger chunk), VRAM flat at ~1.8 GB; embedding norm rises with resolution (67.3 -> 76.4), so features are not simply rescaled

Extrapolated at fwd+bwd ~3x forward: 256px ~21 h/pass, 384px ~35 h, 512px ~65 h over 5.32M, versus ViT-B at ~160 img/s / ~9.2 h — the 460M tower at 256px is ~2.3x the current backbone

Open question: whether 256px keeps enough detail for small-object tags (ribbons, earrings, text, hand gestures). Merged tokens at 256px are an 8x8 grid; only a frozen-feature control against DINOv3-B at the same resolution settles it

For 33 img/s: one `grid_thw` call (`QWEN_FWD_CHUNK` large, not 1), OFT rotations cached per microbatch outside checkpointing, compiled forward only, fp32 Cayley solve, bf16 tower weights. CAUTION: mixed-size batches must score per-image (chunk=1), since packed `grid_thw` from mixed sizes breaks rotary sizing; native inputs must resample up to a %32 multiple, grid built from the SAME size

## Why one experiment is still worth it

The sister task is pairwise similarity, barely sensitive to backbone width. Ours is 12476-way multi-label classification, where the head must carve thousands of semantic directions; a wider tower (5120-d merged vs our 3072-d pool) could matter more

Cheapest test before ~123 h: frozen-feature linear probe — same subset, same linear head on Qwen-tower and DINOv3-B features, scored at 0.40 on the 50k nested cache. No OFT, no full pass, a few hours. Better separation justifies the long run; a match or a loss as for DreamSim costs a few hours instead of five days

## That test was run on 2026-10-22: closed for our task

Frozen features from the mmproj tower, 512px, merged pooling, 5120 dimensions, 60k images in 19 minutes through `scratch/sophie/siglip_probe.py`. Every row is the same head recipe: 10k train cache, 4000 steps, Adam 1e-3, ASL gamma 4.0, scored on the 50k nested val at 0.40. `H` is one hidden layer of that width, no H is linear.

| features | head | macro F1 | micro F1 |
|---|---|---:|---:|
| SigLIP2-B/16-512 | H768 | 0.0457 | 0.1676 |
| DINOv3-B/16 | H768 | 0.0380 | 0.1754 |
| Qwen3.8 tower | H768 | 0.0291 | 0.1438 |
| Qwen3.8 tower | linear | 0.0337 | 0.1576 |
| DINOv3 + SigLIP2 | H768 | 0.0508 | 0.2139 |
| DINOv3 + SigLIP2 + Qwen | H768 | 0.0425 | 0.1793 |
| SigLIP2 + Qwen | H768 | 0.0389 | 0.1656 |

Weakest of the three at every head shape, including a 64M linear head, so this is not a capacity artefact. Worse than that, it dilutes: adding it to the DINOv3 plus SigLIP2 pair costs 16% macro and 16% micro, and pairing it with SigLIP2 alone scores below SigLIP2 alone, because 5120 of 6656 input dimensions belong to the weakest signal and a fixed-budget head spends most of its weights there.

The wider-tower hypothesis this document raised is refuted for tag prediction: 5120 dimensions of Qwen3.8 features are worth less than 768 from either other tower, on a task where the head must indeed carve thousands of semantic directions. The reason is probably that the merger output is trained to be read by a language model rather than to be linearly separable per tag, but that is a guess.

Open, and cheap: only 512px merged was measured. Premerge 1152, 384px and 256px were not, and the spatial-merge step is the most likely place tag detail is lost. If someone wants to reopen this, that is the four-cell sweep to run first, not a training arm.

Cost avoided by closing now: 54.5 GB of cached vectors and about 25 hours of extraction at 512px, or 12 hours at 384.

## Environment gotchas

bitsandbytes is unusable here: it ships up to `libbitsandbytes_cuda130.dll`, torch is cu132, so 4-bit loading the full VLM dies with "Configured CUDA binary not found". No `BNB_CUDA_VERSION` workaround exists; the GGUF route sidesteps it

Existing launchers (`startllamaservergpu\src\start_server.py`, `VLM-imageArena\src\config_local.py`) hardcode `Qwen3.8-27B-Q5_K_M.gguf`, absent from disk; the real file is `Q5_K_S`. One path patch before either runs
