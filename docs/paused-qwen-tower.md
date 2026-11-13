# PAUSED: Qwen3.8-27B vision tower as a backbone for the tagger

Status: paused 2026-10-22, before any training. Nothing is running and nothing was launched. The idea is fully costed below; resuming is mechanical.

Paused because a more fundamental question came up first: whether the head matters at all. The tower experiment answers "does a different 460M backbone help", which only makes sense once we know the deficit is a backbone/representation problem rather than a head problem. Pausing also avoided spending hours on OFT-on-a-foreign-tower while the head question was open.

## The idea

Use the vision encoder of Qwen3.8-27B as a replacement backbone for `DinoTagger`, with OFTv2 adapters trained on it, exactly as our DINOv3-B runs do. No language model, no token generation, no llama.cpp at inference. The 888 MB mmproj GGUF contains the vision tower + merger and nothing else.

## What is already measured and known (do not redo)

Loads and runs in our env. `C:\Python314`, torch 2.16 cu132, transformers 5.16.1. `dreamsim_oft.qwen_vision.load_vision_tower()` strict-loads 334 GGUF tensors -> 333 HF params, 460730096 params, 0 unmappable. ~30 s from the mmproj on I:
Forward-only throughput at 512px: 67.5 img/s (16.6 ms/img) at chunk 16, peak VRAM 1.79 GB. At 384px: 124.1 img/s. At 256px: 210.4 img/s (chunk 64)
Parameters per layer, exact: 27 blocks x 15.240M = 411.467M; merger 44.839M; patch_embed 1.771M; pos_embed 2.654M. Per block: qkv 3.985M (1152->3456 fused), proj 1.328M, fc1 4.963M (1152->4304), fc2 4.959M, norms 0.004M
OFT surface at block_size 32 is 2317896 params (~2.32M) across 108 modules. `fc2`'s 4304-d input is not divisible by 32, so `_adjust_block_size` silently uses 16 there (269 blocks); the other three use 32
Three silent-failure traps in this path (documented in `docs/qwen-vision-tower.md`): patch-embed `Conv3d` stored as 2 temporal frames, patch order MERGED-BLOCK not raster, and non-%32 sides must resample up before building `grid_thw`
Chunking is mandatory: the tower packs a batch into ONE sequence, so attention is quadratic in `batch * patches`; 120 images at 512px (123k tokens) wedged a 24 GB card

## Why it might not work (the honest prior)

The sister project `dreamsim-dinov3` already tried this tower as a backbone on a closely related visual task (FGResQ 2AFC preference) and it lost after OFT training: 71.86/71.16 vs DINOv3-B's 73.54/72.76, at 3.13x wall and 2.2x VRAM. Frozen it was a dead heat (+0.96pp, CI crossing zero). Their verdict: "QWEN PIVOT CLOSED".

Counter-argument for still trying: their task is pairwise similarity, a relative judgement that may be insensitive to feature width; ours is 12476-way multi-label classification, which must carve the space into thousands of distinct semantic directions and could reward a wider (5120-d) representation more.

## Decisions still open (must be settled before launch)

1. Head shape. Options: (a) project 5120->768 and reuse `cls_mean_max_std` (head stays 38.33M, matches our ViT-B head exactly); (b) bar-analogue: mean-pool 5120 -> LayerNorm -> dropout -> `Linear(5120,12476)` = 63.9M; (c) mean+max+std over the 8x8 token grid -> 15360 -> 191.6M.
2. Tower frozen or partly unfrozen. One block is 15.24M, so unfreezing 6 blocks adds 91M trainable against OFT's 2.32M. Recommendation was frozen + OFT only.
3. Resolution. 256px is cheap (~21 h/pass estimated) but off-distribution (native 768) and gives only an 8x8 merged token grid — a real risk for small-object tags. 512px is ~65 h/pass.

## To resume

1. Read `docs/qwen-vision-tower.md` (loader, OFT surface, traps, cost).
2. Settle the three decisions above.
3. Write a `TowerBackbone` adapter in `wdtag/model.py` exposing `forward(pixel_values) -> pool vector`, plus a `backbone="qwen"` branch in `TrainConfig` so `validate.py` / `load_run_model` can rebuild it (a tower checkpoint cannot load through the DINOv3 path).
4. Stage a 5k-step probe like `configs/probes/c2_vitl_fresh_5k.json` (~15 min at 256px, ~50 min at 512px).
5. Score on the 50k nested cache at 0.40 with `python -m wdtag.validate`, same protocol.

## Related files

`docs/qwen-vision-tower.md` — the full mechanism, measurements, traps, and cost tables
The probe scripts that produced the load/speed/resolution/parameter numbers (`qwen_tower_smoke.py`, `qwen_tower_res.py`, `param_breakdown.py`) were deleted in the 2026-10-22 automation cleanup; the numbers and the method that produced them are recorded in `docs/qwen-vision-tower.md`
Sister project: `<sister-project>`, `src/dreamsim_oft/qwen_vision.py` and `scratch/sophie/qwen38_frozen/FINDINGS.md`
