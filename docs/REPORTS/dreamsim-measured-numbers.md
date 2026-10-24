# Measured numbers found in the repo

| # | number | what it measures | source |
|---|--------|------------------|--------|
| 1 | 60 → 5.5 s/step (11×), peak 15.24 → 12.86 GB | packed batch forward + OFT rotation cache + `forward_pair` | `MEMORY.md:56-66` |
| 2 | 14 s/step → ~5.5 s/step | chunk 1→60 packing alone (Qwen tower) | `MEMORY.md:57-60` |
| 3 | ~80 calls/step × 108 modules ≈ 10k redundant kernel chains/step | what the rotation cache removed ("the GPU-idle signature") | `oft.py:184-196` |
| 4 | 0.43 → 1.14 Mpx/s, peak 19375 → 15240 MiB | A/B: `compile(model)` vs `compile(model.forward_pair)`+cache, same step 74→75, same run | `scratch/sophie/rotasim/abit_phaseA.log` vs `abit_phaseB2.log` |
| 5 | 0.687 steps/s, ~14.0 Mpx/s, peak 5056 MiB | steady state, b60 × 288², blk32 | `runs/20261003-183132-rotasim-fgresq60-288-blk32-qkvoupdown-seed1234/steps.jsonl` |
| 6 | 1.276→1.456 Mpx/s (+14.1%) / 1.290 (+1.1%) / 1.228 (−3.8%) / no-op | `NVIDIA_TF32_OVERRIDE=1`+benchmark / benchmark alone / `cudaMallocAsync` / `expandable_segments`, batch 8 | `docs/startupflags-costs.md:4-11` |
| 7 | 9 → 13 GB/s | H2D with `pin_memory`; `non_blocking` degrades to sync on pageable memory | `data.py:903-905` |
| 8 | 24/25 s vs 30 s (u8); eval 21.2 vs 24.4 s | fp32 288 cache vs uint8 cache vs none, 15 train steps, identical accuracy | `pixel_cache.py:67-74` |
| 9 | 1214 → 750 MiB per tower | grad-ckpt silently OFF in eval mode vs engaging in train mode | `backbones.py:450-455` |
| 10 | 4316 / 5639 / 6959 MiB (b12p/b16p/b20p @288²) | DINOv3-B VRAM ladder; b24p fits, b28p/b32p OOM | `MEMORY.md:629-632` |
| 11 | ~30% compute, 0.4 GiB/triplet @544 | price and saving of tower gradient checkpointing | `train.py:278-281`, `backbones.py:304` |
| 12 | drift 1264 (bound 2), `‖RᵀR−I‖` 0.156 | truncated Neumann diverging vs exact Cayley for OFTv2 | `train.py:186-195` |
| 13 | 11.06× / 7.7460× | SOFT's hidden effective-LR division (blk32 / blk16) and the compensation applied | `train.py:389-405`, `MEMORY.md:621-622` |
| 14 | ~16 s worker respawn; "16 batches of host commit" | loader teardown that wedged a run (~30 s/batch after, 65% of wall) | `train.py:1310-1318`, `data.py:907-913` |

Items 1–2, 4–6, 9–11, 14 are quotations from the repo's own logs/docs; each was traced to a file and line and the code path confirmed, but I ran no benchmarks (read-only audit).
