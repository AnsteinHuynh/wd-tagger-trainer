# WHY ALL-OFT TRAINING RUNS AT ~101 img/s: the adapter-backward cost, measured

2026-10-27 evening. The user expected 150-160 img/s for DINOv3-B at 512 on the 4090; three pilots, two benches and one controlled experiment attribute every missing img/s to a single autograd fact. Read this before promising or hunting throughput for any run that carries OFT adapters

## The numbers, one configuration each

All on DINOv3-B/16 at 512 squash, u0 (no unfrozen blocks), batch 192 x 2 accum, exact Cayley, compile default, seed 7, unless the row says otherwise

| what | img/s | detail |
| --- | --- | --- |
| full trainer, with fp32 siglip feat cache | 101 | run faa09caa, steady steps 20-500 |
| full trainer, no cache (768 head) | 101 | pilot 84d2d37b: the frozen second view costs ~0 |
| pure GPU step, synthetic input, no disk | 103 | bench_step: fwd 604 + bwd 1245 + opt 8 ms per 192 |
| stream alone, no model | 271 | bench_stream: 3.68 ms/img, 8 decode threads off X: |
| compile max-autotune-no-cudagraphs | 113 | pilot 84d2d37b: +12%, VRAM 14.4 GB, tuner ~8 min |
| u1 at batch 64 x 6 (the record's production shape) | 99-103 | benchmark.md J2, run 62902e8d |
| adapters requires_grad=False | 324 | bench_step freeze_oft probe: fwd 592 + bwd 0.5 ms, VRAM 3.1 GB |

## The diagnosis

The adapters cost ~2% in forward (592 vs 604 ms; rotations and exact Cayley solves are cheap). The entire gap is backward: 1245 of 1857 ms. Setting requires_grad=True on the 73 OFT tensors puts the whole 12-block tower inside the autograd graph, because the adapter in block 0 cannot receive its gradient without differentiating back through every block above it. Gradient checkpointing then makes backward pay the forward AGAIN: bwd 1245 = recompute ~592 + vector-Jacobian products ~2x592. The activation memory (10 GB of the 13.2 peak) exists only to serve this backward; the frozen probe runs at 3.1 GB

Consequences that follow and are all measured: depth is speed-neutral (u0 = u1 = u12 within noise, because the traversal is owned by the adapters, not the unfrozen weights); the fp32 feature cache and the 1536-wide head cost nothing measurable; the tar stream has 2.7x headroom and is never the limit while the step is compute-bound; compile autotune buys 12% and no compile mode can buy more because the ratio is structural

The historical 150-160 figures are real but are other animals: benchmark.md B measured compile ~150 on an old batch-32/block-32 launch-bound config, and the 160 in section G is the no-adapter wide-head reference. A tower with zero trainable params inside it does 324 img/s in this bench

## The one honest lever: partial-block placement

Backward descends only to the LOWEST adapted block. Adapters in blocks 6-11 leave blocks 0-5 out of the graph: no recompute, no VJPs, no stored activations for them. Projection: bwd 1245 -> ~620 ms -> ~155-165 img/s at default compile. Unmeasured as of writing; the accuracy question (does top-half adaptation carry DINOv3) is open and needs a 1000-step screen with a validation read. Do not quote 155 as a measurement

## Traps hit while measuring

PowerShell `$_.Value -eq ""` loose-equals 0 and $false: a config rewrite meant to drop empty strings silently deleted every zero-valued key, including unfreeze_blocks, and a "u1 pilot" ran as u0. Hand-write pilot configs or use strict comparison. Caught because the run banner prints the unfrozen count; always read the banner, not the intent

unfreeze_blocks > 0 changes grad_checkpoint_blocks semantics: "true" wraps only the UNFROZEN blocks ("checkpointing on the last 1 of 12"), not all of them. u1 at batch 192 then runs 11 unwrapped blocks of activations into the 22.5 GB cliff and looks alive while crawling. At depth u>0 with big micro-batches, set grad_checkpoint_blocks to the full block count explicitly

TrainConfig silently drops unknown config keys (it prints one "ignoring unknown key(s)" line). feat_val_cache was dropped this way and only the banner line "[val] frozen view from ..." proves the key took effect. A config whose only difference is an unknown key also hashes to the same run dir

## The instrument

scratch/sophie/bench_step.py --config <cfg> --batch 192 --steps 10 --compile --opt: full step split into fwd/bwd/clip/opt, synthetic input, no disk, honors freeze_oft. scratch/sophie/bench_stream.py --batches 60: stream-only ceiling. The freeze_oft probe (requires_grad=False on everything matching "oft") is the cleanest isolation of adapter-backward cost from everything else; a bwd under 1 ms with fwd unchanged proves the graph, not the math, is the price
