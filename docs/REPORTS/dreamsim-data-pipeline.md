# Data pipeline

`num_workers=4` default; 8 is known-bad (`train.py:97`; `MEMORY.md:622`): WDDM/host-commit pressure, not CPU. → Low (6 ms/step)
`pin_memory` + `non_blocking=True` everywhere (`data.py:891-905`: "Measured 9→13 GB/s … non_blocking silently degrades to synchronous on pageable memory"; H2D `train.py:1430-1431`, `1559-1560`, `train_rotasim.py:504-506`). → Medium-high if any H2D is pageable
`persistent_workers` + `prefetch_factor=1` are WDDM mitigation (`data.py:907-913`, defaults off `train.py:260-261`); teardown/re-priming costs 16 batches of host commit. → Low
Epoch-boundary hygiene: `cuda_settle()` = `synchronize()`+`gc`, no `empty_cache()` (`train.py:1258-1264`); epoch e+1's loader built in a background thread during epoch e (`1306-1328`, `1395-1408`), so the ~16 s respawn is off the critical path
Caches, gated: per-image `.pt` decode cache (`data.py:233-259`); 288 cache checked for bit-identity against a live decode+resize (`pixel_cache.py:193-224`) plus a directory method token (`118-124`). → The gate transfers, the cache does not
fp32 vs uint8 cache, measured (`pixel_cache.py:67-74`): "training 24 s vs 25 s (inside noise, GPU-bound), eval wall 21.2 vs 24.4 s … u8 is 4× smaller but SLOWER on the training path (30 s — dequantising costs more than the decode+resize it saves)". Confound: that u8 path dequantises on the CPU (`187-190`). → Keep uint8, convert after H2D
No `sampler`; custom `batch_sampler`s (`AspectBatchSampler`, `data.py:831-888`): deterministic in `seed+epoch`, plan built in `__iter__` before workers spawn, `skip_batches` slices the consumed prefix so a resume decodes nothing (`841-844`). → Medium for resume; plan is a pure function of `(seed, epoch)`
Bucketing beats accumulation (`data.py:834-836`): "bucketing or per-sample forwards with accumulation (strictly worse: forfeits batched attention)". → Batch-32 over grad-accum
Successor loop drops `DataLoader`: offset-indexed shards (`shardio.py:26-44`: 32 B sha + u64 offset
  u32 length), fd cache LRU-8 (`rotasim_data.py:33-60`), synchronous `pread` + `torch.load(BytesIO)`
`stack` (`131-157`). → Low at seconds/step, but that manifest suits your `shard#####-N.pt` cache — store raw uint8 blobs so `np.memmap` slices skip `torch.load` per record
