# Reading image datasets off a spinning hard drive

Image formats, tar/shard layouts, feeder design and training-side choices for data on an HDD. Measured on a 5.3M-image Danbooru tagger; the reasoning generalises.

Legend: (measured) = timed on the machine below; (estimate) = arithmetic or experience. Time every number on your own disk — HDDs vary >3x.

Machine: Windows, RTX 4090 (24 GB), 47.9 GB RAM, 16 CPU threads. Dataset: 213 `.tar` shards, 316.8 GB, 5321713 images → ~60 KB/image. Archive: 8 TB WD Blue (WDC WD80EAAZ), SATA, spinning rust, ~45 MB/s sequential (measured). Cache/checkpoints: NVMe.

---

## 1. The physics you are fighting

| access pattern | what it costs | realistic throughput |
|---|---|---|
| Long sequential read | pure transfer, head never moves | 100-250 MB/s (7200 rpm); 45 MB/s measured here |
| 1 MB sequential reads | one seek + transfer | tens of MB/s |
| 60 KB random reads | ~one seek each | 100-150 IOPS = 6-9 MB/s (estimate) |
| 4 KB random reads | pure seek overhead | <1 MB/s (estimate) |

Seek ~8-9 ms at 7200 rpm, ~15 ms on a 5400 rpm laptop drive. Random small-file access buys no bandwidth — only 100-150 chances per second to read anything.

> Convert random access into sequential access, or hide the latency with concurrency. Never let a random-access pattern touch an HDD per image.

### The arithmetic that decides your design

Four numbers, before choosing anything:

```
images/s needed   = batch_size / step_seconds          # set by the GPU
bytes/s needed    = images/s × bytes_per_image_stored
decodes/s needed  = images/s                           # if stored compressed
seeks/s needed    = images/s                           # if stored one file per image
```

Then check each against hardware. Worked example from this project:

```
GPU: 6.7 ms/image  -> need 150 images/s
stored size: 60 KB -> need 9 MB/s        (drive does 45)      OK
decode: need 150/s  -> 8 threads give 250/s                   OK
```

GPU is the bottleneck, drive 20% utilised: the goal state.

Raw 512x512 uint8 = 786 KB/image → 118 MB/s at 150 images/s, 2.6x what this drive gives. Uncompressed shards are I/O-infeasible here: on a slow disk compression is what makes the throughput possible, not a space saving.

---

## 2. Do not store one file per image

Five million small files on an HDD = a 40 images/s pipeline that cannot be debugged.

Seek per file. 8 ms → 150 images/s needs 1.2 s of seek time per second
Metadata — `CreateFile`/`open`, directory lookup, NTFS records: 50-200 us per file (estimate), plus fragmentation as the directory grows
No locality — consecutive samples scattered across the platter; any shuffle is a seek storm
Tail latency — one slow directory block stalls the feeder

Directory structure does not help (2-level fanout still pays metadata per file). Only fixes: a container, or SSD.

Rule: `images × 60 KB` above a few GB on HDD → pack it.

---

## 3. Container formats, ranked for HDD

### 3.1 Recommended: uncompressed tar shards (webdataset-style)

What we use, and what I would reach for again

Sequential by construction — concatenated members, one forward pass, head never reverses
No index — any byte offset, headers parsed forward; resume trivial and exact (§5.3)
Image and annotation adjacent — `key.jpg` then `key.json`, one read for both
Splittable across workers — disjoint shard subsets, zero coordination
Skip without reading — `tarfile` jumps member payloads via the size field, so skipping costs header reads only. (measured: ~2600 samples/s header-only walk of a 1.5 GB tar, vs ~250/s decoding.) That is what makes resume-without-replay cheap

Rules for making tar fast:

Never compress the tars. `tar.gz` decompresses everything before your sample and destroys seekability. Plain `.tar`, or per-member compression (`.zst` per file), never container compression. (Plain-tar read measured ~5800 samples/s, faster than decode.)
Shard 1-4 GB — amortises open/seek, still parallelisable and checkpointable. Ours: 1.5 GB / 25000 images
`--format=gnu`, not `pax` — pax adds extended headers per long filename, bloating the stream and slowing header parsing
Short flat names — `000123.webp`, not `dataset/train/class_a/very/long/path/000123.webp`
Deterministic order inside a shard — any sample's position computable without an index
Companion index (`shard-00001.count`), or lazy counts cached (§5.3). Scanning a tar for counts costs ~10 s per 1.5 GB (measured) — not on every start

### 3.2 Memory-mapped array shards (`.npy`/`.npz`/tensor shards)

What our cache uses: `torch.save`/`npy` files holding a contiguous `uint8[N,3,H,W]` block, opened with `mmap`.

Excellent on SSD — no decode, no copy, the OS pages in exactly what you touch
HDD only uncompressed. 786 KB/image at 512x512 = 118 MB/s at 150 images/s: infeasible at 45 MB/s, trivial on NVMe. The whole corpus would cache to 4.18 TB
Loses JPEG's 10-15x compression — ~13x the space
Fixed shape — mixed resolutions need extra shards

Cache on SSD, never archive on HDD. Here: HDD holds 317 GB of JPEG tars, NVMe a decoded 10k-image cache.

### 3.3 LMDB / database files

B+tree + mmap: good locality, transactional, one big file
HDD reads still random — IOPS-limited unless the pattern is sequential
Readahead + sequential keys fine; shuffled keys seek constantly
Worth it for key-random access (e.g. sampling by label) on SSD

### 3.4 Zip / squashfs / other

Zip — central directory: per-file seek to a local header, then into the data; deflate also decompresses. Acceptable on SSD, mediocre on HDD
squashfs — read-only, random access, block compression. Good for deployment images, still random on HDD
Parquet/Arrow — excellent for labels and metadata (columnar, compressed, splittable, cheap subsets). Not image bytes
In-RAM index — fine if small (int64 offset per image for 5.3M images = 42 MB)

### 3.5 Decision table

`S` = stored bytes per image, `R` = images/s you need:

| situation | choice |
|---|---|
| HDD, R×S well under drive speed, images compressed | uncompressed tar shards |
| HDD, need uncompressed arrays | add a decoded cache on SSD |
| SSD/NVMe, random access by key | array shards or LMDB |
| SSD/NVMe, decoded, fixed shape | mmap'd array shards |
| mixed resolutions | per-resolution tar shards, or one cache per shape |
| labels only | Parquet |

Sanity: `R × S` under ~50% of measured sequential throughput, leaving room for latency spikes and housekeeping.

---

## 4. Image-side choices

### 4.1 Store at the resolution you train at

Decode cost scales with pixels: 4K→512 costs 10-20x a 640-wide JPEG for the same tensor.

Store ~1.0-1.3x training resolution, not 4x "for safety"
Our corpus was pre-downscaled upstream to 640px wide, ~60 KB/image — much of why the pipeline is cheap

### 4.2 Decode tricks

`Image.draft()` — for JPEG, decode at 1/2, 1/4 or 1/8 in the DCT domain. Big win when source is ≥2x target (estimate: 2-4x decode throughput); nothing when source ≈ target, as here
Baseline JPEG, not progressive. Progressive decodes in multiple passes, meaningfully slower (estimate: 1.5-2x). Once GPU-bound, optimise decode, not bytes
Plain JPEG 4:2:0 for photographic/anime art. WebP saves 20-30% bytes but decodes slower — measure both. PNG only for alpha/line art
Normalise at build time — RGB, EXIF orientation applied, profiles stripped. Never pay `convert()` or EXIF per epoch; CMYK JPEGs are especially slow
One resize filter, chosen once. `Image.BILINEAR` fast default, `LANCZOS` slower but sharper. Never mix per-image
Resize in one shot to final size; two-step costs more and blurs more

### 4.3 What to store for the label

Small annotations (a few dozen ids): per-image JSON inside the tar, one stream
Large or relational annotations: separate Parquet/Arrow file, integer id in the tar
Precompute the label encoding (indices, multi-hot) at build time rather than parsing JSON per epoch, if the space is affordable

---

## 5. The feeder

One batch every `step_seconds`, forever, without stalling. Design for latency hiding, not peak bandwidth.

### 5.1 Threads or processes?

Threads, when the work is C/C++ that releases the GIL. PIL decode and resize do: real parallelism, no process overhead.

(measured) On this 16-thread machine, decode+resize to 512x512:

| workers | throughput |
|---|---|
| 1 thread | 55 images/s |
| 8 threads | 250-289 images/s |
| 14 threads (inside training) | 121 images/s — slower |

More threads made it worse: same physical cores as the training process, cache thrash. Pool size `ceil(R / per_thread_rate × 1.5)`, capped at (physical cores − 2).

End-to-end feeder throughput in the loop: 143-156 images/s at batch 48.

Processes (`DataLoader(num_workers=N)`) for Python-heavy work or a GIL-holding library. Costs: memory per worker, pickling, a full spawn per worker on Windows. Pure decode: threads win.

### 5.2 Bounded prefetch, not unlimited

Queue small and fixed (4-8 batches): absorbs ~10 ms HDD latency spikes, no RAM balloon.

Buffer reuse matters as much as buffering:

Fresh `37.7 MB` buffer per step → ~7.5 GB of page faults over 200 steps, forced a synchronous pageable H2D copy
Fix: a ring of preallocated pinned buffers, one per queue slot plus slack. Fill `i`, yield it, move to `i+1`. Constant memory (6 x 37.7 MB), asynchronous H2D
Fresh pinned tensor per batch: 24.3 ms per 48-image batch (measured) — 5x the actual transfer. Pin once, reuse: free

Never clear a buffer after handing it out — the tensor aliases the numpy view. We shipped that bug for one test run and it silently zeroed every label (`loss 0.0000`). Clear the next slot.

### 5.3 Exact resume without replaying data

1. Order is a pure function of (seed, pass). Seeded shard-order shuffle, fixed within-shard order. No dependence on wall-clock, thread scheduling or iteration order.
2. The cursor is derived, not stored. `samples_consumed = global_step × batch_size` — a checkpoint already holds everything needed.

Resume maps a global sample offset to `(shard, offset_in_shard)`, needing per-shard counts (expensive: 10 s per 1.5 GB tar (measured)). In order of preference:

Store a tiny count/manifest file alongside the shards at build time. Do this
Learn counts lazily on first resume, cache incrementally so each later resume is cheaper
Never decode-and-discard to skip. Skipping reads headers only

### 5.4 Shuffling without seeking

This is where people destroy their HDD pipeline: shuffle a global index, then seek to each sample.

Shuffle the shard order, not the sample order. Zero cost, perfect locality
Bounded in-memory shuffle buffer (1000-4000 samples) for more mixing: fill from the current shard, emit randomly
Accept coarser-than-true-random "epoch" shuffling. Above ~1M samples it does not matter — the buffer plus shard-level shuffling decorrelates far more than SGD needs

### 5.5 A minimal feeder skeleton

```python
def produce(shards, bs, ring, counts, skip=0):
    order = seeded_shuffle(shards, seed, pass_index)
    start_i, start_off = locate(order, counts, skip)
    slot = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i in range(start_i, len(order)):
            it = iter_tar(order[i], start_off if i == start_i else 0)
            for chunk in chunks(it, 256):                 # batch the decode calls
                decoded = list(pool.map(decode_resize, chunk))   # GIL released here
                for img, labels in zip(chunk, decoded):
                    ring.x[slot][fill] = img                # pinned, preallocated
                    if fill == bs:
                        yield ring.x[slot], ring.y[slot]
                        slot = (slot + 1) % len(ring)
                        ring.y[slot][:] = 0                 # clear the NEXT slot

def batches(...):
    q = Queue(maxsize=4)
    threading.Thread(target=lambda: [q.put(b) for b in produce(...)], daemon=True).start()
    while (b := q.get()) is not None:
        yield b
```

The producer thread overlaps HDD latency and decode with GPU work; a plain generator only advances when the consumer calls it.

---

## 6. Training-side choices that interact

### 6.1 First determine whether you are I/O-bound at all

Cheap test: a few hundred steps, same model, synthetic batch (random or zeros, no loader). Same steps/sec → compute-bound, no feeder work helps. (measured: our GPU does 6.7 ms/image at ~505 GFLOP — ~75 TFLOPS, ~45% of a 4090's BF16 peak; the loader supplies an image every 0.5 ms of wall time — compute-bound, 13x feeder headroom.)

Only after that test should you touch the loader.

### 6.2 Batch size

(measured) 512x512: batch 32 → 11.2 GB; batch 48 → 16.6 GB, i.e. ~0.34 GB per extra image. Throughput did not improve from 32 to 48 on the cached path, so batch size was chosen for step count, not speed. Do not assume bigger is faster; measure.

### 6.3 Checkpoints

Checkpoint bytes are dominated by optimizer state. Frozen backbone + 20M trainable params gave 78 MB per checkpoint (measured).

```
checkpoints per pass = total_steps / ckpt_every
disk used            = checkpoints per pass × ckpt_bytes
```

A 110k-step pass every 200 steps would be ~55 GB; every 5000 steps is ~2.6 GB. Pick `ckpt_every` from your disk budget, and prune old checkpoints.

### 6.4 Compilation and steady state

`torch.compile(mode="max-autotune")` gave ~20% over the default mode here, at the cost of minutes of autotuning at start and shape specialisation (so avoid changing input shape mid-run — §7.4). Measure steady state, exclude the first tens of steps, or autotune and warmup dominate and you "optimise" the wrong thing.

### 6.5 Validation cadence

Validate every N steps on a small cached split, never from the HDD stream. Our 1000-image validation costs ~7 s (measured) against a 27-minute val interval; streaming it from HDD would contend with training reads.

---

## 7. Operational notes

### 7.1 Put each thing on the right medium

Archive (compressed, sequential, huge) → HDD, the cheapest byte
Cache (decoded, random-access) → SSD/NVMe
Checkpoints → SSD (frequent writes, small)
Archive and cache on different physical devices so decoding does not fight cache reads

### 7.2 Never let the corpus exceed RAM expectations

Page cache helps only if the working set fits. 318 GB on a 48 GB machine means every epoch is a cold read; do not tune as if the second pass will be faster.

### 7.3 Avoid concurrent I/O and CPU contention

Throughput varied between 126 and 155 images/s purely on whether the user was gaming on the same machine — a 20% swing with nothing else changed. Re-measure under identical load before concluding "format X is 20% faster"; for a stable number, close everything and run a controlled A/B.

### 7.4 Avoid mid-run shape changes

Changing resolution or aspect mid-run forces recompilation and can turn a compute-bound job into a compile-bound one. If you need variable shapes, use a small fixed set of buckets, one compiled graph per bucket. DINOv3-style RoPE backbones accept arbitrary H x W natively (no positional interpolation), which makes this feasible — measured: 640x384 → 965 tokens, 448x576 → 1013 tokens, both forward fine.

### 7.5 Instrument before you optimise

Time each stage separately and print it:

```
tar read      -> MB/s and images/s
decode+resize -> images/s at your thread count
H2D transfer  -> ms per batch
GPU step      -> ms per batch
```

The stage nearest 100% utilisation is the one to work on. Here that was the GPU by a factor of five, so every loader optimisation after the first was wasted effort.

### 7.6 Defragmentation and layout

Write read-mostly, build-once datasets on a freshly formatted (or short-stroked) partition so shards are physically contiguous. A tar on a fragmented volume reads at a fraction of its potential. Never append to shards; write new ones.

---

## 8. Anti-patterns (all of these have bitten real projects)

| anti-pattern | why it hurts |
|---|---|
| 5M loose JPEGs on HDD | a seek per image; IOPS-limited at ~100-150/s |
| `tar.gz` / `zip` of images | seek requires decompressing everything before it |
| Shuffling a global index and seeking | turns sequential reads into a seek storm |
| Decoding 4K then resizing to 256 | 10-20x decode cost for no information gain |
| Progressive JPEG in training data | slower decode, no benefit |
| Fresh `pin_memory()` tensor per batch | a full extra copy per batch (24 ms measured) |
| Unbounded prefetch | OOM, and hides the real bottleneck |
| Streaming validation from the archive disk | steals I/O from training |
| Storing uncompressed arrays on slow HDD | e.g. 786 KB/img at 150/s = 118 MB/s |
| Trusting a throughput number taken under different load | 20% swings are normal |
| Optimising the loader when the GPU is the bottleneck | wasted days |
