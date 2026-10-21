# Reading image datasets off a spinning hard drive

A field guide to image formats, tar/shard layouts, feeder design and training-side
choices when your data lives on an HDD. Written from measurements on one project
(a 5.3M-image Danbooru tagger), but the reasoning generalises.

**How to read the numbers in this document.** Anything marked *(measured)* was
timed on the machine described below. Anything marked *(estimate)* is arithmetic or
experience. Do not trust a number you have not timed on your own disk — HDDs vary by
more than 3x between a healthy 7200 rpm drive and a tired external one.

Reference machine: Windows, RTX 4090 (24 GB), 47.9 GB RAM, 16 CPU threads.
Dataset: 213 `.tar` shards, **316.8 GB**, **5,321,713 images** → ~60 KB/image.
Archive disk: **8 TB WD Blue (WDC WD80EAAZ), SATA, spinning rust**, measured
**~45 MB/s** sequential. Cache/checkpoint disk: NVMe.

---

## 1. The physics you are fighting

An HDD is a rotating platter with one head. Two consequences dominate everything:

| access pattern | what it costs | realistic throughput |
|---|---|---|
| Long sequential read | pure transfer, head never moves | 100-250 MB/s (7200 rpm); **45 MB/s measured here** |
| 1 MB sequential reads | one seek + transfer | tens of MB/s |
| 60 KB random reads | ~one seek each | **100-150 IOPS** = 6-9 MB/s *(estimate)* |
| 4 KB random reads | pure seek overhead | <1 MB/s *(estimate)* |

Seek time on a 7200 rpm drive is ~8-9 ms; a 5400 rpm laptop drive is worse, ~15 ms.
Note what that means for small-file random access: **you do not get bandwidth, you
get 100-150 chances per second to read anything.**

The single most important rule in this document:

> **Convert random access into sequential access, or hide the latency with
> concurrency. Never let a random-access pattern touch an HDD per image.**

### The arithmetic that decides your design

Before choosing anything, compute these four numbers:

```
images/s needed   = batch_size / step_seconds          # set by the GPU
bytes/s needed    = images/s × bytes_per_image_stored
decodes/s needed  = images/s                           # if stored compressed
seeks/s needed    = images/s                           # if stored one file per image
```

Then check each against the hardware. Worked example from this project:

```
GPU: 6.7 ms/image  -> need 150 images/s
stored size: 60 KB -> need 9 MB/s        (drive does 45)      OK
decode: need 150/s  -> 8 threads give 250/s                   OK
```

The GPU is the bottleneck and the drive is 20% utilised. That is the goal state.

Now the counterfactual that kills most naive plans: if we had stored **raw**
512x512 uint8 arrays instead of JPEGs, each image would be 786 KB, so 150 images/s
would need **118 MB/s** — 2.6x more than this drive provides. On this HDD,
uncompressed shards are *I/O-infeasible* at the target rate. Compression is not just
a space optimisation on a slow disk; it is what makes the throughput possible.

---

## 2. Do not store one file per image

Five million small files on an HDD is the classic way to build a pipeline that runs
at 40 images/s and is impossible to debug.

Per-image costs of loose files:

* **A seek per file.** At 8 ms, 150 images/s would need 1.2 s of seek time per second.
* **Filesystem metadata**: `CreateFile`/`open`, directory lookup, NTFS records.
  Roughly 50-200 us per file *(estimate)*, plus fragmentation as the directory grows.
* **No locality**: consecutive samples are physically scattered across the platter,
  so any shuffling is a seek storm.
* **Terrible tail latency**: one slow directory block stalls the whole feeder.

Directory *structure* does not save you (2-level fanout still needs metadata per
file). The only real fixes are packing into a container, or putting the files on SSD.

**Rule:** if `images × 60 KB` exceeds a few GB and the data is on HDD, pack it.

---

## 3. Container formats, ranked for HDD

### 3.1 Recommended: uncompressed tar shards (webdataset-style)

This is what we use and what I would reach for again.

**Why it works:**

* **Sequential by construction.** Members are concatenated; reading the dataset is
  one forward pass. The head never reverses.
* **No index needed.** You can start streaming at any byte offset and just parse
  headers forward. That makes resume trivial and exact (see §5.3).
* **Image and annotation are adjacent**, so one read yields both. Store
  `key.jpg` then `key.json` and you never open a second file.
* **Splittable across workers.** Each process can take a disjoint subset of shards
  with zero coordination.
* **Skip without reading.** `tarfile` advances over member payload using the size
  field, so skipping N images costs header reads only. *(measured: ~2,600
  samples/s for header-only walking of a 1.5 GB tar, vs ~250/s when decoding.)*
  This is what makes "resume without replaying images" cheap.

**Rules for making tar fast:**

* **Never compress the tars.** `tar.gz` forces sequential decompression of
  everything before your sample and destroys seekability. Use plain `.tar`, or
  worst case per-member compression (`.zst` per file), never container compression.
  *(We measured the plain-tar read at ~5,800 samples/s, faster than decode.)*
* **Shard size 1-4 GB.** Big enough to amortise open/seek costs, small enough to
  parallelise and to checkpoint transfers. Ours are 1.5 GB / 25,000 images.
* **Use `--format=gnu`, not `pax`.** Pax adds extended headers per long filename,
  bloating the stream and slowing header parsing.
* **Keep member names short and flat.** `000123.webp` not
  `dataset/train/class_a/very/long/path/000123.webp`.
* **Deterministic order inside a shard.** It lets you compute any sample's position
  without an index.
* **Store a companion index file** (`shard-00001.count`) or, better, learn counts
  lazily and cache them (see §5.3). Scanning a whole tar for header counts costs
  ~10 s per 1.5 GB *(measured)*, which you do not want on every start.

### 3.2 Memory-mapped array shards (`.npy`/`.npz`/tensor shards)

What our *cache* uses: one or more `torch.save`/`npy` files holding a contiguous
`uint8[N,3,H,W]` block, opened with `mmap`.

* **Excellent on SSD**: no decode, no copy, the OS pages in exactly what you touch.
* **Fine on HDD only if you accept no compression.** 786 KB/image at 512x512 means
  118 MB/s at 150 images/s. On a 45 MB/s drive this is infeasible; on NVMe it is trivial.
* **Loses JPEG's 10-15x compression**, so you need ~13x the space.
* Fixed shape is a real constraint (no mixed resolutions without extra shards).

**Use this as your *cache* on SSD, not as your *archive* on HDD.** That split is
exactly what this project does: HDD holds 317 GB of JPEG tars, NVMe holds a
decoded 10k-image cache, and each is on the medium where it makes sense.

### 3.3 LMDB / database files

* B+tree with memory mapping: good locality, transactional, one big file.
* On HDD the reads are still **random**, so you get IOPS-limited performance unless
  your access pattern happens to be sequential.
* With a large readahead and sequential keys it behaves well; with shuffled keys it
  seeks constantly.
* Worth it when you need random access by key (e.g. sampling by label) and the data
  is on SSD.

### 3.4 Zip / squashfs / other

* **Zip**: has a central directory, so per-file access seeks to a local header, then
  into the data. Deflate also means decompression. Acceptable on SSD, mediocre on HDD.
* **squashfs**: read-only, random-access with block compression. Great for
  deployment images, still random on HDD.
* **Parquet/Arrow**: excellent for *labels and metadata* (columnar, compressed,
  splittable, cheap to read a subset). Not for image bytes.
* **Anything requiring an index in RAM**: fine, but the index must stay small
  (an int64 offset per image for 5.3M images is 42 MB, which is acceptable).

### 3.5 Decision table

Given `S` = stored bytes per image, `R` = images/s you need:

| situation | choice |
|---|---|
| HDD, R×S well under drive speed, images compressed | **uncompressed tar shards** |
| HDD, need uncompressed arrays | add a **decoded cache on SSD** |
| SSD/NVMe, random access by key | array shards or LMDB |
| SSD/NVMe, decoded, fixed shape | mmap'd array shards |
| mixed resolutions | per-resolution tar shards, or one cache per shape |
| labels only | Parquet |

Sanity check with numbers: `R × S` must stay below ~50% of measured sequential
throughput so that latency spikes and housekeeping have room.

---

## 4. Image-side choices

### 4.1 Store at the resolution you train at

Decode cost scales roughly with pixel count. Decoding a 4K JPEG to resize it to 512
is 10-20x the work of decoding a 640-wide JPEG, and produces the same tensor.

* Store images at ~1.0-1.3x your training resolution. Not 4x "for safety".
* Our corpus was pre-downscaled to 640px wide upstream, ~60 KB/image, and that is a
  large part of why the pipeline is cheap.

### 4.2 Decode tricks

* **`Image.draft()`**: for JPEG, decode at 1/2, 1/4 or 1/8 scale directly in the DCT
  domain. If your source is ≥2x your target, this is a large speedup *(estimate:
  2-4x decode throughput)*. It does nothing when source ≈ target, which is our case.
* **Use baseline JPEG, not progressive.** Progressive decodes in multiple passes and
  is meaningfully slower per image *(estimate: 1.5-2x)*. Optimise for decode, not
  for bytes, once the pipeline is GPU-bound.
* **Prefer plain JPEG 4:2:0** for photographic/anime art. WebP usually saves 20-30%
  bytes but decodes slower; measure both. PNG is only worth it for alpha/line art.
* **Normalise at build time**: convert to RGB, apply EXIF orientation, strip
  profiles. Never pay for `convert()` or EXIF handling per epoch. CMYK JPEGs are
  especially slow.
* **One resize filter, chosen once.** `Image.BILINEAR` is the fast default;
  `LANCZOS` is slower but sharper. Do not mix per-image.
* **Resize in one shot** from the decoded size to the final size. Two-step resizes
  cost more and blur more.

### 4.3 What to store for the label

* If annotations are small (a few dozen ids), per-image JSON inside the tar is fine
  and keeps everything in one stream.
* If annotations are large or relational, keep them in a separate Parquet/Arrow file
  and store only an integer id in the tar.
* Precompute the label encoding (indices, multi-hot) at build time rather than
  parsing JSON per epoch if you can afford the space.

---

## 5. The feeder

The feeder's job: hand the GPU one batch every `step_seconds`, forever, without
stalling. Design for *latency hiding*, not for peak bandwidth.

### 5.1 Threads or processes?

**Threads, when the work is in C/C++ that releases the GIL.** PIL decode and resize
release the GIL, so threads genuinely parallelise and you avoid process overhead.

*(measured)* On this 16-thread machine, decode+resize to 512x512:

| workers | throughput |
|---|---|
| 1 thread | 55 images/s |
| **8 threads** | **250-289 images/s** |
| 14 threads (inside training) | **121 images/s — slower** |

More threads made it *worse*, because they competed with the training process for
the same physical cores and thrashed cache. Size the pool at
`ceil(R / per_thread_rate × 1.5)` and **cap it at (physical cores − 2)**.

Processes (`DataLoader(num_workers=N)`) are worth it when your per-sample work is
Python-heavy, or when a library does not release the GIL. Costs: memory per worker,
pickling, and on Windows a full spawn per worker. For pure decode, threads win.

### 5.2 Bounded prefetch, not unlimited

Use a queue with a **small, fixed** depth (4-8 batches). It absorbs the ~10 ms
latency spikes of an HDD without letting a slow consumer balloon your RAM.

We measured that buffer *reuse* matters as much as buffering:

* Allocating a fresh `37.7 MB` batch buffer per step caused **~7.5 GB of page faults
  over 200 steps** and forced a synchronous pageable H2D copy.
* **Fix: a ring of preallocated pinned buffers**, one per queue slot plus slack.
  Fill buffer `i`, yield it, move to `i+1`. Memory is then constant (6 x 37.7 MB)
  and the H2D copy can be asynchronous.
* Pinning a *fresh* tensor per batch cost **24.3 ms per 48-image batch** *(measured)*
  — 5x the actual transfer time. Pinning once and reusing is free.

If you use a ring, **never clear a buffer after handing it out**: the tensor aliases
the numpy view. We shipped that bug for one test run and it silently zeroed every
label (`loss 0.0000`). Clear the *next* slot, not the current one.

### 5.3 Exact resume without replaying data

Two properties you want, and they are cheap if you design for them:

1. **Order is a pure function of (seed, pass).** Shuffle the shard order with a
   seeded RNG; keep within-shard order fixed. No dependence on wall-clock, thread
   scheduling or iteration order.
2. **The cursor is derived, not stored.** `samples_consumed = global_step ×
   batch_size`, so a checkpoint already contains everything needed to resume.

Then resume = map a global sample offset to `(shard, offset_in_shard)`. That mapping
needs per-shard counts, which are expensive to compute (10 s per 1.5 GB tar
*(measured)*). Options, in order of preference:

* Store a tiny count/manifest file alongside the shards at build time. **Do this.**
* Learn counts lazily on first resume, and **cache them incrementally** so each
  later resume is cheaper than the last.
* Never decode-and-discard to skip. Skipping should read headers only.

### 5.4 Shuffling without seeking

This is where people accidentally destroy their HDD pipeline: they shuffle a global
index and then seek to each sample.

* **Shuffle the shard order**, not the sample order. Cost: zero. Locality: perfect.
* Add a **bounded in-memory shuffle buffer** (1,000-4,000 samples) if you need more
  mixing. Fill it with samples from the current shard, emit randomly.
* Accept that "epoch" shuffling is coarser than true random sampling. For datasets
  above ~1M samples this does not matter at all — the buffer plus shard-level
  shuffling decorrelates far more than SGD needs.

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

The producer thread is what overlaps HDD latency and decode with GPU work. A plain
generator cannot: it only advances when the consumer calls it.

---

## 6. Training-side choices that interact

### 6.1 First determine whether you are I/O-bound at all

Cheap test: run a few hundred steps with the same model but a synthetic batch
(random or zeros, no loader). If steps/sec is the same, you are **compute-bound**
and no feeder work will help. *(measured: our GPU does 6.7 ms/image; the loader
supplies an image every 0.5 ms of wall time, so it is compute-bound and the feeder
has 13x headroom.)*

Only after that test should you touch the loader.

### 6.2 Batch size

Measure VRAM per image rather than guessing:

*(measured)* 512x512, batch 32 → 11.2 GB; batch 48 → 16.6 GB, i.e. **~0.34 GB per
extra image**. Throughput did **not** improve from 32 to 48 on the cached path, so
batch size was chosen for step count, not speed. Do not assume bigger batches are
faster; measure.

### 6.3 Checkpoints

Checkpoint bytes are dominated by optimizer state. Frozen backbone + 20M trainable
params gave **78 MB** per checkpoint *(measured)*. Then:

```
checkpoints per pass = total_steps / ckpt_every
disk used            = checkpoints per pass × ckpt_bytes
```

A 110k-step pass at every 200 steps would have been ~55 GB. Every 5,000 steps is
~2.6 GB. Pick `ckpt_every` from your disk budget, and prune old checkpoints.

### 6.4 Compilation and steady state

`torch.compile(mode="max-autotune")` gave **~20%** over the default mode here, at
the cost of minutes of autotuning at start and shape specialisation (so **avoid
changing input shape mid-run** — see §7.4). Always measure steady state and exclude
the first tens of steps from your throughput numbers; otherwise autotune and warmup
dominate and you will "optimise" the wrong thing.

### 6.5 Validation cadence

Validate every N steps on a small **cached** split, never from the HDD stream. Our
1,000-image validation costs ~7 s *(measured)*, which is nothing against a 27-minute
val interval. Streaming validation from HDD would contend with training reads.

---

## 7. Operational notes

### 7.1 Put each thing on the right medium

* **Archive (compressed, sequential, huge)** → HDD. It is the cheapest byte.
* **Cache (decoded, random-access)** → SSD/NVMe.
* **Checkpoints** → SSD (frequent writes; small).
* Keep the archive and the cache on *different* physical devices so decoding does
  not fight with cache reads.

### 7.2 Never let the corpus exceed RAM expectations

Page cache helps only if the working set fits. 318 GB of data on a 48 GB machine
means every epoch is a cold read; do not tune as if the second pass will be faster.

### 7.3 Avoid concurrent I/O and CPU contention

Our throughput varied between **126 and 155 images/s** depending only on whether the
user was playing a game on the same machine — a 20% swing with nothing else changed.
Before you conclude "format X is 20% faster", re-measure under identical load.
If you need a stable number, close everything and run a controlled A/B.

### 7.4 Avoid mid-run shape changes

Changing resolution or aspect mid-run forces recompilation and can turn a
compute-bound job into a compile-bound one. If you must have variable shapes, use a
small fixed set of buckets and accept one compiled graph per bucket. Note that
DINOv3-style **RoPE** backbones accept arbitrary H x W natively (no positional
interpolation needed), which makes this feasible — measured: 640x384 → 965 tokens,
448x576 → 1013 tokens, both forward fine.

### 7.5 Instrument before you optimise

Time each stage separately and print it:

```
tar read      -> MB/s and images/s
decode+resize -> images/s at your thread count
H2D transfer  -> ms per batch
GPU step      -> ms per batch
```

Whichever stage is nearest 100% utilisation is the one to work on. In this project
that was the GPU by a factor of five, so every loader optimisation after the first
was wasted effort.

### 7.6 Defragmentation and layout

For read-mostly datasets built once, write them on a freshly formatted (or
short-stroked) partition so the shards are physically contiguous. A tar written to
a fragmented volume reads at a fraction of its potential. Never append to shards;
write new ones.

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

---

## 9. Checklist for a new project

1. Compute `images/s`, `bytes/s`, `decodes/s` from your GPU's step time.
2. Measure your disk's *sequential* throughput and its random IOPS (not its spec sheet).
3. Choose the container: tar shards for HDD, array shards/LMDB for SSD.
4. Store images at ~1.0-1.3x training resolution, baseline JPEG, RGB, stripped.
5. Write shards sequentially on a fresh volume, 1-4 GB each, uncompressed, GNU tar.
6. Write a per-shard count manifest next to the shards.
7. Feeder: threads sized to `ceil(R / per_thread × 1.5)`, capped at cores − 2;
   bounded queue; preallocated pinned ring.
8. Order = f(seed, pass); resume cursor = `step × batch_size`; skip by headers only.
9. Verify: measure stage utilisation; confirm the GPU is the busy one.
10. Re-measure under your real working conditions before believing any delta.

---

## 10. Cheat sheet of the numbers measured here

| quantity | value |
|---|---|
| Corpus | 213 tars, 316.8 GB, 5,321,713 images, ~60 KB/image |
| HDD sequential read | ~45 MB/s (8 TB WD Blue, SATA) |
| Bytes/s needed for 150 img/s | 9 MB/s (20% of the drive) |
| Decode+resize, 1 thread | 55 images/s |
| Decode+resize, 8 threads | 250-289 images/s |
| Decode+resize, 14 threads under training | 121 images/s (worse) |
| Plain-tar payload read | ~5,800 samples/s |
| Tar header-only walk | ~2,600 samples/s (10 s per 1.5 GB tar) |
| Feeder in-loop throughput | 143-156 images/s at batch 48 |
| GPU cost per image | ~505 GFLOP, ~6.7 ms at 150 img/s (~75 TFLOPS, ~45% of a 4090's BF16 peak) |
| VRAM | 11.2 GB @ bs32, 16.6 GB @ bs48 (~0.34 GB/image) |
| Pinning a fresh batch | 24.3 ms per 48 images |
| Checkpoint (frozen backbone) | 78 MB |
| 1,000-image validation | ~7 s |
| Load-induced throughput swing | 126 <-> 155 images/s (20%) |
| Raw-array alternative | 786 KB/image → 4.18 TB cached, 118 MB/s at 150 img/s |
