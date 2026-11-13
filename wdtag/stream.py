"""Stream training data from the webdataset tars.

5321713 images at 512x512 uint8 = ~4.2 TB, fitting neither drive, so the full corpus cannot use `wdtag.cache` shards: sequential tar reads, thread-pool decode, cached batch shape. Order is a pure function of `(seed, pass_index)` (seeded tar shuffle, file order); resume from `n_consumed` reads headers only, never decoding or replaying.
"""
from __future__ import annotations

import io
import json
import os
import queue
import random
import tarfile
import threading
from concurrent.futures import ThreadPoolExecutor
from itertools import islice

import numpy as np
import torch
from PIL import Image

from .cache import CACHE_ROOT, DATASET_ROOT, SIZE, SPLITS
from .config import RUNS_ROOT
from .labels import load_label_space, tag_indices

IMG_EXT = (".webp", ".jpg", ".png")


# --------------------------------------------------------------------------- manifest
def _count_samples(tar_path: str) -> int:
    """Number of complete (image, json) pairs in a tar, from headers only."""
    pending: set[str] = set()
    n = 0
    with tarfile.open(tar_path, "r:") as tf:
        for m in tf:
            if not m.isfile():
                continue
            stem, ext = os.path.splitext(os.path.basename(m.name))
            if ext in IMG_EXT:
                pending.add(stem)
            elif ext == ".json" and stem in pending:
                pending.discard(stem)
                n += 1
    return n


def tar_dir(split: str) -> str:
    return os.path.join(DATASET_ROOT, SPLITS[split]["subdir"])


def _manifest_dir() -> str:
    """Durable home for the per-tar count table. NOT CACHE_ROOT (disposable): losing it costs a header scan per resume, ~10 s/tar, ~19 min at train's halfway mark (~97 tars)."""
    d = os.path.join(RUNS_ROOT, "_manifests")
    os.makedirs(d, exist_ok=True)
    return d


def _manifest_candidates(split: str) -> list[str]:
    """Durable copy, then legacy cache location (read-only migration)."""
    name = f"manifest_{split}.json"
    return [os.path.join(_manifest_dir(), name), os.path.join(CACHE_ROOT, name)]


# Corpus sizes from the dataset README, so the LR schedule need not scan tars.
KNOWN_TOTAL = {"train": 5321713, "valid": 296957, "test": 295926}


def _known_total(split: str) -> int:
    path = os.path.join(CACHE_ROOT, f"manifest_{split}.json")
    try:
        man = json.load(open(path, encoding="utf-8"))
        if man.get("split") == split and man.get("total"):
            return int(man["total"])
    except Exception:
        pass
    return KNOWN_TOTAL[split]


# --------------------------------------------------------------------------- decoding
def _decode_pair(item: tuple[bytes, dict], space, n_labels: int):
    """One (image bytes, annotation) -> uint8 CHW plus label indices."""
    blob, meta = item
    im = Image.open(io.BytesIO(blob))
    if im.mode != "RGB":
        im = im.convert("RGB")
    im = im.resize((SIZE, SIZE), Image.BILINEAR)
    arr = np.asarray(im, dtype=np.uint8).transpose(2, 0, 1)  # HWC -> CHW
    idx, _unknown = tag_indices(meta, space)
    return arr, idx


def _iter_tar(path: str, skip: int = 0):
    """Yield (stem, image_bytes, meta) from a tar, skipping the first `skip`."""
    pending: dict[str, bytes] = {}
    skip_pending: set[str] = set()  # skips tracked by name only, never by content
    seen = 0
    with tarfile.open(path, "r:") as tf:
        for m in tf:
            if not m.isfile():
                continue
            stem, ext = os.path.splitext(os.path.basename(m.name))
            if ext in IMG_EXT:
                if seen < skip:
                    # Skip without reading the blob: doing so pulled a whole tar
                    # (~1.5 GB) through the drive per resume to discard it; the name
                    # pairs with the following json.
                    skip_pending.add(stem)
                else:
                    pending[stem] = tf.extractfile(m).read()
            elif ext == ".json":
                if stem in skip_pending:
                    skip_pending.discard(stem)
                    seen += 1
                    continue
                if stem in pending:
                    blob = pending.pop(stem)
                    yield stem, blob, json.loads(tf.extractfile(m).read().decode("utf-8"))


def _chunked(it, n: int):
    it = iter(it)
    while True:
        chunk = list(islice(it, n))
        if not chunk:
            return
        yield chunk


# ----------------------------------------------------------------------------- stream
class TarStream:
    """Batching, shuffled, resumable stream over one split's tars; yields `(images uint8[B,3,SIZE,SIZE], labels float32[B,n_labels], consumed)`, including this batch -- save it, pass back as `skip`."""

    def __init__(self, split: str = "train", seed: int = 1337, pass_index: int = 0,
                 batch_size: int = 32, n_labels: int | None = None, workers: int = 8,
                 prefetch: int = 4, skip: int = 0, max_samples: int | None = None,
                 hold: int = 1, log=None):
        # `hold` is how many batches the consumer keeps alive at once. With gradient
        # accumulation it is `grad_accum`: the trainer holds the whole group before it steps.
        # The ring has to cover the queue, the held batches and the slot being written, or the
        # producer wraps and overwrites a batch the trainer is still going to use, which shows up
        # as two runs of the same config drifting apart at a random step.
        self.hold = max(1, int(hold))
        self.split = split
        self.seed = seed
        self.pass_index = pass_index
        self.batch_size = batch_size
        self.n_labels = n_labels
        self.workers = max(1, workers)
        self.prefetch = max(1, prefetch)
        self.skip = skip
        self.max_samples = max_samples
        self.log = log or (lambda *a, **k: None)
        self._tars = sorted(f for f in os.listdir(tar_dir(split)) if f.endswith(".tar"))
        self._total = _known_total(split)
        # Manifest makes resume seek instant, else skip walks tar headers (a few
        # minutes, resume only).
        self._counts = self._load_counts()

    def _load_counts(self) -> dict | None:
        """Per-tar counts, learned lazily on resume and remembered; partial is fine."""
        for path in _manifest_candidates(self.split):
            try:
                man = json.load(open(path, encoding="utf-8"))
                if man.get("split") == self.split:
                    return {k: int(v) for k, v in man["tars"].items()}
            except Exception:
                continue
        return None

    def _save_counts(self) -> None:
        try:
            complete = self._counts and set(self._counts) == set(self._tars)
            total = int(sum(self._counts.values())) if complete else self._total
            payload = {"split": self.split, "tars": self._counts, "total": total}
            # Durable copy authoritative; cache copy for older readers.
            # Durable first, then best-effort cache.
            for path in _manifest_candidates(self.split):
                try:
                    tmp = path + ".tmp"
                    with open(tmp, "w", encoding="utf-8") as f:
                        json.dump(payload, f)
                    os.replace(tmp, path)
                except Exception:
                    continue
        except Exception:
            pass  # never fail training over a cache

    # ------------------------------------------------------------------ ordering
    def _order(self) -> list[str]:
        # One definition of order lives in wdtag.manifest, because a cache shard labelled with the
        # wrong tar is silently wrong while still looking plausible.
        from .manifest import order as _tar_order
        return _tar_order(self.split, self.seed, self.pass_index, names=self._tars)

    def total_samples(self) -> int:
        return min(self._total, self.max_samples) if self.max_samples else self._total

    def _locate(self, order: list[str], skip: int) -> tuple[int, int]:
        """Global offset -> (tar index, offset within)."""
        if skip <= 0:
            return 0, 0  # a fresh pass never needs to know any tar's length
        if self._counts is None:
            self._counts = {}
        left = skip
        for i, t in enumerate(order):
            c = self._counts.get(t)
            if c is None:
                c = _count_samples(os.path.join(tar_dir(self.split), t))
                self._counts[t] = c
                self._save_counts()  # remember it, so the next resume is cheaper
            if left < c:
                return i, left
            left -= c
        return len(order), 0

    # ------------------------------------------------------------------ producing
    def _produce(self):
        order = self._order()
        start_pos, start_skip = self._locate(order, self.skip)
        if start_pos:
            self.log(f"[stream] pass {self.pass_index}: skipping {self.skip:,} samples "
                     f"-> starts at tar {start_pos + 1}/{len(order)}")
        space = load_label_space()
        n_labels = self.n_labels or len(space)
        emitted = self.skip

        # Preallocated pinned ring: a fresh 37.7 MB batch per step cost ~7.5 GB of page
        # faults over a few hundred steps and forced a synchronous pageable H2D copy.
        # Slots in use at once: prefetch in the queue + hold by the consumer + 1 being written.
        ring = self.prefetch + self.hold + 1
        bx = [torch.empty((self.batch_size, 3, SIZE, SIZE), dtype=torch.uint8,
                          pin_memory=True) for _ in range(ring)]
        bxn = [t.numpy() for t in bx]  # writable uint8 HWC-free views
        # Labels were pageable numpy, which makes the consumer's non_blocking H2D copy a
        # synchronous one on the same stream as the forward. Pinned like the image slots.
        by = [torch.zeros((self.batch_size, n_labels), dtype=torch.float32, pin_memory=True)
              for _ in range(ring)]
        slot = 0
        fill = 0
        buf_src: list[tuple[str, str]] = []  # (sample key, tar) for the debug dump

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for ti in range(start_pos, len(order)):
                tar = order[ti]
                skip_here = start_skip if ti == start_pos else 0
                path = os.path.join(tar_dir(self.split), tar)
                for chunk in _chunked(_iter_tar(path, skip_here), self.workers * 32):
                    items = [(blob, meta) for _stem, blob, meta in chunk]
                    decoded = list(pool.map(lambda it: _decode_pair(it, space, n_labels),
                                            items))
                    for (stem, _blob, _meta), (arr, idx) in zip(chunk, decoded):
                        bxn[slot][fill] = arr
                        if idx:
                            by[slot][fill, idx] = 1.0
                        buf_src.append((stem, tar))
                        fill += 1
                        if fill == self.batch_size:
                            consumed = emitted + fill
                            x = bx[slot]
                            y = by[slot]  # pinned, so the trainer's copy can overlap
                            src = buf_src
                            buf_src = []
                            fill = 0
                            emitted = consumed
                            yield x, y, consumed, src
                            # Clear the NEXT slot: `y` aliases by[slot], so clearing this
                            # would zero labels the trainer is about to use; next slot
                            # >=2 ahead of the consumer.
                            slot = (slot + 1) % ring
                            by[slot][:] = 0.0
                            if self.max_samples and emitted >= self.max_samples:
                                return
                self.log(f"[stream] pass {self.pass_index}: finished tar "
                         f"{ti + 1}/{len(order)} ({emitted:,} samples)")

            # The tail after the last full batch used to be dropped silently: a cache pass covered
            # floor(total/batch)*batch images, 241 short of the 5321713 corpus, and left zeros in the
            # rows it claimed. Emit it so a pass over the corpus covers the corpus.
            if fill > 0:
                consumed = emitted + fill
                yield bx[slot], by[slot], consumed, buf_src
                emitted = consumed

    # ------------------------------------------------------------------ consuming
    def batches(self):
        """Yield batches from a producer thread so decoding overlaps training."""
        q: queue.Queue = queue.Queue(maxsize=self.prefetch)
        err: list[BaseException] = []

        def run():
            try:
                for item in self._produce():
                    q.put(item)
            except BaseException as e:  # surface it in the consumer
                err.append(e)
            finally:
                q.put(None)

        t = threading.Thread(target=run, name="tar-stream", daemon=True)
        t.start()
        while True:
            item = q.get()
            if item is None:
                if err:
                    raise err[0]
                return
            yield item



