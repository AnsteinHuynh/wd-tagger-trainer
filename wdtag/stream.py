"""Stream training data straight from the webdataset tars.

Why this exists: 5,321,713 images at 512x512 uint8 is ~4.2 TB, which does not fit on
either drive, so the full-corpus run cannot use the shard cache that `wdtag.cache`
builds. Instead this module reads one tar at a time (sequential reads only, which is
what the drive likes), decodes and resizes in a thread pool, and yields batches of the
exact same shape the cached path yields, so the trainer is unchanged downstream.

Two properties the trainer depends on:

  * **Determinism.** The order of samples is a pure function of `(seed, pass_index)`:
    the tar order is a seeded shuffle, and within a tar samples come in file order.
    Nothing depends on timing or thread scheduling.
  * **Exact resume.** A cursor of `n_consumed` samples identifies the tar and the
    offset inside it, so a resumed run skips straight there (reading tar headers only,
    never decoding the images it already saw) and never replays a picture.
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
from .labels import load_label_space, tag_indices

IMG_EXT = (".webp", ".jpg", ".png")


# --------------------------------------------------------------------------- manifest
def _count_samples(tar_path: str) -> int:
    """Number of complete (image, json) pairs in a tar, by reading headers only."""
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


# Corpus sizes, from the dataset README. Known up front so the LR schedule does not
# depend on having scanned the tars.
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
    """Decode one (image bytes, annotation) into a uint8 CHW array plus label indices."""
    blob, meta = item
    im = Image.open(io.BytesIO(blob))
    if im.mode != "RGB":
        im = im.convert("RGB")
    im = im.resize((SIZE, SIZE), Image.BILINEAR)
    arr = np.asarray(im, dtype=np.uint8).transpose(2, 0, 1)  # HWC -> CHW
    idx, _unknown = tag_indices(meta, space)
    return arr, idx


def _iter_tar(path: str, skip: int = 0):
    """Yield (stem, image_bytes, meta) pairs from a tar, skipping the first `skip`."""
    pending: dict[str, bytes] = {}
    seen = 0
    with tarfile.open(path, "r:") as tf:
        for m in tf:
            if not m.isfile():
                continue
            stem, ext = os.path.splitext(os.path.basename(m.name))
            if ext in IMG_EXT:
                pending[stem] = tf.extractfile(m).read()
            elif ext == ".json" and stem in pending:
                blob = pending.pop(stem)
                if seen < skip:
                    seen += 1
                    continue
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
    """Batching, shuffled, resumable stream over the tars of one split.

    Yields `(images uint8[B,3,SIZE,SIZE], labels float32[B,n_labels], consumed)`
    where `consumed` is the running count of samples emitted *including* this batch.
    Save `consumed` and pass it back as `skip` to resume exactly.
    """

    def __init__(self, split: str = "train", seed: int = 1337, pass_index: int = 0,
                 batch_size: int = 32, n_labels: int | None = None, workers: int = 8,
                 prefetch: int = 4, skip: int = 0, max_samples: int | None = None,
                 log=None):
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
        # Optional accelerator: if a manifest exists it makes resume seeking instant,
        # otherwise the skip walks tar headers (a few minutes, paid only on resume).
        self._counts = self._load_counts()

    def _load_counts(self) -> dict | None:
        """Per-tar sample counts, if any have been learned already.

        This is only needed to seek on resume, so it is filled in lazily: every tar
        counted during a resume is remembered, which makes each later resume cheaper
        than the last. A partial map is fine.
        """
        path = os.path.join(CACHE_ROOT, f"manifest_{self.split}.json")
        try:
            man = json.load(open(path, encoding="utf-8"))
            if man.get("split") == self.split:
                return {k: int(v) for k, v in man["tars"].items()}
        except Exception:
            pass
        return None

    def _save_counts(self) -> None:
        path = os.path.join(CACHE_ROOT, f"manifest_{self.split}.json")
        try:
            complete = self._counts and set(self._counts) == set(self._tars)
            total = int(sum(self._counts.values())) if complete else self._total
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"split": self.split, "tars": self._counts, "total": total}, f)
            os.replace(tmp, path)
        except Exception:
            pass  # a cache: never fail training over it

    # ------------------------------------------------------------------ ordering
    def _order(self) -> list[str]:
        rng = random.Random(self.seed * 1000003 + self.pass_index)
        tars = list(self._tars)
        rng.shuffle(tars)
        return tars

    def total_samples(self) -> int:
        return min(self._total, self.max_samples) if self.max_samples else self._total

    def _locate(self, order: list[str], skip: int) -> tuple[int, int]:
        """Map a global sample offset to (position in the tar order, offset within it)."""
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

        # Ring of preallocated pinned buffers. Allocating a fresh 37.7 MB batch every
        # step cost ~7.5 GB of page faults over a few hundred steps and forced a
        # synchronous pageable H2D copy; reusing pinned buffers removes both.
        ring = self.prefetch + 2
        bx = [torch.empty((self.batch_size, 3, SIZE, SIZE), dtype=torch.uint8,
                          pin_memory=True) for _ in range(ring)]
        bxn = [t.numpy() for t in bx]  # writable uint8 HWC-free views
        by = [np.zeros((self.batch_size, n_labels), dtype=np.float32) for _ in range(ring)]
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
                            y = torch.from_numpy(by[slot])  # shares memory with by[slot]
                            src = buf_src
                            buf_src = []
                            fill = 0
                            emitted = consumed
                            yield x, y, consumed, src
                            # Clear the NEXT slot, not this one: `y` above aliases
                            # by[slot], so clearing it here would zero the labels the
                            # trainer is about to use. The next slot is at least two
                            # ahead of the consumer, so it is always safe to reuse.
                            slot = (slot + 1) % ring
                            by[slot][:] = 0.0
                            if self.max_samples and emitted >= self.max_samples:
                                return
                self.log(f"[stream] pass {self.pass_index}: finished tar "
                         f"{ti + 1}/{len(order)} ({emitted:,} samples)")

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
