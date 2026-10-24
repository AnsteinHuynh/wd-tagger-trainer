"""Dataset + loader over the .pt image caches.

Shards open lazily per process with `mmap=True`: a mapped tensor held as an attribute would make the DataLoader pickle gigabytes of pixels into every worker on Windows spawn.
"""
from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import Dataset

from .cache import CACHE_ROOT, IMAGES_PER_SHARD, SPLITS, _shard_name
from .labels import load_label_space


def shard_paths(split: str, mode: str) -> list[str]:
    total = SPLITS[split]["n"]
    n_shards = (total + IMAGES_PER_SHARD - 1) // IMAGES_PER_SHARD
    return [_shard_name(split, mode, k, total) for k in range(n_shards)]


def shard_len(split: str, k: int) -> int:
    total = SPLITS[split]["n"]
    return min(IMAGES_PER_SHARD, total - k * IMAGES_PER_SHARD)


# One mapping per shard per process: opening a shard twice costs another full copy of
# its pages on Windows, so every consumer (dataset, label stats, targets) uses this cache.
_SHARD_CACHE: dict[str, dict] = {}


def open_shard(path: str) -> dict:
    s = _SHARD_CACHE.get(path)
    if s is None:
        s = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        _SHARD_CACHE[path] = s
    return s


def close_shards() -> None:
    _SHARD_CACHE.clear()


class CacheDataset(Dataset):
    """Read-only, lazy mmap view over one split's shards."""

    def __init__(self, split: str, mode: str, n_labels: int):
        self.split, self.mode = split, mode
        self.paths = shard_paths(split, mode)
        self.n_labels = n_labels
        self.index: list[tuple[int, int]] = []
        for k in range(len(self.paths)):
            n = shard_len(split, k)
            self.index.extend([(k, i) for i in range(n)])
        self._pid: int | None = None
        self._shards: list[dict] = []

    # ---------------------------------------------------------------- opening
    def _ensure_open(self) -> None:
        if self._pid == os.getpid() and self._shards:
            return
        self._shards = [open_shard(p) for p in self.paths]
        self._pid = os.getpid()

    @property
    def shards(self) -> list[dict]:
        self._ensure_open()
        return self._shards

    def __len__(self) -> int:
        return len(self.index)

    # ---------------------------------------------------------------- pickling
    def __getstate__(self):
        """Never ship mapped tensors to DataLoader workers: pickling a mapped tensor serialises its contents -- the whole 7.9 GB train cache into every spawned worker. Workers get paths and open their own maps."""
        state = self.__dict__.copy()
        state["_shards"] = []
        state["_pid"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._shards = []
        self._pid = None

    def meta(self, i: int) -> tuple[str, str, int]:
        k, j = self.index[i]
        s = self.shards[k]
        return s["keys"][j], s["tars"][j], int(s["ids"][j])

    def __getitem__(self, i: int):
        k, j = self.index[i]
        s = self.shards[k]
        img = s["images"][j]  # uint8 [3,512,512], memmapped
        lab = torch.zeros(self.n_labels, dtype=torch.float32)
        lo, hi = int(s["ptr"][j]), int(s["ptr"][j + 1])
        if hi > lo:
            lab[s["pos"][lo:hi]] = 1.0
        return img, lab


class MultiHotCollator:
    """Stack uint8 images and multi-hot targets; normalisation on GPU."""

    def __init__(self, n_labels: int):
        self.n_labels = n_labels

    def __call__(self, batch):
        imgs = torch.stack([b[0] for b in batch])   # uint8 [B,3,H,W]
        labs = torch.stack([b[1] for b in batch])
        return imgs, labs


def label_priors(split: str, mode: str, n_labels: int) -> np.ndarray:
    """Per-label positive rate, for head-bias init."""
    paths = shard_paths(split, mode)
    counts = np.zeros(n_labels, dtype=np.int64)
    n = 0
    for p in paths:
        s = open_shard(p)
        n += s["images"].shape[0]
        counts += np.bincount(s["pos"].numpy(), minlength=n_labels)
    return counts / max(n, 1)


def load_targets(split: str, mode: str, n_labels: int) -> np.ndarray:
    """Dense bool [N, L] ground truth for evaluation."""
    paths = shard_paths(split, mode)
    out = np.zeros((SPLITS[split]["n"], n_labels), dtype=bool)
    base = 0
    for k, p in enumerate(paths):
        s = open_shard(p)
        n = s["images"].shape[0]
        pos, ptr = s["pos"].numpy(), s["ptr"].numpy()
        for j in range(n):
            out[base + j, pos[ptr[j]:ptr[j + 1]]] = True
        base += n
    return out
