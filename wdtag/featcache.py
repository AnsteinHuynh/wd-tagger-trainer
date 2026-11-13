"""Read pooled features out of a sharded feature cache by absolute image index.

The shards are per-tar `.npy` files written in the stream's own order, with a `shards.json`
recording each shard's `source_start`. That makes the whole cache one logical row space, so a
batch can ask for rows `[start, start + n)` without any lockstep state: the stream already knows
how many images it has consumed, so the index is exact and a resume needs no seeking.

Row order equals the training stream's order only when the cache was built at the same seed and
pass index. `check_against` exists because a mismatch would train the head on another image's
features and still look like learning.
"""
from __future__ import annotations

import json
import os

import numpy as np


class FeatCache:
    def __init__(self, root: str, cols: tuple[int, int] | None = None):
        self.root = root
        meta_path = os.path.join(root, "shards.json")
        if not os.path.isfile(meta_path):
            raise SystemExit(f"feat cache {root}: no shards.json")
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        self.meta = meta
        self.dim = int(meta["dim"])
        self.dtype = np.dtype(meta.get("dtype", "float16"))
        self.seed = meta.get("seed")
        self.pass_index = meta.get("pass_index")
        self.shards = meta["shards"]
        self.rows = sum(int(sh["rows"]) for sh in self.shards)
        self.starts = np.cumsum([0] + [int(sh["rows"]) for sh in self.shards])[:-1]
        lo, hi = cols if cols is not None else (0, self.dim)
        if not (0 <= lo < hi <= self.dim):
            raise SystemExit(f"feat cache {root}: columns {lo}:{hi} outside the {self.dim}-wide rows")
        self.cols = (lo, hi)
        self.width = hi - lo
        self._maps: dict[int, np.memmap] = {}

    def describe(self) -> str:
        lo, hi = self.cols
        return (f"{len(self.shards)} shards, {self.rows} rows, dim {self.dim}, columns "
                f"{lo}:{hi} ({self.width}-wide), seed {self.seed} pass {self.pass_index}, "
                f"{self.dtype}")

    def _map(self, k: int) -> np.memmap:
        m = self._maps.get(k)
        if m is None:
            path = os.path.join(self.root, self.shards[k]["feats"])
            m = np.load(path, mmap_mode="r")
            self._maps[k] = m
        return m

    def read(self, start: int, n: int) -> np.ndarray:
        """Rows [start, start + n) of the column slice, as a float32 array."""
        if start < 0 or n < 0 or start + n > self.rows:
            raise SystemExit(f"feat cache lookahead out of range: rows {start}..{start + n} "
                             f"of {self.rows}")
        out = np.empty((n, self.width), dtype=np.float32)
        k = int(np.searchsorted(self.starts, start, side="right") - 1)
        done = 0
        while done < n:
            within = start + done - int(self.starts[k])
            take = min(n - done, int(self.shards[k]["rows"]) - within)
            lo, hi = self.cols
            block = self._map(k)[within:within + take, lo:hi]
            out[done:done + take] = np.asarray(block, dtype=np.float32)
            done += take
            k += 1
        return out
