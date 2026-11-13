"""Shared inference helpers (used for our model and for the baseline)."""
from __future__ import annotations

import numpy as np
import torch

from .data import CacheDataset, SPLITS


@torch.no_grad()
def predict_probs(
    forward_fn,
    split: str,
    mode: str,
    n_labels: int,
    mean,
    std,
    batch_size: int = 32,
    device: str = "cuda",
    bf16: bool = True,
    log=print,
) -> np.ndarray:
    """Run `forward_fn(pixel_values)->logits` over a cached split, return sigmoid probs."""
    ds = CacheDataset(split, mode, n_labels)
    n = len(ds)
    out = np.empty((n, n_labels), dtype=np.float32)
    mean_t = torch.tensor(mean, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    done = 0
    for start in range(0, n, batch_size):
        idx = list(range(start, min(start + batch_size, n)))
        imgs = torch.stack([ds.shards[ds.index[i][0]]["images"][ds.index[i][1]] for i in idx])
        x = imgs.to(device, non_blocking=True).float().div_(255.0)
        x = ((x - mean_t) / std_t)
        if bf16:
            x = x.to(torch.bfloat16)
        logits = forward_fn(x)
        probs = torch.sigmoid(logits.float())
        out[start:start + len(idx)] = probs.cpu().numpy()
        done += len(idx)
        if done % (batch_size * 8) == 0 or done == n:
            log(f"  [{split}/{mode}] {done}/{n}")
    return out
