"""Compare the classifier head's row norms between two trained runs.

Sinkhorn preconditioning forces every row of a gradient matrix to equal RMS, so if normed
momentum is applied to the head it should flatten the per-tag weight scales. Frequent tags
normally earn bigger rows than rare ones, so a flat distribution is evidence the geometry
switch is doing real damage there, and a varied one is evidence it is not.
"""
from __future__ import annotations

import sys

import torch


def head_rows(path: str) -> torch.Tensor:
    d = torch.load(path, map_location="cpu", weights_only=False)
    sd = d["adapter"]
    w = sd["head.1.weight"].float()
    return w.norm(dim=1)


def describe(name: str, norms: torch.Tensor) -> None:
    q = torch.quantile(norms, torch.tensor([0.01, 0.25, 0.5, 0.75, 0.99]))
    cv = float(norms.std() / norms.mean())
    ratio = float(norms.max() / norms.min())
    print(f"{name}: n {norms.numel()} mean {norms.mean():.4f} cv {cv:.4f} "
          f"max/min {ratio:.2f} p1 {q[0]:.4f} p25 {q[1]:.4f} p50 {q[2]:.4f} "
          f"p75 {q[3]:.4f} p99 {q[4]:.4f}", flush=True)


if __name__ == "__main__":
    for spec in sys.argv[1:]:
        label, path = spec.split("=", 1)
        describe(label, head_rows(path))
