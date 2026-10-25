"""One-step displacement of each ladder rung, measured directly on CPU.

The question this answers is whether a rung's score difference is the switch changing the
direction of the update or the switch changing how far it steps. A constant gradient removes
the gradient from the comparison, so any difference is the operator, not the data.

Displacement is the Frobenius norm of (param before - param after), summed across tensors,
measured on the two shapes that matter here: a 768x768 adapter-sized matrix and the
12476x768 classifier head, which is where a per-row Sinkhorn has the most to say.
"""
from __future__ import annotations

import math

import torch

from adv_optm import SinkSGD_adv

SHAPES = {"adapter_768": (768, 768), "head_12476": (12476, 768)}
LR = 5e-4
G = 1e-3  # constant gradient entry


def _grad(shape, dtype):
    g = torch.Generator().manual_seed(7)
    return torch.randn(shape, generator=g, dtype=dtype) * G


def build(shape: tuple[int, int], dtype=torch.float32):
    p = torch.nn.Parameter(torch.randn(shape, dtype=dtype) * 0.02)
    return p


def run(kwargs: dict, shape: tuple[int, int], steps: int = 50) -> float:
    p = build(shape)
    opt = SinkSGD_adv([{"params": [p], "lr": LR}], momentum=0.95, weight_decay=0.0,
                      sinkhorn_iterations=5, stochastic_rounding=True,
                      compiled_optimizer=False, **kwargs)
    last = 0.0
    for i in range(steps):
        before = p.detach().clone()
        p.grad = _grad(shape, p.dtype)
        opt.step()
        if i == steps - 1:
            last = float((before - p.detach()).norm())
    return last


RUNGS = {
    "r0 plain fp32": {},
    "ntm only": {"normed_momentum": True},
    "r1 +ntm+snr": {"normed_momentum": True, "snr_cond": True},
    "r2 +orthograd": {"normed_momentum": True, "snr_cond": True,
                      "orthogonal_gradient": "iterative"},
    "r3 +ortho_sinkhorn": {"normed_momentum": True, "snr_cond": True,
                           "orthogonal_gradient": "iterative", "orthogonal_sinkhorn": True},
    "r4 +spectral": {"normed_momentum": True, "snr_cond": True,
                     "orthogonal_gradient": "iterative", "orthogonal_sinkhorn": True,
                     "spectral_normalization": True},
    "nesterov only": {"nesterov": True},
    "spectral only": {"spectral_normalization": True},
    "orthograd only": {"orthogonal_gradient": "iterative"},
    "ortho_sinkhorn only": {"orthogonal_sinkhorn": True},
}

if __name__ == "__main__":
    for shape_name, shape in SHAPES.items():
        base = None
        print(f"[{shape_name}] shape {shape}", flush=True)
        for name, kw in RUNGS.items():
            try:
                d = run(kw, shape)
            except Exception as exc:  # noqa: BLE001 - report and keep going
                print(f"  {name:22s} FAILED {type(exc).__name__}: {exc}", flush=True)
                continue
            if base is None:
                base = d
            print(f"  {name:22s} displacement {d:.6e}  ratio {d / base:.4f}", flush=True)


