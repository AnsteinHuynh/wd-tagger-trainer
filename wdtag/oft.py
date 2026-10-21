"""OFTv2 (Orthogonal Fine-Tuning, v2) adapters.

Ported from OneTrainer's `modules/module/oft_utils.py` + `LoRAModule.OFTModule`
(`PeftType.OFT_2`), adapted to wrap arbitrary `nn.Linear` / `nn.Conv2d` modules
with a frozen base weight.

OFT parametrises a block-diagonal orthogonal rotation R as the Cayley transform
of a skew-symmetric matrix Q built from a learned parameter (per block, upper
triangle).  v2 differs from v1 in that the Cayley transform is expanded as a
truncated Neumann series instead of `torch.linalg.solve`, which is much cheaper
and fully differentiable.  At init the parameter is zero, so R = I exactly and
the adapted model starts out identical to the base model.

* `Linear`: rotating the *input* is mathematically equivalent to rotating the
  weights, and is cheaper, so we rotate activations.
* `Conv2d`: rotating the input would destroy spatial structure, so we rotate the
  (flattened) weight instead.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiplicativeDropoutLayer(nn.Module):
    """Randomly replaces learned rotation blocks with identity during training."""

    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.p > 0:
            d, h, _ = x.shape
            if d == 1:
                return x
            keep = 1.0 - self.p
            mask = torch.empty(d, 1, 1, device=x.device, dtype=x.dtype).bernoulli_(p=keep)
            eye = torch.eye(h, device=x.device, dtype=x.dtype).repeat(d, 1, 1)
            x = mask * x + (1 - mask) * eye
        return x


class OFTRotationModule(nn.Module):
    """Holds the OFT parameter and turns it into the `r` block rotations."""

    def __init__(
        self,
        r: int,
        n_elements: int,
        block_size: int,
        in_features: int,
        block_share: bool = False,
        oft_scaled: bool = False,
        use_cayley_neumann: bool = True,
        num_cayley_neumann_terms: int = 5,
        dropout_probability: float = 0.0,
    ):
        super().__init__()
        self.r = r
        self.n_elements = n_elements
        self.block_size = block_size
        self.in_features = in_features
        self.block_share = block_share
        self.oft_scaled = oft_scaled
        self.use_cayley_neumann = use_cayley_neumann
        self.num_cayley_neumann_terms = num_cayley_neumann_terms
        self.weight = nn.Parameter(torch.zeros(r, n_elements))
        rows, cols = torch.triu_indices(block_size, block_size, 1)
        self.register_buffer("rows", rows, persistent=False)
        self.register_buffer("cols", cols, persistent=False)
        self.dropout = MultiplicativeDropoutLayer(p=dropout_probability)

    def _skew(self, vec: torch.Tensor) -> torch.Tensor:
        b = vec.shape[0]
        m = torch.zeros(b, self.block_size, self.block_size, device=vec.device, dtype=vec.dtype)
        idx = torch.arange(b, device=vec.device)[:, None]
        m = m.index_put((idx, self.rows, self.cols), vec)
        return m - m.transpose(-2, -1)

    def _cayley_batch(self) -> torch.Tensor:
        q = self.weight
        if self.oft_scaled:
            q = q / (2 * math.sqrt(self.block_size - 1))
        q_skew = self._skew(q).to(torch.float32)  # keep the rotation in fp32
        b = q_skew.shape[0]
        if not self.use_cayley_neumann:
            eye = torch.eye(self.block_size, device=q_skew.device, dtype=q_skew.dtype)
            eye = eye.unsqueeze(0).expand(b, -1, -1)
            return torch.linalg.solve(eye + q_skew, eye - q_skew, left=False)

        n = self.num_cayley_neumann_terms
        r = torch.eye(self.block_size, device=q_skew.device, dtype=q_skew.dtype).repeat(b, 1, 1)
        r.add_(q_skew, alpha=2.0)
        if n > 2:
            q_pow = torch.bmm(q_skew, q_skew)
            r.add_(q_pow, alpha=2.0)
            for _ in range(3, n):
                q_pow = torch.bmm(q_pow, q_skew)
                r.add_(q_pow, alpha=2.0)
            q_pow = torch.bmm(q_pow, q_skew)
            r.add_(q_pow)
        return r

    def rotate_vector(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the block rotation to the last dimension of `x`."""
        rot = self.dropout(self._cayley_batch()).to(x.dtype)
        in_dtype = x.dtype
        x = x.to(rot.dtype)
        *lead, d = x.shape
        rank = d // self.block_size
        xr = x.reshape(*lead, rank, self.block_size)
        if self.block_share:
            rot = rot[0]
            out = torch.einsum("...rk,kc->...rc", xr, rot)
        else:
            rot = rot[:rank] if rot.shape[0] >= rank else rot.repeat(rank, 1, 1)
            out = torch.einsum("...rk,rkc->...rc", xr, rot)
        return out.reshape(*lead, d).to(in_dtype)

    def rotate_matrix(self, w: torch.Tensor) -> torch.Tensor:
        """Rotate the input axis of a weight matrix [out, in]."""
        rot = self.dropout(self._cayley_batch()).to(w.dtype)
        out_f, in_f = w.shape
        rank = in_f // self.block_size
        if self.block_share:
            rot = rot[0]
            wr = w.reshape(out_f, rank, self.block_size)
            return torch.einsum("ork,kc->orc", wr, rot).reshape(out_f, in_f)
        rot = rot[:rank]
        wr = w.reshape(out_f, rank, self.block_size)
        return torch.einsum("ork,rkc->orc", wr, rot).reshape(out_f, in_f)


class OFTLinear(nn.Module):
    """`nn.Linear` with a frozen weight whose *input* is orthogonally rotated."""

    def __init__(self, base: nn.Linear, block_size: int, block_share: bool,
                 oft_scaled: bool, dropout: float):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        in_f = base.in_features
        block_size = _adjust_block_size(in_f, block_size)
        self.rank = in_f // block_size
        self.oft = OFTRotationModule(
            r=1 if block_share else self.rank,
            n_elements=block_size * (block_size - 1) // 2,
            block_size=block_size,
            in_features=in_f,
            block_share=block_share,
            oft_scaled=oft_scaled,
            dropout_probability=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(self.oft.rotate_vector(x), self.base.weight, self.base.bias)

    # compatibility with code that pokes at `.weight` / `.bias` (e.g. HF models)
    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias


class OFTConv2d(nn.Module):
    """`nn.Conv2d` with a frozen kernel whose flattened weight is rotated."""

    def __init__(self, base: nn.Conv2d, block_size: int, block_share: bool,
                 oft_scaled: bool, dropout: float):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        in_f = base.in_channels * base.kernel_size[0] * base.kernel_size[1]
        if base.dilation[0] > 1 or base.dilation[1] > 1:
            raise ValueError("dilated Conv2d is not supported by OFT")
        block_size = _adjust_block_size(in_f, block_size)
        self.rank = in_f // block_size
        self.oft = OFTRotationModule(
            r=1 if block_share else self.rank,
            n_elements=block_size * (block_size - 1) // 2,
            block_size=block_size,
            in_features=in_f,
            block_share=block_share,
            oft_scaled=oft_scaled,
            dropout_probability=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o, i, kh, kw = self.base.weight.shape
        w = self.base.weight.detach().reshape(o, i * kh * kw)
        w = self.oft.rotate_matrix(w).reshape(o, i, kh, kw)
        return F.conv2d(x, w, self.base.bias, self.base.stride,
                        self.base.padding, self.base.dilation, self.base.groups)

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias


def _adjust_block_size(in_features: int, block_size: int) -> int:
    """Pick a valid block size dividing `in_features` (mirrors OneTrainer)."""
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if in_features % block_size == 0 and block_size <= in_features:
        return block_size
    if block_size > in_features:
        return in_features
    hi = block_size
    while hi <= in_features and in_features % hi != 0:
        hi += 1
    lo = block_size
    while lo > 1 and in_features % lo != 0:
        lo -= 1
    if hi > in_features:
        return lo
    return lo if (block_size - lo) <= (hi - block_size) else hi


def iter_oft_modules(module: nn.Module):
    for m in module.modules():
        if isinstance(m, (OFTLinear, OFTConv2d)):
            yield m
