"""Single-image tag inference example for the wdtag-ensemble-dinov3-siglip2 release.

Model: https://huggingface.co/ansteinhuynh/wdtag-ensemble-dinov3-siglip2

What the model is: a two-tower Danbooru-style multi-label tagger.
  Tower 1 (trained): DINOv3-B/16 with OFTv2 orthogonal adapters on every Linear and on
    the patch-embed Conv, plus one unfrozen final block, feeding a trained linear head.
  Tower 2 (frozen): SigLIP2-B/16 (also adapter-wrapped during its own training run, the
    weights are in siglip2-tower-ckpt-27718.pt). Only its pooled CLS feature is used,
    concatenated with tower 1's pooled CLS feature before the head.

The forward pass therefore needs BOTH checkpoints: the ensemble checkpoint alone expects
the frozen SigLIP2 features as a second input. This script computes that frozen view live
in fp32, exactly like the training-time caches did.

What you need on disk:
  1. ensemble-ckpt-13859.pt       (from the HF repo above)
  2. siglip2-tower-ckpt-27718.pt  (from the HF repo above)
  3. A WD-tagger v4 baseline model directory holding config.json + selected_tags.csv,
     e.g. the convnextv2_huge.dbv4-full release. This defines the 12,476-tag label
     space and its canonical order; the ensemble was trained against exactly it.
  4. The two base towers from HuggingFace (downloaded automatically on first run):
     facebook/dinov3-vitb16-pretrain-lvd1689m and google/siglip2-base-patch16-512.

Dependencies: torch, transformers (>= 5.16, needs DINOv3), pillow, numpy.
This file is fully self-contained: it inlines the minimal OFTv2 adapter and tower
definitions, so no wdtag package install is required.

Usage: edit the CONFIG block below, then `python infer_example.py`.
Command-line overrides are also accepted, e.g.
  python infer_example.py --image C:/pics/example.png --top-n 30
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# ============================================================================
# CONFIG: edit these paths to match your machine, then just run the script.
# ============================================================================

# The image you want tagged. Any common format (png/jpg/webp/bmp).
IMAGE_PATH = r"C:/path/to/your/image.png"

# The two checkpoints downloaded from the HF model repo.
ENSEMBLE_CKPT = r"C:/path/to/ensemble-ckpt-13859.pt"    # trained DINOv3 tower + head
SIGLIP_CKPT = r"C:/path/to/siglip2-tower-ckpt-27718.pt"  # frozen SigLIP2 tower weights

# WD-tagger v4 baseline model dir (config.json + selected_tags.csv), defines the label
# space. E.g. https://huggingface.co/SmilingWolf/wd-convnextv2-tagger-v3 style releases,
# for this model use the v4 convnextv2_huge.dbv4-full directory.
BASELINE_DIR = r"C:/path/to/convnextv2_huge.dbv4-full"

# Base towers. Either local dirs or HF repo ids (auto-downloaded on first use).
# NOTE: facebook/dinov3-vitb16-pretrain-lvd1689m is license-gated on HF; accept its
# license once on the model page, or point DINO_DIR at a local copy.
DINO_DIR = "facebook/dinov3-vitb16-pretrain-lvd1689m"
SIGLIP_DIR = "google/siglip2-base-patch16-512"

# Input resolution and preprocessing mode. The model was trained at 512x512 with the
# aspect ratio ignored (image squashed to a square). Do not change these.
IMAGE_SIZE = 512   # training resolution, keep at 512
SQUASH = True      # resize to a square without preserving aspect ratio, as in training

# Output tuning.
TOP_N = 30                 # how many general tags to print
GENERAL_THRESHOLD = 0.35   # drop general tags below this probability
CHARACTER_THRESHOLD = 0.75  # drop character tags below this probability
DEVICE = "auto"            # "auto" | "cuda" | "cpu"; bf16 is used on cuda, fp32 on cpu

# ============================================================================
# Adapter hyperparameters. These MUST match how the checkpoints were trained;
# they are listed here only so the OFT wrapper builds tensors of the right shape.
# Do not change them.
# ============================================================================
BLOCK_SIZE = 128
BLOCK_SHARE = False
OFT_SCALED = True
OFT_DROPOUT = 0.0
OFT_CONV = True      # the patch-embed Conv2d is adapted too
CAYLEY_TERMS = 0     # 0 = exact Cayley solve, what both runs were trained with

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
N_REGISTERS = 4      # DINOv3 prefixes patch tokens with cls + 4 register tokens

# Danbooru-style category ids used by selected_tags.csv: 9 = rating, 0 = general,
# 4 = character.
CAT_RATING, CAT_GENERAL, CAT_CHARACTER = 9, 0, 4


# ============================================================================
# OFTv2 adapters (minimal inline port of wdtag/oft.py; OneTrainer PeftType.OFT_2).
# A learnt block-diagonal orthogonal rotation, applied to the input of every frozen
# Linear and to the flattened weight of the patch-embed Conv. Parameter starts at
# zero, i.e. R = identity, so the wrapped tower initially computes exactly the
# pretrained function.
# ============================================================================


class OFTRotationModule(nn.Module):
    """OFT parameter -> r block rotations via the Cayley transform."""

    def __init__(self, r, n_elements, block_size, in_features, block_share,
                 oft_scaled, use_cayley_neumann, num_cayley_neumann_terms,
                 dropout_probability):
        super().__init__()
        self.r = r
        self.block_size = block_size
        self.block_share = block_share
        self.oft_scaled = oft_scaled
        self.use_cayley_neumann = use_cayley_neumann
        self.num_cayley_neumann_terms = num_cayley_neumann_terms
        self.weight = nn.Parameter(torch.zeros(r, n_elements))
        rows, cols = torch.triu_indices(block_size, block_size, 1)
        self.register_buffer("rows", rows, persistent=False)
        self.register_buffer("cols", cols, persistent=False)
        # dropout stays inert at inference (module in eval mode); kept for parity

    def _skew(self, vec):
        b = vec.shape[0]
        m = torch.zeros(b, self.block_size, self.block_size, device=vec.device, dtype=vec.dtype)
        idx = torch.arange(b, device=vec.device)[:, None]
        m = m.index_put((idx, self.rows, self.cols), vec)
        return m - m.transpose(-2, -1)

    def _cayley_batch(self):
        q = self.weight
        if self.oft_scaled:
            q = q / (2 * math.sqrt(self.block_size - 1))
        q_skew = self._skew(q).to(torch.float32)  # rotation always computed in fp32
        b = q_skew.shape[0]
        eye = torch.eye(self.block_size, device=q_skew.device, dtype=q_skew.dtype)
        eye = eye.unsqueeze(0).expand(b, -1, -1)
        # exact solve (CAYLEY_TERMS = 0): (I-Q)^-1 (I+Q); order matters
        return torch.linalg.solve(eye - q_skew, eye + q_skew)

    def rotate_vector(self, x):
        rot = self._cayley_batch().to(x.dtype)
        in_dtype = x.dtype
        x = x.to(rot.dtype)
        *lead, d = x.shape
        rank = d // self.block_size
        xr = x.reshape(*lead, rank, self.block_size)
        rot = rot[:rank] if rot.shape[0] >= rank else rot.repeat(rank, 1, 1)
        out = torch.einsum("...rk,rkc->...rc", xr, rot)
        return out.reshape(*lead, d).to(in_dtype)

    def rotate_matrix(self, w):
        rot = self._cayley_batch().to(w.dtype)
        out_f, in_f = w.shape
        rank = in_f // self.block_size
        rot = rot[:rank]
        wr = w.reshape(out_f, rank, self.block_size)
        return torch.einsum("ork,rkc->orc", wr, rot).reshape(out_f, in_f)


class OFTLinear(nn.Module):
    """nn.Linear with a frozen weight and an orthogonally rotated input."""

    def __init__(self, base: nn.Linear):
        super().__init__()
        self.base = base
        in_f = base.in_features
        bs = _adjust_block_size(in_f, BLOCK_SIZE)
        self.oft = OFTRotationModule(
            r=in_f // bs,
            n_elements=bs * (bs - 1) // 2,
            block_size=bs,
            in_features=in_f,
            block_share=BLOCK_SHARE,
            oft_scaled=OFT_SCALED,
            use_cayley_neumann=CAYLEY_TERMS > 0,
            num_cayley_neumann_terms=max(5, CAYLEY_TERMS),
            dropout_probability=OFT_DROPOUT,
        )

    def forward(self, x):
        return F.linear(self.oft.rotate_vector(x), self.base.weight, self.base.bias)

    # `.weight` / `.bias` compatibility for HF code that inspects them
    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias


class OFTConv2d(nn.Module):
    """nn.Conv2d with a frozen kernel whose flattened weight is rotated."""

    def __init__(self, base: nn.Conv2d):
        super().__init__()
        self.base = base
        in_f = base.in_channels * base.kernel_size[0] * base.kernel_size[1]
        bs = _adjust_block_size(in_f, BLOCK_SIZE)
        self.oft = OFTRotationModule(
            r=in_f // bs,
            n_elements=bs * (bs - 1) // 2,
            block_size=bs,
            in_features=in_f,
            block_share=BLOCK_SHARE,
            oft_scaled=OFT_SCALED,
            use_cayley_neumann=CAYLEY_TERMS > 0,
            num_cayley_neumann_terms=max(5, CAYLEY_TERMS),
            dropout_probability=OFT_DROPOUT,
        )

    def forward(self, x):
        o, i, kh, kw = self.base.weight.shape
        w = self.base.weight.reshape(o, i * kh * kw)
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
    """Pick the divisor of in_features closest to block_size (mirrors OneTrainer)."""
    if in_features % block_size == 0 and block_size <= in_features:
        return block_size
    if block_size > in_features:
        return in_features
    hi, lo = block_size, block_size
    while hi <= in_features and in_features % hi != 0:
        hi += 1
    while lo > 1 and in_features % lo != 0:
        lo -= 1
    if hi > in_features:
        return lo
    return lo if (block_size - lo) <= (hi - block_size) else hi


def wrap_oft(backbone: nn.Module) -> None:
    """Wrap every Linear (and the patch-embed Conv when OFT_CONV) in an OFT module.

    The wrapping order and naming here are identical to the training code, so the
    checkpoint's parameter keys line up one-to-one with this module tree.
    """
    for name, mod in list(backbone.named_modules()):
        if isinstance(mod, nn.Linear):
            new = OFTLinear(mod)
        elif isinstance(mod, nn.Conv2d) and OFT_CONV:
            new = OFTConv2d(mod)
        else:
            continue
        parent_name, _, child = name.rpartition(".")
        parent = backbone.get_submodule(parent_name) if parent_name else backbone
        setattr(parent, child, new)


def load_trained_params(module: nn.Module, ckpt_path: str,
                        prefix: str = "") -> None:
    """Load a run checkpoint's `adapter` dict (all trained tensors) into `module`.

    Keys are matched by exact name; every checkpoint tensor must land somewhere,
    otherwise the architecture and the checkpoint disagree. `prefix` selects a
    subtree ("backbone." for the frozen SigLIP2 tower, whose checkpoint also
    carries that run's own training head, which inference does not need).
    """
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = blob["adapter"]
    if prefix:
        state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        if not state:
            raise RuntimeError(f"{ckpt_path}: no {prefix}* tensors to load")
    own = dict(module.named_parameters())
    used = 0
    with torch.no_grad():
        for k, v in state.items():
            if k in own and own[k].shape == v.shape:
                own[k].copy_(v)
                used += 1
    n_missed = len(state) - used
    if n_missed:
        raise RuntimeError(
            f"{ckpt_path}: {n_missed}/{len(state)} tensors did not match this module "
            f"tree; the OFT hyperparameters above likely diverged from training")


# ============================================================================
# The two towers (minimal inline port of wdtag/model.py, inference-only).
# Parameter names mirror the training class exactly so checkpoints load as-is.
# ============================================================================


class EnsembleTagger(nn.Module):
    """Trained DINOv3-B/16 tower (OFT-adapted, last block unfrozen) + linear head.

    forward() takes the raw uint8 image tensor and the precomputed frozen SigLIP2
    CLS feature, normalises each tower with its own LayerNorm (they have different
    output statistics), concatenates, and applies the head.
    """

    def __init__(self, n_labels: int, dtype: torch.dtype):
        super().__init__()
        from transformers import DINOv3ViTModel

        self.backbone = DINOv3ViTModel.from_pretrained(DINO_DIR, dtype=dtype)
        wrap_oft(self.backbone)  # rotations start at identity
        self.patch_offset = 1 + N_REGISTERS  # unused for cls pooling, kept for parity
        hidden = self.backbone.config.hidden_size  # 768 for ViT-B
        # the frozen SigLIP2 side is 768-wide; head input is the concatenation
        self.norm = nn.LayerNorm(hidden)          # tower 1
        self.norm2 = nn.LayerNorm(hidden)         # frozen tower 2
        # head in fp32 (head_fp32 in training); Sequential index 1 = the Linear,
        # matching the training naming so head.1.* checkpoint keys resolve
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(2 * hidden, n_labels))
        self.head = self.head.to(torch.float32)
        self.norm = self.norm.to(torch.float32)
        self.norm2 = self.norm2.to(torch.float32)
        self.register_buffer(
            "in_mean", torch.tensor(IMAGENET_MEAN, dtype=dtype).view(1, 3, 1, 1),
            persistent=False)
        self.register_buffer(
            "in_std", torch.tensor(IMAGENET_STD, dtype=dtype).view(1, 3, 1, 1),
            persistent=False)

    def _prep(self, x_uint8: torch.Tensor) -> torch.Tensor:
        """uint8 [0,255] -> ImageNet-normalised, tower dtype."""
        x = x_uint8.to(self.in_mean.dtype).div_(255.0)
        return (x - self.in_mean) / self.in_std

    def forward(self, x_uint8: torch.Tensor, frozen_feats: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(pixel_values=self._prep(x_uint8)).pooler_output
        feats = torch.cat(
            [self.norm(feats.float()), self.norm2(frozen_feats.float())], dim=-1)
        return self.head(feats)


class SiglipTower(nn.Module):
    """Frozen SigLIP2-B/16 tower; only its pooled CLS feature is exposed.

    Built in fp32 (the training caches held fp32 features, so inference computes
    the same view live in fp32). Its weights come from SIGLIP_CKPT.
    """

    def __init__(self):
        super().__init__()
        from transformers import SiglipVisionModel

        self.backbone = SiglipVisionModel.from_pretrained(SIGLIP_DIR, dtype=torch.float32)
        wrap_oft(self.backbone)

    def features(self, x_uint8: torch.Tensor) -> torch.Tensor:
        # SigLIP2 normalisation: (x/255 - 0.5) / 0.5, in fp32
        x = x_uint8.to(torch.float32).div_(255.0)
        x = (x - 0.5) / 0.5
        return self.backbone(pixel_values=x).pooler_output


# ============================================================================
# Label space and image IO.
# ============================================================================


def load_labels(baseline_dir: str) -> tuple[list[str], list[int]]:
    """12,476 tags in the canonical order baked into the WD-tagger v4 baseline."""
    with open(os.path.join(baseline_dir, "config.json"), encoding="utf-8") as f:
        names = json.load(f)["tags"]
    with open(os.path.join(baseline_dir, "selected_tags.csv"), encoding="utf-8",
              newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != len(names) or [r["name"] for r in rows] != names:
        raise RuntimeError(
            "baseline dir mismatch: selected_tags.csv row order differs from "
            "config.json['tags']")
    return names, [int(r["category"]) for r in rows]


def load_image(path: str) -> torch.Tensor:
    """PIL load -> RGB -> squash-resize to IMAGE_SIZE -> uint8 [1,3,H,W] tensor."""
    img = Image.open(path).convert("RGB")
    if SQUASH:
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC)
    else:
        # not used by the released model, provided for completeness
        scale = IMAGE_SIZE / max(img.size)
        img = img.resize((round(img.width * scale), round(img.height * scale)),
                         Image.BICUBIC)
        canvas = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE))
        canvas.paste(img, ((IMAGE_SIZE - img.width) // 2, (IMAGE_SIZE - img.height) // 2))
        img = canvas
    arr = np.asarray(img, dtype=np.uint8)          # H,W,3 uint8
    t = torch.from_numpy(arr.copy()).permute(2, 0, 1).contiguous()  # 3,H,W
    return t.unsqueeze(0)                           # 1,3,H,W


def print_tags(names, categories, probs) -> None:
    """Print the best rating, plus general and character tags sorted by confidence."""
    order = np.argsort(probs)[::-1]
    rating = [i for i in order if categories[i] == CAT_RATING][:1]
    general = [i for i in order
               if categories[i] == CAT_GENERAL and probs[i] >= GENERAL_THRESHOLD][:TOP_N]
    chars = [i for i in order
             if categories[i] == CAT_CHARACTER and probs[i] >= CHARACTER_THRESHOLD]

    print("\n===== rating =====")
    for i in rating:
        print(f"  {names[i]:30s} {probs[i]:.4f}")
    print("===== top general tags =====")
    for i in general:
        print(f"  {names[i]:30s} {probs[i]:.4f}")
    print("===== characters =====")
    if chars:
        for i in chars:
            print(f"  {names[i]:30s} {probs[i]:.4f}")
    else:
        print("  (none above threshold)")
    print(f"\n{len(general)} general tag(s) at threshold {GENERAL_THRESHOLD}, "
          f"{len(chars)} character tag(s) at threshold {CHARACTER_THRESHOLD}")


# ============================================================================
# Main.
# ============================================================================


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--image", default=IMAGE_PATH, help="image file to tag")
    ap.add_argument("--ensemble-ckpt", default=ENSEMBLE_CKPT)
    ap.add_argument("--siglip-ckpt", default=SIGLIP_CKPT)
    ap.add_argument("--baseline-dir", default=BASELINE_DIR)
    ap.add_argument("--dino-dir", default=DINO_DIR)
    ap.add_argument("--siglip-dir", default=SIGLIP_DIR)
    ap.add_argument("--device", default=DEVICE,
                    help='"auto", "cuda" or "cpu" (bf16 on cuda, fp32 on cpu)')
    ap.add_argument("--top-n", type=int, default=TOP_N)
    args = ap.parse_args()
    globals()["TOP_N"] = args.top_n
    globals()["DINO_DIR"] = args.dino_dir
    globals()["SIGLIP_DIR"] = args.siglip_dir

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    tower_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    names, categories = load_labels(args.baseline_dir)
    print(f"label space: {len(names)} tags from {args.baseline_dir}")

    # frozen SigLIP2 view, always fp32
    print(f"loading frozen SigLIP2 tower: {SIGLIP_DIR} + {os.path.basename(args.siglip_ckpt)}")
    sig = SiglipTower().to(device)
    load_trained_params(sig.backbone, args.siglip_ckpt, prefix="backbone.")
    sig.eval()

    # trained DINOv3 ensemble tower + head
    print(f"loading ensemble tower: {DINO_DIR} + {os.path.basename(args.ensemble_ckpt)}")
    ens = EnsembleTagger(len(names), tower_dtype).to(device)
    load_trained_params(ens, args.ensemble_ckpt)
    ens.eval()

    print(f"tagging {args.image} on {device} ({tower_dtype})")
    x = load_image(args.image).to(device)

    with torch.no_grad():
        frozen_feats = sig.features(x)      # second view, fp32 CLS pool
        logits = ens(x, frozen_feats)       # two-tower forward
        probs = torch.sigmoid(logits.float())[0].cpu().numpy()

    print_tags(names, categories, probs)


if __name__ == "__main__":
    main()
