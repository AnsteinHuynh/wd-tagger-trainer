"""DINOv3-ViT/B + OFTv2 tagger."""
from __future__ import annotations

import torch
import torch.nn as nn

from .config import DINO_DIR, TrainConfig
from .oft import OFTConv2d, OFTLinear, iter_oft_modules

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
N_REGISTERS = 4


def _replace_child(parent: nn.Module, name: str, new: nn.Module) -> None:
    setattr(parent, name, new)


def apply_oft_to_backbone(backbone: nn.Module, cfg: TrainConfig) -> int:
    """Wrap every transformer Linear (and optionally the patch-embed Conv) in OFTv2."""
    targets: list[tuple[nn.Module, str, nn.Module]] = []
    for name, mod in backbone.named_modules():
        if isinstance(mod, nn.Linear):
            targets.append((mod, name, "linear"))
        elif isinstance(mod, nn.Conv2d) and cfg.oft_conv:
            targets.append((mod, name, "conv"))

    for mod, name, kind in targets:
        parent_name, _, child = name.rpartition(".")
        parent = backbone.get_submodule(parent_name) if parent_name else backbone
        if kind == "linear":
            new = OFTLinear(mod, cfg.block_size, cfg.block_share, cfg.oft_scaled, cfg.oft_dropout)
        else:
            new = OFTConv2d(mod, cfg.block_size, cfg.block_share, cfg.oft_scaled, cfg.oft_dropout)
        _replace_child(parent, child, new)
    return len(targets)


class DinoTagger(nn.Module):
    """Frozen DINOv3-B/16 with OFTv2 adapters and a multi-label head."""

    def __init__(self, cfg: TrainConfig, n_labels: int, dtype=torch.bfloat16):
        super().__init__()
        from transformers import DINOv3ViTModel

        self.cfg = cfg
        self.n_labels = n_labels
        self.backbone = DINOv3ViTModel.from_pretrained(DINO_DIR, dtype=dtype)
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.n_oft = apply_oft_to_backbone(self.backbone, cfg)
        self.backbone.gradient_checkpointing = False

        hidden = self.backbone.config.hidden_size
        pool_dim = hidden * (2 if cfg.head_pool == "cls_mean" else 1)
        self.norm = nn.LayerNorm(pool_dim)
        layers: list[nn.Module] = [nn.Dropout(cfg.head_dropout)]
        if cfg.head_hidden > 0:
            layers += [nn.Linear(pool_dim, cfg.head_hidden), nn.GELU(),
                       nn.Dropout(cfg.head_dropout), nn.Linear(cfg.head_hidden, n_labels)]
        else:
            layers += [nn.Linear(pool_dim, n_labels)]
        self.head = nn.Sequential(*layers)
        self.head = self.head.to(dtype)
        self.norm = self.norm.to(dtype)
        self.register_buffer("in_mean", torch.tensor(IMAGENET_MEAN, dtype=dtype).view(1, 3, 1, 1),
                             persistent=False)
        self.register_buffer("in_std", torch.tensor(IMAGENET_STD, dtype=dtype).view(1, 3, 1, 1),
                             persistent=False)

    def _prep(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """uint8 [0,255] -> normalised bf16, done on the GPU when given uint8."""
        if pixel_values.dtype == torch.uint8:
            x = pixel_values.to(self.in_mean.dtype).div_(255.0)
            return (x - self.in_mean) / self.in_std
        return pixel_values

    # ---------------------------------------------------------------- features
    def forward_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.backbone(pixel_values=pixel_values)
        cls = out.pooler_output                       # already LayerNorm'ed CLS
        if self.cfg.head_pool == "cls":
            return cls
        patches = out.last_hidden_state[:, 1 + N_REGISTERS:, :]
        mean = patches.mean(dim=1)
        if self.cfg.head_pool == "mean":
            return mean
        return torch.cat([cls, mean], dim=-1)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.head(self.norm(self.forward_features(self._prep(pixel_values))))

    def init_head_bias(self, priors) -> None:
        """Initialise the output bias to the log-odds of each tag's prior.

        With a 0.26% positive rate a zero bias makes the first many steps
        nothing but 'predict zero'; starting at the prior skips that.
        """
        import numpy as _np

        last = None
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                last = m
        if last is None or last.bias is None or last.out_features != len(priors):
            return
        p = _np.clip(_np.asarray(priors, dtype=_np.float64), 1e-6, 1 - 1e-6)
        bias = _np.log(p / (1 - p))
        with torch.no_grad():
            last.bias.copy_(torch.tensor(bias, dtype=last.bias.dtype, device=last.bias.device))

    # -------------------------------------------------------------- parameters
    def trainable_parameters(self):
        oft, head = [], []
        for m in iter_oft_modules(self.backbone):
            oft.append(m.oft.weight)
        head.extend(self.norm.parameters())
        head.extend(self.head.parameters())
        return oft, head

    def adapter_state_dict(self) -> dict:
        state = {k: v.detach().clone() for k, v in self.named_parameters()
                 if k.endswith("oft.weight")}
        state.update({f"head.{k}": v.detach().clone() for k, v in self.head.state_dict().items()})
        state.update({f"norm.{k}": v.detach().clone() for k, v in self.norm.state_dict().items()})
        return state

    def load_adapter_state_dict(self, state: dict) -> None:
        missing, unexpected = [], []
        own = dict(self.named_parameters())
        for k, v in state.items():
            if k in own and own[k].shape == v.shape:
                with torch.no_grad():
                    own[k].copy_(v)
            else:
                missing.append(k)
        for k in own:
            if k not in state and own[k].requires_grad:
                unexpected.append(k)
        if missing or unexpected:
            raise RuntimeError(f"adapter state mismatch: missing={missing[:4]} unexpected={unexpected[:4]}")

    def count_params(self) -> dict:
        oft, head = self.trainable_parameters()
        return {
            "oft": sum(p.numel() for p in oft),
            "head": sum(p.numel() for p in head),
            "backbone": sum(p.numel() for p in self.backbone.parameters()),
        }
