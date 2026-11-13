"""Vision transformer + OFTv2 tagger, with DINOv3-ViT or SigLIP2 as the adapted tower."""
from __future__ import annotations

import torch
import torch.nn as nn

from .config import DINO_DIR, TrainConfig
from .oft import OFTConv2d, OFTLinear, iter_oft_modules

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
N_REGISTERS = 4

# Head stats in `head_pool` order: cls_mean 2x768, cls_mean_max_std 4x768=3072, near ConvNeXt-V2-Huge stage-4 2816.
POOL_PARTS = ("cls", "mean", "max", "std")


def backbone_kind(path: str) -> str:
    """Which tower lives at `path`, read from its own config.json rather than from a cfg field:
    a new field would change the sha8 of every config for the sake of one boolean."""
    import json
    import os

    try:
        with open(os.path.join(path, "config.json"), encoding="utf-8") as fh:
            raw = json.load(fh)
    except OSError:
        return "dino"
    return "siglip2" if str(raw.get("model_type", "")).startswith("siglip") else "dino"


def backbone_layers(backbone: nn.Module) -> list[nn.Module]:
    """Block list for either tower: DINOv3ViTModel keeps them in `.model.layer`,
    SiglipVisionModel in `.encoder.layers` (transformers 5.16 verified)."""
    if hasattr(backbone, "encoder"):
        return list(backbone.encoder.layers)
    return list(backbone.model.layer)


def pool_parts(name: str) -> tuple[str, ...]:
    parts = tuple(p for p in name.split("_") if p)
    bad = [p for p in parts if p not in POOL_PARTS]
    if bad or not parts:
        raise ValueError(f"head_pool {name!r} has unknown parts {bad}; "
                         f"known parts are {POOL_PARTS}")
    return parts


def _replace_child(parent: nn.Module, name: str, new: nn.Module) -> None:
    setattr(parent, name, new)


def apply_oft_to_backbone(backbone: nn.Module, cfg: TrainConfig) -> int:
    """Wrap transformer Linears in OFTv2 (patch-embed Conv too when `oft_conv`)."""
    targets: list[tuple[nn.Module, str, nn.Module]] = []
    for name, mod in backbone.named_modules():
        if isinstance(mod, nn.Linear):
            targets.append((mod, name, "linear"))
        elif isinstance(mod, nn.Conv2d) and cfg.oft_conv:
            targets.append((mod, name, "conv"))

    terms = int(getattr(cfg, "cayley_terms", 5))
    for mod, name, kind in targets:
        parent_name, _, child = name.rpartition(".")
        parent = backbone.get_submodule(parent_name) if parent_name else backbone
        if kind == "linear":
            new = OFTLinear(mod, cfg.block_size, cfg.block_share, cfg.oft_scaled,
                            cfg.oft_dropout, terms)
        else:
            new = OFTConv2d(mod, cfg.block_size, cfg.block_share, cfg.oft_scaled,
                            cfg.oft_dropout, terms)
        _replace_child(parent, child, new)
    return len(targets)


def unfreeze_last_blocks(backbone: nn.Module, n: int) -> int:
    """Unfreeze the last `n` blocks' base weights; their OFT rotation stays at theta=0 (Cayley(0)=I), avoiding a redundant rotation-beside-weight parameterisation."""
    layers = backbone_layers(backbone)
    if n < 1 or n > len(layers):
        raise ValueError(f"unfreeze_blocks={n} outside 1..{len(layers)}")
    for blk in layers[len(layers) - n:]:
        for name, p in blk.named_parameters():
            p.requires_grad_(not name.endswith("oft.weight"))
    return n


def _parse_blocks(spec: str, n_layers: int) -> list[int]:
    out = []
    for part in str(spec or "").replace(" ", "").split(","):
        if not part:
            continue
        i = int(part)
        if not 0 <= i < n_layers:
            raise ValueError(f"block {i} outside 0..{n_layers - 1}")
        out.append(i)
    return out


def apply_block_grad_overrides(backbone: nn.Module, cfg) -> dict:
    """Arms whose handle on a block is its rotation rather than its weights.

    `unfreeze_last_blocks` gives an unfrozen block exactly one handle by freezing its OFT
    rotation, because a free weight beside a free rotation is gauge-redundant: the rotation is
    absorbed by the trained weight and the two just fight over the same product. Some arms want
    the opposite split (weights frozen, rotation trained), or a rotation enabled in a block above
    the unfreeze window, and a block count cannot express that. Applied after the count:
      freeze_block_weights="11"   freeze these blocks' weights, biases and LayerNorms
      train_oft_blocks="11"       re-enable these blocks' rotation parameters
      train_oft_pooler=1          re-enable the pooler's rotation parameters (backbone.head)
    Counts are returned so a mistyped spec shows up in the log/cfg before training starts.
    """
    layers = backbone_layers(backbone)
    frozen_w = train_oft = 0
    for i in _parse_blocks(getattr(cfg, "freeze_block_weights", ""), len(layers)):
        for name, p in layers[i].named_parameters():
            if not name.endswith("oft.weight"):
                p.requires_grad_(False)
                frozen_w += 1
    for i in _parse_blocks(getattr(cfg, "train_oft_blocks", ""), len(layers)):
        for name, p in layers[i].named_parameters():
            if name.endswith("oft.weight"):
                p.requires_grad_(True)
                train_oft += 1
    if getattr(cfg, "train_oft_pooler", False):
        for name, p in backbone.head.named_parameters():
            if name.endswith("oft.weight"):
                p.requires_grad_(True)
                train_oft += 1
    return {"frozen_weight_tensors": frozen_w, "trainable_rotations": train_oft}


def enable_block_checkpointing(backbone: nn.Module, n: int) -> int:
    """Recompute activations for the last `n` blocks.

    At 512px one block is ~1.5 GB at batch 48 (1029x768x48x~10 bf16), which OOMs well before
    four, while this keeps only the block's input (~76 MB) and pays one extra forward.

    A tower with no unfrozen block still needs this when OFT rotations sit inside the blocks:
    the weights are frozen but the graph runs through the rotations, so every block's
    activations are retained exactly as if it were unfrozen. The gating in `DinoTagger` used to
    assume otherwise, which made `grad_checkpoint_blocks` silently do nothing on an OFT-only run.
    """
    import torch.utils.checkpoint as cp

    layers = backbone_layers(backbone)
    if n < 1 or n > len(layers):
        raise ValueError(f"checkpoint blocks={n} outside 1..{len(layers)}")
    for blk in layers[len(layers) - n:]:
        if getattr(blk, "_ckpt_wrapped", False):
            continue
        orig = blk.forward

        def wrapped(*args, _fn=orig, **kwargs):
            return cp.checkpoint(_fn, *args, use_reentrant=False, **kwargs)

        blk.forward = wrapped
        blk._ckpt_wrapped = True
    return n


class DinoTagger(nn.Module):
    """Frozen-or-adapted vision tower with OFTv2 adapters and a multi-label head.

    The tower is DINOv3-B/16 by default and SigLIP2-B/16 when `cfg.backbone_dir` points at a
    siglip checkpoint; the class name is historical.
    """

    def __init__(self, cfg: TrainConfig, n_labels: int, dtype=torch.bfloat16):
        super().__init__()
        from transformers import DINOv3ViTModel

        self.cfg = cfg
        self.n_labels = n_labels
        self.kind = backbone_kind(cfg.backbone_dir or DINO_DIR)
        if self.kind == "siglip2":
            from transformers import SiglipVisionModel
            self.backbone = SiglipVisionModel.from_pretrained(cfg.backbone_dir, dtype=dtype)
        else:
            self.backbone = DINOv3ViTModel.from_pretrained(cfg.backbone_dir or DINO_DIR, dtype=dtype)
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.n_oft = apply_oft_to_backbone(self.backbone, cfg)
        self.n_unfrozen = 0
        if getattr(cfg, "unfreeze_blocks", 0) > 0:
            # after the OFT wrap: it freezes the rotations that wrap just created
            self.n_unfrozen = unfreeze_last_blocks(self.backbone, cfg.unfreeze_blocks)
        # per-block overrides for arms that adapt a block by rotation rather than by weights
        self.grad_overrides = apply_block_grad_overrides(self.backbone, cfg)
        self.backbone.gradient_checkpointing = False
        # DINOv3 prefixes its patch tokens with a cls token and four registers; SigLIP2 has
        # neither, and its 1024 tokens are all image patches.
        self.patch_offset = 1 + N_REGISTERS if self.kind == "dino" else 0
        # An OFT-only run keeps a graph through every block, so checkpointing the whole encoder
        # is what buys memory; without adapters there is nothing to recompute for and the gate
        # stays off.
        _ck = getattr(cfg, "grad_checkpoint_blocks", False)
        if _ck and (self.n_unfrozen > 0 or self.n_oft > 0):
            n_layers = len(backbone_layers(self.backbone))
            if isinstance(_ck, bool):  # True = every block that carries a backward
                n_ck = self.n_unfrozen if self.n_unfrozen > 0 else n_layers
            else:                      # int N = only the last N, for speed at known VRAM
                n_ck = int(_ck)
            n_ck = enable_block_checkpointing(self.backbone, n_ck)
            print(f"[ckpt] gradient checkpointing on the last {n_ck} of {n_layers} block(s)", flush=True)

        hidden = self.backbone.config.hidden_size
        self.parts = pool_parts(cfg.head_pool)
        pool_dim = hidden * len(self.parts)
        # Second tower, frozen with no adapter inside, so it runs under no_grad: no retained
        # activations, no backward, and its features are exactly what a precomputed cache would
        # hold. Measured at +1.6% macro and +18.5% micro over the best single tower on cached
        # features, and it is the only way two towers fit in 24.5 GB at batch 48.
        self.siglip = None
        self.sig_dim = 0
        if getattr(cfg, "siglip_dir", ""):
            from transformers import SiglipVisionModel
            self.siglip = SiglipVisionModel.from_pretrained(cfg.siglip_dir, dtype=dtype).eval()
            ckpt_path = getattr(cfg, "siglip_ckpt", "")
            if ckpt_path:
                # Bake a trunk we trained into the frozen side. The OFT wrap must happen before
                # the load, because the rotation tensors only exist once the modules are wrapped,
                # and the wrapping is what makes the checkpoint's keys line up at all.
                apply_oft_to_backbone(self.siglip, cfg)
                blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                state = blob.get("adapter") or blob.get("model") or blob
                state = {k[len("backbone."):]: v for k, v in state.items()
                         if k.startswith("backbone.")}
                if not state:
                    raise SystemExit(f"siglip_ckpt {ckpt_path}: no backbone.* tensors to load")
                missing, unexpected = self.siglip.load_state_dict(state, strict=False)
                bad = [k for k in missing if "rotations" not in k]
                if bad:
                    raise SystemExit(
                        f"siglip_ckpt {ckpt_path}: {len(bad)} tensors missing from the frozen "
                        f"tower's own state, first {bad[:3]}; the OFT settings of this config do "
                        f"not match the checkpoint it was trained with")
                print(f"[siglip] baked {len(state)} trained tensors into the frozen tower "
                      f"({ckpt_path}), {len(missing)} keys left at init", flush=True)
            for p in self.siglip.parameters():
                p.requires_grad_(False)
            self.siglip.eval()
            self.sig_dim = self.siglip.config.hidden_size
            pool_dim += self.sig_dim
        # Precomputed features for the frozen side: the training loop supplies them and the tower
        # above is then only used when they are absent, which keeps validation identical to what
        # the cache holds. The width must match, or the head would be built for the wrong input.
        self.feat_cache = None
        if getattr(cfg, "feat_cache", ""):
            from .featcache import FeatCache
            spec = getattr(cfg, "feat_cache_cols", "") or ""
            lo, hi = (int(x) for x in spec.split(":")) if spec else (0, None)
            self.feat_cache = FeatCache(cfg.feat_cache, (lo, hi) if hi else None)
            if self.sig_dim and self.feat_cache.width != self.sig_dim:
                raise SystemExit(
                    f"feat_cache {cfg.feat_cache} cols {spec or 'all'} is "
                    f"{self.feat_cache.width}-wide but the frozen tower is {self.sig_dim}; the "
                    f"head would be built for the wrong input width")
            if not self.sig_dim:
                self.sig_dim = self.feat_cache.width
                pool_dim += self.sig_dim
            print(f"[feat_cache] {self.feat_cache.describe()}", flush=True)
        # Head fp32, rest bf16: many small updates need precision, bulk compute does not.
        # bf16 head update ~2 bf16 steps at peak LR but ~1/50 after cosine decay, where
        # only stochastic rounding prevents a stall. Head ~0.02% of FLOPs.
        self.head_dtype = (torch.float32 if getattr(cfg, "head_fp32", False) else dtype)
        # One LayerNorm per tower, not one over the concatenation: DINOv3's cls arrives already
        # normalised by its own pooler while SigLIP2's pooling output does not, so a shared norm
        # couples their scales and lets the wider side dominate the head's input statistics.
        self.norm = nn.LayerNorm(pool_dim - self.sig_dim)
        self.norm2 = nn.LayerNorm(self.sig_dim) if self.sig_dim else None
        layers: list[nn.Module] = [nn.Dropout(cfg.head_dropout)]
        if cfg.head_hidden > 0:
            layers += [nn.Linear(pool_dim, cfg.head_hidden), nn.GELU(),
                       nn.Dropout(cfg.head_dropout), nn.Linear(cfg.head_hidden, n_labels)]
        else:
            layers += [nn.Linear(pool_dim, n_labels)]
        self.head = nn.Sequential(*layers)
        self.head = self.head.to(self.head_dtype)
        self.norm = self.norm.to(self.head_dtype)
        if self.norm2 is not None:
            self.norm2 = self.norm2.to(self.head_dtype)
        # No learned per-class bias: init_head_bias() sets logit(prior), BALMS's NTR bias
        # is fixed from the prior; both double-suppressed the tail at lr_head.
        # See losses.py, scratch/sophie/loss-audit.md.
        self.register_buffer("in_mean", torch.tensor(
            (0.5, 0.5, 0.5) if self.kind == "siglip2" else IMAGENET_MEAN, dtype=dtype).view(1, 3, 1, 1),
            persistent=False)
        self.register_buffer("in_std", torch.tensor(
            (0.5, 0.5, 0.5) if self.kind == "siglip2" else IMAGENET_STD, dtype=dtype).view(1, 3, 1, 1),
            persistent=False)

    def _prep(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """uint8 [0,255] -> normalised bf16."""
        if pixel_values.dtype == torch.uint8:
            x = pixel_values.to(self.in_mean.dtype).div_(255.0)
            return (x - self.in_mean) / self.in_std
        return pixel_values

    def _siglip_prep(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """SigLIP2 wants (x/255 - 0.5)/0.5. Undo whatever normalisation the caller applied, so
        both paths reach the same tensor."""
        if pixel_values.dtype == torch.uint8:
            x01 = pixel_values.to(torch.float32).div_(255.0)
        else:
            x01 = pixel_values.float() * self.in_std.float() + self.in_mean.float()
        return ((x01 - 0.5) / 0.5).to(self.siglip.dtype)

    # ---------------------------------------------------------------- features
    def forward_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.backbone(pixel_values=pixel_values)
        need_patches = any(p != "cls" for p in self.parts)
        patches = out.last_hidden_state[:, self.patch_offset:, :] if need_patches else None
        feats = []
        for p in self.parts:
            if p == "cls":
                feats.append(out.pooler_output)          # already LayerNorm'ed CLS
            elif p == "mean":
                feats.append(patches.mean(dim=1))
            elif p == "max":
                # peak over patches: "concept present anywhere", what a multi-label
                # tag decision needs
                feats.append(patches.amax(dim=1))
            elif p == "std":
                # spread over patches; fp32 because bf16 variance over 1024 tokens
                # loses too much precision
                feats.append(patches.float().std(dim=1, unbiased=False).to(patches.dtype))
        return feats[0] if len(feats) == 1 else torch.cat(feats, dim=-1)

    def forward(self, pixel_values: torch.Tensor, frozen_feats: torch.Tensor | None = None
                ) -> torch.Tensor:
        feats = self.forward_features(self._prep(pixel_values))
        if self.siglip is not None or self.sig_dim:
            if frozen_feats is not None:
                # Precomputed second view: same columns the tower would produce, already pooled
                # by the same trunk, so no forward is paid for it. Only valid when the cache was
                # built at this run's seed and pass index, which FeatCache does not check itself.
                f2 = frozen_feats
                if f2.dim() != 2 or f2.shape[1] != self.sig_dim:
                    raise SystemExit(f"cached features are {tuple(f2.shape)} but this model's "
                                     f"frozen side is {self.sig_dim}-wide")
            else:
                with torch.no_grad():
                    f2 = self.siglip(pixel_values=self._siglip_prep(pixel_values)).pooler_output
            # normalise each tower on its own, then concatenate
            feats = torch.cat([self.norm(feats.to(self.head_dtype)),
                               self.norm2(f2.to(self.head_dtype))], dim=-1)
            return self.head(feats)
        if feats.dtype != self.head_dtype:
            feats = feats.to(self.head_dtype)
        return self.head(self.norm(feats))

    def init_head_bias(self, priors) -> None:
        """Set the output bias to logit(prior); at 0.26% positives a zero bias predicts only zero for many early steps."""
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
        oft, head, blocks = [], [], []
        for m in iter_oft_modules(self.backbone):
            if m.oft.weight.requires_grad:
                oft.append(m.oft.weight)
        head.extend(self.norm.parameters())
        head.extend(self.head.parameters())
        # unfrozen base weights: everything requiring grad that is not an OFT rotation
        for name, p in self.backbone.named_parameters():
            if p.requires_grad and not name.endswith("oft.weight"):
                blocks.append(p)
        return oft, head, blocks

    def adapter_state_dict(self) -> dict:
        # Save all params, not just requires_grad: resume loses unfrozen weights (load
        # hard-fails when a trainable param is absent). Frozen OFT rotations must survive
        # forks (freezing preserves value, no reset): a requires_grad filter dropped them,
        # reload paired a fine-tuned backbone with R=I, micro 0.19 vs 0.58 (live model
        # fine). Name predicates rot (earlier `".oft." in k` over (name,param) missed).
        # Frozen base ~86M/129M, file under 2x.
        # The second tower is excluded: it is re-loaded from disk at construction, never
        # modified, and saving 93.5M dead weights would double every checkpoint.
        return {k: v.detach().clone() for k, v in self.named_parameters()
                if not k.startswith("siglip.")}

    def load_adapter_state_dict(self, state: dict, strict: bool = True) -> None:
        """Load a checkpoint's trainable parameters. `strict=False` keeps absent trainable parameters at their values -- forking needs it: newly unfrozen blocks are absent from the old checkpoint and hold pretrained weights; any other mismatch fails."""
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
        if missing or (unexpected and strict):
            raise RuntimeError(f"adapter state mismatch: missing={missing[:4]} unexpected={unexpected[:4]}")
        if unexpected:
            print(f"[load] {len(unexpected)} newly trainable parameter(s) kept at their "
                  f"pretrained values (fork)", flush=True)

    def count_params(self) -> dict:
        oft, head, blocks = self.trainable_parameters()
        return {
            "oft": sum(p.numel() for p in oft),
            "head": sum(p.numel() for p in head),
            "blocks": sum(p.numel() for p in blocks),
            "backbone": sum(p.numel() for p in self.backbone.parameters()),
        }
