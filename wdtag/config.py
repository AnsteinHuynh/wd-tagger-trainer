"""Training configuration + run-directory naming."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field, asdict

RUNS_ROOT = r"C:\Users\kerok\apps\wd-tagger-trainer\runs"
DINO_DIR = r"I:\Models\dinov3-vitb16-pretrain-lvd1689m"
CHECKPOINT_EVERY = 200


@dataclass
class TrainConfig:
    # data
    mode: str = "squash"              # "squash" (default) or "pad"; tied in accuracy, see REPORT.md
    batch_size: int = 64
    num_workers: int = 0
    # adapter
    block_size: int = 32
    block_share: bool = False
    oft_scaled: bool = True
    oft_dropout: float = 0.0
    oft_conv: bool = True             # also adapt the patch-embed conv
    # head
    head_pool: str = "cls"            # "cls" | "cls_mean" | "mean"
    head_hidden: int = 0
    head_dropout: float = 0.1
    # optimisation
    optimizer: str = "sinksgd_adv"
    lr_oft: float = 0.02
    lr_head: float = 0.002
    weight_decay: float = 0.0
    momentum: float = 0.95
    sinkhorn_iterations: int = 5
    orthogonal_sinkhorn: bool = False
    warmup_steps: int = 50
    epochs: int = 8
    grad_clip: float = 1.0
    bf16: bool = True
    compile_model: bool = True
    compiled_optimizer: bool = True
    # staged training: stage 1 leaves the adapters at identity so the head is
    # trained against fixed features; stage 2 warm-starts from it with --init-from.
    freeze_oft: bool = False
    # misc
    seed: int = 1337
    labels_per_image_smoothing: float = 0.0
    log_every: int = 20

    def to_dict(self) -> dict:
        return asdict(self)

    def sha8(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:8]

    def run_dir(self) -> str:
        import os
        d = os.path.join(RUNS_ROOT, self.sha8())
        os.makedirs(d, exist_ok=True)
        return d

    @classmethod
    def from_json(cls, path: str) -> "TrainConfig":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


