"""Training configuration + run-directory naming."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, asdict

RUNS_ROOT = r"C:\Users\kerok\apps\wd-tagger-trainer\runs"
DINO_DIR = r"I:\Models\dinov3-vitb16-pretrain-lvd1689m"
CHECKPOINT_EVERY = 200


@dataclass
class TrainConfig:
    # data
    mode: str = "squash"              # "squash" (best-known, default) or "pad" (tied once lr is right; REPORT.md)
    batch_size: int = 32
    num_workers: int = 0
    # adapter
    block_size: int = 32
    block_share: bool = False
    oft_scaled: bool = True
    oft_dropout: float = 0.0
    oft_conv: bool = True             # also adapt the patch-embed conv
    # head
    head_pool: str = "cls_mean"       # "cls" | "cls_mean" | "mean"; cls_mean is best-known
    head_hidden: int = 0
    head_dropout: float = 0.1
    # optimisation
    optimizer: str = "sinksgd_adv"
    lr_oft: float = 0.0005            # 0.02 (old default) destroys the model: micro F1 0.14
    lr_head: float = 0.00012
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
    # run directory whose adapter+head warm-start this run. Part of the run identity on
    # purpose: a staged run and a cold run with otherwise identical settings are different
    # experiments, and must not collide in one directory.
    init_from: str = ""
    # Read training data straight from the tars instead of the shard cache. Part of the
    # run identity for the same reason as init_from: a streamed full-corpus pass and a
    # cached 10k run are different experiments and must not share a directory.
    stream: bool = False
    stream_images: int = 0   # 0 = one full pass over the split
    # misc
    seed: int = 1337
    log_every: int = 20

    def to_dict(self) -> dict:
        d = asdict(self)
        pin = getattr(self, "_sha8_pin", None)
        if pin:
            d["_sha8"] = pin
        return d

    def sha8(self) -> str:
        pin = getattr(self, "_sha8_pin", None)
        if pin:
            return pin
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
        cfg = cls(**{k: v for k, v in d.items() if k in known})
        # A run's directory name is pinned in its config.json (`_sha8`), so
        # adding/removing fields later cannot re-key the run and silently make a
        # resume start a brand-new directory.  Templates under configs/ are not
        # pinned: every new run gets its own pin when train.py creates it.
        if isinstance(d.get("_sha8"), str):
            cfg._sha8_pin = d["_sha8"]
        return cfg


