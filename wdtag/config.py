"""Training configuration + run-directory naming."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, asdict

RUNS_ROOT = r"C:\Users\kerok\apps\wd-tagger-trainer\runs"
DINO_DIR = r"I:\Models\dinov3-vitb16-pretrain-lvd1689m"
CHECKPOINT_EVERY = 1000


@dataclass
class TrainConfig:
    # data
    mode: str = "squash"              # "squash" (best-known) or "pad" (tied once lr is right)
    batch_size: int = 48
    num_workers: int = 0
    # Input resolution. Must be paired with WDTAG_SIZE in the environment, because the cache
    # tree, the streamer's ring buffer and the validator all bind SIZE at import time. It is
    # in the config so two runs that differ only in resolution cannot share a hash.
    image_size: int = 512
    # adapter
    block_size: int = 128
    block_share: bool = False
    oft_scaled: bool = True
    oft_dropout: float = 0.0
    oft_conv: bool = True             # adapt the patch-embed conv too
    # head
    head_pool: str = "cls_mean"       # "cls" | "cls_mean" | "mean"; cls_mean is best
    head_hidden: int = 0
    head_dropout: float = 0.1
    # DINOv3 backbone; "" = ViT-B/16 in DINO_DIR. Run identity: never mix sizes.
    backbone_dir: str = ""
    # Second frozen tower (SigLIP2 vision), features concatenated before the head. "" = DINOv3
    # alone. Frozen and adapter-free by design: it runs under no_grad, so it adds no backward
    # and no retained activations. Run identity.
    siglip_dir: str = ""
    # Run identity: it changes the numerics of every head update.
    head_fp32: bool = False
    # fine-tune the last N blocks (OFT rotations at identity). 0 = backbone frozen.
    unfreeze_blocks: int = 0
    # Neumann terms; 0 = exact. Valid only while ||Q|| is small: >=15 or exact; 5 = old default.
    cayley_terms: int = 5
    # lr for those blocks' base weights (~per-parameter movement/step); keep far below lr_oft
    # or pretrained features die.
    lr_unfreeze: float = 0.0
    # recompute activations instead of storing them: ~1.5 GB/block at 512px, batch 48.
    # True = every block that carries a backward (all of them when OFT is on, which is what the
    # old gate got wrong); an int N = only the last N blocks, keeping the rest's activations so
    # the recompute is paid N times instead of 24. VRAM is linear in the blocks left unwrapped.
    grad_checkpoint_blocks: bool | int = False
    # optimisation
    # `optimizer` as a string nothing read is a lie in the hash.
    # bce | asl | cb_ntr | asl_ntr; see wdtag/losses.py.
    loss: str = "bce"
    # ASL gamma_neg. 2.0 measured better than 4.0 on 2026-10-22: same macro, +11.9% micro.
    loss_gamma: float = 2.0
    ntr_lambda: float = 2.0
    # CB beta, 0.9 per BALMS; at 0.999 the (1-beta)/(1-beta**n) floor swallows >n~6000.
    cb_beta: float = 0.9
    # v_i = -kappa*logit(p_i). 0.0: init_head_bias sets logit(prior), kappa>0 double-suppresses
    # tails.
    ntr_kappa: float = 0.0
    lr_oft: float = 0.0005            # 0.02 (old default) destroys the model: micro F1 0.14
    lr_head: float = 0.00012
    weight_decay: float = 0.0
    momentum: float = 0.95
    sinkhorn_iterations: int = 5
    orthogonal_sinkhorn: bool = False
    warmup_steps: int = 50
    # "cosine" spends the learning rate across the whole schedule, so the horizon has to be
    # chosen up front. "const" holds the peak and anneals only over the final `decay_steps`,
    # which makes a run extendable by raising `stream_images`.
    lr_schedule: str = "cosine"
    decay_steps: int = 0
    epochs: int = 8
    grad_clip: float = 1.0
    # Micro-batches per optimiser step. 2 with batch 24 spends 48 images per update at 24 images
    # of VRAM, which is how a 24 block tower reaches base's effective batch on a 24.5 GB card.
    grad_accum: int = 1
    bf16: bool = True
    compile_model: bool = True
    compiled_optimizer: bool = True
    # stage 1 leaves adapters at identity (head on fixed features); stage 2 uses --init-from.
    freeze_oft: bool = False
    # run dir to warm-start from; run identity, so staged and cold runs never collide.
    init_from: str = ""
    # stream from tars, not the shard cache; run identity like init_from (full pass vs
    # cached 10k).
    stream: bool = False
    stream_images: int = 0   # 0 = one full pass over the split
    # in-run val size, keep SMALL: a step holds ~21 GB of the 24.5 GB card. The 50k eval is a
    # separate process (wdtag.validate, batch 256). <=1000 .pt cache, >1000 .npy memmap.
    val_images: int = 1000
    # misc
    seed: int = 1337
    log_every: int = 20

    def to_dict(self) -> dict:
        d = asdict(self)
        # image_size and the schedule fields were added after dozens of runs already existed.
        # Their historical defaults stay out of the hashed blob so every old sha8 survives while
        # a 384 config or a const-schedule config still gets a unique one. Never drop a field
        # that was in the hash from the start.
        if d.get("image_size") == 512:
            d.pop("image_size", None)
        if d.get("lr_schedule") == "cosine":
            d.pop("lr_schedule", None)
        if not d.get("decay_steps"):
            d.pop("decay_steps", None)
        if d.get("grad_accum", 1) == 1:
            d.pop("grad_accum", None)
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
        import os
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(k for k in d if k not in known and not k.startswith("_"))
        if unknown:
            # Silent filtering hid a stale "optimizer" key for weeks; warn instead.
            print(f"[config] ignoring unknown key(s) in {os.path.basename(path)}: "
                  f"{', '.join(unknown)}", flush=True)
        cfg = cls(**{k: v for k, v in d.items() if k in known})
        # Dir name pinned in config.json (`_sha8`) so later fields cannot re-key it; configs/
        # templates unpinned.
        if isinstance(d.get("_sha8"), str):
            cfg._sha8_pin = d["_sha8"]
        return cfg




