"""convnextv2_huge.dbv4-full baseline: load, predict, score.

This is the bar to beat.  It is evaluated on exactly the same cached pixels as
our model (per resize arm), so the comparison is apples-to-apples.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from .config import RUNS_ROOT
from .infer import predict_probs
from .labels import load_label_space
from .metrics import apply_thresholds, compute_metrics, sweep_thresholds

BASELINE_DIR = r"C:\MyApps\ComfyUIRoot\convnextv2_huge.dbv4-full"
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def load_baseline(device: str = "cuda", dtype=torch.bfloat16):
    import timm
    from safetensors.torch import load_file

    with open(os.path.join(BASELINE_DIR, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    model = timm.create_model(
        cfg["architecture"], pretrained=False, num_classes=cfg["num_classes"], **cfg["model_args"]
    )
    sd = load_file(os.path.join(BASELINE_DIR, "model.safetensors"))
    model.load_state_dict(sd, strict=True)
    return model.to(device=device, dtype=dtype).eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["test", "valid", "all"])
    ap.add_argument("--mode", default="squash")
    ap.add_argument("--batch-size", type=int, default=16)
    a = ap.parse_args()

    space = load_label_space()
    out_dir = os.path.join(RUNS_ROOT, "_baseline", "convnextv2_huge", a.mode)
    os.makedirs(out_dir, exist_ok=True)

    model = load_baseline()
    print(f"baseline loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params", flush=True)

    def fwd(x):
        return model(x)

    splits = ["valid", "test"] if a.split == "all" else [a.split]
    for split in splits:
        print(f"predicting {split} [{a.mode}] ...", flush=True)
        probs = predict_probs(fwd, split, a.mode, len(space), MEAN, STD,
                              batch_size=a.batch_size, bf16=True)
        np.save(os.path.join(out_dir, f"{split}_probs.npy"), probs)
        print(f"probs saved for {split}", flush=True)


if __name__ == "__main__":
    main()

