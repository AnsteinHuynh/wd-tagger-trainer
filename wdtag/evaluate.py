"""Final evaluation on the cached test split.

    python -m wdtag.evaluate --run runs/<sha8>
    python -m wdtag.evaluate --baseline

Writes eval_test.json (+ probability arrays) into the run directory.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from .config import RUNS_ROOT, TrainConfig
from .data import load_targets
from .infer import predict_probs
from .labels import load_label_space
from .model import IMAGENET_MEAN, IMAGENET_STD, DinoTagger
from .score import fmt_table, score_probs


def _dump(path: str, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def eval_run(run_dir: str, split: str = "test") -> dict:
    cfg = TrainConfig.from_json(os.path.join(run_dir, "config.json"))
    space = load_label_space()
    model = DinoTagger(cfg, len(space)).cuda()
    ck = torch.load(os.path.join(run_dir, "best_val.pt"), map_location="cuda", weights_only=False)
    model.load_adapter_state_dict(ck["adapter"])
    model.eval()
    print(f"loaded best_val.pt (step {ck['step']}, val metrics {ck.get('metrics')})", flush=True)

    def fwd(x):
        return model(x)

    print(f"predicting {split} ...", flush=True)
    probs = predict_probs(fwd, split, cfg.mode, len(space), IMAGENET_MEAN, IMAGENET_STD,
                          batch_size=cfg.batch_size * 2, bf16=True)
    print("predicting valid (for threshold tuning) ...", flush=True)
    vprobs = predict_probs(fwd, "valid", cfg.mode, len(space), IMAGENET_MEAN, IMAGENET_STD,
                           batch_size=cfg.batch_size * 2, bf16=True)

    tgt = load_targets(split, cfg.mode, len(space))
    vtgt = load_targets("valid", cfg.mode, len(space))
    score = score_probs(probs, tgt, vprobs, vtgt)
    score["split"], score["mode"], score["model"] = split, cfg.mode, "dinov3-b/16+oftv2"
    score["step"] = int(ck["step"])
    _dump(os.path.join(run_dir, f"eval_{split}.json"), score)
    np.save(os.path.join(run_dir, f"{split}_probs.npy"), probs)
    np.save(os.path.join(run_dir, "valid_probs.npy"), vprobs)
    print(fmt_table(score), flush=True)
    return score


def eval_baseline(mode: str, split: str = "test") -> dict:
    space = load_label_space()
    d = os.path.join(RUNS_ROOT, "_baseline", "convnextv2_huge", mode)
    probs = np.load(os.path.join(d, f"{split}_probs.npy"))
    vprobs = np.load(os.path.join(d, "valid_probs.npy"))
    tgt = load_targets(split, mode, len(space))
    vtgt = load_targets("valid", mode, len(space))
    score = score_probs(probs, tgt, vprobs, vtgt)
    score["split"], score["mode"], score["model"] = split, mode, "convnextv2_huge.dbv4-full"
    _dump(os.path.join(d, f"eval_{split}.json"), score)
    print(fmt_table(score), flush=True)
    return score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--mode", default="squash")
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    if a.baseline:
        eval_baseline(a.mode, a.split)
    elif a.run:
        eval_run(a.run, a.split)
    else:
        ap.error("give --run DIR or --baseline")


if __name__ == "__main__":
    main()

