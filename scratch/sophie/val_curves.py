"""Validation curves for every run: do we ever see a real decline?

One val point per epoch (312 steps at 10k images), so this is the coarsest
possible view of overfitting. Train loss is at the top of each row.
"""
import glob
import json
import os

RUNS = sorted(glob.glob(r"runs\*\config.json"))
print(f"{'run':10s} {'lr_oft':>8s} {'seed':>5s} {'mode':>7s}  "
      f"val micro F1 per epoch                                    peak")
for cfg_path in RUNS:
    d = os.path.dirname(cfg_path)
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    log = os.path.join(d, "train_log.jsonl")
    if not os.path.exists(log):
        continue
    rows = [json.loads(x) for x in open(log, encoding="utf-8") if x.strip()]
    if not rows:
        continue
    vals = [r["micro_f1"] for r in rows]
    peak = int(max(range(len(vals)), key=lambda i: vals[i]))
    curve = " ".join(f"{v:.3f}" for v in vals)
    drop = vals[-1] - vals[peak]
    print(f"{os.path.basename(d):10s} {cfg.get('lr_oft'):>8} {cfg.get('seed'):>5} "
          f"{cfg.get('mode'):>7s}  {curve:56s} ep{peak} ({drop:+.3f})")
