"""How does rotation drift grow with steps and with lr?

If drift ~ steps^b, then running 10x more steps at the same lr multiplies drift
by 10^b, and the lr that keeps drift constant at 100k images is lr/10^b.
Measured from the per-200-step checkpoints we already have.
"""
import glob
import os

import numpy as np
import torch

D = 32


def cayley(vec, d=D):
    b = vec.shape[0]
    q = torch.zeros(b, d, d, dtype=torch.float32)
    iu = torch.triu_indices(d, d, offset=1)
    q[:, iu[0], iu[1]] = vec.float()
    q = q - q.transpose(1, 2)
    eye = torch.eye(d).expand(b, d, d)
    return torch.linalg.solve(eye + q, eye - q, left=False)


def drift_of(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    vals = []
    for k, v in ck["adapter"].items():
        if not k.endswith("oft.weight"):
            continue
        r = cayley(v)
        eye = torch.eye(D).expand_as(r)
        vals.append((torch.linalg.matrix_norm(r - eye).mean() / D ** 0.5).item())
    return float(np.mean(vals))


for run, label in ((r"runs\e6bc2879", "lr_oft 0.0005"), (r"runs\539d8880", "lr_oft 0.002")):
    ckpts = sorted(glob.glob(os.path.join(run, "ckpt_step*.pt")),
                   key=lambda p: int(os.path.basename(p).split("step")[1].split(".")[0]))
    steps, drifts = [], []
    for p in ckpts:
        s = int(os.path.basename(p).split("step")[1].split(".")[0])
        steps.append(s)
        drifts.append(drift_of(p))
    steps, drifts = np.array(steps), np.array(drifts)
    # fit only the unsaturated part (drift < 1.2) -- the ratio is meaningless once
    # the rotation has wandered most of the way round
    m = drifts < 1.2
    if m.sum() >= 3:
        b = np.polyfit(np.log(steps[m]), np.log(drifts[m]), 1)[0]
    else:
        b = float("nan")
    print(f"\n{label}  ({os.path.basename(run)})")
    for s, d in zip(steps, drifts):
        print(f"  step {s:5d}  drift {d:.3f}")
    print(f"  exponent b (drift ~ steps^b) = {b:.3f}")

print("\nInterpretation: at 100k images (10x steps) the same lr multiplies drift by "
      "10^b.\nTo hold drift constant the lr must be divided by 10^b.")
