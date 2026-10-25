"""One fact: are the OFT blocks synchronised in an arm that used normed momentum?

The adapter weight of an OFT module is (k, C(b,2)): k rows, one per diagonal block. If normed
momentum plus snr equalises them, the coefficient of variation of the per-row L2 norm should be
far smaller than in the plain arm. CPU only, reads the saved adapter state.
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]

_args = sys.argv[1:]
ARMS = ([( _args[i], Path(_args[i + 1])) for i in range(0, len(_args), 2)] if _args else [
    ("plain 49 step4000", ROOT / "runs" / "b0ba3dcd" / "ckpt_step004000.pt"),
    ("scoped ntm 71 step4000", ROOT / "runs" / "4ec33484" / "ckpt_step004000.pt"),
])


def row_cv(weight: torch.Tensor) -> float:
    norms = weight.float().norm(dim=1)
    mean = norms.mean()
    return float(norms.std() / mean) if mean > 0 else float("nan")


def head_cv(weight: torch.Tensor) -> float:
    norms = weight.float().norm(dim=1)
    return float(norms.std() / norms.mean())


for label, path in ARMS:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    adapter = ck["adapter"]
    cvs = [row_cv(v) for k, v in adapter.items() if v.dim() == 2 and v.shape[0] <= 32]
    head_key = next((k for k in adapter if k.endswith("1.weight") and adapter[k].shape[0] > 1000), None)
    head = head_cv(adapter[head_key]) if head_key else float("nan")
    print(f"{label}: block row-norm CV mean {sum(cvs)/len(cvs):.3f} over {len(cvs)} tensors, "
          f"min {min(cvs):.3f} max {max(cvs):.3f}; head row-norm CV {head:.3f}")

# the spread of the per-block norms themselves, for one representative tensor each
for label, path in ARMS:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    adapter = ck["adapter"]
    k = next(k for k, v in adapter.items() if v.dim() == 2 and v.shape[0] <= 32)
    norms = adapter[k].float().norm(dim=1).tolist()
    print(f"{label}: {k} per-row norms " + " ".join(f"{n:.4f}" for n in norms))
