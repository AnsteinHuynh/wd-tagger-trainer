"""How saturated are the OFT adapters at a checkpoint?

Per block: ||Q||, rotation angle in degrees, and the relative error the truncated Neumann
series would have made. Production uses exact Cayley, so the error column is a warning
gauge rather than a live defect.
"""
import math
import sys

import torch

CKPT = sys.argv[1] if len(sys.argv) > 1 else r"runs\459e2a02\ckpt_step009457.pt"
BLOCK = 128
SCALED = True
TERMS = 5

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
ad = ck["adapter"]

rows, cols = torch.triu_indices(BLOCK, BLOCK, 1)


def stats(theta):
    theta = theta.detach().float()
    r, n_el = theta.shape
    if SCALED:
        theta = theta / (2.0 * math.sqrt(BLOCK - 1))
    m = torch.zeros(r, BLOCK, BLOCK)
    m.index_put_((torch.arange(r)[:, None], rows, cols), theta, accumulate=False)
    q = m - m.transpose(-2, -1)
    eye = torch.eye(BLOCK).expand_as(q)
    exact = torch.linalg.solve(eye - q, eye + q)
    approx = eye.clone().add(q, alpha=2.0)
    if TERMS > 2:
        qp = torch.bmm(q, q)
        approx.add_(qp, alpha=2.0)
        for _ in range(3, TERMS):
            qp = torch.bmm(qp, q)
            approx.add_(qp, alpha=2.0)
        qp = torch.bmm(qp, q)
        approx.add_(qp)
    nq = torch.linalg.matrix_norm(q, ord=2)
    rel = (torch.linalg.matrix_norm(approx - exact, ord=2)
           / torch.linalg.matrix_norm(exact, ord=2))
    return nq, rel


norms, errs, blocks = [], [], 0
per_module = []
for name, t in ad.items():
    if not name.endswith(".oft.weight"):
        continue
    nq, rel = stats(t)
    norms.append(nq)
    errs.append(rel)
    blocks += nq.numel()
    per_module.append((name.split(".backbone.")[-1], nq.numel(), float(nq.median()),
                       float(nq.max()), float(rel.max())))

nq = torch.cat(norms)
rel = torch.cat(errs)
ang = 2.0 * torch.atan(nq) * 180.0 / math.pi
print(f"checkpoint {CKPT}  modules {len(per_module)}  blocks {blocks}")
for label, v in (("||Q||", nq), ("angle deg", ang), ("neumann rel err", rel)):
    q = torch.quantile(v, torch.tensor([0.5, 0.9, 0.99, 1.0]))
    print(f"{label:16s} median {q[0]:.4f}  p90 {q[1]:.4f}  p99 {q[2]:.4f}  max {q[3]:.4f}")
print("blocks with ||Q|| > 0.5:", int((nq > 0.5).sum()), "of", nq.numel())
print("blocks with ||Q|| > 0.8:", int((nq > 0.8).sum()))
print("blocks with angle > 120 deg:", int((ang > 120).sum()))
per_module.sort(key=lambda t: -t[4])
print("worst modules by median ||Q||:")
for name, n, med, mx, em in sorted(per_module, key=lambda t: -t[2])[:5]:
    print(f"  {name:52s} blocks {n:3d} median {med:.3f} max {mx:.3f} err {em:.2e}")
