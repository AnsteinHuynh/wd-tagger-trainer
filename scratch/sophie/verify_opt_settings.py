"""CPU check of the four settings the user asked for. No GPU, no training.

Two questions only:
  1. does SinkSGD_adv accept the kwargs and actually hold fp32 state
  2. how much does the stack move the weights per step vs the old stack

A constant gradient (loss = sum(W * G)) makes the second question exact: every
difference in displacement is the optimizer, not the data.
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from adv_optm.optim import SinkSGD_adv

ROOT = Path(__file__).resolve().parents[2]

OLD = dict(momentum=0.95, weight_decay=0.0, sinkhorn_iterations=5,
           orthogonal_sinkhorn=False, normed_momentum=False, snr_cond=False,
           nesterov=False, state_precision="auto", stochastic_rounding=True,
           compiled_optimizer=False)
NEW = dict(OLD, momentum=0.95, normed_momentum=True, snr_cond=True,
           nesterov=True, state_precision="fp32")


def cfg_fields(name):
    c = json.loads((ROOT / "configs" / name).read_text())
    return {k: c[k] for k in ("optimizer_normed_momentum", "optimizer_snr_cond",
                              "optimizer_nesterov", "optimizer_state_precision")}


def build(kw, shape=(64, 64), seed=0):
    torch.manual_seed(seed)
    p = torch.nn.Parameter(torch.randn(*shape) * 0.02)
    opt = SinkSGD_adv([p], lr=5e-4, **kw)
    return p, opt


def probe(kw, steps=30, shape=(64, 64)):
    p, opt = build(kw, shape)
    g = torch.Generator().manual_seed(1234)
    G = torch.randn(*shape, generator=g)
    moves = []
    for _ in range(steps):
        before = p.detach().clone()
        p.grad = G.clone()
        opt.step()
        opt.zero_grad(set_to_none=True)
        moves.append(float((p.detach() - before).norm()))
    st = opt.state[p]
    dtypes = sorted({str(v.dtype) for v in st.values() if torch.is_tensor(v)})
    return moves, dtypes


def main():
    print("=== config keys ===")
    for name in ("best.json", "full_corpus_5p3m.json"):
        print(f"  {name}: {cfg_fields(name)}")

    print("\n=== state precision: bf16 param, auto vs fp32 ===")
    for prec in ("auto", "fp32"):
        torch.manual_seed(0)
        p = torch.nn.Parameter((torch.randn(32, 32) * 0.02).to(torch.bfloat16))
        opt = SinkSGD_adv([p], lr=5e-4, momentum=0.95, normed_momentum=True,
                          snr_cond=True, nesterov=True, state_precision=prec,
                          compiled_optimizer=False)
        p.grad = torch.randn(32, 32).to(torch.bfloat16)
        opt.step()
        st = opt.state[p]
        dtypes = sorted({str(v.dtype) for v in st.values() if torch.is_tensor(v)})
        print(f"  state_precision={prec:4s} -> param {p.dtype}, state {dtypes}, "
              f"actual {opt.param_groups[0].get('actual_state_precision')}")

    print("\n=== plumbing: does each stack construct and step ===")
    for label, kw in (("old", OLD), ("new", NEW)):
        moves, dtypes = probe(kw, steps=5)
        print(f"  {label}: ok, state dtypes {dtypes}, first move {moves[0]:.6f}")

    print("\n=== displacement per step, constant gradient, lr 5e-4, 64x64 ===")
    a, _ = probe(OLD)
    b, _ = probe(NEW)
    import statistics as s
    ma, mb = s.mean(a), s.mean(b)
    print(f"  old  mean {ma:.6e}  median {s.median(a):.6e}  max {max(a):.6e}")
    print(f"  new  mean {mb:.6e}  median {s.median(b):.6e}  max {max(b):.6e}")
    print(f"  ratio new/old = {mb / ma:.4f}")

    print("\n=== isolate one switch at a time (new stack minus the others) ===")
    for label, extra in (("normed_momentum only", dict(normed_momentum=True)),
                         ("nesterov only", dict(nesterov=True)),
                         ("normed+snr", dict(normed_momentum=True, snr_cond=True)),
                         ("normed+snr+nesterov", dict(normed_momentum=True, snr_cond=True,
                                                     nesterov=True))):
        m, _ = probe(dict(OLD, **extra), steps=30)
        print(f"  {label:22s} mean {s.mean(m):.6e}  ratio {s.mean(m) / ma:.4f}")

    print("\n=== 1-D parameter (the rank gate: head bias) ===")
    a1, _ = probe(OLD, shape=(256,))
    b1, _ = probe(NEW, shape=(256,))
    print(f"  old mean {s.mean(a1):.6e}   new mean {s.mean(b1):.6e}   "
          f"ratio {s.mean(b1) / s.mean(a1):.4f}")


if __name__ == "__main__":
    main()
