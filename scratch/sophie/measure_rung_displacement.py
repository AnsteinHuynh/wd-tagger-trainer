"""CPU-only effective step size of each optimizer-ablation rung.

Extends verify_opt_settings.py: same constant-gradient trick (loss = sum(W*G),
so every difference is the optimizer), same average-over-steps instead of a
single cold step, but now over the real live-parameter inventory of a u1
base-384 run and with the per-group lr train.py uses.

No CUDA: CUDA_VISIBLE_DEVICES is cleared before torch is imported.
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import json
import sys
from dataclasses import fields
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from adv_optm import SinkSGD_adv
from wdtag.config import TrainConfig

PHASE3 = ROOT / "configs" / "heads" / "phase3"
CONFIGS = [
    ("42_a1", "42_u1-C-L-asl-g2.0-siglip2b384-200k-b48-a1.json"),
    ("48_gold145", "48_u1-C-L-asl-g2.0-siglip2b384-200k-b48-gold145.json"),
    ("49_plainfp32", "49_u1-C-L-asl-g2.0-siglip2b384-200k-b48-r0-plainfp32.json"),
    ("50_ntmsnr", "50_u1-C-L-asl-g2.0-siglip2b384-200k-b48-r1-ntmsnr.json"),
    ("51_orthograd", "51_u1-C-L-asl-g2.0-siglip2b384-200k-b48-r2-orthograd.json"),
    ("52_sinkhorn", "52_u1-C-L-asl-g2.0-siglip2b384-200k-b48-r3-sinkhorn.json"),
    ("53_spectral", "53_u1-C-L-asl-g2.0-siglip2b384-200k-b48-r4-spectral.json"),
    ("54_wd1e5", "54_u1-C-L-asl-g2.0-siglip2b384-200k-b48-w0-wd1e5.json"),
    ("55_geowd", "55_u1-C-L-asl-g2.0-siglip2b384-200k-b48-w1-geowd.json"),
    ("56_cautiouswd", "56_u1-C-L-asl-g2.0-siglip2b384-200k-b48-w2-cautiouswd.json"),
]

# Live inventory of the phase3 arm: u1, base-384, block 128, siglip2-base.
# OFT weight is (rank, block_size*(block_size-1)/2) = (r, 8128); r = in_features/128.
# u1 freezes the last encoder block's 6 rotations, so 11 blocks are live, plus the
# frozen tower's attention-pooling head (4 Linears) and the patch-embed conv.
OFT_768 = (6, 8128)
OFT_3072 = (24, 8128)
BLOCK_SPEC = [
    ("block", (768, 768)), ("block", (768, 768)), ("block", (768, 768)),
    ("block", (768, 768)), ("block", (768,)), ("block", (768,)), ("block", (768,)),
    ("block", (768,)), ("block", (3072, 768)), ("block", (3072,)),
    ("block", (768, 3072)), ("block", (768,)),
    ("block", (768,)), ("block", (768,)), ("block", (768,)), ("block", (768,)),
]
HEAD_SPEC = [
    ("head", (768,)), ("head", (768,)),
    ("head", (12476, 768)), ("head", (12476,)),
]
SHAPES = [
    ("1024x768", (1024, 768)),
    ("768x768", (768, 768)),
    ("12476x768", (12476, 768)),
    ("1-D 12476", (12476,)),
    ("OFT 6x8128", (6, 8128)),
    ("OFT 24x8128", (24, 8128)),
]


def inventory():
    """(group, shape) for every live tensor in the phase3 arm."""
    spec = []
    for _ in range(11):  # live encoder blocks
        spec += [("oft", OFT_768)] * 5 + [("oft", OFT_3072)]
    spec += [("oft", OFT_768)] * 3 + [("oft", OFT_3072)]  # pooling head
    spec += [("oft", OFT_768)]  # patch-embed conv
    spec += BLOCK_SPEC + HEAD_SPEC
    return spec


def load_cfg(name):
    raw = json.loads((PHASE3 / name).read_text())
    known = {f.name for f in fields(TrainConfig)}
    unknown = set(raw) - known
    if unknown:
        raise SystemExit(f"{name}: unknown config keys {sorted(unknown)}")
    return TrainConfig(**raw)


def kw_from_cfg(c):
    """Exactly the SinkSGD_adv call of wdtag/train.py:574, minus torch.compile."""
    return dict(
        momentum=c.momentum,
        weight_decay=c.weight_decay,
        sinkhorn_iterations=c.sinkhorn_iterations,
        orthogonal_sinkhorn=c.orthogonal_sinkhorn,
        normed_momentum=c.optimizer_normed_momentum,
        snr_cond=c.optimizer_snr_cond,
        nesterov=c.optimizer_nesterov,
        orthogonal_gradient=c.optimizer_orthogonal_gradient,
        state_precision=c.optimizer_state_precision,
        spectral_normalization=c.optimizer_spectral_normalization,
        geometric_wd=c.optimizer_geometric_wd,
        cautious_wd=c.optimizer_cautious_wd,
        stochastic_rounding=True,
        compiled_optimizer=False,
    )


def make_params(spec):
    """Identical init and identical constant gradients across all configs."""
    torch.manual_seed(1234)
    params = []
    for _group, shape in spec:
        params.append(torch.nn.Parameter(torch.randn(*shape) * 0.02))
    g = torch.Generator().manual_seed(4321)
    grads = [torch.randn(*shape, generator=g) for _g, shape in spec]
    return params, grads


def run_full(cfg, spec, steps=140, warm=60, freeze=False):
    params, gts = make_params(spec)
    oft = [p for (g, _s), p in zip(spec, params) if g == "oft"]
    block = [p for (g, _s), p in zip(spec, params) if g == "block"]
    head = [p for (g, _s), p in zip(spec, params) if g == "head"]
    groups = [{"params": oft, "lr": cfg.lr_oft},
              {"params": block, "lr": cfg.lr_unfreeze},
              {"params": head, "lr": cfg.lr_head}]
    opt = SinkSGD_adv(groups, **kw_from_cfg(cfg))
    group_of = ["oft"] * len(oft) + ["block"] * len(block) + ["head"] * len(head)
    total, grp, per_tensor = [], {"oft": [], "block": [], "head": []}, [0.0] * len(params)
    for step in range(steps):
        before = [p.detach().clone() for p in params]
        for p, gt in zip(params, gts):
            p.grad = gt.clone()
        opt.step()
        opt.zero_grad(set_to_none=True)
        sq = {"oft": 0.0, "block": 0.0, "head": 0.0}
        ds = []
        for i, p in enumerate(params):
            d = float((p.detach() - before[i]).norm())
            ds.append(d)
            sq[group_of[i]] += d * d
            if step >= warm:
                per_tensor[i] += d / (steps - warm)
            if freeze:
                p.detach().copy_(before[i])
        if freeze:
            del before
        total.append(sum(d * d for d in ds) ** 0.5)
        for k in sq:
            grp[k].append(sq[k] ** 0.5)
    window = slice(warm, steps)
    out = {
        "total_mean": sum(total[window]) / (steps - warm),
        "total_first": total[0],
        "total_last": total[-1],
        "group_mean": {k: sum(v[window]) / (steps - warm) for k, v in grp.items()},
        "per_tensor_mean": per_tensor,
        "total_series": total,
    }
    return out


def run_shapes(cfg, shapes, lr=5e-4, steps=140, warm=60, freeze=False):
    """Each shape is its own parameter in one group at a fixed lr, so the shape
    ratio is the optimizer switch alone; 48's 1.45x lr is therefore excluded."""
    spec = [("s", s) for _n, s in shapes]
    params, gts = make_params(spec)
    opt = SinkSGD_adv([{"params": params, "lr": lr}], **kw_from_cfg(cfg))
    per_tensor = [0.0] * len(params)
    for step in range(steps):
        before = [p.detach().clone() for p in params]
        for p, gt in zip(params, gts):
            p.grad = gt.clone()
        opt.step()
        opt.zero_grad(set_to_none=True)
        for i, p in enumerate(params):
            d = float((p.detach() - before[i]).norm())
            if step >= warm:
                per_tensor[i] += d / (steps - warm)
            if freeze:
                p.detach().copy_(before[i])
        del before
    return per_tensor


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", action="store_true",
                    help="restore the weights after each measured step (step at fixed weights)")
    ap.add_argument("--steps", type=int, default=140)
    ap.add_argument("--warm", type=int, default=60)
    a = ap.parse_args()
    freeze, steps, warm = a.freeze, a.steps, a.warm
    tag = "freeze" if freeze else "drift"
    spec = inventory()
    n = sum(shape[0] * (shape[1] if len(shape) > 1 else 1) for _g, shape in spec)
    print(f"mode {tag}: inventory {len(spec)} tensors, {n} params, "
          f"{steps} steps, window {warm}..{steps}", flush=True)
    cfg_all = [(label, load_cfg(name)) for label, name in CONFIGS]
    results = {}
    for label, cfg in cfg_all:
        r = run_full(cfg, spec, steps=steps, warm=warm, freeze=freeze)
        results[label] = r
        print(f"{label:14s} mean {r['total_mean']:.9e}  first {r['total_first']:.3e}  "
              f"groups {r['group_mean']}", flush=True)
    ref = results["42_a1"]["total_mean"]
    base = results["49_plainfp32"]["total_mean"]
    print("\n=== total displacement ===")
    for label, _cfg in cfg_all:
        m = results[label]["total_mean"]
        print(f"{label:14s} {m:.6e}  /42 {m/ref:.4f}  /49 {m/base:.4f}  restore {ref/m:.4f}")

    print(f"\n=== per shape, common lr 5e-4, {tag}, ratio to rung 49 ===")
    shape_res = {}
    for label, cfg in cfg_all:
        shape_res[label] = run_shapes(cfg, SHAPES, steps=steps, warm=warm, freeze=freeze)
        print(label, [f"{x:.4e}" for x in shape_res[label]], flush=True)
    s49 = shape_res["49_plainfp32"]
    print("shape ratios to 49")
    for label, _cfg in cfg_all:
        vals = shape_res[label]
        print(f"{label:14s} " + " ".join(
            f"{nm}:{vals[i]/s49[i]:.4f}" for i, (nm, _s) in enumerate(SHAPES)))

    out = {
        "mode": tag,
        "configs": [{"label": l, "total_mean": results[l]["total_mean"],
                     "total_first": results[l]["total_first"],
                     "total_last": results[l]["total_last"],
                     "group_mean": results[l]["group_mean"],
                     "per_tensor_mean": results[l]["per_tensor_mean"]}
                    for l, _c in cfg_all],
        "spec": [{"group": g, "shape": list(s)} for g, s in spec],
        "shape_names": [nm for nm, _s in SHAPES],
        "shape_values": {l: shape_res[l] for l, _c in cfg_all},
    }
    path = ROOT / "scratch" / "sophie" / f"rung-displacement-{tag}.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {path.name}")


if __name__ == "__main__":
    main()
