"""Check the loss implementations against independent reference formulas.

Self-contained on purpose: this file used to import `losses_corrected.py`, a second copy of the
losses that existed only to be compared against. Two copies meant the test could agree with the copy
instead of the shipped code. The reference formulas are written out here in plain torch, next to
nothing else.

Run: python scratch/sophie/test_losses.py
"""
import sys

import torch

sys.path.insert(0, r"C:\Users\kerok\apps\wd-tagger-trainer")
from wdtag.losses import asl, asl_ntr, bce_pos_weight, cb_ntr, class_balanced_weights  # noqa: E402


def ref_asl(x, y, gamma_pos=0.0, gamma_neg=4.0, clip=0.05):
    """ASL written out independently: probability clip, one-sided gamma, detached focal."""
    p = torch.sigmoid(x)
    xs = (1.0 - p + clip).clamp(max=1.0)
    pos = y * torch.log(p)
    neg = (1.0 - y) * torch.log(xs)
    pt = p * y + xs * (1.0 - y)
    gamma = gamma_pos * y + gamma_neg * (1.0 - y)
    return (-(pos + neg) * (1.0 - pt).detach().pow(gamma)).mean()


def ref_cb_ntr(x, y, lam=2.0, gamma=2.0):
    """BALMS Eq 6/8: lambda scales negative logits AND divides their weight."""
    z_eff = torch.where(y > 0.5, x, x * lam)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(z_eff, y, reduction="none")
    w = torch.where(y > 0.5, torch.ones_like(y), torch.full_like(y, 1.0 / lam))
    return (w * (1.0 - torch.exp(-bce)).pow(gamma) * bce).mean()


def main() -> int:
    torch.manual_seed(0)
    x = torch.randn(16, 64)
    y = (torch.rand(16, 64) < 0.15).float()
    ok = True

    for g in (1.0, 2.0, 4.0):
        a, b = float(asl(x, y, gamma_neg=g)), float(ref_asl(x, y, gamma_neg=g))
        same = abs(a - b) < 1e-6
        ok &= same
        print(f"asl gamma_neg={g}: impl {a:.6f} ref {b:.6f} {'OK' if same else 'MISMATCH'}")

    c, r = float(cb_ntr(x, y, ntr_lambda=2.0, gamma=2.0)), float(ref_cb_ntr(x, y))
    ok &= abs(c - r) < 1e-6
    print(f"cb_ntr: impl {c:.6f} ref {r:.6f} {'OK' if abs(c - r) < 1e-6 else 'MISMATCH'}")

    d = float(asl_ntr(x, y, gamma_neg=4.0, ntr_lambda=1.0)) - float(asl(x, y, gamma_neg=4.0))
    ok &= abs(d) < 1e-9
    print(f"asl_ntr(lambda=1) - asl = {d:.2e} {'OK' if abs(d) < 1e-9 else 'MISMATCH'}")

    xg = x.clone().requires_grad_(True)
    asl_ntr(xg, y, gamma_neg=4.0, ntr_lambda=2.0).backward()
    finite = bool(torch.isfinite(xg.grad).all()) and float(xg.grad.abs().sum()) > 0
    ok &= finite
    print(f"asl_ntr gradient finite, non-zero: {finite}")

    counts = torch.tensor([1.0, 10.0, 1000.0, 6840.0])
    w = class_balanced_weights(counts, beta=0.9, normalize="pos_mass")
    ok &= float((w * counts).sum()) > 0
    print(f"class_balanced_weights range {float(w.min()):.2f}..{float(w.max()):.2f}")

    pw = float(bce_pos_weight(x, y, pos_weight=torch.full((64,), 3.0)))
    ok &= pw > 0
    print(f"bce_pos_weight: {pw:.6f}")

    print("\nALL OK" if ok else "\nFAILURES ABOVE")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
