"""Multi-label metrics: Macro/Micro F1, MCC, Precision, Recall.

Conventions (applied identically to every model, ours and the baseline):

* a label with no positive and no predicted positive contributes P=R=F1=MCC=0
  (`0/0 -> 0`), so the macro averages are over the *entire* 12,476-label space;
* micro totals pool TP/FP/FN over the whole (images x labels) matrix, and micro
  MCC is the correlation over that flattened matrix;
* `supported_*` variants average only over labels with at least one positive in
  the split, which is reported alongside because with 1k images many labels are
  simply absent.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12


def _prf(tp, fp, fn):
    p = np.where(tp + fp > 0, tp / np.maximum(tp + fp, EPS), 0.0)
    r = np.where(tp + fn > 0, tp / np.maximum(tp + fn, EPS), 0.0)
    f = np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, EPS), 0.0)
    return p, r, f


def _mcc(tp, fp, fn, tn):
    num = tp * tn - fp * fn
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return np.where(den > 0, num / np.maximum(den, EPS), 0.0)


def compute_metrics(pred: np.ndarray, tgt: np.ndarray) -> dict[str, float]:
    """pred/tgt: boolean arrays [n_images, n_labels]."""
    pred = pred.astype(bool)
    tgt = tgt.astype(bool)
    n = pred.shape[0]
    tp = (pred & tgt).sum(0).astype(np.float64)
    fp = (pred & ~tgt).sum(0).astype(np.float64)
    fn = (~pred & tgt).sum(0).astype(np.float64)
    tn = n - tp - fp - fn

    p, r, f = _prf(tp, fp, fn)
    mcc = _mcc(tp, fp, fn, tn)

    sup = tp + fn > 0
    out: dict[str, float] = {}
    out["macro_f1"] = float(f.mean())
    out["macro_mcc"] = float(mcc.mean())
    out["macro_precision"] = float(p.mean())
    out["macro_recall"] = float(r.mean())
    if sup.any():
        out["supported_macro_f1"] = float(f[sup].mean())
        out["supported_macro_mcc"] = float(mcc[sup].mean())
        out["supported_macro_precision"] = float(p[sup].mean())
        out["supported_macro_recall"] = float(r[sup].mean())
    out["n_supported_labels"] = int(sup.sum())

    TP, FP, FN = tp.sum(), fp.sum(), fn.sum()
    TN = n * pred.shape[1] - TP - FP - FN
    mp, mr, mf = _prf(np.array([TP]), np.array([FP]), np.array([FN]))
    mmcc = _mcc(np.array([TP]), np.array([FP]), np.array([FN]), np.array([TN]))
    out["micro_f1"] = float(mf[0])
    out["micro_mcc"] = float(mmcc[0])
    out["micro_precision"] = float(mp[0])
    out["micro_recall"] = float(mr[0])
    return out


def sweep_thresholds(
    probs: np.ndarray,
    tgt: np.ndarray,
    grid: np.ndarray | None = None,
) -> np.ndarray:
    """Per-label F1-optimal threshold on `probs`/`tgt` (both [n, L]).

    Returns thresholds [L]. Labels with no positives fall back to 0.5.
    """
    if grid is None:
        grid = np.round(np.arange(0.02, 0.99, 0.02), 3)
    tgt = tgt.astype(bool)
    n_lab = probs.shape[1]
    best_thr = np.full(n_lab, 0.5, dtype=np.float32)
    best_f1 = np.zeros(n_lab, dtype=np.float32)
    pos = tgt.sum(0)
    for t in grid:
        pred = probs >= t
        tp = (pred & tgt).sum(0).astype(np.float64)
        fp = (pred & ~tgt).sum(0).astype(np.float64)
        fn = (~pred & tgt).sum(0).astype(np.float64)
        denom = 2 * tp + fp + fn
        f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, EPS), 0.0)
        better = f1 > best_f1
        best_f1 = np.where(better, f1, best_f1)
        best_thr = np.where(better, t, best_thr)
    best_thr[pos == 0] = 0.5
    return best_thr


def apply_thresholds(probs: np.ndarray, thr: np.ndarray) -> np.ndarray:
    return probs >= thr[None, :]
