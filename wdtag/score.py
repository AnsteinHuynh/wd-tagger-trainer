"""Scoring helpers shared by our model and the baseline."""
from __future__ import annotations

import numpy as np

from .labels import CAT_CHARACTER, CAT_GENERAL, CAT_RATING, load_label_space
from .metrics import apply_thresholds, compute_metrics, sweep_thresholds

CAT_NAME = {CAT_RATING: "rating", CAT_GENERAL: "general", CAT_CHARACTER: "character"}


def score_probs(
    probs: np.ndarray,
    tgt: np.ndarray,
    valid_probs: np.ndarray | None = None,
    valid_tgt: np.ndarray | None = None,
) -> dict:
    """Full scorecard: fixed 0.5 plus per-label thresholds tuned on val."""
    out: dict = {"threshold_0.5": compute_metrics(probs >= 0.5, tgt)}

    if valid_probs is not None and valid_tgt is not None:
        thr = sweep_thresholds(valid_probs, valid_tgt)
        out["threshold_tuned"] = compute_metrics(apply_thresholds(probs, thr), tgt)

    # per-category macro at 0.5 (micro over that category's labels)
    space = load_label_space()
    cats = np.array(space.categories)
    by_cat = {}
    for c, name in CAT_NAME.items():
        cols = np.where(cats == c)[0]
        m = compute_metrics((probs >= 0.5)[:, cols], tgt[:, cols])
        by_cat[name] = {k: v for k, v in m.items() if k.startswith("macro")}
    out["by_category_threshold_0.5"] = by_cat
    out["n_images"] = int(probs.shape[0])
    out["n_labels"] = int(probs.shape[1])
    return out


def fmt_table(score: dict) -> str:
    lines = []
    for key in ("threshold_0.5", "threshold_tuned"):
        if key not in score:
            continue
        m = score[key]
        lines.append(f"| {key} | macro | {m['macro_f1']:.4f} | {m['macro_mcc']:.4f} | "
                     f"{m['macro_precision']:.4f} | {m['macro_recall']:.4f} |")
        lines.append(f"| {key} | micro | {m['micro_f1']:.4f} | {m['micro_mcc']:.4f} | "
                     f"{m['micro_precision']:.4f} | {m['micro_recall']:.4f} |")
    return "\n".join(lines)
