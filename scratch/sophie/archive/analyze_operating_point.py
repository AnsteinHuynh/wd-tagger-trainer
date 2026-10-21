"""Where is our model actually operating? Global vs per-label thresholds.

Everything is chosen on VALID and reported on TEST, identically for the baseline
and for our run, so the comparison stays honest.
"""
import json
import numpy as np
from wdtag.data import load_targets
from wdtag.metrics import compute_metrics, sweep_thresholds
from wdtag.labels import load_label_space

N = len(load_label_space())
tgt_te = load_targets("test", "pad", N).astype(bool)
tgt_va = load_targets("valid", "pad", N).astype(bool)

RUNS = {
    "baseline": (r"runs\_baseline\convnextv2_huge\pad\valid_probs.npy",
                 r"runs\_baseline\convnextv2_huge\pad\test_probs.npy"),
    "oftv2-pad": (r"runs\54f4af8e\valid_probs.npy", r"runs\54f4af8e\test_probs.npy"),
}

def show(tag, m):
    print(f"  {tag:34s} macro F1 {m['macro_f1']:.4f}  MCC {m['macro_mcc']:.4f}  "
          f"P {m['macro_precision']:.4f}  R {m['macro_recall']:.4f} | "
          f"micro F1 {m['micro_f1']:.4f}  MCC {m['micro_mcc']:.4f}  "
          f"P {m['micro_precision']:.4f}  R {m['micro_recall']:.4f}", flush=True)

GRID = np.round(np.arange(0.02, 0.99, 0.02), 3)
for name, (vpath, tpath) in RUNS.items():
    pv = np.load(vpath); pt = np.load(tpath)
    print(f"\n=== {name} ===")
    show("fixed 0.5", compute_metrics(pt >= 0.5, tgt_te))

    # one global threshold, chosen on valid
    best, best_f1 = 0.5, -1
    for t in GRID:
        f1 = compute_metrics(pv >= t, tgt_va)["micro_f1"]
        if f1 > best_f1:
            best_f1, best = f1, float(t)
    print(f"  [global threshold picked on val: {best:.2f}  (val micro F1 {best_f1:.4f})]")
    show("global-tuned", compute_metrics(pt >= best, tgt_te))

    # per-label thresholds, but only trust them where val has enough support
    thr = sweep_thresholds(pv, tgt_va, GRID)
    for min_sup in (2, 8, 32):
        t = thr.copy()
        t[tgt_va.sum(0) < min_sup] = 0.5
        show(f"per-label (val support >= {min_sup})", compute_metrics(pt >= t[None, :], tgt_te))

    # threshold-free ranking quality: predict the top-k labels per image
    order = np.argsort(-pt, axis=1)
    for k in (10, 20, 30, 50):
        pred = np.zeros_like(tgt_te)
        np.put_along_axis(pred, order[:, :k], True, axis=1)
        m = compute_metrics(pred, tgt_te)
        print(f"  {'top-' + str(k) + ' per image':34s} micro F1 {m['micro_f1']:.4f}  "
              f"P {m['micro_precision']:.4f}  R {m['micro_recall']:.4f}  "
              f"avg preds/img {pred.sum(1).mean():.1f}", flush=True)
    print(f"  avg true labels/img: {tgt_te.sum(1).mean():.1f}")
