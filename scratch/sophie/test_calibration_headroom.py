"""One purpose: how much macro@0.40 is sitting in per-tag calibration, and how much survives a fit
on data the evaluation did not see.

The metric is fixed at threshold 0.40, so a tag whose scores sit just under 0.40 costs F1 in both
precision and recall while a tag sitting just over it costs precision. Shifting a tag's scores by a
constant (a per-tag bias, which is what a threshold per tag amounts to) can therefore move macro a
long way without touching the model. This script prices that, from the probability cache, on CPU.

Two numbers come out. The oracle, which picks each tag's best cut on all 50000 images and is an
upper bound that cannot be achieved honestly. And the honest one, which fits each tag's cut on even
rows and evaluates on odd rows.

    python scratch/sophie/test_calibration_headroom.py [--run b6e293f0] [--step 012500]
"""

import argparse
import json
import os

import numpy as np

NESTED = r"I:\.cache\wd-tagger-trainer\valid_nested"
PROBS = r"I:\.cache\wd-tagger-trainer\probs"
NBIN = 40  # cuts at 0, 0.025, ... 1.0: fine enough around 0.40 for a tag with few positives


def load_labels(count: int, n_labels: int) -> np.ndarray:
    lab = np.load(os.path.join(NESTED, "labels.npz"))
    idx, offs = lab["indices"], lab["offsets"]
    y = np.zeros((count, n_labels), dtype=bool)
    for i in range(count):
        y[i, idx[int(offs[i]):int(offs[i + 1])]] = True
    return y


def binned_counts(probs, y, mask):
    """[bin, label] counts of scores for positives and negatives, over the rows in `mask`."""
    pos = np.zeros((NBIN + 1, y.shape[1]), dtype=np.int64)
    neg = np.zeros((NBIN + 1, y.shape[1]), dtype=np.int64)
    rows = np.flatnonzero(mask)
    n_lab = y.shape[1]
    chunk = 2000
    for s in range(0, len(rows), chunk):
        r = rows[s:s + chunk]
        p = np.asarray(probs[r], dtype=np.float32)
        b = np.minimum((p * NBIN).astype(np.int32), NBIN)
        t = y[r]
        # combined index so one bincount does every label at once
        bp = (b * n_lab + np.arange(n_lab, dtype=np.int32)).ravel()
        pos += np.bincount(bp[t.ravel()], minlength=(NBIN + 1) * n_lab).reshape(NBIN + 1, n_lab)
        neg += np.bincount(bp[~t.ravel()], minlength=(NBIN + 1) * n_lab).reshape(NBIN + 1, n_lab)
    return pos, neg


def f1_grid(pos, neg):
    """F1 for every cut position k, where the cut accepts bins >= k."""
    # cumulative from the top: tp[k] = positives in bins k..NBIN
    tp = np.cumsum(pos[::-1], 0)[::-1]
    fp = np.cumsum(neg[::-1], 0)[::-1]
    npos = pos.sum(0, keepdims=True)
    fn = npos - tp
    return 2 * tp / np.maximum(2 * tp + fp + fn, 1e-12)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="b6e293f0")
    ap.add_argument("--step", default="012500")
    a = ap.parse_args()

    path = os.path.join(PROBS, f"{a.run}_{a.step}.npy")
    meta = json.load(open(path + ".json", encoding="utf-8"))
    count = int(meta.get("valid", meta.get("count", 50000)))
    probs = np.load(path, mmap_mode="r")
    probs = probs[:count]
    n_labels = probs.shape[1]
    y = load_labels(count, n_labels)
    print(f"{path}: {probs.shape} float16, {count} labelled rows, meta {meta}")

    k40 = int(round(0.40 * NBIN))
    all_rows = np.ones(count, dtype=bool)

    pos, neg = binned_counts(probs, y, all_rows)
    f1 = f1_grid(pos, neg)
    at40 = f1[k40].mean()
    oracle = f1.max(0).mean()
    print(f"all {count} rows: macro@0.40 {at40:.4f}   per-tag oracle {oracle:.4f}   "
          f"headroom {100 * (oracle / at40 - 1):+.1f}%")

    even = np.zeros(count, dtype=bool)
    even[::2] = True
    odd = ~even
    pe, ne = binned_counts(probs, y, even)
    po, no = binned_counts(probs, y, odd)
    fe, fo = f1_grid(pe, ne), f1_grid(po, no)
    best_k = fe.argmax(0)  # per-tag cut chosen on even rows
    cols = np.arange(n_labels)
    honest = fo[best_k, cols].mean()
    base_odd = fo[k40].mean()
    print(f"fit on even rows, evaluate on odd: base@0.40 {base_odd:.4f}   per-tag cut {honest:.4f}   "
          f"gain {100 * (honest / base_odd - 1):+.1f}%")
    moved = (best_k != k40)
    print(f"tags whose cut moved off 0.40: {int(moved.sum())} of {n_labels} "
          f"({100 * moved.mean():.1f}%)")
    shifts = (best_k - k40) * (1.0 / NBIN)
    for q in (1, 5, 25, 50, 75, 95, 99):
        print(f"  shift p{q:02d} {np.percentile(shifts, q):+.3f}")

    # one global knob: a single output bias fitted on even rows, which is what temperature
    # scaling amounts to at a fixed decision threshold, and it has one parameter instead of 12476
    curve_e, curve_o = fe.mean(1), fo.mean(1)
    print("global cut curve (even rows), macro F1 by cut:")
    for k in range(0, NBIN + 1, 4):
        print(f"  cut {k / NBIN:.2f}  even {curve_e[k]:.4f}  odd {curve_o[k]:.4f}")
    kbest = int(curve_e.argmax())
    print(f"single global cut fitted on even rows: {kbest / NBIN:.3f} -> odd macro "
          f"{curve_o[kbest]:.4f} against {base_odd:.4f} at 0.40, gain "
          f"{100 * (curve_o[kbest] / base_odd - 1):+.1f}%")


if __name__ == "__main__":
    main()
