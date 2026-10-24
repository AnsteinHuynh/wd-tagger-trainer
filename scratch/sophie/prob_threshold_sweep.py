"""Is ASL's macro win real ranking quality, or a global operating-point shift?

Cheap companion to `threshold_sweep.py` (which re-runs inference): this reads the cached probability
matrices, so it costs no GPU time.

Why it matters: the tag-bin analysis showed ASL trading precision for recall in *every* bin, and at
the 1000+ tags F1 did not move (0.4100 -> 0.4098) while precision fell 0.534 -> 0.366 and recall rose
0.351 -> 0.471. A ~1:1 trade with flat F1 at the head is the signature of a monotone score shift, not
of better ordering. If that is all ASL is doing, then stacking more negative-side relaxation (NTR,
CSL, SPLC) will keep shifting the operating point and eventually collapse precision -- which is
exactly the risk the loss research flagged.

Method: one pass per run's cached probabilities, accumulating per-(threshold, label) counts over a
GRID OF GLOBAL THRESHOLDS. Global only -- per-tag best would be tuning, and our protocol forbids it.

    python scratch/sophie/prob_threshold_sweep.py
"""
import json
import os

import numpy as np

CACHE = r"I:\.cache\wd-tagger-trainer"
PROBS = os.path.join(CACHE, "probs")
NESTED = os.path.join(CACHE, "valid_nested")
RUNS = {"h10_CMSS": "7ee12110_005000", "arm01_cls": "83231ec3_005200"}
THR = [0.05, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60]
CHUNK = 2000


def main() -> int:
    lab = np.load(os.path.join(NESTED, "labels.npz"))
    idx_all, offs = lab["indices"], lab["offsets"]
    meta = json.load(open(os.path.join(NESTED, "meta.json"), encoding="utf-8"))
    n = min(50000, int(meta["count"]))
    n_labels = int(idx_all.max()) + 1
    T = len(THR)

    results = {}
    for name, tag in RUNS.items():
        p = os.path.join(PROBS, tag + ".npy")
        if not os.path.exists(p):
            print(f"MISSING {p}")
            return 1
        m = np.load(p, mmap_mode="r")
        tp = np.zeros((T, n_labels))
        fp = np.zeros((T, n_labels))
        fn = np.zeros((T, n_labels))
        for start in range(0, n, CHUNK):
            end = min(start + CHUNK, n)
            y = np.zeros((end - start, n_labels), dtype=bool)
            for i in range(start, end):
                y[i - start, idx_all[int(offs[i]):int(offs[i + 1])]] = True
            block = np.asarray(m[start:end])
            for ti, t in enumerate(THR):
                pred = block > t
                tp[ti] += (pred & y).sum(0)
                fp[ti] += (pred & ~y).sum(0)
                fn[ti] += (~pred & y).sum(0)
        del m
        results[name] = (tp, fp, fn)

        print(f"\n=== {name} ===")
        print(f"{'thr':>6} {'macroF1':>9} {'macroP':>8} {'macroR':>8} "
              f"{'microF1':>9} {'microP':>8} {'microR':>8}")
        best = (None, -1.0)
        for ti, t in enumerate(THR):
            den = 2 * tp[ti] + fp[ti] + fn[ti]
            f1 = np.where(den > 0, 2 * tp[ti] / np.maximum(den, 1), np.nan)
            prec = np.where(tp[ti] + fp[ti] > 0,
                            tp[ti] / np.maximum(tp[ti] + fp[ti], 1), np.nan)
            rec = np.where(tp[ti] + fn[ti] > 0,
                           tp[ti] / np.maximum(tp[ti] + fn[ti], 1), np.nan)
            TP, FP, FN = tp[ti].sum(), fp[ti].sum(), fn[ti].sum()
            mf = float(np.nanmean(f1))
            print(f"{t:6.2f} {mf:9.4f} {np.nanmean(prec):8.4f} {np.nanmean(rec):8.4f} "
                  f"{2 * TP / max(2 * TP + FP + FN, 1):9.4f} "
                  f"{TP / max(TP + FP, 1):8.4f} {TP / max(TP + FN, 1):8.4f}")
            if mf > best[1]:
                best = (t, mf)
        print(f"best GLOBAL threshold -> macro F1 {best[1]:.4f} at thr {best[0]:.2f}")

    print("\n=== the decisive comparison ===")
    for t in (0.30, 0.35, 0.40):
        pass
    ctp, cfp, cfn = results["control_BCE"]
    for ti, t in enumerate(THR):
        den = 2 * ctp[ti] + cfp[ti] + cfn[ti]
        cf = float(np.nanmean(np.where(den > 0, 2 * ctp[ti] / np.maximum(den, 1), np.nan)))
        if abs(t - 0.40) < 1e-9:
            print(f"  control at its OWN best-vs-ASL operating point: see rows above")
    print("  If control's best global threshold approaches ASL's 0.4097, the ASL gain is an")
    print("  operating-point shift that a global threshold on BCE could also reach -- which the")
    print("  fixed-0.40 protocol forbids us from exploiting, and which explains the precision loss.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
