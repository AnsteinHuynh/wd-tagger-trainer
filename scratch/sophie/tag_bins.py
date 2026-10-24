"""Where does ASL's +51% macro actually come from? Per-tag, from the probability dumps.

Compares h1 (BCE control, run 470e6fa9) against h10 (ASL, run 7ee12110) on the same 50,000
validation images at threshold 0.40, using the cached probability matrices -- no inference needed.

Binning is by the number of POSITIVES IN THE 50k SUBSET, which is directly observable and needs no
train-set count. The question it answers: is ASL's macro gain concentrated in the rarest tags (the
tail hypothesis we have been asserting) or spread evenly?

    python scratch/sophie/tag_bins.py [--thr 0.40] [--size 50000]
"""
import argparse
import json
import os

import numpy as np

CACHE = r"I:\.cache\wd-tagger-trainer"
PROBS = os.path.join(CACHE, "probs")
NESTED = os.path.join(CACHE, "valid_nested")
RUNS = {"control_BCE": "470e6fa9_005000", "asl": "7ee12110_005000"}
BINS = [(1, 9), (10, 49), (50, 199), (200, 999), (1000, 10 ** 9)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--thr", type=float, default=0.40)
    ap.add_argument("--size", type=int, default=50000)
    ap.add_argument("--chunk", type=int, default=2000)
    a = ap.parse_args()

    lab = np.load(os.path.join(NESTED, "labels.npz"))
    idx_all, offs = lab["indices"], lab["offsets"]
    meta = json.load(open(os.path.join(NESTED, "meta.json"), encoding="utf-8"))
    n = min(a.size, int(meta["count"]))
    n_labels = int(lab["n_labels"]) if "n_labels" in lab else int(idx_all.max()) + 1
    print(f"{n} images, {n_labels} labels, threshold {a.thr}", flush=True)

    # per-tag positive counts in the subset (needed for binning, and cheap to get)
    pos = np.zeros(n_labels, dtype=np.int64)
    for i in range(n):
        pos[idx_all[int(offs[i]):int(offs[i + 1])]] += 1

    counts = {k: dict(tp=np.zeros(n_labels), fp=np.zeros(n_labels), fn=np.zeros(n_labels))
              for k in RUNS}
    for name, tag in RUNS.items():
        p = os.path.join(PROBS, tag + ".npy")
        if not os.path.exists(p):
            print(f"MISSING {p}")
            return 1
        m = np.load(p, mmap_mode="r")
        print(f"{name}: {m.shape} {m.dtype}", flush=True)
        for start in range(0, n, a.chunk):
            end = min(start + a.chunk, n)
            y = np.zeros((end - start, n_labels), dtype=bool)
            for i in range(start, end):
                y[i - start, idx_all[int(offs[i]):int(offs[i + 1])]] = True
            pred = np.asarray(m[start:end]) > a.thr
            c = counts[name]
            c["tp"] += (pred & y).sum(0)
            c["fp"] += (pred & ~y).sum(0)
            c["fn"] += (~pred & y).sum(0)
        del m

    def f1(c):
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        denom = 2 * tp + fp + fn
        return np.where(denom > 0, 2 * tp / np.maximum(denom, 1), np.nan)

    f_ctrl, f_asl = f1(counts["control_BCE"]), f1(counts["asl"])

    def pr(c):
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        p = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), np.nan)
        r = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), np.nan)
        return p, r

    p_ctrl, r_ctrl = pr(counts["control_BCE"])
    p_asl, r_asl = pr(counts["asl"])

    print(f"\n{'bin (val positives)':>22} {'tags':>6} {'ctrl':>8} {'asl':>8} {'delta':>8}")
    for lo, hi in BINS:
        sel = (pos >= lo) & (pos <= hi)
        if not sel.any():
            continue
        c = np.nanmean(f_ctrl[sel])
        s = np.nanmean(f_asl[sel])
        label = f"{lo}-{hi}" if hi < 10 ** 9 else f"{lo}+"
        print(f"{label:>22} {int(sel.sum()):6d} {c:8.4f} {s:8.4f} {s - c:+8.4f}")

    print(f"\n{'bin':>10} {'ctrl P':>8} {'asl P':>8} {'dP':>8} {'ctrl R':>8} {'asl R':>8} {'dR':>8}")
    for lo, hi in BINS:
        sel = (pos >= lo) & (pos <= hi)
        if not sel.any():
            continue
        label = f"{lo}-{hi}" if hi < 10 ** 9 else f"{lo}+"
        cp, sp = np.nanmean(p_ctrl[sel]), np.nanmean(p_asl[sel])
        cr, sr = np.nanmean(r_ctrl[sel]), np.nanmean(r_asl[sel])
        print(f"{label:>10} {cp:8.4f} {sp:8.4f} {sp - cp:+8.4f} "
              f"{cr:8.4f} {sr:8.4f} {sr - cr:+8.4f}")

    ok = ~np.isnan(f_ctrl) & ~np.isnan(f_asl)
    print(f"\noverall macro@0.40  ctrl {np.nanmean(f_ctrl):.4f}  asl {np.nanmean(f_asl):.4f}")
    improved = int((f_asl[ok] > f_ctrl[ok]).sum())
    print(f"tags improved {improved}/{int(ok.sum())} "
          f"({100*improved/max(int(ok.sum()),1):.1f}%), "
          f"fired at all: ctrl {int((~np.isnan(f_ctrl)).sum())}, "
          f"asl {int((~np.isnan(f_asl)).sum())}")

    delta = np.where(ok, f_asl - f_ctrl, np.nan)
    order = np.argsort(np.nan_to_num(delta, nan=-9))[::-1]
    print("\ntop 10 tag gains:")
    for j in order[:10]:
        print(f"  label {j:5d}  pos {int(pos[j]):5d}  ctrl {f_ctrl[j]:.3f} -> asl {f_asl[j]:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
