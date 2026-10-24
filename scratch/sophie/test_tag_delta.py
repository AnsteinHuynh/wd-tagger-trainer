"""One purpose: where the large tower loses to base, tag by tag.

Both runs are scored on the same 50000 image nested cache at the fixed 0.40 threshold, so their
probability caches are directly comparable row for row. This prints F1 per tag frequency bucket and
the biggest single-tag wins and losses, which is what decides whether the wide tower is short of
data on the tail or simply worse everywhere.

    python scratch/sophie/test_tag_delta.py
"""

import os

import numpy as np

NESTED = r"I:\.cache\wd-tagger-trainer\valid_nested"
PROBS = r"I:\.cache\wd-tagger-trainer\probs"
RUNS = (("base u0", "b6e293f0_012500"), ("large u0", "f63f6f54_030000"))
# second pair on the command line: `--pair runA_step runB_step` compares any two cached runs
import sys
if "--pair" in sys.argv:
    i = sys.argv.index("--pair")
    RUNS = (("A", sys.argv[i + 1]), ("B", sys.argv[i + 2]))
elif "--solo" in sys.argv:
    RUNS = (("solo", sys.argv[sys.argv.index("--solo") + 1]), ("solo2", sys.argv[sys.argv.index("--solo") + 1]))
THR = 0.40
N_LABELS = 12476
def labels(n: int) -> np.ndarray:
    lab = np.load(os.path.join(NESTED, "labels.npz"))
    idx, offs = lab["indices"], lab["offsets"]
    y = np.zeros((n, N_LABELS), dtype=bool)
    for i in range(n):
        y[i, idx[int(offs[i]):int(offs[i + 1])]] = True
    return y


def per_tag_f1(path: str, y: np.ndarray) -> tuple:
    probs = np.load(path, mmap_mode="r")
    n = probs.shape[0]
    tp = np.zeros(N_LABELS, dtype=np.int64)
    fp = np.zeros(N_LABELS, dtype=np.int64)
    fn = np.zeros(N_LABELS, dtype=np.int64)
    for s in range(0, n, 2000):
        p = np.asarray(probs[s:s + 2000], dtype=np.float32) >= THR
        t = y[s:s + 2000]
        tp += (p & t).sum(0)
        fp += (p & ~t).sum(0)
        fn += (~p & t).sum(0)
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    return f1, tp + fn


def main() -> None:
    meta = os.path.join(PROBS, RUNS[0][1] + ".npy.json")
    import json
    n = int(json.load(open(meta, encoding="utf-8"))["rows"])
    y = labels(n)
    print(f"{n} images, {int((y.sum(0) > 0).sum())} labels with positives, threshold {THR}")

    out = {}
    for name, tag in RUNS:
        f1, supp = per_tag_f1(os.path.join(PROBS, tag + ".npy"), y)
        out[name] = (f1, supp)
        print(f"{name:9s} macro F1 {f1.mean():.4f} over all labels")

    base, supp = out[RUNS[0][0]]
    large, _ = out[RUNS[1][0]]
    d = large - base
    print(f"delta (large - base): mean {d.mean():+.4f}, better on {int((d > 0).sum())} labels, "
          f"worse on {int((d < 0).sum())}")
    print("by support bucket (images containing the tag):")
    edges = [0, 5, 20, 50, 200, 1000, 5000, 10 ** 9]
    for lo, hi in zip(edges, edges[1:]):
        m = (supp >= lo) & (supp < hi)
        if m.sum() == 0:
            continue
        print(f"  support {lo:>5}-{hi if hi < 10 ** 9 else '+':>5}: {int(m.sum()):5d} labels  "
              f"base {base[m].mean():.4f}  large {large[m].mean():.4f}  delta {d[m].mean():+.4f}")
    order = np.argsort(d)
    from wdtag.labels import load_label_space
    names = load_label_space().names
    print("worst 8 for the large tower:")
    for i in order[:8]:
        print(f"  {names[i]:40s} support {int(supp[i]):5d}  base {base[i]:.3f} large {large[i]:.3f}")
    print("best 8 for the large tower:")
    for i in order[::-1][:8]:
        print(f"  {names[i]:40s} support {int(supp[i]):5d}  base {base[i]:.3f} large {large[i]:.3f}")


if __name__ == "__main__":
    main()

