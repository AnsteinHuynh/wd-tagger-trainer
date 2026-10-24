"""Does the 1-9 positive bin actually matter to macro? Arithmetic, no GPU.

The research note asked for this before any further loss work: macro is the mean over the 12,454
scored tags, so a bin's influence is (tags in bin / 12454) x its F1, NOT its tag count. F1 values
are the measured ones from tag_bins.py (BCE control vs ASL, 50k cache, threshold 0.40).

The point: the 1-9 bin is the largest population (37% of tags) but may contribute almost nothing to
the macro score, in which case aiming gamma at it is a tidy-looking mistake.
"""
N = 12454
BINS = [
    # name, tags, control F1, ASL F1
    ("1-9", 4641, 0.0002, 0.0028),
    ("10-49", 5093, 0.0034, 0.0189),
    ("50-199", 1682, 0.0413, 0.0906),
    ("200-999", 779, 0.1471, 0.1834),
    ("1000+", 259, 0.4100, 0.4098),
]


def main():
    totals = {"ctrl": 0.0, "asl": 0.0}
    print(f"{'bin':>8} {'tags':>6} {'ctrl F1':>8} {'asl F1':>8} "
          f"{'ctrl contrib':>13} {'asl contrib':>12} {'delta':>8} {'% of ASL':>9}")
    for name, n, c, a in BINS:
        cc, ca = n / N * c, n / N * a
        totals["ctrl"] += cc
        totals["asl"] += ca
        print(f"{name:>8} {n:6d} {c:8.4f} {a:8.4f} {cc:13.5f} {ca:12.5f} "
              f"{ca - cc:+8.5f} {100 * ca / 0.0410:8.1f}%")
    print(f"{'TOTAL':>8} {N:6d} {'':8} {'':8} {totals['ctrl']:13.5f} "
          f"{totals['asl']:12.5f} {totals['asl'] - totals['ctrl']:+8.5f}")

    print("\nwhat-if levers (macro change):")
    print(f"  1-9 bin F1 x5      (0.0028 -> 0.0140): "
          f"{4641 / N * (0.0140 - 0.0028):+.5f}")
    print(f"  10-49 bin F1 x2    (0.0189 -> 0.0378): "
          f"{5093 / N * (0.0378 - 0.0189):+.5f}")
    print(f"  50-199 bin F1 +50% (0.0906 -> 0.1359): "
          f"{1682 / N * 0.0453:+.5f}")
    print(f"  200-999 bin +50%   (0.1834 -> 0.2751): "
          f"{779 / N * 0.0917:+.5f}")
    print(f"  1000+ bin +10%     (0.4098 -> 0.4508): "
          f"{259 / N * 0.0410:+.5f}")


if __name__ == "__main__":
    main()
