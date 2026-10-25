"""One fact: how often does an effective batch see each tag at least once.

Reads the dataset's own per-tag train counts and reports, for a range of effective batch
sizes, the probability that a tag appears in a single update, averaged over tags (macro
weights every tag equally, so the unweighted mean over tags is the number that matters).

    python scratch/sophie/tag_coverage.py
"""

import csv
import math

TAGS = r"X:\unique\datasets\animetimm-danbooru-wdtagger-v4-w640-ws-full\repo\tags.csv"
TRAIN_N = 5321713
BATCHES = [48, 168, 336, 672, 1344, 2688, 5376, 10752]
BUCKETS = [(0, 5), (5, 20), (20, 50), (50, 200), (200, 1000), (1000, 5000), (5000, 10**9)]


def main() -> None:
    rows = list(csv.DictReader(open(TAGS, encoding="utf-8")))
    counts = [int(r["train_count"]) for r in rows]
    n = len(counts)
    supported = [c for c in counts if c > 0]
    print(f"{n} tags, {len(supported)} with train support, corpus {TRAIN_N}")

    def cov(c: int, b: int) -> float:
        return 1.0 - (1.0 - c / TRAIN_N) ** b

    print("\nmean appearance probability over tags")
    print("| effective batch | mean over all tags | mean over supported | P>=50% | P>=75% | P>=95% |")
    print("|---|---:|---:|---:|---:|---:|")
    prev = None
    for b in BATCHES:
        allm = sum(cov(c, b) for c in counts) / n
        supm = sum(cov(c, b) for c in supported) / len(supported)
        g50 = sum(1 for c in counts if cov(c, b) >= 0.50) / n
        g75 = sum(1 for c in counts if cov(c, b) >= 0.75) / n
        g95 = sum(1 for c in counts if cov(c, b) >= 0.95) / n
        gain = "" if prev is None else f"{100*(allm-prev):+.1f} pts"
        print(f"| {b} | {allm:.4f} | {supm:.4f} | {100*g50:.1f}% | {100*g75:.1f}% | "
              f"{100*g95:.1f}% | {gain}")
        prev = allm

    print("\nmean appearance probability by train-support bucket")
    head = " | ".join(f"{lo}-{hi if hi < 10**9 else 'up'}" for lo, hi in BUCKETS)
    print(f"| effective batch | {head} |")
    print("|---|" + "---:|" * len(BUCKETS))
    for b in BATCHES:
        cells = []
        for lo, hi in BUCKETS:
            cs = [c for c in counts if lo <= c < hi]
            cells.append(f"{sum(cov(c, b) for c in cs)/len(cs):.4f}" if cs else "-")
        print(f"| {b} | " + " | ".join(cells) + " |")

    print("\nhow many tags would need what batch for a 75% chance per update")
    for target in (0.75, 0.95):
        need = [math.log(1 - target) / math.log(1 - c / TRAIN_N) for c in supported]
        need.sort()
        print(f"  target {target:.0%}: median tag {need[len(need)//2]:,.0f} images, "
              f"90th percentile {need[int(len(need)*0.9)]:,.0f}, max {need[-1]:,.0f}")

    print("\njoint probability that every tag appears in one update")
    for b in BATCHES:
        joint = 1.0
        for c in counts:
            joint *= cov(c, b)
            if joint < 1e-12:
                break
        print(f"  batch {b}: {'below 1e-12' if joint < 1e-12 else f'{joint:.3e}'}")

    print("\njitter: positives per update follow Poisson(lambda = c*B/M)")
    print("| effective batch | median tag lambda | median tag rel std | median tag P(zero) | mean rel std |")
    print("|---:|---:|---:|---:|---:|")
    med_c = sorted(counts)[len(counts) // 2]
    for b in BATCHES:
        lam = med_c * b / TRAIN_N
        mean_cv = sum(1 / math.sqrt(max(c * b / TRAIN_N, 1e-12)) for c in counts) / n
        print(f"| {b} | {lam:.3f} | {1/math.sqrt(lam):.2f} | {math.exp(-lam):.3f} | "
              f"{mean_cv:.2f} |")


if __name__ == "__main__":
    main()
