import json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
g = lambda n: [json.loads(l) for l in open(os.path.join(HERE, n), encoding="utf-8")]
br, tc, te = g("geom_train_broad.jsonl"), g("geom_train_cache.jsonl"), g("geom_test_cache.jsonl")

print("total rows:", len(br), len(tc), len(te))
print("meta==PIL:", sum((r["meta_w"], r["meta_h"]) == (r["w"], r["h"]) for r in br + tc + te),
      "/", len(br) + len(tc) + len(te))

for name, rows in (("train_cache", tc), ("train_broad", br), ("test", te)):
    w = np.array([r["w"] for r in rows], float); h = np.array([r["h"] for r in rows], float)
    d = np.maximum(w, h) / np.minimum(w, h)
    print(f"\n{name}: d>3 {int((d>3).sum())} ({100*(d>3).mean():.2f}%)  d>2.5 {int((d>2.5).sum())} "
          f"({100*(d>2.5).mean():.2f}%)")
    i = int(np.argmax(d))
    r = rows[i]
    print(f"  worst: {r['w']}x{r['h']} d={d[i]:.2f} key={r['key']} shard={r['shard']}")
    print(f"  width p1 {np.percentile(w,1):.0f} p50 {np.percentile(w,50):.0f} "
          f"p99 {np.percentile(w,99):.0f} max {w.max():.0f}")
    print(f"  height p1 {np.percentile(h,1):.0f} p99 {np.percentile(h,99):.0f}")
    top = sorted(range(len(rows)), key=lambda j: -d[j])[:3]
    print("  top3 d:", [(rows[j]["w"], rows[j]["h"], round(d[j], 2), rows[j]["shard"]) for j in top])

def union(rows, tags):
    n = 0
    for r in rows:
        tg = {t.replace("_", " ") for t in r["general"]}
        if tg & tags:
            n += 1
    return n

print("\n40k union letterboxed|pillarboxed|white border:",
      union(br, {"letterboxed", "pillarboxed", "white border"}), "/40000 =",
      round(100 * union(br, {"letterboxed", "pillarboxed", "white border"}) / len(br), 3), "%")
print("40k union any border family:",
      union(br, {"letterboxed", "pillarboxed", "border", "white border", "outside border",
                 "black border", "grey border", "inset border", "ornate border",
                 "borderless panels"}), "/40000 =",
      round(100 * union(br, {"letterboxed", "pillarboxed", "border", "white border", "outside border",
                             "black border", "grey border", "inset border", "ornate border",
                             "borderless panels"}) / len(br), 3), "%")
print("40k letterboxed only:", union(br, {"letterboxed"}))
print("40k pillarboxed only:", union(br, {"pillarboxed"}))
print("40k white border:", union(br, {"white border"}))
print("40k border exact:", union(br, {"border"}))
