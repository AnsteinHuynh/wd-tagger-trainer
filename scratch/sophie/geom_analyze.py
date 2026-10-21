"""Stage 2: tag forensics + aspect-ratio stats + joinable CSVs.

Reads the JSONL produced by geom_sample.py plus the dataset's own tags.csv
(full 12,476-tag vocabulary with corpus-wide counts).
Writes scratch/sophie/{test_aspect,train_sample_aspect,train_broad_aspect}.csv
and scratch/sophie/geom_stats.json ; prints a full human-readable summary.
"""
import csv, json, math, os, re, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = r"X:\unique\datasets\animetimm-danbooru-wdtagger-v4-w640-ws-full\repo"
NP = np

def load_jsonl(name):
    with open(os.path.join(HERE, name), encoding="utf-8") as f:
        return [json.loads(l) for l in f]

def canon(s):
    return s.lower().strip().replace("_", " ")

# ---------------------------------------------------------------- vocabulary
vocab = []  # (name, count, selected, train, test, val)
with open(os.path.join(REPO, "tags.csv"), encoding="utf-8") as f:
    rd = csv.DictReader(f)
    for r in rd:
        vocab.append((r["name"], int(r["count"]), int(r["test_count"]), int(r["train_count"]), int(r["val_count"])))
print(f"vocabulary: {len(vocab)} tags, total count {sum(v[1] for v in vocab):,}")

CANON = {v[0]: canon(v[0]) for v in vocab}

# ---------------------------------------------------------------- samples
tr_br = load_jsonl("geom_train_broad.jsonl")   # 40,000 images, 100 shards
tr_ca = load_jsonl("geom_train_cache.jsonl")   # 10,000 = exact training set
te_ca = load_jsonl("geom_test_cache.jsonl")    # 1,000  = exact test order
for g in (tr_br, tr_ca, te_ca):
    for r in g:
        r["tags"] = [canon(t) for t in r["general"]]
print(f"samples: train_broad={len(tr_br):,} train_cache={len(tr_ca):,} test_cache={len(te_ca):,}")

# ---------------------------------------------------------------- patterns
STRETCH_PRIMARY = ["stretch", "disproportion", "bad aspect ratio", "aspect ratio"]
STRETCH_ALT = ["distort", "squashed", "squished", "elongat", "fattened", "thin",
               "deformed", "bad proportions", "messed up", "bad anatomy"]
PAD_PRIMARY = ["white bar", "side panel", "white background panel"]
PAD_ALT = ["bar", "border", "letterbox", "pillarbox", "panel", "frame", "padding",
           "white background", "simple background", "cropped", "crop", "out of frame",
           "split screen", "multiple views", "two side", "side by side", "comic",
           "2koma", "4koma", "manga"]

def classify(tag_canon, pat):
    """returns 'exact' | 'token' | 'sub' | None"""
    if tag_canon == pat:
        return "exact"
    tp, tt = pat.split(), tag_canon.split()
    if len(tt) >= len(tp):
        for i in range(len(tt) - len(tp) + 1):
            if tt[i:i + len(tp)] == tp:
                return "token"
    if pat in tag_canon:
        return "sub"
    return None

def pattern_report(pat, group_name, pats_row):
    hits = {"exact": [], "token": [], "sub": []}
    for name, cnt, tcnt, trcnt, vcnt in vocab:
        c = classify(CANON[name], pat)
        if c:
            hits[c].append((name, cnt))
    for k in hits:
        hits[k].sort(key=lambda x: -x[1])
    # per-image counts on the broad sample
    img = {"exact": 0, "token": 0, "sub": 0}
    tagimg = {}
    for r in tr_br:
        best = None
        for t in r["tags"]:
            c = classify(t, pat)
            if c is None:
                continue
            tagimg.setdefault(t, 0)
            tagimg[t] += 1
            rank = {"exact": 3, "token": 2, "sub": 1}[c]
            if best is None or rank > best:
                best = rank
        if best is not None:
            img[["sub", "token", "exact"][best - 1]] += 1
    sample_total = {"exact": 0, "token": 0, "sub": 0}
    for r in tr_br:
        for t in r["tags"]:
            c = classify(t, pat)
            if c:
                sample_total[c] += 1
    top = sorted(tagimg.items(), key=lambda x: -x[1])[:8]
    row = dict(group=group_name, pattern=pat,
               vocab_exact=[h[0] for h in hits["exact"]],
               vocab_token=[h[0] for h in hits["token"]],
               vocab_sub=[h[0] for h in hits["sub"]],
               vocab_any=len(hits["exact"]) + len(hits["token"]) + len(hits["sub"]),
               img_exact=img["exact"], img_token=img["token"], img_sub=img["sub"],
               img_any=img["exact"] + img["token"] + img["sub"],
               tagimg_total=sum(sample_total.values()),
               top_tags=top)
    pats_row.append(row)
    return row

rows = []
for p in STRETCH_PRIMARY:
    pattern_report(p, "stretch-primary", rows)
for p in STRETCH_ALT:
    pattern_report(p, "stretch-alt", rows)
for p in PAD_PRIMARY:
    pattern_report(p, "pad-primary", rows)
for p in PAD_ALT:
    pattern_report(p, "pad-alt", rows)

N = len(tr_br)
def pct(k):
    return 100.0 * k / N

print("\n=== TASK A: pattern scan over 40,000 sampled train images ===")
hdr = f"{'group':16s} {'pattern':22s} {'vocab(ex/tok/sub)':20s} {'imgs any':>9s} {'%':>8s}  matched tags by image count"
print(hdr)
for r in rows:
    v = f"{len(r['vocab_exact'])}/{len(r['vocab_token'])}/{len(r['vocab_sub'])}"
    tops = ", ".join(f"{t}:{c}" for t, c in r["top_tags"][:5])
    print(f"{r['group']:16s} {r['pattern']:22s} {v:20s} {r['img_any']:9,d} {pct(r['img_any']):7.3f}%  {tops}")

print("\n=== patterns with ZERO vocabulary hits (string appears in no tag at all) ===")
none = [r["pattern"] for r in rows if r["vocab_any"] == 0]
print(none)

print("\n=== full vocabulary scan: every tag containing these roots ===")
ROOTS = ["stretch", "disproport", "aspect", "distort", "squash", "squish", "elong",
         "fatten", "deform", "proportion", "bad anatomy", "messed",
         "letterbox", "pillarbox", "white bar", "side panel", " out of frame",
         "split screen", "multiple views", "side by side", "2koma", "4koma", "comic", "manga",
         "border", "padding", "panel", " cropped", " crop", "frame"]
vocab_scan = {}
for root in ROOTS:
    hits = [(n, c) for n, c, *_ in vocab if root in CANON[n]]
    hits.sort(key=lambda x: -x[1])
    vocab_scan[root] = hits
    print(f"  {root!r}: {len(hits)} tags -> {hits[:12]}")

# ---------------------------------------------------------------- Task B
def stats(group, name):
    w = NP.array([r["w"] for r in group], dtype=float)
    h = NP.array([r["h"] for r in group], dtype=float)
    a = w / h
    d = NP.maximum(w, h) / NP.minimum(w, h)
    waste = 1.0 - (w * h) / NP.maximum(w, h) ** 2
    qs = [0, 1, 5, 25, 50, 75, 95, 99, 100]
    out = dict(
        name=name, n=len(group),
        aspect_pct={q: float(NP.percentile(a, q)) for q in qs},
        aspect_min=float(a.min()), aspect_max=float(a.max()),
        h_pct={q: float(NP.percentile(h, q)) for q in qs},
        w_pct={q: float(NP.percentile(w, q)) for q in qs},
        d_pct={q: float(NP.percentile(d, q)) for q in qs},
        waste_pct={q: float(NP.percentile(waste, q)) for q in qs},
        waste_mean=float(waste.mean()),
        frac_w640=float((w == 640).mean()),
        frac_h640=float((h == 640).mean()),
        frac_min640=float((NP.minimum(w, h) == 640).mean()),
        frac_min_lt640=float((NP.minimum(w, h) < 640).mean()),
        frac_max_eq640=float((NP.maximum(w, h) == 640).mean()),
        distinct_w=int(len(NP.unique(w))), distinct_h=int(len(NP.unique(h))),
        distinct_aspect=int(len(NP.unique(NP.round(a, 4)))),
    )
    ab = NP.abs(a - 1.0)
    buckets = [(0.0, 0.02), (0.02, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 1e9)]
    out["absdev_buckets"] = [
        dict(lo=lo, hi=hi, n=int(((ab >= lo) & (ab < hi)).sum()),
             pct=float(100 * ((ab >= lo) & (ab < hi)).mean()))
        for lo, hi in buckets]
    targets = [("1.00 square", 1.00), ("1.33 (4:3)", 1.3333), ("1.50 (3:2)", 1.50),
               ("1.78 (16:9)", 1.7778), ("0.75 (3:4)", 0.75), ("0.56 (9:16)", 0.5625)]
    out["common"] = [dict(label=lab, target=t,
                          n=int((NP.abs(a - t) <= 0.02).sum()),
                          pct=float(100 * (NP.abs(a - t) <= 0.02).mean()))
                     for lab, t in targets]
    out["orientation"] = dict(
        square=int((a == 1.0).sum()), portrait=int((a < 1).sum()), landscape=int((a > 1).sum()),
        portrait_pct=float(100 * (a < 1).mean()), landscape_pct=float(100 * (a > 1).mean()),
        square_pct=float(100 * (a == 1).mean()))
    out["d_frac"] = {t: dict(n=int((d > t).sum()), pct=float(100 * (d > t).mean()))
                     for t in (1.25, 1.5, 1.75, 2.0)}
    # exact top sizes / reduced ratios
    from collections import Counter
    c = Counter((int(x), int(y)) for x, y in zip(w, h))
    out["top_sizes"] = [[[s[0], s[1]], n] for s, n in c.most_common(12)]
    rc = Counter()
    for x, y in zip(w, h):
        g = math.gcd(int(x), int(y))
        rc[(int(x) // g, int(y) // g)] += 1
    out["top_ratios"] = [[[rr[0], rr[1]], n] for rr, n in rc.most_common(12)]
    return out

S_br = stats(tr_br, "train_broad(40k)")
S_ca = stats(tr_ca, "train_cache(10k = training set)")
S_te = stats(te_ca, "test_cache(1k = eval order)")

def show(s):
    print(f"\n--- {s['name']} (n={s['n']:,}) ---")
    p = s["aspect_pct"]
    print(f"aspect w/h: min {p[0]:.4f}  p5 {p[5]:.4f}  p25 {p[25]:.4f}  p50 {p[50]:.4f}  "
          f"p75 {p[75]:.4f}  p95 {p[95]:.4f}  p99 {p[99]:.4f}  max {p[100]:.4f}")
    hp = s["h_pct"]
    print(f"height: min {hp[0]:.0f}  p5 {hp[5]:.0f}  p25 {hp[25]:.0f}  p50 {hp[50]:.0f}  "
          f"p75 {hp[75]:.0f}  p95 {hp[95]:.0f}  max {hp[100]:.0f}")
    print(f"width:  min {s['w_pct'][0]:.0f}  p50 {s['w_pct'][50]:.0f}  max {s['w_pct'][100]:.0f}  "
          f"| w==640 {100*s['frac_w640']:.1f}%  h==640 {100*s['frac_h640']:.1f}%  "
          f"min(w,h)==640 {100*s['frac_min640']:.2f}%  min<640 {100*s['frac_min_lt640']:.2f}%  "
          f"max(w,h)==640 {100*s['frac_max_eq640']:.1f}%")
    print(f"distinct w/h/aspect: {s['distinct_w']}/{s['distinct_h']}/{s['distinct_aspect']}")
    print("|aspect-1| buckets: " + "  ".join(f"[{b['lo']},{b['hi']}): {b['n']:,} ({b['pct']:.2f}%)"
                                             for b in s["absdev_buckets"]))
    print("common aspects (|a-t|<=0.02): " + "  ".join(f"{c['label']}: {c['n']:,} ({c['pct']:.2f}%)"
                                                       for c in s["common"]))
    print(f"orientation: portrait {s['orientation']['portrait']:,} ({s['orientation']['portrait_pct']:.2f}%)"
          f"  landscape {s['orientation']['landscape']:,} ({s['orientation']['landscape_pct']:.2f}%)"
          f"  square {s['orientation']['square']:,} ({s['orientation']['square_pct']:.2f}%)")
    dp = s["d_pct"]
    print(f"distortion d: p5 {dp[5]:.4f} p25 {dp[25]:.4f} p50 {dp[50]:.4f} p75 {dp[75]:.4f} "
          f"p95 {dp[95]:.4f} p99 {dp[99]:.4f} max {dp[100]:.4f}")
    print("d>1.25: %d (%.2f%%)  d>1.5: %d (%.2f%%)  d>1.75: %d (%.2f%%)  d>2: %d (%.2f%%)" % (
        s["d_frac"][1.25]["n"], s["d_frac"][1.25]["pct"],
        s["d_frac"][1.5]["n"], s["d_frac"][1.5]["pct"],
        s["d_frac"][1.75]["n"], s["d_frac"][1.75]["pct"],
        s["d_frac"][2.0]["n"], s["d_frac"][2.0]["pct"]))
    wp = s["waste_pct"]
    print(f"letterbox blank fraction: mean {100*s['waste_mean']:.2f}%  p50 {100*wp[50]:.2f}%  "
          f"p75 {100*wp[75]:.2f}%  p95 {100*wp[95]:.2f}%  max {100*wp[100]:.2f}%")
    print("top sizes:", s["top_sizes"][:8])
    print("top reduced ratios:", s["top_ratios"][:8])

show(S_br); show(S_ca); show(S_te)

# ---------------------------------------------------------------- CSVs
def write_csv(path, group):
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["row", "w", "h", "aspect", "key", "shard"])
        for r in group:
            wr.writerow([r["row"], r["w"], r["h"], f"{r['w']/r['h']:.6f}", r["key"], r["shard"]])
    print(f"wrote {path} ({len(group)} rows)")

write_csv(os.path.join(HERE, "test_aspect.csv"), te_ca)
write_csv(os.path.join(HERE, "train_sample_aspect.csv"), tr_ca)
write_csv(os.path.join(HERE, "train_broad_aspect.csv"), tr_br)

# ---------------------------------------------------------------- save stats
with open(os.path.join(HERE, "geom_stats.json"), "w", encoding="utf-8") as f:
    json.dump(dict(vocab_size=len(vocab), n_broad=N, patterns=rows,
                   stats=[S_br, S_ca, S_te]), f, indent=1)
print("\nsaved geom_stats.json")
