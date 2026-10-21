"""Stage 2b: extra geometry numbers + a direct pixel check for baked-in bars.

Numbers: size-rule confirmation, upscale factors, tails, height|w=640.
Pixels: decode 4,000 images (10 shards x 400) and measure flat edge strips,
including *symmetric* bars (equal on opposite sides), which is the signature of
real letterbox/pillarbox padding rather than a plain background.
CPU + disk only.
"""
import io, json, os, time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from wdtag.cache import build_plan, _read_samples, DATASET_ROOT, SPLITS

HERE = os.path.dirname(os.path.abspath(__file__))
br = [json.loads(l) for l in open(os.path.join(HERE, "geom_train_broad.jsonl"), encoding="utf-8")]
te = [json.loads(l) for l in open(os.path.join(HERE, "geom_test_cache.jsonl"), encoding="utf-8")]

def extra(group, name):
    w = np.array([r["w"] for r in group], float)
    h = np.array([r["h"] for r in group], float)
    a, d = w / h, np.maximum(w, h) / np.minimum(w, h)
    sw, sh = 512.0 / w, 512.0 / h
    print(f"\n--- {name} n={len(group):,} ---")
    print(f"w==640 {100*(w==640).mean():.2f}%  w<640 {100*(w<640).mean():.2f}%  w>640 {100*(w>640).mean():.2f}%")
    print(f"h==640 {100*(h==640).mean():.2f}%  h<640 {100*(h<640).mean():.2f}%  h>640 {100*(h>640).mean():.2f}%")
    print(f"min(w,h)==640 (downscaled) {100*(np.minimum(w,h)==640).mean():.2f}%  "
          f"min(w,h)<640 (native small) {100*(np.minimum(w,h)<640).mean():.2f}%")
    m = w == 640
    if m.any():
        hp = np.percentile(h[m], [5, 25, 50, 75, 95, 100])
        print(f"height | w==640 (n={m.sum():,}): p5 {hp[0]:.0f} p25 {hp[1]:.0f} p50 {hp[2]:.0f} "
              f"p75 {hp[3]:.0f} p95 {hp[4]:.0f} max {hp[5]:.0f}")
    print(f"squash scale 512/w: p5 {np.percentile(sw,5):.3f} p50 {np.percentile(sw,50):.3f} "
          f"p95 {np.percentile(sw,95):.3f} | 512/h: p5 {np.percentile(sh,5):.3f} "
          f"p50 {np.percentile(sh,50):.3f} p95 {np.percentile(sh,95):.3f}")
    print(f"upscaling after squash: 512/w>1 {100*(sw>1).mean():.2f}%  512/h>1 {100*(sh>1).mean():.2f}%  "
          f"both>1 {100*((sw>1)&(sh>1)).mean():.2f}%  max upscale {max(sw.max(), sh.max()):.2f}x")
    for t in (1.25, 1.5, 1.75, 2.0, 3.0):
        print(f"  d>{t}: {int((d>t).sum()):,} ({100*(d>t).mean():.2f}%)")
    print(f"aspect<0.4: {int((a<0.4).sum()):,} ({100*(a<0.4).mean():.2f}%)   "
          f"aspect>2.5: {int((a>2.5).sum()):,} ({100*(a>2.5).mean():.2f}%)")
    i = int(np.argmin(a)); j = int(np.argmax(a))
    print(f"  most extreme: {group[i]['w']}x{group[i]['h']} key {group[i]['key']} shard {group[i]['shard']} | "
          f"{group[j]['w']}x{group[j]['h']} key {group[j]['key']} shard {group[j]['shard']}")
    return dict(d_frac={str(t): float((d > t).mean()) for t in (1.25, 1.5, 1.75, 2.0, 3.0)})

E_br = extra(br, "train_broad")
E_te = extra(te, "test_cache")

# ------------------------------------------------------------------ pixel bars
def flat_rows(arr, from_top=True, max_frac=0.45, tol=6.0):
    H = arr.shape[0]
    lim = int(H * max_frac)
    ref = (arr[0] if from_top else arr[-1]).mean(axis=0)
    n = 0
    for k in range(lim):
        row = arr[k] if from_top else arr[H - 1 - k]
        if row.std(axis=0).max() > tol or np.abs(row.mean(axis=0) - ref).max() > tol:
            break
        n += 1
    return n, ref

def flat_cols(arr, from_left=True, max_frac=0.45, tol=6.0):
    W = arr.shape[1]
    lim = int(W * max_frac)
    ref = (arr[:, 0] if from_left else arr[:, -1]).mean(axis=0)
    n = 0
    for k in range(lim):
        col = arr[:, k] if from_left else arr[:, W - 1 - k]
        if col.std(axis=0).max() > tol or np.abs(col.mean(axis=0) - ref).max() > tol:
            break
        n += 1
    return n, ref

def analyse(blob):
    with Image.open(io.BytesIO(blob)) as im:
        arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
    H, W, _ = arr.shape
    t, tc = flat_rows(arr, True)
    b, bc = flat_rows(arr, False)
    l, lc = flat_cols(arr, True)
    r, rc = flat_cols(arr, False)
    white = lambda c: bool((c > 235).all())
    return dict(H=H, W=W, top=t, bottom=b, left=l, right=r,
                hbar=min(t, b), vbar=min(l, r),
                hsym=abs(t - b) <= max(2, int(0.02 * H)), vsym=abs(l - r) <= max(2, int(0.02 * W)),
                hbar_white=(white(tc) and white(bc)), vbar_white=(white(lc) and white(rc)),
                dark=(tc.mean() < 30 and bc.mean() < 30) or (lc.mean() < 30 and rc.mean() < 30))

plan = build_plan("train")
shards = list(np.linspace(0, 212, 10).round().astype(int))
t0 = time.time()
jobs = []
for si in shards:
    e = plan[si]
    jobs += _read_samples(os.path.join(DATASET_ROOT, SPLITS["train"]["subdir"], e.shard), 0.0, 400)
print(f"\nbar check: read {len(jobs)} images from {len(shards)} shards in {time.time()-t0:.1f}s")
t0 = time.time()
with ThreadPoolExecutor(max_workers=8) as pool:
    res = list(pool.map(lambda s: analyse(s[1]), jobs))
print(f"decoded+measured in {time.time()-t0:.1f}s")

n = len(res)
def frac(f):
    return 100.0 * sum(1 for x in res if f(x)) / n

hb = np.array([x["hbar"] for x in res]); vb = np.array([x["vbar"] for x in res])
hh = np.array([x["hbar"] / x["H"] for x in res]); vv = np.array([x["vbar"] / x["W"] for x in res])
print(f"flat strip >=5px top&bottom: {100*(hb>=5).mean():.2f}%   left&right: {100*(vb>=5).mean():.2f}%")
print(f"flat strip >=2% of dim  top&bottom: {100*(hh>=0.02).mean():.2f}%   left&right: {100*(vv>=0.02).mean():.2f}%")
print(f"SYMMETRIC flat strip >=2% top&bottom: {frac(lambda x: x['hbar']/x['H']>=0.02 and x['hsym']):.2f}%")
print(f"SYMMETRIC flat strip >=2% left&right: {frac(lambda x: x['vbar']/x['W']>=0.02 and x['vsym']):.2f}%")
print(f"SYMMETRIC white strip >=2% top&bottom: {frac(lambda x: x['hbar']/x['H']>=0.02 and x['hsym'] and x['hbar_white']):.2f}%")
print(f"SYMMETRIC white strip >=2% left&right: {frac(lambda x: x['vbar']/x['W']>=0.02 and x['vsym'] and x['vbar_white']):.2f}%")
print(f"SYMMETRIC strip >=5% top&bottom: {frac(lambda x: x['hbar']/x['H']>=0.05 and x['hsym']):.2f}%  "
      f"left&right: {frac(lambda x: x['vbar']/x['W']>=0.05 and x['vsym']):.2f}%")

# cross-tab against the dataset's own tags (canonical space form)
def tags_of(meta):
    return {t.lower().replace("_", " ") for t in meta.get("general_tags", [])}
KEYS = ["letterboxed", "pillarboxed", "border", "white border", "white background",
        "simple background", "comic", "4koma", "multiple views"]
tab = {k: dict(n=0, flat=0, sym=0) for k in KEYS}
has_any_border = 0
for (key, blob, meta), x in zip(jobs, res):
    tg = tags_of(meta)
    flat = (x["hbar"] / x["H"] >= 0.02) or (x["vbar"] / x["W"] >= 0.02)
    sym = ((x["hbar"] / x["H"] >= 0.02 and x["hsym"]) or (x["vbar"] / x["W"] >= 0.02 and x["vsym"]))
    if any(k in tg for k in ("letterboxed", "pillarboxed", "border", "white border", "black border", "outside border")):
        has_any_border += 1
    for k in KEYS:
        if k in tg:
            tab[k]["n"] += 1
            tab[k]["flat"] += int(flat)
            tab[k]["sym"] += int(sym)
print("\ntag -> pixel-bar cross-tab (n in this 4k subset / with flat strip / with SYMMETRIC strip):")
for k in KEYS:
    d = tab[k]
    print(f"  {k:18s} n={d['n']:4d}  flat={d['flat']:4d}"
          f" ({100*d['flat']/d['n']:5.1f}%)  sym={d['sym']:4d} ({100*d['sym']/d['n']:5.1f}%)" if d["n"] else
          f"  {k:18s} n=0")
print(f"  any border-family tag: n={has_any_border}")
base_flat = frac(lambda x: (x["hbar"] / x["H"] >= 0.02) or (x["vbar"] / x["W"] >= 0.02))
base_sym = frac(lambda x: (x["hbar"] / x["H"] >= 0.02 and x["hsym"]) or (x["vbar"] / x["W"] >= 0.02 and x["vsym"]))
print(f"BASE RATE over all {n}: flat >=2% {base_flat:.2f}%   symmetric {base_sym:.2f}%")

out = dict(n=n, base_flat=base_flat, base_sym=base_sym,
           hbar5=float((hb >= 5).mean()), vbar5=float((vb >= 5).mean()),
           sym_h2=frac(lambda x: x['hbar']/x['H'] >= 0.02 and x['hsym']),
           sym_v2=frac(lambda x: x['vbar']/x['W'] >= 0.02 and x['vsym']),
           sym_h2_white=frac(lambda x: x['hbar']/x['H'] >= 0.02 and x['hsym'] and x['hbar_white']),
           sym_v2_white=frac(lambda x: x['vbar']/x['W'] >= 0.02 and x['vsym'] and x['vbar_white']),
           extra_train=E_br, extra_test=E_te, crosstab=tab, any_border=has_any_border)
with open(os.path.join(HERE, "geom_bars.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1)
print("\nsaved geom_bars.json")
