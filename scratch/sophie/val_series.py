import json

rows = [json.loads(l) for l in open(r"runs\459e2a02\train_log.jsonl", encoding="utf-8") if l.strip()]
print("keys:", sorted(rows[-1].keys()))
vals = [r for r in rows if r.get("macro_f1") is not None and r.get("micro_f1") is not None]
print("n val rows:", len(vals))
prev = None
for r in vals[-16:]:
    s = r.get("step")
    m = r["macro_f1"]
    u = r["micro_f1"]
    p = r.get("macro_precision")
    rc = r.get("macro_recall")
    d = "" if prev is None else f" dmacro {m - prev[0]:+.4f} dmicro {u - prev[1]:+.4f}"
    extra = "" if p is None else f" P {p:.4f} R {rc:.4f}"
    print(f"step {s:5d} macro {m:.4f} micro {u:.4f}{extra}{d}")
    prev = (m, u)
