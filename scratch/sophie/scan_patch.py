import zstandard, json, os, re

ROOT = r"C:\Users\kerok\.dsh\sessions\--C-Users-kerok-apps-wd-tagger-trainer--"
d = zstandard.ZstdDecompressor()

def texts(o):
    t = o.get("type"); data = o.get("data") or {}
    if t == "user/message":
        for b in data.get("content") or []:
            if isinstance(b, dict) and b.get("type") == "text":
                yield data.get("role", "user"), b.get("text", "")
    elif t == "assistant/message":
        msg = data.get("message") or {}
        for b in msg.get("content") or []:
            if isinstance(b, dict) and b.get("type") == "text":
                yield msg.get("role", "assistant"), b.get("text", "")

rows = []
for dirname in sorted(os.listdir(ROOT)):
    p = os.path.join(ROOT, dirname, "session.v4.jsonl.zstd")
    if not os.path.exists(p):
        continue
    with open(p, "rb") as f:
        raw = d.stream_reader(f).read()
    for i, ln in enumerate(raw.decode("utf-8", "replace").splitlines()):
        try:
            o = json.loads(ln)
        except Exception:
            continue
        for role, txt in texts(o):
            if not txt:
                continue
            low = txt.lower()
            # signature: patch-token cache discussion
            if "patch token" in low and "cach" in low:
                rows.append((o.get("time") or 0, dirname, i, role, txt))

rows.sort(key=lambda r: (r[0], r[1], r[2]))
print("count:", len(rows))
for t, dirname, i, role, txt in rows:
    print("\n" + "=" * 100)
    print(f"session={dirname} line={i} role={role} time={t} chars={len(txt)}")
    print("-" * 100)
    print(txt[:3000])
