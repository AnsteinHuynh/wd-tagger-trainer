import zstandard, json, os, re

ROOT = r"C:\Users\kerok\.dsh\sessions\--C-Users-kerok-apps-wd-tagger-trainer--"
d = zstandard.ZstdDecompressor()

SIZE = re.compile(r"\b\d+(?:[.,]\d+)?\s?(?:KB|MB|GB|kB|kb)\b", re.I)

def blocks(o):
    t = o.get("type")
    data = o.get("data") or {}
    if t == "user/message":
        for b in data.get("content") or []:
            if isinstance(b, dict) and b.get("type") == "text":
                yield data.get("role", "user"), b.get("text", ""), (data.get("source") or {}).get("kind")
    elif t == "assistant/message":
        msg = data.get("message") or {}
        for b in msg.get("content") or []:
            if isinstance(b, dict) and b.get("type") == "text":
                yield msg.get("role", "assistant"), b.get("text", ""), None

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
        for role, txt, kind in blocks(o):
            if role != "assistant" or not txt:
                continue
            low = txt.lower()
            if "cach" not in low:
                continue
            sizes = SIZE.findall(txt)
            if not sizes:
                continue
            rows.append((dirname, i, o.get("time"), len(txt), sizes[:12], txt))

print("assistant cache+size messages:", len(rows))
for r in rows:
    print(f"\n### {r[0]} line={r[1]} t={r[2]} chars={r[3]} sizes={r[4]}")
    # print sentence-ish contexts around each size
    for m in SIZE.finditer(r[5]):
        s = max(0, m.start() - 260); e = min(len(r[5]), m.end() + 260)
        print("   ...", r[5][s:e].replace("\n", " | "), "...")
