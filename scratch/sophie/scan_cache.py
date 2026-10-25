import zstandard, json, os, re, collections

ROOT = r"C:\Users\kerok\.dsh\sessions\--C-Users-kerok-apps-wd-tagger-trainer--"
d = zstandard.ZstdDecompressor()

SIZE = re.compile(r"\b\d+(?:\.\d+)?\s?(?:KB|MB|GB|kB|kb)\b", re.I)
CACHE = re.compile(r"cach", re.I)
EXTRA = re.compile(r"(embed|pool|cls|mean|max|patch token|per image|level|rung|tier|cheap|free|no cache)", re.I)

def texts(o):
    """yield (role, text, meta) from an event object"""
    t = o.get("type")
    data = o.get("data") or {}
    if t in ("user/message",):
        role = data.get("role", "user")
        for b in data.get("content") or []:
            if isinstance(b, dict) and b.get("type") == "text":
                yield role, b.get("text", ""), o
        return
    if t == "assistant/message":
        msg = data.get("message") or {}
        role = msg.get("role", "assistant")
        for b in msg.get("content") or []:
            if isinstance(b, dict) and b.get("type") == "text":
                yield role, b.get("text", ""), o
        return
    if t == "compaction/summary":
        # may carry text
        s = json.dumps(data)
        yield "compaction", s, o

hits = []
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
        for role, txt, ev in texts(o):
            if not txt:
                continue
            lc = CACHE.findall(txt)
            if not lc:
                continue
            sizes = SIZE.findall(txt)
            ex = set(m.group(0).lower() for m in EXTRA.finditer(txt))
            score = len(sizes) * 5 + len(ex)
            if len(sizes) >= 1 and len(ex) >= 2:
                hits.append((dirname, i, role, ev.get("time"), score, len(txt), txt))

hits.sort(key=lambda h: (h[0], h[1]))
print("TOTAL HITS:", len(hits))
for h in hits:
    print("\n" + "=" * 100)
    print(f"session={h[0]} line={h[1]} role={h[2]} time={h[3]} score={h[4]} chars={h[5]}")
    print("-" * 100)
    print(h[6][:4000])
