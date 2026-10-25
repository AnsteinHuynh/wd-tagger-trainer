import zstandard, json, os, re

ROOT = r"C:\Users\kerok\.dsh\sessions\--C-Users-kerok-apps-wd-tagger-trainer--"
d = zstandard.ZstdDecompressor()

PAT = re.compile(r"(\d+(?:\.\d+)?)\s?(KB|kB|kb|kilobyte)", re.I)

def blocks(o):
    t = o.get("type")
    data = o.get("data") or {}
    if t == "user/message":
        for b in data.get("content") or []:
            if isinstance(b, dict) and b.get("type") == "text":
                yield data.get("role", "user"), b.get("text", ""), "user"
    elif t == "assistant/message":
        msg = data.get("message") or {}
        for b in msg.get("content") or []:
            if isinstance(b, dict) and b.get("type") == "text":
                yield msg.get("role", "assistant"), b.get("text", ""), "assistant"
    elif t == "compaction/summary":
        yield "summary", json.dumps(data), "summary"

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
            for m in PAT.finditer(txt or ""):
                ctx = txt[max(0, m.start()-200):m.end()+200].replace("\n", " | ")
                if any(k in txt[max(0,m.start()-200):m.end()+200].lower() for k in ("cach","embed","image","pool","tag")):
                    print(f"[{dirname} L{i} {kind} t={o.get('time')}] {m.group(0)} :: {ctx}")
                    print()
