import zstandard, json, re

SESS = r"C:\Users\kerok\.dsh\sessions\--C-Users-kerok-apps-wd-tagger-trainer--\session-35636ebe-eae5-4a06-9d89-6974c58541ac\session.v4.jsonl.zstd"
d = zstandard.ZstdDecompressor()
with open(SESS, "rb") as f:
    raw = d.stream_reader(f).read()
lines = raw.decode("utf-8", "replace").splitlines()

def blocks(o):
    t = o.get("type"); data = o.get("data") or {}
    if t == "user/message":
        for b in data.get("content") or []:
            if isinstance(b, dict) and b.get("type") == "text":
                yield data.get("role", "user"), b.get("text", ""), (data.get("source") or {}).get("kind")
    elif t == "assistant/message":
        msg = data.get("message") or {}
        for b in msg.get("content") or []:
            if isinstance(b, dict) and b.get("type") == "text":
                yield msg.get("role", "assistant"), b.get("text", ""), None

for i in range(7200, 10600):
    if i >= len(lines):
        break
    try:
        o = json.loads(lines[i])
    except Exception:
        continue
    for role, txt, kind in blocks(o):
        if not txt:
            continue
        low = txt.lower()
        if "cach" in low and role in ("assistant", "user"):
            print(f"\n@@@@ L{i} {role} kind={kind} t={o.get('time')} len={len(txt)}")
            print(txt[:1200])
