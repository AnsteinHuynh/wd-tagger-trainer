import zstandard, json, os, re

SESS = r"C:\Users\kerok\.dsh\sessions\--C-Users-kerok-apps-wd-tagger-trainer--\session-35636ebe-eae5-4a06-9d89-6974c58541ac\session.v4.jsonl.zstd"
d = zstandard.ZstdDecompressor()
with open(SESS, "rb") as f:
    raw = d.stream_reader(f).read()
lines = raw.decode("utf-8", "replace").splitlines()

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
    elif t == "compaction/summary":
        yield "summary", json.dumps(data)

PAT = re.compile(r"(no cache|four ways|4 ways|levels of cach|ways to cach|way to cach|image tags|tag vector|predicted tags|first.{0,30}free)", re.I)

for i, ln in enumerate(lines):
    try:
        o = json.loads(ln)
    except Exception:
        continue
    for role, txt in texts(o):
        for m in PAT.finditer(txt or ""):
            ctx = txt[max(0, m.start()-400):m.end()+400]
            print(f"\n########## L{i} {role} t={o.get('time')} match={m.group(0)!r}")
            print(ctx)
