import zstandard, json, datetime

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

WANT = [7281, 7282, 7289, 7290, 7299, 9551, 9552, 10583, 10592]
out = []
for i in WANT:
    o = json.loads(lines[i])
    ts = o.get("time")
    iso = datetime.datetime.fromtimestamp(ts / 1000, datetime.timezone.utc).isoformat()
    for role, txt, kind in blocks(o):
        out.append("=" * 90)
        out.append(f"LINE {i} role={role} kind={kind} time_ms={ts} utc={iso} chars={len(txt)}")
        out.append("-" * 90)
        out.append(txt)
        out.append("")

with open(r"C:\Users\kerok\apps\wd-tagger-trainer\scratch\sophie\verbatim.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out))
print("wrote", sum(len(x) for x in out), "chars")
