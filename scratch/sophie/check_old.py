import zstandard, json, datetime

P = r"C:\Users\kerok\.dsh\sessions\--C-Users-kerok-apps-wd-tagger-trainer--\session-2645804d-bf0d-4aaf-a26c-dcfe10504be6\session.v4.jsonl.zstd"
d = zstandard.ZstdDecompressor()
with open(P, "rb") as f:
    raw = d.stream_reader(f).read()
lines = raw.decode("utf-8", "replace").splitlines()

def blocks(o):
    t = o.get("type"); data = o.get("data") or {}
    if t == "assistant/message":
        msg = data.get("message") or {}
        for b in msg.get("content") or []:
            if isinstance(b, dict) and b.get("type") == "text":
                yield msg.get("role", "assistant"), b.get("text", "")

for i in [11815, 14600, 15329, 15592, 15646, 15938]:
    o = json.loads(lines[i])
    iso = datetime.datetime.fromtimestamp(o["time"]/1000, datetime.timezone.utc).isoformat()
    for role, txt in blocks(o):
        print("\n" + "=" * 100)
        print(f"L{i} {role} {iso} chars={len(txt)}")
        print("-" * 100)
        print(txt)
