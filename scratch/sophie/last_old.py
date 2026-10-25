import zstandard, json, datetime

P = r"C:\Users\kerok\.dsh\sessions\--C-Users-kerok-apps-wd-tagger-trainer--\session-2645804d-bf0d-4aaf-a26c-dcfe10504be6\session.v4.jsonl.zstd"
d = zstandard.ZstdDecompressor()
with open(P, "rb") as f:
    raw = d.stream_reader(f).read()
lines = raw.decode("utf-8", "replace").splitlines()
print("total lines:", len(lines))
for i in [22487]:
    o = json.loads(lines[i])
    data = o.get("data") or {}
    msg = data.get("message") or {}
    content = msg.get("content") or data.get("content") or []
    txt = "\n".join(b.get("text","") for b in content if isinstance(b, dict) and b.get("type")=="text")
    iso = datetime.datetime.fromtimestamp(o["time"]/1000, datetime.timezone.utc).isoformat()
    print(f"\nL{i} {o.get('type')} {iso} len={len(txt)}")
    print(txt)
