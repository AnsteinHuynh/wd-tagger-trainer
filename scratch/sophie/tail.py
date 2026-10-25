import zstandard, json, datetime

P = r"C:\Users\kerok\.dsh\sessions\--C-Users-kerok-apps-wd-tagger-trainer--\session-35636ebe-eae5-4a06-9d89-6974c58541ac\session.v4.jsonl.zstd"
d = zstandard.ZstdDecompressor()
with open(P, "rb") as f:
    raw = d.stream_reader(f).read()
lines = raw.decode("utf-8", "replace").splitlines()
print("total lines:", len(lines))
for i in range(len(lines)-25, len(lines)):
    try:
        o = json.loads(lines[i])
    except Exception:
        print(i, "unparsed"); continue
    t = o.get("type")
    if t in ("assistant/message","user/message"):
        data = o.get("data") or {}
        msg = data.get("message") or {}
        content = msg.get("content") or data.get("content") or []
        txt = " ".join(b.get("text","") for b in content if isinstance(b, dict) and b.get("type")=="text")
        iso = datetime.datetime.fromtimestamp(o["time"]/1000, datetime.timezone.utc).isoformat()
        print(f"\nL{i} {t} {iso} len={len(txt)}")
        print(txt[:400])
    else:
        print(f"L{i} {t}")
