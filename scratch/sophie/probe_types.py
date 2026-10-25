import zstandard, json, collections

p = r"C:\Users\kerok\.dsh\sessions\--C-Users-kerok-apps-wd-tagger-trainer--\session-2645804d-bf0d-4aaf-a26c-dcfe10504be6\session.v4.jsonl.zstd"
d = zstandard.ZstdDecompressor()
with open(p, "rb") as f:
    raw = d.stream_reader(f).read()
lines = raw.decode("utf-8", "replace").splitlines()
c = collections.Counter()
shown = 0
for ln in lines:
    try:
        o = json.loads(ln)
    except Exception:
        c["UNPARSED"] += 1
        continue
    t = o.get("type")
    c[t] += 1
    if t and ("message" in t or "chat" in t) and shown < 6:
        print("###", t)
        print(json.dumps(o, indent=1)[:1800])
        shown += 1
print()
for k, v in c.most_common(40):
    print(v, k)
