import zstandard, json, sys

p = r"C:\Users\kerok\.dsh\sessions\--C-Users-kerok-apps-wd-tagger-trainer--\session-2645804d-bf0d-4aaf-a26c-dcfe10504be6\session.v4.jsonl.zstd"
d = zstandard.ZstdDecompressor()
with open(p, "rb") as f:
    raw = d.stream_reader(f).read()
lines = raw.decode("utf-8", "replace").splitlines()
print("lines:", len(lines))
for i, ln in enumerate(lines[:4]):
    try:
        o = json.loads(ln)
    except Exception as e:
        print(i, "PARSE FAIL", e, ln[:300]); continue
    print("=== line", i, "top keys:", list(o.keys()))
    print(json.dumps(o, indent=1)[:2500])
