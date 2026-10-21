import io, json, os, struct, tarfile, time
import torch
from PIL import Image

R = r"X:\unique\datasets\animetimm-danbooru-wdtagger-v4-w640-ws-full\repo"

p = os.path.join(R, "train", "00001.tar")
print("size GB: %.2f" % (os.path.getsize(p) / 1e9))

# inspect raw first two headers
with open(p, "rb") as f:
    for i in range(4):
        h = f.read(512)
        name = h[0:100].rstrip(b"\x00").decode("utf8", "replace")
        size = h[124:136].rstrip(b"\x00 ").decode("ascii", "replace")
        typeflag = h[156:157]
        magic = h[257:265]
        print(i, "name=", name, "size=", size, "type=", typeflag, "magic=", magic)

t0 = time.time()
n = 0
with tarfile.open(p, "r:") as tf:
    for m in tf:
        if n < 3:
            print("MEMBER", m.name, m.size, m.type)
            if m.name.endswith(".json"):
                d = json.loads(tf.extractfile(m).read().decode("utf-8"))
                print("  json keys:", list(d.keys()))
                print("  general_tags[:8]:", d["general_tags"][:8])
                print("  character_tags[:5]:", d["character_tags"][:5])
                print("  rating:", d["rating"], "w/h:", d["width"], d["height"], "id:", d["id"])
        n += 1
        if n > 6:
            break
print("open+iterate 6 members: %.2fs" % (time.time() - t0))

# time: read first 50 images sequentially
t0 = time.time()
imgs = []
with tarfile.open(p, "r:") as tf:
    for m in tf:
        if m.name.endswith(".webp"):
            imgs.append(tf.extractfile(m).read())
            if len(imgs) >= 50:
                break
print("read 50 webp bytes: %.2fs, avg %.0f KB" % (time.time() - t0, sum(len(b) for b in imgs) / 50 / 1024))

t0 = time.time()
sizes = []
for b in imgs[:20]:
    im = Image.open(io.BytesIO(b)).convert("RGB")
    sizes.append(im.size)
print("decode 20 webp: %.3fs -> %.1f img/s" % (time.time() - t0, 20 / (time.time() - t0)))
print("sizes sample:", sizes[:6])

# resize cost
t0 = time.time()
for b in imgs[:20]:
    im = Image.open(io.BytesIO(b)).convert("RGB")
    w, h = im.size
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    canvas.paste(im, ((side - w) // 2, (side - h) // 2))
    canvas = canvas.resize((512, 512), Image.BILINEAR)
print("pad+resize 20: %.3fs -> %.1f img/s" % (time.time() - t0, 20 / (time.time() - t0)))
