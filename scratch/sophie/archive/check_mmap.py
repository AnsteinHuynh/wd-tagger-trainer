import os, time, torch
p = r"I:\.cache\wd-tagger-trainer\train\pad\shard00000-10000.pt"

def probe(**kw):
    t0 = time.time()
    o = torch.load(p, map_location="cpu", **kw)
    img = o["images"]
    st = img.untyped_storage()
    fn = getattr(st, "filename", "n/a")
    print(f"{kw}: {time.time()-t0:.3f}s shape {tuple(img.shape)} "
          f"storage.filename={fn} nbytes={st.nbytes()/1e9:.2f}GB "
          f"ptr={img.data_ptr()}")
    return o

o1 = probe(mmap=True, weights_only=True)
del o1
o2 = probe(mmap=True, weights_only=False)
print("keys ok:", list(o2.keys()))
