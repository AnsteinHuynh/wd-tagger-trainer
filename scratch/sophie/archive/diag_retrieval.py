import os, numpy as np, torch
from PIL import Image
from wdtag.probe import PROBE_DIR
from wdtag.data import CacheDataset
from wdtag.labels import load_label_space
from wdtag.metrics import compute_metrics

space = load_label_space()
tr = np.load(os.path.join(PROBE_DIR, "train_pad.npz")); va = np.load(os.path.join(PROBE_DIR, "valid_pad.npz"))
Xt = torch.nn.functional.normalize(torch.from_numpy(tr["cls"].astype(np.float32)).cuda(), dim=1)
Xv = torch.nn.functional.normalize(torch.from_numpy(va["cls"].astype(np.float32)).cuda(), dim=1)

ds_tr = CacheDataset("train", "pad", len(space))
ds_va = CacheDataset("valid", "pad", len(space))

queries = [0, 7, 21, 44]
rows = []
for q in queries:
    sim = Xv[q] @ Xt.T
    nn = sim.topk(3).indices.tolist()
    imgs = [ds_va[q][0].numpy().transpose(1, 2, 0)] + [ds_tr[j][0].numpy().transpose(1, 2, 0) for j in nn]
    tags_q = set(ds_va[q][1].nonzero().flatten().tolist())
    agree = []
    for j in nn:
        tags_n = set(ds_tr[j][1].nonzero().flatten().tolist())
        agree.append(len(tags_q & tags_n) / max(1, len(tags_n)))
    print(f"query {q}: sim {[round(float(s),3) for s in sim.topk(3).values]} tag-overlap {[round(a,2) for a in agree]}")
    rows.append(np.concatenate([np.asarray(Image.fromarray(im).resize((256, 256), Image.BILINEAR)) for im in imgs], 1))

grid = np.concatenate(rows, 0)
Image.fromarray(grid).save("scratch/sophie/dump/retrieval.png")
print("saved scratch/sophie/dump/retrieval.png (query | nn1 | nn2 | nn3)")
