import numpy as np, torch, os
from PIL import Image
from wdtag.data import CacheDataset
from wdtag.labels import load_label_space

space = load_label_space()
ds = CacheDataset("valid", "pad", len(space))
os.makedirs("scratch/sophie/dump", exist_ok=True)
tiles = []
for i in [0, 1, 2, 3, 4, 5]:
    img, lab = ds[i]
    a = img.numpy().transpose(1, 2, 0)
    print(i, ds.meta(i), "n_pos", int(lab.sum()))
    Image.fromarray(a).save(f"scratch/sophie/dump/v{i}.png")
    tiles.append(a)
grid = np.concatenate([np.concatenate(tiles[:3], 1), np.concatenate(tiles[3:], 1)], 0)
Image.fromarray(grid).resize((1536, 1024), Image.BILINEAR).save("scratch/sophie/dump/grid.png")

# what are the most frequent tags among these?
idx = torch.topk(ds[0][1], 12).indices.tolist()
print("image 0 top tags (by index order, not strength):", [space.names[i] for i in idx])
print("image 0 all positives:", [space.names[i] for i in ds[0][1].nonzero().flatten().tolist()][:30])
