import os, numpy as np, torch
from wdtag.data import load_targets
from wdtag.labels import load_label_space
from wdtag.probe import PROBE_DIR

space = load_label_space(); n_labels = len(space)
tr = np.load(os.path.join(PROBE_DIR, "train_pad.npz")); va = np.load(os.path.join(PROBE_DIR, "valid_pad.npz"))
Ytr = load_targets("train", "pad", n_labels); Yva = load_targets("valid", "pad", n_labels)

for pool in ("cls", "mean"):
    Xt = torch.from_numpy(tr[pool].astype(np.float32)).cuda()
    Xv = torch.from_numpy(va[pool].astype(np.float32)).cuda()
    Xt = torch.nn.functional.normalize(Xt, dim=1); Xv = torch.nn.functional.normalize(Xv, dim=1)
    print(pool, "cos sim between distinct train images: mean %.3f" % (
        (Xt[:1000] @ Xt[1000:2000].T).mean().item()))
    sim = Xv[:300] @ Xt.T          # [300, 10000]
    topk = sim.topk(10, dim=1).indices.cpu().numpy()
    for name in ("1girl", "solo", "long_hair", "smile"):
        j = space.name2idx[name]
        for thr in (0.3, 0.5):
            pred = Ytr[topk, j].mean(1) >= thr
            tgt = Yva[:300, j]
            tp = (pred & tgt).sum(); fp = (pred & ~tgt).sum(); fn = (~pred & tgt).sum()
            f1 = 2*tp/max(2*tp+fp+fn, 1)
            if thr == 0.5:
                print(f"   kNN {name:10s} f1 {f1:.4f} (prev {tgt.mean():.3f})")
