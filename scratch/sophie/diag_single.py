import os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from wdtag.data import load_targets
from wdtag.labels import load_label_space
from wdtag.probe import PROBE_DIR

space = load_label_space(); n_labels = len(space)
tr = np.load(os.path.join(PROBE_DIR, "train_pad.npz")); va = np.load(os.path.join(PROBE_DIR, "valid_pad.npz"))
Ytr = load_targets("train", "pad", n_labels); Yva = load_targets("valid", "pad", n_labels)

def probe_single(pool, name):
    j = space.name2idx[name]
    ytr = Ytr[:, j].astype(np.float32); yva = Yva[:, j].astype(np.float32)
    Xt = torch.from_numpy(tr[pool].astype(np.float32)).cuda(); yt = torch.from_numpy(ytr).cuda()
    Xv = torch.from_numpy(va[pool].astype(np.float32)).cuda()
    torch.manual_seed(0)
    head = nn.Sequential(nn.LayerNorm(Xt.shape[1]), nn.Linear(Xt.shape[1], 1)).cuda()
    opt = torch.optim.AdamW(head.parameters(), lr=3e-3)
    for ep in range(10):
        perm = torch.randperm(len(Xt), device="cuda")
        for i in range(0, len(perm) - 256 + 1, 256):
            idx = perm[i:i+256]
            loss = F.binary_cross_entropy_with_logits(head(Xt[idx]).squeeze(-1), yt[idx])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    with torch.no_grad():
        p = torch.sigmoid(head(Xv).squeeze(-1)).cpu().numpy()
    pred = p >= 0.5
    tgt = yva.astype(bool)
    tp = (pred & tgt).sum(); fp = (pred & ~tgt).sum(); fn = (~pred & tgt).sum()
    f1 = 2*tp/max(2*tp+fp+fn, 1)
    acc = (pred == tgt).mean()
    print(f"{name:16s} [{pool:8s}] prev {yva.mean():.3f} acc {acc:.4f} f1 {f1:.4f} auc-ish")
    return acc

for name in ("1girl", "solo", "long_hair", "smile", "blush"):
    for pool in ("cls", "mean"):
        probe_single(pool, name)
