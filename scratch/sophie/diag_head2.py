import os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from wdtag.data import load_targets
from wdtag.metrics import compute_metrics
from wdtag.probe import PROBE_DIR

n_labels = 12476
d = np.load(os.path.join(PROBE_DIR, "train_pad.npz")); v = np.load(os.path.join(PROBE_DIR, "valid_pad.npz"))
Xtr, Xva = d["cls"].astype(np.float32), v["cls"].astype(np.float32)
Ytr = load_targets("train", "pad", n_labels).astype(np.float32)
Yva = load_targets("valid", "pad", n_labels).astype(np.float32)
freq = Ytr.mean(0)

def train(lr=3e-3, wd=0.0, epochs=30, cols=None, bs=256, dropout=0.0):
    torch.manual_seed(0)
    Xt = torch.from_numpy(Xtr).cuda(); Yt = torch.from_numpy(Ytr).cuda()
    if cols is not None:
        Yt = Yt[:, torch.from_numpy(cols).cuda()]
    layers = [nn.LayerNorm(Xtr.shape[1])]
    if dropout: layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(Xtr.shape[1], Yt.shape[1]))
    head = nn.Sequential(*layers).cuda()
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    for ep in range(epochs):
        head.train()
        perm = torch.randperm(len(Xt), device="cuda")
        for i in range(0, len(perm) - bs + 1, bs):
            idx = perm[i:i+bs]
            loss = F.binary_cross_entropy_with_logits(head(Xt[idx]), Yt[idx])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    head.eval()
    with torch.no_grad():
        pv = torch.sigmoid(head(torch.from_numpy(Xva).cuda()).float()).cpu().numpy()
    yv = Yva if cols is None else Yva[:, cols]
    out = {}
    for t in (0.5, 0.3, 0.2, 0.1, 0.05):
        out[t] = compute_metrics(pv >= t, yv)
    best = max(out.values(), key=lambda m: m["micro_f1"])
    print(f"  lr {lr} wd {wd} ep {epochs} drop {dropout} ncols {Yt.shape[1]}: "
          f"micro_f1@0.5 {out[0.5]['micro_f1']:.4f} | best-thr micro_f1 {best['micro_f1']:.4f} "
          f"macro_f1@0.5 {out[0.5]['macro_f1']:.4f} | preds/img {[round(float((pv>=t).sum(1).mean()),1) for t in (0.5,0.2,0.1)]}")
    return out

print("== all 12476 labels, regularisation sweep ==")
train(wd=0.0, epochs=10)
train(wd=0.1, epochs=10)
train(wd=0.1, epochs=30, dropout=0.2)
train(lr=1e-3, wd=0.5, epochs=30, dropout=0.2)

print("== top-1000 most frequent labels only ==")
cols = np.argsort(-freq)[:1000]
train(wd=0.0, epochs=10, cols=cols)
train(wd=0.1, epochs=30, cols=cols, dropout=0.2)

print("== top-100 labels only ==")
cols = np.argsort(-freq)[:100]
train(wd=0.0, epochs=10, cols=cols)
