"""Control experiments: raw-pixel probe, and DINOv3 at 224 vs 512."""
import os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from wdtag.data import CacheDataset, load_targets
from wdtag.labels import load_label_space
from wdtag.metrics import compute_metrics
from wdtag.probe import PROBE_DIR

space = load_label_space(); n_labels = len(space)
Ytr_full = load_targets("train", "pad", n_labels); Yva = load_targets("valid", "pad", n_labels)
freq = Ytr_full.mean(0)
cols = np.argsort(-freq)[:1000]

def probe(Xtr, Xva, name, epochs=15, lr=3e-3, wd=0.0):
    Xt = torch.from_numpy(np.asarray(Xtr, dtype=np.float32)).cuda()
    Xv = torch.from_numpy(np.asarray(Xva, dtype=np.float32)).cuda()
    Yt = torch.from_numpy(Ytr_full[:, cols].astype(np.float32)).cuda()
    yv = Yva[:, cols]
    torch.manual_seed(0)
    head = nn.Sequential(nn.LayerNorm(Xtr.shape[1]), nn.Linear(Xtr.shape[1], len(cols))).cuda()
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    for ep in range(epochs):
        perm = torch.randperm(len(Xt), device="cuda")
        for i in range(0, len(perm) - 256 + 1, 256):
            idx = perm[i:i+256]
            loss = F.binary_cross_entropy_with_logits(head(Xt[idx]), Yt[idx])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    head.eval()
    with torch.no_grad():
        p = torch.sigmoid(head(Xv)).cpu().numpy()
    m = compute_metrics(p >= 0.5, yv)
    print(f"{name:24s}: top1000 micro_f1 {m['micro_f1']:.4f} macro_f1 {m['macro_f1']:.4f}")
    return m

# ---- 1. raw pixel control (32x32 and 64x64)
for size in (32, 64):
    def pixels(split):
        ds = CacheDataset(split, "pad", n_labels)
        out = np.empty((len(ds), size * size * 3), dtype=np.float16)
        for i in range(len(ds)):
            im = ds[i][0].float().unsqueeze(0)
            im = F.interpolate(im, size=(size, size), mode="bilinear", antialias=True)
            out[i] = im.numpy().ravel() / 255.0
        return out
    ptr = pixels("train"); pva = pixels("valid")
    probe(ptr, pva, f"raw pixels {size}x{size}")

# ---- 2. DINOv3 features at 224 native resolution
def feats224(split):
    from transformers import DINOv3ViTModel
    path = os.path.join(PROBE_DIR, f"{split}_224.npz")
    if os.path.exists(path):
        return np.load(path)["cls"]
    m = DINOv3ViTModel.from_pretrained(r"I:\Models\dinov3-vitb16-pretrain-lvd1689m",
                                       dtype=torch.bfloat16).cuda().eval()
    ds = CacheDataset(split, "pad", n_labels)
    mean = torch.tensor([0.485, 0.456, 0.406]).cuda().view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).cuda().view(1, 3, 1, 1)
    out = np.empty((len(ds), 768), dtype=np.float16)
    with torch.no_grad():
        for s in range(0, len(ds), 64):
            ims = torch.stack([ds[i][0] for i in range(s, min(s + 64, len(ds)))]).cuda().float() / 255.0
            ims = F.interpolate(ims, size=(224, 224), mode="bilinear", antialias=True)
            x = ((ims - mean) / std).to(torch.bfloat16)
            out[s:s + len(x)] = m(pixel_values=x).pooler_output.float().cpu().numpy().astype(np.float16)
    np.savez(path, cls=out)
    return out

tr224, va224 = feats224("train"), feats224("valid")
probe(tr224.astype(np.float32), va224.astype(np.float32), "dinov3 cls @224")

tr = np.load(os.path.join(PROBE_DIR, "train_pad.npz")); va = np.load(os.path.join(PROBE_DIR, "valid_pad.npz"))
for pool in ("cls", "mean", "cls_mean"):
    probe(tr[pool].astype(np.float32), va[pool].astype(np.float32), f"dinov3 {pool} @512")
