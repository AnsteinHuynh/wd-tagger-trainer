import time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from wdtag.data import load_targets
from wdtag.metrics import compute_metrics
from wdtag.probe import PROBE_DIR
import os

d = np.load(os.path.join(PROBE_DIR, "train_pad.npz"))
v = np.load(os.path.join(PROBE_DIR, "valid_pad.npz"))
print("feature files:", {k: d[k].shape for k in d.files})
Xtr = d["cls"].astype(np.float32); Xva = v["cls"].astype(np.float32)
print("feature stats: mean %.3f std %.3f min %.2f max %.2f" % (Xtr.mean(), Xtr.std(), Xtr.min(), Xtr.max()))

n_labels = 12476
Ytr = load_targets("train", "pad", n_labels).astype(np.float32)
Yva = load_targets("valid", "pad", n_labels)
print("train positives/image %.1f, labels with support %d" % (Ytr.sum(1).mean(), (Ytr.sum(0) > 0).sum()))

# most common label baseline: always predict the top-k tags?
prev = Ytr.mean(0)
print("top-10 prevalences:", np.round(np.sort(prev)[::-1][:10], 3))

def run(opt_name, lr, epochs=30, bs=256):
    torch.manual_seed(0)
    head = nn.Sequential(nn.LayerNorm(Xtr.shape[1]), nn.Linear(Xtr.shape[1], n_labels)).cuda()
    Xt = torch.from_numpy(Xtr).cuda(); Yt = torch.from_numpy(Ytr).cuda()
    if opt_name == "adamw":
        opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.0)
    else:
        from adv_optm import SinkSGD_adv
        opt = SinkSGD_adv(head.parameters(), lr=lr, momentum=0.95)
    losses = []
    for ep in range(epochs):
        perm = torch.randperm(len(Xt), device="cuda")
        for i in range(0, len(perm) - bs + 1, bs):
            idx = perm[i:i+bs]
            loss = F.binary_cross_entropy_with_logits(head(Xt[idx]), Yt[idx])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            losses.append(loss.item())
    head.eval()
    with torch.no_grad():
        pv = torch.sigmoid(head(torch.from_numpy(Xva).cuda()).float()).cpu().numpy()
    m = compute_metrics(pv >= 0.5, Yva)
    pos_per_img = (pv >= 0.5).sum(1).mean()
    print(f"{opt_name} lr {lr}: loss {losses[0]:.4f}->{np.mean(losses[-20:]):.4f} "
          f"val micro_f1 {m['micro_f1']:.4f} macro_f1 {m['macro_f1']:.4f} "
          f"pred_pos/img {pos_per_img:.1f}")

t0 = time.time()
run("adamw", 1e-3)
run("adamw", 3e-3)
run("adamw", 1e-2)
print("total %.0fs" % (time.time() - t0))
