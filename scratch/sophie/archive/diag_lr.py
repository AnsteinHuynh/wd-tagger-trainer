import os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from wdtag.data import load_targets
from wdtag.labels import load_label_space
from wdtag.metrics import compute_metrics
from wdtag.probe import PROBE_DIR
from adv_optm import SinkSGD_adv

space = load_label_space(); n_labels = len(space)
tr = np.load(os.path.join(PROBE_DIR, "train_pad.npz")); va = np.load(os.path.join(PROBE_DIR, "valid_pad.npz"))
Ytr = load_targets("train", "pad", n_labels); Yva = load_targets("valid", "pad", n_labels)
freq = Ytr.mean(0); cols = np.argsort(-freq)[:1000]

def run(pool, lr, opt_name, epochs=15, wd=0.0, mom=0.95):
    d = tr[pool].shape[1]
    torch.manual_seed(0)
    head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, len(cols))).cuda()
    Xt = torch.from_numpy(tr[pool].astype(np.float32)).cuda()
    Xv = torch.from_numpy(va[pool].astype(np.float32)).cuda()
    Yt = torch.from_numpy(Ytr[:, cols].astype(np.float32)).cuda()
    if opt_name == "adamw":
        opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    else:
        opt = SinkSGD_adv(head.parameters(), lr=lr, momentum=mom, weight_decay=wd)
    for ep in range(epochs):
        perm = torch.randperm(len(Xt), device="cuda")
        for i in range(0, len(perm) - 256 + 1, 256):
            idx = perm[i:i+256]
            loss = F.binary_cross_entropy_with_logits(head(Xt[idx]), Yt[idx])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    head.eval()
    with torch.no_grad():
        p = torch.sigmoid(head(Xv)).cpu().numpy()
    m = compute_metrics(p >= 0.5, Yva[:, cols])
    print(f"{pool:9s} {opt_name:8s} lr {lr:<7} wd {wd:<4} mom {mom}: micro_f1 {m['micro_f1']:.4f} "
          f"macro_f1 {m['macro_f1']:.4f}", flush=True)
    return m["micro_f1"]

for lr in (0.002, 0.005, 0.01, 0.02, 0.05):
    run("cls_mean", lr, "sinksgd")
run("cls_mean", 0.003, "adamw")
run("cls_mean", 0.01, "adamw")
run("cls_mean", 0.005, "sinksgd", wd=0.01)
run("cls_mean", 0.005, "sinksgd", mom=0.0)
