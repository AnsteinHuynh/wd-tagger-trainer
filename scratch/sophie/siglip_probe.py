"""Does a second frozen tower carry tag information DINOv3 does not? Two stages, one number each.

    python scratch/sophie/siglip_probe.py extract siglip --split train
    python scratch/sophie/siglip_probe.py extract siglip --split val --n 50000
    python scratch/sophie/siglip_probe.py extract dino   --split train
    python scratch/sophie/siglip_probe.py extract dino   --split val --n 50000
    python scratch/sophie/siglip_probe.py fit --features siglip_train --val-features siglip_val
    python scratch/sophie/siglip_probe.py fit --features dino_train,siglip_train --val-features dino_val,siglip_val

Features are pooled, frozen, and cached as fp16 .npy under the probe dir. `fit` trains the same ASL
head on whichever feature set it is given and scores the 50k val at the fixed 0.40 threshold, so the
only thing that varies between runs is the features.
"""
import argparse
import os

import numpy as np
import torch

from wdtag.data import CacheDataset, label_priors, load_targets

CACHE = r"I:\.cache\wd-tagger-trainer"
PROBE = os.path.join(CACHE, "probe")
N_LABELS = 12476
SIGLIP = r"I:\Models\siglip2-base-patch16-512"
SIGLIP_L = r"I:\Models\siglip2-large-patch16-512"
VITL = r"I:\Models\dinov3-vitl16-pretrain-lvd1689m"
THR = 0.40


def split_dataset(split: str, n: int):
    if split == "train":
        return CacheDataset("train", "squash", N_LABELS)
    from wdtag.train import NestedValSet
    return NestedValSet(os.path.join(CACHE, "valid_nested"), n, N_LABELS)


def build_tower(name: str, res: int = 512, pool: str = "merged"):
    """Frozen extractor, its own input preparation, and its pooling."""
    if name in ("siglip", "siglip_large"):
        from transformers import SiglipVisionModel
        src = SIGLIP_L if name == "siglip_large" else SIGLIP
        m = SiglipVisionModel.from_pretrained(src).eval()

        def run(x):                      # SigLIP: mean=std=0.5, range -1..1
            return m(x.float().div(127.5).sub(1.0)).pooler_output
    elif name == "qwen":
        import sys
        sys.path.insert(0, r"C:\Users\kerok\apps\dreamsim-dinov3\src")
        from dreamsim_oft.qwen_vision import load_vision_tower, qwen_vision_embed
        m, info = load_vision_tower()
        print(f"qwen tower: {info}", flush=True)

        def run(x):                      # Qwen: (x/255 - .5)/.5, same [-1,1] range
            x01 = x.float().div(127.5).sub(1.0)
            if res != x01.shape[-1]:
                x01 = torch.nn.functional.interpolate(x01, size=(res, res), mode="bicubic",
                                                      align_corners=False, antialias=True)
            return qwen_vision_embed(m, x01, pool)
    else:
        from wdtag.config import TrainConfig
        from wdtag.model import DinoTagger
        # "dino" is ViT-B/16, "vitl" the same family at 24 blocks and 1024 wide: the capacity
        # question, asked of the feature bank rather than of a 45-hour training run
        backbone = VITL if name == "vitl" else ""
        cfg = TrainConfig(unfreeze_blocks=0, freeze_oft=True, head_pool="cls",
                          backbone_dir=backbone)
        m = DinoTagger(cfg, N_LABELS).eval()

        def run(x):
            return m.forward_features(m._prep(x))   # _prep takes raw uint8
    for p in m.parameters():
        p.requires_grad_(False)
    return m.cuda(), run


def extract(a) -> None:
    os.makedirs(PROBE, exist_ok=True)
    ds = split_dataset(a.split, a.n)
    model, run = build_tower(a.tower, a.res, a.pool)
    tag = a.tower + a.tag
    path = os.path.join(PROBE, f"{tag}_{a.split}.npy")
    out = None
    for start in range(0, len(ds), a.batch):
        idx = range(start, min(start + a.batch, len(ds)))
        x = torch.stack([ds[i][0] for i in idx]).cuda()
        with torch.no_grad():
            feats = run(x).float()
        if out is None:
            out = np.zeros((len(ds), feats.shape[1]), dtype=np.float16)
            print(f"{tag} {a.split}: {len(ds)} rows, {feats.shape[1]} dims", flush=True)
        out[start:start + len(x)] = feats.half().cpu().numpy()
        if start and start % 10000 == 0:
            print(f"  {start}/{len(ds)}", flush=True)
    np.save(path, out)
    print(f"wrote {path} {out.shape}")


def fit(a) -> None:
    from wdtag.losses import asl
    from wdtag.metrics import compute_metrics
    from wdtag.train import NestedValSet

    x = np.concatenate([np.load(os.path.join(PROBE, f"{f}.npy")) for f in a.features.split(",")],
                       axis=1).astype(np.float32)
    xv = np.concatenate([np.load(os.path.join(PROBE, f"{f}.npy"))
                         for f in a.val_features.split(",")], axis=1).astype(np.float32)
    y = np.asarray(load_targets("train", "squash", N_LABELS), dtype=np.float32)
    yv = NestedValSet(os.path.join(CACHE, "valid_nested"), xv.shape[0], N_LABELS).dense_targets()
    yv = np.asarray(yv, dtype=np.float32).reshape(xv.shape[0], N_LABELS)   # torch or bool -> float

    torch.manual_seed(1337)
    xt, yt = torch.from_numpy(x).cuda(), torch.from_numpy(y).cuda()
    priors = torch.from_numpy(label_priors("train", "squash", N_LABELS)).cuda().clamp(1e-6, 1 - 1e-6)
    if a.hidden:
        # a bottleneck between the pooled features and the logits is the only adaptation a cache allows
        head = torch.nn.Sequential(torch.nn.Linear(xt.shape[1], a.hidden), torch.nn.GELU(),
                                   torch.nn.Linear(a.hidden, N_LABELS)).cuda()
    else:
        head = torch.nn.Linear(xt.shape[1], N_LABELS).cuda()
    last = head[-1] if a.hidden else head
    with torch.no_grad():
        last.bias.copy_(torch.log(priors / (1 - priors)))
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps)
    step, order = 0, torch.randperm(xt.shape[0])
    while step < a.steps:
        for i in range(0, xt.shape[0] - a.batch + 1, a.batch):
            idx = order[i:i + a.batch]
            loss = asl(head(xt[idx]), yt[idx], gamma_neg=4.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            if step % 1000 == 0:
                print(f"  step {step} loss {loss.item():.4f}", flush=True)
            if step >= a.steps:
                break
        order = torch.randperm(xt.shape[0])

    head.eval()
    probs = np.zeros((xv.shape[0], N_LABELS), dtype=np.float32)
    with torch.no_grad():
        xvt = torch.from_numpy(xv).cuda()
        for i in range(0, xvt.shape[0], 1024):
            probs[i:i + 1024] = torch.sigmoid(head(xvt[i:i + 1024])).float().cpu().numpy()
    m = compute_metrics(probs >= THR, yv >= 0.5)
    print(f"{a.features} | macro F1 {m['macro_f1']:.4f} MCC {m['macro_mcc']:.4f} "
          f"P {m['macro_precision']:.4f} R {m['macro_recall']:.4f} | "
          f"micro F1 {m['micro_f1']:.4f} P {m['micro_precision']:.4f} R {m['micro_recall']:.4f}",
          flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("tower", choices=("siglip", "siglip_large", "dino", "vitl", "qwen"))
    e.add_argument("--split", choices=("train", "val"), required=True)
    e.add_argument("--n", type=int, default=0)
    e.add_argument("--batch", type=int, default=64)
    e.add_argument("--res", type=int, default=512)
    e.add_argument("--pool", default="merged")
    e.add_argument("--tag", default="", help="extra suffix for the .npy name, e.g. 512merged")
    f = sub.add_parser("fit")
    f.add_argument("--features", required=True)
    f.add_argument("--val-features", required=True)
    f.add_argument("--steps", type=int, default=4000)
    f.add_argument("--batch", type=int, default=48)
    f.add_argument("--hidden", type=int, default=0, help="0 = linear head, else one hidden layer")
    a = ap.parse_args()
    if a.cmd == "extract":
        extract(a)
    else:
        fit(a)


if __name__ == "__main__":
    main()



