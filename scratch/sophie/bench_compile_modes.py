"""Is a better torch.compile mode worth anything?

Measures ms/image for eager vs compile(default) vs compile(max-autotune) on the
real training step. Everything else (model, optimizer, data) is identical, so the
ratios hold even if another job is sharing the GPU.
"""
import time

import torch

from wdtag.config import TrainConfig
from wdtag.data import CacheDataset, MultiHotCollator
from wdtag.labels import load_label_space
from wdtag.model import DinoTagger

B, STEPS = 8, 6
cfg = TrainConfig.from_json("configs/squash_lr5e4.json")
n_labels = len(load_label_space())
ds = CacheDataset("train", cfg.mode, n_labels)
coll = MultiHotCollator(n_labels)
xb, yb = coll([ds[i] for i in range(B)])
xg, yg = xb.cuda(), yb.cuda()


def bench(tag, fwd, model, steps=STEPS):
    for _ in range(2):
        out = fwd(xg)
        torch.nn.functional.binary_cross_entropy_with_logits(out.float(), yg).backward()
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(steps):
        out = fwd(xg)
        torch.nn.functional.binary_cross_entropy_with_logits(out.float(), yg).backward()
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    ms = (time.time() - t0) / steps * 1000
    print(f"{tag:22s} {ms:8.1f} ms/step  {ms/B:6.2f} ms/img  "
          f"{B*1000/ms:6.0f} img/s", flush=True)
    return ms / B


results = {}
for mode in [None, "default", "max-autotune"]:
    model = DinoTagger(cfg, n_labels).cuda()
    fwd = model if mode is None else torch.compile(model, mode=mode)
    try:
        results[str(mode)] = bench(f"eager" if mode is None else f"compile({mode})", fwd, model)
    except Exception as e:
        print(f"{mode}: failed ({type(e).__name__}: {str(e)[:90]})", flush=True)
    del model, fwd
    torch.cuda.empty_cache()

base = results.get("None")
if base:
    for k, v in results.items():
        if k != "None":
            print(f"{k:22s} speedup vs eager: {base/v:.2f}x")
