"""Does a bigger batch actually pay? Measures img/s and peak VRAM at several
batch sizes, with the same model/optimizer/compile path training uses.

The point: ms/step grows sublinearly with batch (the ~28 ms compiled-optimizer
dispatch is flat), so img/s should improve -- but steps/epoch shrink, and with a
normalised (muon-like) update it is steps, not images, that make progress.
"""
import argparse
import time

import torch

from wdtag.config import TrainConfig
from wdtag.data import CacheDataset, MultiHotCollator, label_priors, load_targets
from wdtag.labels import load_label_space
from wdtag.model import DinoTagger

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/squash_lr5e4.json")
ap.add_argument("--batches", default="32,48,64,96")
ap.add_argument("--steps", type=int, default=8)
a = ap.parse_args()

cfg = TrainConfig.from_json(a.config)
device = torch.device("cuda")
n_labels = len(load_label_space())

model = DinoTagger(cfg, n_labels).to(device)
model.init_head_bias(label_priors("train", cfg.mode, n_labels))
forward = torch.compile(model, mode="default")
oft_params, head_params = model.trainable_parameters()

torch._dynamo.config.cache_size_limit = 1_000_000
torch._dynamo.config.recompile_limit = 1_000_000
torch._dynamo.config.accumulated_recompile_limit = 1_000_000
torch._dynamo.config.force_parameter_static_shapes = False
from adv_optm import SinkSGD_adv
opt = SinkSGD_adv([{"params": oft_params, "lr": cfg.lr_oft},
                   {"params": head_params, "lr": cfg.lr_head}],
                  momentum=cfg.momentum, weight_decay=cfg.weight_decay,
                  sinkhorn_iterations=cfg.sinkhorn_iterations,
                  orthogonal_sinkhorn=cfg.orthogonal_sinkhorn,
                  stochastic_rounding=True, compiled_optimizer=cfg.compiled_optimizer)

ds = CacheDataset("train", cfg.mode, n_labels)
coll = MultiHotCollator(n_labels)
y = load_targets("train", cfg.mode, n_labels)

print(f"{'batch':>6s} {'ms/step':>9s} {'img/s':>7s} {'peak VRAM':>10s} "
      f"{'steps/ep(10k)':>14s} {'steps/ep(100k)':>15s}")
for b in [int(x) for x in a.batches.split(",")]:
    try:
        idx = list(range(b))
        xb, yb = coll([ds[i] for i in idx])
        x = xb.to(device)
        yb = yb.to(device)
        torch.cuda.reset_peak_memory_stats()
        for _ in range(2):  # warm up compile + cudnn
            out = forward(x)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(out.float(), yb)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(a.steps):
            out = forward(x)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(out.float(), yb)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        dt = (time.time() - t0) / a.steps
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"{b:6d} {dt*1000:9.1f} {b/dt:7.0f} {peak:9.1f}G "
              f"{10000//b:14d} {100000//b:15d}", flush=True)
        del x, yb, out, loss
    except torch.cuda.OutOfMemoryError:
        print(f"{b:6d}   OOM", flush=True)
        torch.cuda.empty_cache()
print(f"total VRAM {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")


