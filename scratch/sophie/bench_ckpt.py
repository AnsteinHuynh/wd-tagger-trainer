"""One number: what gradient checkpointing costs in ms/step and buys in GB at a given batch.

Synthetic input, no data reading, so it can run beside a live arm without touching the tar
stream. Eager, not compiled: the ratio between checkpoint counts is what this measures, and
compile changes both sides of it by roughly the same factor.

    python scratch/sophie/bench_ckpt.py --batch 48 --blocks 12
    --blocks -1 = all blocks (what arm 35 runs), 0 = none, N = last N.
"""

import argparse
import os
import sys

os.environ.setdefault("WDTAG_SIZE", "384")

import torch

sys.path.insert(0, r"C:\Users\kerok\apps\wd-tagger-trainer")

from wdtag.config import TrainConfig
from wdtag.losses import asl
from wdtag.model import DinoTagger

CONFIG = r"configs\heads\phase3\35_u0-C-L-asl-g2.0-siglip2large384-600k-b48-ckpt.json"
N_LABELS = 12476


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--blocks", type=int, default=-1)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--compile", action="store_true")
    a = ap.parse_args()

    cfg = TrainConfig.from_json(a.config)
    cfg.batch_size = a.batch
    cfg.stream = False
    cfg.grad_checkpoint_blocks = True if a.blocks < 0 else a.blocks

    free, total = torch.cuda.mem_get_info()
    print(f"[bench] batch {a.batch}, blocks {cfg.grad_checkpoint_blocks}, "
          f"{free/2**30:.1f} of {total/2**30:.1f} GB free before load", flush=True)

    model = DinoTagger(cfg, N_LABELS).to("cuda").train()
    fwd = torch.compile(model) if a.compile else model

    gen = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(a.batch, 3, cfg.image_size, cfg.image_size, device="cuda",
                    dtype=torch.bfloat16)
    y = (torch.rand(a.batch, N_LABELS, device="cuda", generator=gen) < 0.01).float()

    def one() -> float:
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        logits = fwd(x)
        loss = asl(logits.float(), y, gamma_pos=0.0, gamma_neg=2.0, clip=0.05,
                   detach_focal=True)
        t0.record()
        loss.backward()
        t1.record()
        torch.cuda.synchronize()
        model.zero_grad(set_to_none=True)
        return t0.elapsed_time(t1)

    one()  # first touch, not timed
    torch.cuda.reset_peak_memory_stats()
    times = [one() for _ in range(a.steps)]

    peak = torch.cuda.max_memory_allocated() / 2**30
    best = min(times)
    print(f"[bench] batch {a.batch} blocks {cfg.grad_checkpoint_blocks}: "
          f"{best:.1f} ms/backward, {a.batch/best*1000:.0f} img/s eager, "
          f"peak {peak:.2f} GB allocated", flush=True)
    print(f"[bench] all backward ms: {[round(t, 1) for t in times]}", flush=True)


if __name__ == "__main__":
    main()
