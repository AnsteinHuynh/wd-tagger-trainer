"""One number: what a full training step costs, split into forward and backward.

bench_ckpt.py times the backward only, which made every bench-to-trainer comparison a factor of
two wrong. This one records before the forward and after the backward, so its total is comparable
to the img/s the trainer prints. No data reading, synthetic input, so it can run beside a live arm.

    python scratch/sophie/bench_step.py --config <cfg> --batch 48 --cayley 0
    --cayley 0 is the exact Cayley (torch.linalg.solve per module), 5 is the Neumann series.
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

CONFIG = r"configs\heads\phase3\42_u1-C-L-asl-g2.0-siglip2b384-200k-b48-a1.json"
N_LABELS = 12476


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--cayley", type=int, default=0)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--opt", action="store_true", help="also time grad clip and the optimiser")
    ap.add_argument("--clip", action="store_true", help="time the fp32 global-norm clip alone")
    a = ap.parse_args()

    cfg = TrainConfig.from_json(a.config)
    cfg.batch_size = a.batch
    cfg.stream = False
    cfg.cayley_terms = a.cayley

    free, total = torch.cuda.mem_get_info()
    print(f"[step] batch {a.batch}, cayley_terms {a.cayley}, "
          f"{free/2**30:.1f} of {total/2**30:.1f} GB free before load", flush=True)

    model = DinoTagger(cfg, N_LABELS).to("cuda").train()
    fwd = torch.compile(model) if a.compile else model

    params = [p for p in model.parameters() if p.requires_grad]
    opt = None
    if a.opt:
        from adv_optm import SinkSGD_adv
        opt = SinkSGD_adv([{"params": params, "lr": cfg.lr_oft}],
                          momentum=cfg.momentum, weight_decay=cfg.weight_decay,
                          sinkhorn_iterations=cfg.sinkhorn_iterations,
                          orthogonal_sinkhorn=cfg.orthogonal_sinkhorn,
                          stochastic_rounding=True, compiled_optimizer=True)

    gen = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(a.batch, 3, cfg.image_size, cfg.image_size, device="cuda",
                    dtype=torch.bfloat16)
    y = (torch.rand(a.batch, N_LABELS, device="cuda", generator=gen) < 0.01).float()

    def one() -> tuple[float, float, float, float]:
        e = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
        e[0].record()
        logits = fwd(x)
        loss = asl(logits.float(), y, gamma_pos=0.0, gamma_neg=2.0, clip=0.05,
                   detach_focal=True)
        e[1].record()
        loss.backward()
        e[2].record()
        if opt is not None:
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            e[3].record()
            opt.step()
            e[4].record()
        elif a.clip:
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            e[3].record()
            e[4].record()
        else:
            e[3].record()
            e[4].record()
        torch.cuda.synchronize()
        model.zero_grad(set_to_none=True)
        return (e[0].elapsed_time(e[1]), e[1].elapsed_time(e[2]),
                e[3].elapsed_time(e[4]), e[2].elapsed_time(e[3]))

    one()  # first touch, not timed
    torch.cuda.reset_peak_memory_stats()
    pairs = [one() for _ in range(a.steps)]

    peak = torch.cuda.max_memory_allocated() / 2**30
    f_ms = min(p[0] for p in pairs)
    b_ms = min(p[1] for p in pairs)
    o_ms = min(p[2] for p in pairs)
    c_ms = min(p[3] for p in pairs)
    tot = f_ms + b_ms + o_ms + c_ms
    print(f"[step] batch {a.batch} cayley {a.cayley} compile {a.compile} opt {a.opt}: "
          f"fwd {f_ms:.1f} + bwd {b_ms:.1f} + clip {c_ms:.1f} + opt {o_ms:.1f} "
          f"= {tot:.1f} ms/step, {a.batch/tot*1000:.0f} img/s, peak {peak:.2f} GB, "
          f"{tot/a.batch:.2f} ms/image", flush=True)
    print(f"[step] all fwd ms: {[round(p[0], 1) for p in pairs]}", flush=True)
    print(f"[step] all bwd ms: {[round(p[1], 1) for p in pairs]}", flush=True)
    print(f"[step] all opt ms: {[round(p[2], 1) for p in pairs]}", flush=True)
    print(f"[step] all clip ms: {[round(p[3], 1) for p in pairs]}", flush=True)


if __name__ == "__main__":
    main()

