"""One purpose: does 3 micro-batches of 4 give the same gradient as one batch of 12?

Isolates the accumulation math from the training loop. Same model, same images, two ways of
splitting them, compare loss and every gradient.

    python scratch/sophie/test_accum_grad.py
"""

import torch

from wdtag.config import TrainConfig
from wdtag.losses import asl
from wdtag.model import DinoTagger
from wdtag.stream import TarStream


def grads(model):
    return [(n, p.grad.detach().clone()) for n, p in model.named_parameters()
            if p.requires_grad and p.grad is not None]


def main() -> None:
    cfg = TrainConfig.from_json(r"scratch\sophie\test_resume_cfg.json")
    # dropout draws one mask per forward, so a batch of 12 and three of 4 would see different
    # noise and the comparison would measure that instead of accumulation
    cfg.head_dropout = 0.0
    device = "cuda"
    torch.manual_seed(cfg.seed)
    model = DinoTagger(cfg, 12476).to(device)
    model.init_head_bias(torch.full((12476,), 0.01))
    model.train()

    sp = TarStream("train", seed=cfg.seed, pass_index=0, batch_size=12, n_labels=12476,
                   workers=8, prefetch=1, skip=0, max_samples=10 ** 9, log=lambda m: None)
    x, y, consumed, src = next(sp.batches())
    x, y = x.to(device), y.to(device)
    print("batch", x.shape, "loss_fn asl gamma_neg 2.0")

    # one batch of 12
    model.zero_grad(set_to_none=True)
    loss12 = asl(model(x).float(), y, gamma_neg=2.0)
    loss12.backward()
    g12 = grads(model)

    # three micro-batches of 4, same images in the same order
    model.zero_grad(set_to_none=True)
    parts = torch.split(x, 4)
    yparts = torch.split(y, 4)
    total = 0.0
    for xi, yi in zip(parts, yparts):
        li = asl(model(xi).float(), yi, gamma_neg=2.0)
        (li / len(parts)).backward()
        total += li.item() / len(parts)
    gacc = grads(model)

    print(f"loss   batch12 {loss12.item():.9f}   accumulated {total:.9f}   "
          f"rel {abs(total - loss12.item()) / loss12.item():.2e}")
    # max-abs is the wrong metric here: bf16 logits differ by ~1e-2 between a batch of 12 and
    # three of 4, so compare direction (cosine) and magnitude-weighted error instead
    import math
    flat12 = torch.cat([a.float().ravel() for _, a in g12])
    flatacc = torch.cat([b.float().ravel() for _, b in gacc])
    cos = torch.nn.functional.cosine_similarity(flat12, flatacc, dim=0).item()
    rms12 = flat12.pow(2).mean().sqrt().item()
    rmsd = (flat12 - flatacc).pow(2).mean().sqrt().item()
    print(f"grads  cosine {cos:.9f}   rms batch12 {rms12:.3e}   rms(diff) {rmsd:.3e}   "
          f"rms ratio {rmsd / rms12:.3e}")
    for (n, a), (_, b) in zip(g12, gacc):
        sa, sb = a.float(), b.float()
        c = torch.nn.functional.cosine_similarity(sa.ravel(), sb.ravel(), dim=0).item()
        if c < 0.999:
            print(f"  low cosine {c:.6f} on {n}  |a|max {sa.abs().max():.2e}")


if __name__ == "__main__":
    main()


