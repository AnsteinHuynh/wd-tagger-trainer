"""One purpose: locate where two identical runs stop agreeing.

Prints checksums in order, so running it twice in separate processes says whether the divergence
is in initialisation, in the data, in the forward, in the backward, or in the optimiser:

  init   adapter weights after construction and pretrained load, before any step
  batch  the first batch the stream hands over (images and labels)
  loss1  the loss of that batch, 12 decimals
  grad   the gradient checksum after backward
  step1  the adapter checksum after one optimiser step

    python scratch/sophie/test_step_determinism.py [--deterministic]
"""

import argparse
import hashlib

import torch

from wdtag.config import TrainConfig
from wdtag.losses import asl
from wdtag.model import DinoTagger
from wdtag.stream import TarStream


def digest(tensors) -> str:
    h = hashlib.sha256()
    for k, v in tensors:
        h.update(k.encode())
        h.update(v.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def grads(model) -> list:
    return [(n, p.grad) for n, p in model.named_parameters()
            if p.requires_grad and p.grad is not None]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deterministic", action="store_true")
    a = ap.parse_args()
    if a.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    cfg = TrainConfig.from_json(r"scratch\sophie\test_resume_cfg.json")
    cfg.batch_size = 8
    device = "cuda"
    torch.manual_seed(cfg.seed)

    model = DinoTagger(cfg, 12476).to(device)
    model.init_head_bias(torch.full((12476,), 0.01))
    print("init  ", digest(sorted(model.adapter_state_dict().items())))

    sp = TarStream("train", seed=cfg.seed, pass_index=0, batch_size=cfg.batch_size,
                   n_labels=12476, workers=8, prefetch=1, skip=0,
                   max_samples=10 ** 9, log=lambda m: None)
    x, y, consumed, src = next(sp.batches())
    print("batch ", hashlib.sha256(x.numpy().tobytes() + y.numpy().tobytes()).hexdigest()[:16])

    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=5e-4)
    model.train()
    logits = model(x.to(device))
    loss = asl(logits.float(), y.to(device), gamma_pos=0.0, gamma_neg=2.0, clip=0.05,
               detach_focal=True)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    print("loss1  %.12f" % loss.item())
    print("grad  ", digest(sorted(grads(model))))
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
    opt.step()
    print("step1 ", digest(sorted(model.adapter_state_dict().items())))
    print("deterministic", a.deterministic)


if __name__ == "__main__":
    main()
