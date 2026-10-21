import time, torch
from wdtag.config import TrainConfig
from wdtag.model import DinoTagger
from adv_optm import SinkSGD_adv

cfg = TrainConfig()
model = DinoTagger(cfg, 12476).cuda()
oft, head = model.trainable_parameters()
print("params:", len(oft), "OFT tensors +", len(head), "head tensors")

# fake but realistic grads once
params = oft + head
for p in params:
    p.grad = torch.randn_like(p) * 1e-3

def bench(name, opt, n=30):
    for _ in range(3):
        opt.step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        opt.step()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / n
    print(f"{name:52s} {dt*1000:7.1f} ms/step", flush=True)
    return dt

groups = [{"params": oft, "lr": 0.02}, {"params": head, "lr": 0.004}]
bench("SGD (control)", torch.optim.SGD(groups))
bench("SinkSGD default", SinkSGD_adv(groups, momentum=0.95, sinkhorn_iterations=5))
bench("SinkSGD sinkhorn_iterations=3", SinkSGD_adv(groups, momentum=0.95, sinkhorn_iterations=3))
bench("SinkSGD state_precision=bf16_sr", SinkSGD_adv(groups, momentum=0.95, sinkhorn_iterations=5, state_precision="bf16_sr"))
bench("SinkSGD compiled_optimizer", SinkSGD_adv(groups, momentum=0.95, sinkhorn_iterations=5, compiled_optimizer=True), n=15)
bench("SinkSGD sinkhorn=3 + bf16_sr + compiled",
      SinkSGD_adv(groups, momentum=0.95, sinkhorn_iterations=3, state_precision="bf16_sr", compiled_optimizer=True), n=15)
bench("AdamW (reference)", torch.optim.AdamW(groups))
