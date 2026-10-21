import time, torch
from wdtag.config import TrainConfig
from wdtag.model import DinoTagger
from adv_optm import SinkSGD_adv

cfg = TrainConfig(); model = DinoTagger(cfg, 12476).cuda()
oft, head = model.trainable_parameters()
for p in oft + head:
    p.grad = torch.randn_like(p) * 1e-3
groups = [{"params": oft, "lr": 0.02}, {"params": head, "lr": 0.004}]

def bench(name, **kw):
    o = SinkSGD_adv(groups, momentum=0.95, **kw)
    for _ in range(2): o.step()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(20): o.step()
    torch.cuda.synchronize()
    print(f"{name:46s} {(time.perf_counter()-t0)/20*1000:8.2f} ms", flush=True)

bench("SR on (current)", sinkhorn_iterations=5)
bench("SR off", sinkhorn_iterations=5, stochastic_rounding=False)
bench("SR off + compiled", sinkhorn_iterations=5, stochastic_rounding=False, compiled_optimizer=True)
bench("SR off + fp32 states", sinkhorn_iterations=5, stochastic_rounding=False, state_precision="fp32")

# how expensive is the random generation itself?
g = torch.Generator(device="cuda").manual_seed(42)
t = torch.empty(24, 496, device="cuda")
for _ in range(3): torch.randint(0, 1 << 16, t.shape, device="cuda", dtype=torch.int32, generator=g)
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(100): torch.randint(0, 1 << 16, t.shape, device="cuda", dtype=torch.int32, generator=g)
torch.cuda.synchronize()
print(f"{'torch.randint with explicit generator (per call)':46s} {(time.perf_counter()-t0)/100*1000:8.3f} ms")
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(100): torch.randint(0, 1 << 16, t.shape, device="cuda", dtype=torch.int32)
torch.cuda.synchronize()
print(f"{'torch.randint without generator':46s} {(time.perf_counter()-t0)/100*1000:8.3f} ms")
