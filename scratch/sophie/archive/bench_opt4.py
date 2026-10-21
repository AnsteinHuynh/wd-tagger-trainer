import time, torch
torch._dynamo.config.cache_size_limit = 1_000_000
torch._dynamo.config.recompile_limit = 1_000_000
torch._dynamo.config.accumulated_recompile_limit = 1_000_000
torch._dynamo.config.force_parameter_static_shapes = False

from wdtag.config import TrainConfig
from wdtag.model import DinoTagger
from adv_optm import SinkSGD_adv

cfg = TrainConfig(); model = DinoTagger(cfg, 12476).cuda()
oft, head = model.trainable_parameters()

# ---------- 1. does cost scale with tensor COUNT or with element count? ----------
print("=== scaling: k adapter tensors ===", flush=True)
for k in (10, 25, 50, 73):
    ps = oft[:k]
    for p in ps:
        p.grad = torch.randn_like(p) * 1e-3
    o = SinkSGD_adv([{"params": ps, "lr": 0.02}], momentum=0.95)
    for _ in range(2): o.step()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(15): o.step()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / 15
    print(f"  k={k:3d}  {dt*1000:7.2f} ms  -> {dt*1000/k:6.3f} ms/tensor", flush=True)

print("=== same tensors, 8x bigger (element scaling control) ===", flush=True)
big = [torch.zeros(96, 496, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(73)]
for p in big:
    p.grad = torch.randn_like(p) * 1e-3
o = SinkSGD_adv([{"params": big, "lr": 0.02}], momentum=0.95)
for _ in range(2): o.step()
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(15): o.step()
torch.cuda.synchronize()
print(f"  73 x [96,496] (8x bigger elements): {(time.perf_counter()-t0)/15*1000:7.2f} ms", flush=True)

# ---------- 2. compiled optimizer with raised dynamo limits ----------
print("=== compiled_optimizer with raised dynamo limits, 120 steps ===", flush=True)
for p in oft + head:
    p.grad = torch.randn_like(p) * 1e-3
groups = [{"params": oft, "lr": 0.02}, {"params": head, "lr": 0.004}]
try:
    o = SinkSGD_adv(groups, momentum=0.95, compiled_optimizer=True)
    for blk in range(6):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(20): o.step()
        torch.cuda.synchronize()
        print(f"  block {blk}: {(time.perf_counter()-t0)/20*1000:7.1f} ms/step", flush=True)
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {str(e)[:300]}", flush=True)
