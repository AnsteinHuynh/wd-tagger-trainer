import time, torch, torch.nn.functional as F
from wdtag.config import TrainConfig
from wdtag.model import DinoTagger
from adv_optm import SinkSGD_adv

cfg = TrainConfig(); model = DinoTagger(cfg, 12476).cuda()
oft, head = model.trainable_parameters()
for p in oft + head:
    p.grad = torch.randn_like(p) * 1e-3
groups = [{"params": oft, "lr": 0.02}, {"params": head, "lr": 0.004}]

print("=== compiled optimizer, 200 steps ===", flush=True)
o = SinkSGD_adv(groups, momentum=0.95, compiled_optimizer=True, stochastic_rounding=False)
for blk in range(5):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(40): o.step()
    torch.cuda.synchronize()
    print(f"  block {blk}: {(time.perf_counter()-t0)/40*1000:8.1f} ms/step", flush=True)

print("=== eager optimizer, 40 steps ===", flush=True)
o2 = SinkSGD_adv(groups, momentum=0.95)
for _ in range(2): o2.step()
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(40): o2.step()
torch.cuda.synchronize()
print(f"  eager: {(time.perf_counter()-t0)/40*1000:8.1f} ms/step", flush=True)

# ---- model fwd/bwd with and without compile
x = torch.randn(32, 3, 512, 512, device="cuda", dtype=torch.bfloat16)
y = torch.zeros(32, 12476, device="cuda"); y[:, :30] = 1.0
opt = SinkSGD_adv(groups, momentum=0.95)

def step_eager():
    logits = model(x)
    loss = F.binary_cross_entropy_with_logits(logits.float(), y)
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

def bench(name, fn, n=10):
    for _ in range(3): fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize()
    print(f"  {name}: {(time.perf_counter()-t0)/n*1000:7.1f} ms/step", flush=True)

print("=== full step ===", flush=True)
bench("eager", step_eager)
try:
    mc = torch.compile(model, mode="max-autotune-no-cudagraphs")
    def step_compiled():
        logits = mc(x)
        loss = F.binary_cross_entropy_with_logits(logits.float(), y)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    bench("model compiled", step_compiled, n=10)
except Exception as e:
    print("  compile failed:", type(e).__name__, str(e)[:200], flush=True)
