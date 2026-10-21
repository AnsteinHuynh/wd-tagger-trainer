import time, torch, torch.nn.functional as F
from wdtag.config import TrainConfig
from wdtag.model import DinoTagger
from adv_optm import SinkSGD_adv

cfg = TrainConfig(); model = DinoTagger(cfg, 12476).cuda()
oft, head = model.trainable_parameters()
groups = [{"params": oft, "lr": 0.02}, {"params": head, "lr": 0.004}]
opt = SinkSGD_adv(groups, momentum=0.95)
x = torch.randn(32, 3, 512, 512, device="cuda", dtype=torch.bfloat16)
y = torch.zeros(32, 12476, device="cuda"); y[:, :30] = 1.0

def bench(name, fn, n=10):
    for _ in range(3): fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / n
    print(f"  {name:22s} {dt*1000:7.1f} ms/step  {32/dt:5.1f} img/s", flush=True)
    return dt

def make_step(m):
    def step():
        logits = m(x)
        loss = F.binary_cross_entropy_with_logits(logits.float(), y)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return step

e = bench("eager", make_step(model))

for mode in ("default",):
    try:
        t0 = time.time()
        mc = torch.compile(model, mode=mode)
        s = make_step(mc)
        s(); s()  # trigger compile
        print(f"  compiled [{mode}] ready in {time.time()-t0:.1f}s", flush=True)
        bench(f"compiled [{mode}]", s)
    except Exception as e:
        print(f"  compile [{mode}] failed: {type(e).__name__}: {str(e)[:150]}", flush=True)

