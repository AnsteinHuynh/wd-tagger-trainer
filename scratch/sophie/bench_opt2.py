import time, torch
from wdtag.config import TrainConfig
from wdtag.model import DinoTagger
from adv_optm import SinkSGD_adv
from adv_optm.util.sinkhorn import apply_sr_sinkhorn

cfg = TrainConfig(); model = DinoTagger(cfg, 12476).cuda()
oft, head = model.trainable_parameters()
for p in oft + head:
    p.grad = torch.randn_like(p) * 1e-3

def bench(name, groups, n=20):
    o = SinkSGD_adv(groups, momentum=0.95, sinkhorn_iterations=5)
    for _ in range(2): o.step()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n): o.step()
    torch.cuda.synchronize()
    print(f"{name:40s} {(time.perf_counter()-t0)/n*1000:8.2f} ms", flush=True)

bench("head only (4 tensors, 19.2M)", [{"params": head, "lr": 0.004}])
bench("oft only (73 tensors, 1.3M)", [{"params": oft, "lr": 0.02}])

# raw sinkhorn on the head-shaped matrix
m = torch.randn(12476, 1536, device="cuda")
for _ in range(3): apply_sr_sinkhorn(m.clone(), 5)
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(20): apply_sr_sinkhorn(m.clone(), 5)
torch.cuda.synchronize()
print(f"{'raw apply_sr_sinkhorn [12476,1536]':40s} {(time.perf_counter()-t0)/20*1000:8.2f} ms", flush=True)
