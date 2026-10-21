"""Where does a training step actually go?"""
import time, torch, torch.nn.functional as F, numpy as np
from wdtag.config import TrainConfig
from wdtag.model import DinoTagger
from wdtag.data import CacheDataset
from adv_optm import SinkSGD_adv

cfg = TrainConfig()
dev = "cuda"
torch.manual_seed(0)
model = DinoTagger(cfg, 12476).to(dev)
oft, head = model.trainable_parameters()
opt = SinkSGD_adv([{"params": oft, "lr": 0.02}, {"params": head, "lr": 0.004}], momentum=0.95)

ds = CacheDataset("train", "pad", 12476)
perm = torch.randperm(len(ds)).tolist()
BS, STEPS = 32, 15
batches = [[perm[i + j] for j in range(BS)] for i in range(0, BS * (STEPS + 3), BS)]

acc = {k: 0.0 for k in ("data", "h2d", "fwd", "bwd", "opt")}
events = [torch.cuda.Event(True) for _ in range(2)]

def sync():
    torch.cuda.synchronize()

for step, bidx in enumerate(batches):
    t0 = time.perf_counter()
    imgs = torch.stack([ds[j][0] for j in bidx])
    y = torch.stack([ds[j][1] for j in bidx])
    t1 = time.perf_counter()
    x = imgs.to(dev, non_blocking=False)
    yy = y.to(dev, non_blocking=False)
    sync()
    t2 = time.perf_counter()
    logits = model(x)
    sync()
    t3 = time.perf_counter()
    loss = F.binary_cross_entropy_with_logits(logits.float(), yy)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
    sync()
    t4 = time.perf_counter()
    opt.step()
    sync()
    t5 = time.perf_counter()
    if step >= 3:
        acc["data"] += t1 - t0
        acc["h2d"] += t2 - t1
        acc["fwd"] += t3 - t2
        acc["bwd"] += t4 - t3
        acc["opt"] += t5 - t4

n = STEPS - 3
tot = sum(acc.values())
print(f"steps measured: {n}  (batch {BS})")
for k, v in acc.items():
    print(f"  {k:5s} {v/n*1000:7.1f} ms/step  {v/tot*100:5.1f}%")
print(f"  total {tot/n*1000:7.1f} ms/step -> {BS*n/tot:.0f} img/s")

# what if data were free?
print(f"\nwithout data+h2d: {tot/n - (acc['data']+acc['h2d'])/n:.3f}s -> "
      f"{BS/((tot-acc['data']-acc['h2d'])/n):.0f} img/s")
print(f"peak vram {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
