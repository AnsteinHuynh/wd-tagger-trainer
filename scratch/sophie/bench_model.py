import os, time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
from wdtag.config import TrainConfig
from wdtag.model import DinoTagger

free, total = torch.cuda.mem_get_info()
print("vram free/total: %.1f/%.1f GB" % (free / 1e9, total / 1e9), flush=True)
cfg = TrainConfig()
m = DinoTagger(cfg, 12476).cuda()
print("params:", m.count_params(), "oft modules:", m.n_oft, flush=True)
oft, head = m.trainable_parameters()
print("oft params:", sum(p.numel() for p in oft), "tensors:", len(oft), flush=True)
opt = torch.optim.SGD([{"params": oft, "lr": 0.02}, {"params": head, "lr": 0.002}])

bs = int(os.environ.get("BS", "16"))
torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
x = torch.randn(bs, 3, 512, 512, device="cuda", dtype=torch.bfloat16)
y = torch.zeros(bs, 12476, device="cuda"); y[:, :30] = 1.0
for _ in range(2):
    logits = m(x)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), y)
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
torch.cuda.synchronize(); t = time.time(); n = 5
for _ in range(n):
    logits = m(x)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), y)
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
torch.cuda.synchronize()
dt = (time.time() - t) / n
print(f"bs {bs}: {dt*1000:.0f} ms/step, {bs/dt:.0f} img/s, peak allocated "
      f"{torch.cuda.max_memory_allocated()/1e9:.1f} GB, reserved {torch.cuda.max_memory_reserved()/1e9:.1f} GB", flush=True)
