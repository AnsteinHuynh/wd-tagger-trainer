import numpy as np
import torch

from wdtag.config import TrainConfig
from wdtag.model import DinoTagger

CACHE = r"I:\.cache\wd-tagger-trainer\valid_nested"

cfg_a = TrainConfig.from_json(r"runs\459e2a02\config.json")
cfg_b = TrainConfig.from_json(r"runs\841a299d\config.json")
n_labels = len(TrainConfig.from_json(r"runs\459e2a02\config.json").__dict__) and 12476

imgs = np.load(f"{CACHE}\\images.npy", mmap_mode="r")
x = torch.from_numpy(np.ascontiguousarray(imgs[:8])).cuda()

for name, cfg, ck in (("A", cfg_a, r"runs\459e2a02\ckpt_step014000.pt"),
                      ("B", cfg_b, r"runs\841a299d\ckpt_step014000.pt")):
    model = DinoTagger(cfg, n_labels).cuda()
    c = torch.load(ck, map_location="cpu", weights_only=False)
    missing = model.load_adapter_state_dict(c["adapter"], strict=False)
    model.eval()
    with torch.no_grad():
        z = model(x)
        p = torch.sigmoid(z.float())
    print(f"{name}: loaded, missing/unexpected {missing if isinstance(missing, (list, tuple)) else type(missing)}")
    print(f"   logits mean {float(z.mean()):+.3f} std {float(z.std()):.3f} "
          f"min {float(z.min()):+.3f} max {float(z.max()):+.3f}")
    print(f"   probs  mean {float(p.mean()):.5f}  above 0.40 {(p >= 0.40).float().mean() * 100:.1f} percent"
          f"  per image {(p >= 0.40).sum(1).float().mean():.0f} tags")
    del model
    torch.cuda.empty_cache()
