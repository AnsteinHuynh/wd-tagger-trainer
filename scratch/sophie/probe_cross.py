import numpy as np
import torch

from wdtag.config import TrainConfig
from wdtag.model import DinoTagger

CACHE = r"I:\.cache\wd-tagger-trainer\valid_nested"
imgs = np.load(f"{CACHE}\\images.npy", mmap_mode="r")
x = torch.from_numpy(np.ascontiguousarray(imgs[:8])).cuda()

ck_a = torch.load(r"runs\459e2a02\ckpt_step014000.pt", map_location="cpu", weights_only=False)["adapter"]
ck_b = torch.load(r"runs\841a299d\ckpt_step014000.pt", map_location="cpu", weights_only=False)["adapter"]

for cfg_name, cfg_path in (("cfgA", r"runs\459e2a02\config.json"),
                           ("cfgB", r"runs\841a299d\config.json")):
    cfg = TrainConfig.from_json(cfg_path)
    for ck_name, ck in (("ckA", ck_a), ("ckB", ck_b)):
        model = DinoTagger(cfg, 12476).cuda()
        res = model.load_adapter_state_dict(ck, strict=False)
        model.eval()
        with torch.no_grad():
            z = model(x)
            p = torch.sigmoid(z.float())
        print(f"{cfg_name} + {ck_name}: logits mean {float(z.mean()):+.3f}  "
              f"probs mean {float(p.mean()):.5f}  tags/image {(p >= 0.40).sum(1).float().mean():.0f}")
        print(f"    load returned {res}")
        del model
        torch.cuda.empty_cache()
