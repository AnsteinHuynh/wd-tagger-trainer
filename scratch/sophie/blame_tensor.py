"""Which tensor makes checkpoint B predict 8296 tags per image instead of 56?

Bisect by blame: start from A's tensors, substitute one of B's at a time, and see which single
substitution moves the mean logit. A difference this large cannot come from 1000 steps of
training on a 0.19 max weight delta, so one tensor is doing something structural.
"""
import numpy as np
import torch

from wdtag.config import TrainConfig
from wdtag.model import DinoTagger

CACHE = r"I:\.cache\wd-tagger-trainer\valid_nested"
cfg = TrainConfig.from_json(r"runs\459e2a02\config.json")
a = torch.load(r"runs\459e2a02\ckpt_step014000.pt", map_location="cpu", weights_only=False)["adapter"]
b = torch.load(r"runs\841a299d\ckpt_step014000.pt", map_location="cpu", weights_only=False)["adapter"]

imgs = np.load(f"{CACHE}\\images.npy", mmap_mode="r")
x = torch.from_numpy(np.ascontiguousarray(imgs[:8])).cuda()

model = DinoTagger(cfg, 12476).cuda().eval()


def logits_mean(state):
    model.load_adapter_state_dict(state, strict=False)
    with torch.no_grad():
        return float(model(x).mean())


print(f"all A: {logits_mean(a):+.3f}   all B: {logits_mean(b):+.3f}")
diff_keys = [k for k in a if not torch.equal(a[k], b[k])]
print(f"{len(diff_keys)} tensors differ")

blame = []
for k in diff_keys:
    mixed = dict(a)
    mixed[k] = b[k]
    m = logits_mean(mixed)
    blame.append((abs(m - logits_mean(a)), m, k))
blame.sort(reverse=True)
for d, m, k in blame[:6]:
    print(f"  replacing {k:52s} -> logits {m:+.3f} (shift {m - logits_mean(a):+.3f})")
