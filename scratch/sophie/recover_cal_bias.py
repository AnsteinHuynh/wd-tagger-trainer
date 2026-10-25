"""Recover the per-tag calibration bias that arm B trained with but did not save.

B applied a per-tag bias to the logits and let the model's own weights drift to compensate, so
the checkpoint alone is unusable: validated without its bias it fires on 8296 tags per image.
The bias was updated to match each tag's predicted rate to its label rate over training batches,
so re-running that same controller over a fresh sample of training images recovers an
equivalent bias. This is a re-derivation, not the original bytes, and it is fitted on training
data only, never on validation.

Writes runs/841a299d/cal_bias_recovered.npy
"""
import sys

import numpy as np
import torch

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
BATCH = 16
ETA = 2.0
CLIP = 6.0
print_every = 2000

from wdtag.config import TrainConfig
from wdtag.model import DinoTagger
from wdtag.stream import TarStream

device = "cuda"
cfg = TrainConfig.from_json(r"runs\841a299d\config.json")
ck = torch.load(r"runs\841a299d\ckpt_step014000.pt", map_location="cpu", weights_only=False)
model = DinoTagger(cfg, 12476).to(device)
model.load_adapter_state_dict(ck["adapter"], strict=False)
model.eval()
print("model loaded", flush=True)

bias = torch.zeros(12476, device=device, dtype=torch.float32)
st = TarStream("train", seed=424242, pass_index=0, batch_size=BATCH, n_labels=12476,
               workers=12, prefetch=8, max_samples=N)

seen = 0
p_sum = y_sum = None
n = 0
with torch.no_grad():
    for item in st.batches():
        x, y = item[0], item[1]
        x = (x if torch.is_tensor(x) else torch.from_numpy(x)).to(device, non_blocking=True)
        y = (y if torch.is_tensor(y) else torch.from_numpy(y)).to(device) > 0.5
        z = model(x)
        p = torch.sigmoid(z.float() + bias[None, :])
        p_sum = p.sum(0) if p_sum is None else p_sum + p.sum(0)
        y_sum = y.float().sum(0) if y_sum is None else y_sum + y.float().sum(0)
        n += len(x)
        seen += len(x)
        if n >= 384:
            bias.add_(((y_sum - p_sum) / n) * ETA * 8.0).clamp_(-CLIP, CLIP)
            p_sum = y_sum = None
            n = 0
        if seen % print_every < BATCH:
            with torch.no_grad():
                pr = float(p.mean() * 12476)
            print(f"{seen}/{N} images, predicted per image approx {pr:.1f}, "
                  f"bias median {float(bias.median()):+.2f}", flush=True)
        if seen >= N:
            break

np.save(r"runs\841a299d\cal_bias_recovered.npy", bias.detach().cpu().numpy())
print(f"bias median {float(bias.median()):+.2f} p05 {float(bias.quantile(0.05)):+.2f} "
      f"p95 {float(bias.quantile(0.95)):+.2f}", flush=True)
print("saved runs/841a299d/cal_bias_recovered.npy", flush=True)
