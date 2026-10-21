"""Did OFTv2 help or hurt? Compare the trained model with its rotations applied,
with the rotations reset to identity (R = I), and against frozen-probe numbers."""
import numpy as np, torch
from wdtag.config import TrainConfig
from wdtag.model import DinoTagger
from wdtag.data import CacheDataset, load_targets
from wdtag.metrics import compute_metrics
from wdtag.infer import predict_probs
from wdtag.labels import load_label_space

torch.manual_seed(0)
N = len(load_label_space())
cfg = TrainConfig.from_json("configs/pad.json")
run = r"runs\54f4af8e"

model = DinoTagger(cfg, N).cuda()
ck = torch.load(rf"{run}\best_val.pt", map_location="cuda", weights_only=False)
model.load_state_dict(ck["adapter"], strict=False)
tgt = load_targets("test", "pad", N).astype(bool)

def probs():
    return predict_probs(lambda x: model(x), "test", "pad", N, model.in_mean, model.in_std,
                         batch_size=32, device="cuda")

def report(tag, p):
    m = compute_metrics(p >= 0.5, tgt)
    best = max((compute_metrics(p >= t, tgt)["micro_f1"], t) for t in np.arange(0.05, 0.95, 0.05))
    print(f"{tag:24s} micro F1 @0.5 {m['micro_f1']:.4f} | best {best[0]:.4f} @ {best[1]:.2f} | "
          f"macro F1 @0.5 {m['macro_f1']:.4f}", flush=True)
    return p

p_oft = report("OFTv2 as trained", probs())

# identity rotation: zero the adapter weights -> Cayley(0) = I
with torch.no_grad():
    for name, p in model.named_parameters():
        if name.endswith("oft.weight"):
            p.zero_()
p_id = report("rotations reset to R=I", probs())

# a raw frozen backbone + the trained head is what R=I means above, so diffs show
# whether the adapters moved the features somewhere better or worse.
print(f"\nmean |p_oft - p_identity| = {np.abs(p_oft - p_id).mean():.4f}", flush=True)

# restricted to the 1,000 most frequent tags, which is what the frozen probe scored
counts = load_targets("train", "pad", N).sum(0)
top = np.argsort(-counts)[:1000]
for tag, p in (("OFTv2", p_oft), ("R=I", p_id)):
    m = compute_metrics(p[:, top] >= 0.5, tgt[:, top])
    bm = max((compute_metrics(p[:, top] >= t, tgt[:, top])["micro_f1"], t) for t in np.arange(0.05, 0.95, 0.05))
    print(f"top-1000 frequent tags, {tag:6s}: micro F1 @0.5 {m['micro_f1']:.4f}, best {bm[0]:.4f} @ {bm[1]:.2f}")
print("frozen-feature probe on the same 1,000 tags (from earlier sweep): 0.4025")
