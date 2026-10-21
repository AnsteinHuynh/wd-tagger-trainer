"""Enumerate exactly which modules got an OFT rotation and at what cost."""
import collections
from wdtag.config import TrainConfig
from wdtag.model import DinoTagger
from wdtag.oft import OFTLinear, OFTConv2d

cfg = TrainConfig.from_json("configs/squash.json")
m = DinoTagger(cfg, 12476)

rows, shapes = [], collections.Counter()
total = 0
for name, mod in m.backbone.named_modules():
    if not isinstance(mod, (OFTLinear, OFTConv2d)):
        continue
    rot = mod.oft
    total += rot.weight.numel()
    shapes[(rot.block_size, rot.r, tuple(rot.weight.shape))] += 1
    rows.append((name, mod, rot))

print(f"modules: {len(rows)}   trainable rotation params: {total/1e6:.3f} M   "
      f"block_size: {cfg.block_size}")
print(f"free skew entries per block: {cfg.block_size*(cfg.block_size-1)//2} "
      f"(upper triangle of {cfg.block_size}x{cfg.block_size})\n")
print("distinct rotation shapes:")
for (bs, r, shape), n in sorted(shapes.items()):
    print(f"  {n:3d} x weight {shape}   block_size {bs}   r {r} blocks   "
          f"{shape[0]*shape[1]:,} params each")

print("\nper block (all 12 blocks are identical):")
seen = 0
for name, mod, rot in rows:
    if ".layers.0." not in name and "patch" not in name:
        continue
    kind = "Conv2d" if isinstance(mod, OFTConv2d) else "Linear"
    where = "weight" if isinstance(mod, OFTConv2d) else "input"
    base = mod.base
    base_shape = tuple(base.weight.shape)
    print(f"  {name:46s} {kind:6s} {base_shape!s:20s} rotates {where:6s} width "
          f"{rot.in_features:5d} -> r={rot.r:3d} blocks, {tuple(rot.weight.shape)}")
