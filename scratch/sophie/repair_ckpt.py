"""Repair checkpoints written before the frozen-OFT save bug was fixed.

The bug: `adapter_state_dict()` saved only parameters with requires_grad. A fork that
froze rotations a previous stage had already trained (unfreeze_last_blocks does exactly
that) therefore dropped those rotations from every checkpoint it wrote. The live model
kept them in memory, so training and in-run validation looked normal; only reloading was
broken, and it failed quietly -- micro 0.19 instead of 0.58.

The rotations are recoverable because freezing preserves them: they are bit-identical to
the donor checkpoint the fork started from. This grafts them back in, verifying against
the donor before writing.

    python scratch/sophie/repair_ckpt.py --run runs/4a13a120 \
        --donor runs/683cfec3/ckpt_step050519.pt
"""
import argparse
import glob
import os
import sys

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--donor", required=True)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    donor = torch.load(a.donor, map_location="cpu", weights_only=False)
    rots = {k: v for k, v in donor["adapter"].items() if ".oft." in k}
    print(f"donor {a.donor}: {len(rots)} rotation tensors, "
          f"{sum(v.numel() for v in rots.values())/1e6:.2f}M params")

    files = sorted(glob.glob(os.path.join(a.run, "ckpt_step*.pt")),
                   key=lambda p: int(os.path.basename(p).split("ckpt_step")[1].split(".")[0]))
    for path in files:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        have = ck["adapter"]
        missing = [k for k in rots if k not in have]
        if not missing:
            print(f"  {os.path.basename(path):<26} complete ({len(have)} tensors)")
            continue
        bad = [k for k in missing if k in have and have[k].shape != rots[k].shape]
        print(f"  {os.path.basename(path):<26} MISSING {len(missing)} rotations"
              f"{'  SHAPE MISMATCH' if bad else ''}")
        if not a.apply or bad:
            continue
        for k in missing:
            have[k] = rots[k].clone()
        ck["adapter"] = have
        ck["repaired"] = {"donor": a.donor, "added": len(missing),
                          "reason": "frozen OFT rotations omitted by adapter_state_dict"}
        tmp = path + ".tmp"
        torch.save(ck, tmp)
        os.replace(tmp, path)
        print(f"      -> repaired, now {len(have)} tensors")

    if not a.apply:
        print("\n(dry run; pass --apply to write)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
