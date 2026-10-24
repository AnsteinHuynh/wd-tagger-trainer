"""One purpose: are two checkpoints the same model?

    python scratch/sophie/cmp_ckpt.py <a.pt> <b.pt>

Prints the worst absolute difference over the adapter and the head, how many tensors are
bit-identical, and the largest single-tensor deviation with its name, so a divergence can be
named rather than summarised.
"""

import sys

import torch


def main() -> None:
    a = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
    b = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
    sa = a.get("adapter") or a.get("model") or {}
    sb = b.get("adapter") or b.get("model") or {}
    keys = [k for k in sa if k in sb]
    if not keys:
        print("no shared tensors: adapter=%d model=%d" % (len(sa), len(sb)))
        raise SystemExit(2)
    worst, wk, ident = 0.0, None, 0
    for k in keys:
        if sa[k].shape != sb[k].shape:
            print("shape mismatch on", k, sa[k].shape, sb[k].shape)
            continue
        if torch.equal(sa[k], sb[k]):
            ident += 1
            continue
        d = (sa[k].float() - sb[k].float()).abs().max().item()
        if d > worst:
            worst, wk = d, k
    print(f"step {a.get('step')} vs {b.get('step')}: {ident}/{len(keys)} tensors bit-identical, "
          f"worst abs diff {worst:.3e}" + (f" on {wk}" if wk else ""))
    for name, tag in (("a", a), ("b", b)):
        best = tag.get("best")
        print(f"  {name}: step {tag.get('step')} best {best}")


if __name__ == "__main__":
    main()
