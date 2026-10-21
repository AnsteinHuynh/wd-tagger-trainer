"""How far did each run's rotations actually move from identity?

Costs nothing: reads the saved adapter weights and applies OFT's own Cayley map.
A rotation that has wandered far from I has stopped being a small correction to
the frozen backbone.
"""
import torch

RUNS = {
    "lr_oft 0.002 (8 ep, s1337)": r"runs\539d8880\best_val.pt",
    "lr_oft 0.0005 (8 ep, s1337)": r"runs\e6bc2879\best_val.pt",
    "lr_oft 0.0005 (8 ep, s2025)": r"runs\af7c4ac4\best_val.pt",
}
D = 32  # block_size

def cayley(vec: torch.Tensor, d: int = D) -> torch.Tensor:
    b = vec.shape[0]
    q = torch.zeros(b, d, d, dtype=torch.float32)
    iu = torch.triu_indices(d, d, offset=1)
    q[:, iu[0], iu[1]] = vec.float()
    q = q - q.transpose(1, 2)
    eye = torch.eye(d).expand(b, d, d)
    return torch.linalg.solve(eye + q, eye - q, left=False)

print(f"{'run':24s} {'||R-I||F/sqrt(d)':>17s} {'mean angle':>11s} {'max|R-I|':>9s}")
for name, path in RUNS.items():
    ck = torch.load(path, map_location="cpu", weights_only=False)
    vecs = [v for k, v in ck["adapter"].items() if k.endswith("oft.weight")]
    drift, ang, mx = [], [], []
    for v in vecs:
        r = cayley(v)
        eye = torch.eye(D).expand_as(r)
        drift.append((torch.linalg.matrix_norm(r - eye).mean() / D ** 0.5).item())
        # rotation angle of each block: mean over blocks
        tr = r.diagonal(dim1=-2, dim2=-1).sum(-1).clamp(-1, 1)
        ang.append(torch.acos(tr / 3.0).mean().item())   # Frobenius-normalised approx
        mx.append((r - eye).abs().max().item())
    print(f"{name:24s} {sum(drift)/len(drift):17.4f} {sum(ang)/len(ang):11.4f} {max(mx):9.4f}")
print("\n(identity: 0.0 / 0.0 / 0.0)")


