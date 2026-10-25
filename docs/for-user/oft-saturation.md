# OFT saturation: how to measure it, how to read it

Catches an OFT adapter quietly corrupting its own forward pass. Portable to any Orthogonal Fine-Tuning project with a Cayley rotation — the code below has no dependency on this repo.

---

## 1. What "saturation" actually means

OFT rotates a frozen layer orthogonally, storing not the rotation but a small `theta` that builds a skew-symmetric `Q`, converted to an orthogonal `R` by the Cayley transform:

```
R = (I + Q)(I - Q)^-1
```

That inverse is expensive, so many implementations (including OFTv2's default) use a truncated Neumann series, exact only in the limit:

```
(I + Q)^-1 = I - Q + Q^2 - Q^3 + ...
=> R = (I+Q)(I-Q)^-1 = I + 2Q + 2Q^2 + 2Q^3 + ...
```

Converges only while `||Q|| < 1`, and is accurate only well below. `theta` (hence `Q`) grows during training, so a once-accurate adapter drifts into applying a matrix that is no longer the rotation it thinks it is — nothing crashes, no loss spikes, some fraction of blocks just runs a silently wrong transform.

That drift is "saturation", measurable at any checkpoint.

> Terminology warning: "saturation" means two unrelated things below — representational (how much of the rotation range is used) and numerical (how wrong the truncated series is). Only the second one breaks things. Do not conflate them.

---

## 2. The three numbers you need

Per block of every adapted layer:

| quantity | formula | meaning |
|---|---|---|
| `||Q||` | spectral norm (largest singular value) of the scaled skew matrix | how far from identity the rotation has travelled |
| rotation angle | `2 * atan(||Q||)` | the same thing in degrees, because `tan(theta/2) = ||Q||` for a skew matrix |
| relative error | `||R_approx - R_exact||_2 / ||R_exact||_2` | how wrong the truncated series is |

The angle is the interpretable one: 0° = untouched, 180° = maximally rotated. A median of 30° means the adapter is using about a sixth of its range.

---

## 3. Reference code

```python
import math
import torch

def nel_to_block_size(n_elements: int) -> int:
    """Recover the block size b from theta's second dimension, n_elements = b(b-1)/2."""
    b = int(round((1 + math.sqrt(1 + 8 * n_elements)) / 2))
    if b * (b - 1) // 2 != n_elements:
        raise ValueError(f"{n_elements} is not a triangular number")
    return b

def cayley_exact(q: torch.Tensor) -> torch.Tensor:
    """R = (I+Q)(I-Q)^-1 for a batch of skew matrices, shape (r, b, b)."""
    eye = torch.eye(q.shape[-1]).expand_as(q)
    return torch.linalg.solve(eye - q, eye + q)

def cayley_neumann(q: torch.Tensor, n_terms: int) -> torch.Tensor:
    """The truncated series as OFTv2 builds it: I + 2Q + 2Q^2 + ... + 2Q^(n-1) + Q^n."""
    eye = torch.eye(q.shape[-1]).expand_as(q)
    out = eye.clone().add(q, alpha=2.0)
    if n_terms > 2:
        qp = torch.bmm(q, q)
        out.add_(qp, alpha=2.0)
        for _ in range(3, n_terms):
            qp = torch.bmm(qp, q)
            out.add_(qp, alpha=2.0)
        qp = torch.bmm(qp, q)
        out.add_(qp)
    return out

def block_stats(theta: torch.Tensor, block_size: int, oft_scaled: bool, n_terms: int = 5):
    """Per-block ||Q||, angle, and relative truncation error for one OFT module.

    theta: shape (r, n_elements), the raw parameter.
    """
    theta = theta.detach().float()
    r, n_elements = theta.shape
    b = block_size if block_size else nel_to_block_size(n_elements)

    # THE scaling that is easy to forget -- see section 7.
    if oft_scaled:
        theta = theta / (2.0 * math.sqrt(b - 1))

    # scatter the upper-triangle entries into a matrix and skew it
    rows, cols = torch.triu_indices(b, b, 1)
    m = torch.zeros(r, b, b).index_put((torch.arange(r)[:, None], rows, cols), theta)
    q = m - m.transpose(-2, -1)

    exact = cayley_exact(q)
    approx = cayley_neumann(q, n_terms)

    norm_q = torch.linalg.matrix_norm(q, ord=2)
    rel_err = (torch.linalg.matrix_norm(approx - exact, ord=2)
               / torch.linalg.matrix_norm(exact, ord=2))
    return norm_q, rel_err, exact, approx

def summarise(norms, errs):
    s = sorted(norms.tolist())
    e = sorted(errs.tolist())
    pct = lambda xs, p: xs[min(len(xs) - 1, int(p * len(xs)))]
    ang = lambda x: math.degrees(2.0 * math.atan(x))
    print(f"blocks            {len(s)}")
    print(f"||Q||   p50 {pct(s,.5):.3f}  p90 {pct(s,.9):.3f}  max {max(s):.3f}")
    print(f"angle   p50 {ang(pct(s,.5)):.1f} deg  max {ang(max(s)):.1f} deg")
    print(f"error   p50 {pct(e,.5):.2e}  p90 {pct(e,.9):.2e}  max {max(e):.2e}")
    for th in (0.01, 0.05, 0.20):
        print(f"  blocks with >{th*100:4.0f}% error: {sum(x > th for x in e)}")
```

Loop it over every module in the model and pool the blocks:

```python
norms, errs = [], []
for module in oft_modules:                       # however you enumerate them
    n, e, _, _ = block_stats(module.theta, module.block_size,
                             module.oft_scaled, n_terms=5)
    norms += n.tolist()
    errs += e.tolist()
summarise(torch.tensor(norms), torch.tensor(errs))
```

Run it on CPU from a checkpoint — no GPU, no forward pass.

---

## 4. What counts as bad

Read the angle for representational health, the error for correctness.

### `||Q||` / rotation angle (per block)

| range | angle | reading |
|---|---|---|
| `< 0.3` | `< 33°` | healthy, lots of range left |
| `0.3 – 0.6` | `33° – 62°` | adapter is working hard; still fine with exact solve |
| `0.6 – 1.0` | `62° – 90°` | truncated series is seriously wrong here; exact solve still fine |
| `> 1.0` | `> 90°` | past half-turn; any truncation is meaningless |

Median 30° is perfectly healthy — OFT is supposed to rotate; a slowly rising median is normal. A median rising linearly without bound means the rotation LR is too high.

### Truncation error (per block)

| relative error | reading |
|---|---|
| `< 0.1%` | indistinguishable from exact |
| `0.1% – 1%` | fine |
| `1% – 5%` | borderline; the block is not the rotation it should be |
| `5% – 20%` | real corruption |
| `> 20%` | the applied matrix is materially a different transform |
| `> 100%` | not a rotation at all any more |

### The headline rule

> The median lies. Judge the tail.

In our runs the median block was accurate to 0.08% while the worst block was 35% wrong. Report only mean or median and you conclude everything is fine until ~2% of your capacity is quietly broken. Always report:

1. `p50`, `p90`, `p99`, `max` of the error, and
2. the count of blocks above 1% / 5% / 20%.

Trigger: more than ~1% of blocks above 20% error → fix the series.

---

## 5. Exact is usually better, and often free

Exact Cayley (`torch.linalg.solve`): zero truncation error, no threshold to reason about. The only question is cost, and the answer is usually "less than you think".

Measured on an RTX 4090, block size 128, 73 OFT modules, full training step at batch 48 (SinkSGD_ADV, fp32 head), compiled:

| variant | ms/step | img/s | peak VRAM |
|---|---|---|---|
| Neumann 5 terms | 388 | 123.7 | 17.2 GB |
| Neumann 15 terms | 423 | 113.5 | 17.8 GB |
| exact solve | 426 | 112.7 | 17.2 GB |

Exact costs the same as 15 terms (−8.9% throughput vs 5 terms) and used less VRAM. Unless you are desperate for that last 9%, take exact — it permanently deletes a class of silent numerical bug.

Two measurement traps:

Measure compiled, not eager. Eager showed the rotation math at 79 ms/step — 25% of the step — because it is ~500 tiny kernel launches and Windows launch overhead dominates. `torch.compile` fuses them; true cost is a fraction of that
Cost scales as `d·b²` for the rotation alone, negligible against a ViT backbone (~0.6 GFLOP vs ~505 GFLOP at `b=32`; ~1.9% at `b=128`). The bottleneck is launch count, not arithmetic

If you do stay truncated, `n=15` is the practical floor. Measured at `b=32`, step 10k:

| terms | p50 error | max error | blocks >1% | >5% | >20% |
|---|---|---|---|---|---|
| 5 | 7.7e-4 | 35% | 179 | 53 | 3 |
| 10 | 6.1e-7 | 12.3% | 17 | 1 | 0 |
| 15 | 1.8e-7 | 4.3% | 1 | 0 | 0 |
| 20 | 1.8e-7 | 1.5% | 1 | 0 | 0 |

### There is a hard wall, and it is at `||Q|| = 1`

Adding terms is not a universal fix: the geometric series converges only for `||Q|| < 1`, and near that boundary it converges slowly. Measured, block size 32, 64 random blocks, 20 terms versus exact:

| `||Q||` max | rotation angle | Neumann-20 vs exact |
|---|---|---|
| 0.41 | 44.6° | 2.4e-07 |
| 0.99 | 89.6° | 0.176 (17.6%) |
| 1.82 | 122.5° | 4.7e+04 |
| 3.23 | 145.6° | 4.1e+09 |

So the practical rule is sharper than "use 15 terms":

`||Q|| < 0.6` (angle < 62°): a truncated series with enough terms is fine
`||Q|| > 0.8` (angle > 77°): no term count saves you. Past the radius the series diverges; use exact solve

Exact solve is the right default for long runs: rotations grow, and growth walks you straight at the wall.

### The convention bug to check for

Some implementations ship an exact branch that computes the inverse rotation:

```python
# WRONG when it is meant to replace the Neumann series above:
return torch.linalg.solve(eye + q, eye - q, left=False)   # (I-Q)(I+Q)^-1
```

That solves `X(I+Q) = (I-Q)`, giving `(I-Q)(I+Q)^-1` — the transpose of `(I+Q)(I-Q)^-1`, which is what the series approximates. Different rotations: we measured them differing by 1.13 in spectral norm, i.e. completely different. Flipping that flag silently transposes every rotation in the model and destroys a trained adapter. Verify the two branches agree before trusting the switch:

```python
exact   = cayley_exact(q)
approx  = cayley_neumann(q, n_terms=20)
assert (exact - approx).abs().max() < 1e-3
```

Run that guard with `||Q|| < 0.5` (a freshly initialised adapter, or a random skew matrix scaled down). At `||Q|| > 1` it fails for a legitimate reason — the series genuinely cannot converge there — and you will mistake a maths fact for a bug.

---

## 6. Block size: bigger is usually better, but it raises the stakes

`block_size` = size of each orthogonal block. `b=32` in a 768-dim layer gives 24 independent 32×32 rotations — the rotation can never mix information across those groups: a hard expressiveness ceiling, lifted cheaply by widening the blocks.

Parameters scale as `d·b/2`, so 32 → 128 quadruples them:

| block size | params per 768-d layer | adapter total (73 modules) |
|---|---|---|
| 32 | 11904 | 1.30M |
| 64 | 23808 | 2.60M |
| 128 | 47616 | 5.32M |

Throughput is flat. Measured, compiled, batch 48, otherwise identical:

| block size | ms/step | img/s | peak VRAM |
|---|---|---|---|
| 32 | 392 | 122.4 | 17.0 GB |
| 64 | 394 | 121.8 | 17.2 GB |
| 128 | 388 | 123.7 | 17.3 GB |

128 was marginally fastest — halving the number of rotation calls saves more launch overhead than the extra arithmetic costs.

Your LR does not need rescaling. The usual scaling is `theta / (2*sqrt(b-1))` and the spectral norm of a random skew matrix grows like `sqrt(b)`; the two cancel, so `||Q|| ≈ 0.7σ` regardless of `b`. If each element moves `lr` per step then `||dQ|| ≈ lr * sqrt(d/8)`, independent of `b`. Same rotation speed, same LR.

But bigger blocks saturate faster — more free parameters per block and larger reachable angles degrade the truncated series sooner. Measured at step 10k, exact vs 5-term:

| | blocks >20% error | max error |
|---|---|---|
| `b=32` | 3 (0.1%) | 35% |
| `b=128` | 12 (1.8%) | 120% |

The wider you go, the less you can afford a truncated Cayley. Widen blocks → go exact or go home.

---

## 7. Gotchas that cost us time

Do not forget the `oft_scaled` divisor. Reconstructing `Q` from raw `theta` without the `/(2*sqrt(b-1))` factor inflates every norm by ~11× at `b=32` and makes a fine adapter look catastrophically saturated
Compare against the right exact form. `(I+Q)(I-Q)^-1` and `(I-Q)(I+Q)^-1` are transposes, not the same matrix. Backwards turns a healthy 0.08% error into a reported 90%
A short run looks fine. At 5000 steps the median error was 0.02%; the damage arrived with training. Check multiple checkpoints and watch the trend, not one snapshot
The tail grows faster than the median. 10k → 20k steps: median 0.077% → 0.46%, blocks above 20% error 12 → 44. Median-based monitoring reports "fine" throughout
Gauge is CPU-only. Load with `map_location="cpu"`; no forward pass, no GPU memory, safe to run while training continues

---

## 8. Interpretation beyond correctness

Once the series is exact, the angle curve is a diagnostic about your adapter, not just numerics:

Median angle growing steadily, staying under ~45° — adapting without thrashing
Median angle climbing towards 90°+ — the rotation is doing enormous work. Ask why. Here the backbone was frozen, so OFT burned its entire rotation budget compensating for features that could not change: strong argument for unfreezing the last few blocks instead
A sudden jump in `||Q||` for a few blocks — those layers are fighting a loss they cannot represent, or the rotation LR is too high for them

Error tells you whether the maths is right; angle tells you whether the architecture is right.

---

## 9. Suggested cadence

| when | what |
|---|---|
| once, at setup | verify `exact ≈ neumann(20 terms)` (catches the transpose bug) |
| every checkpoint | `||Q||` p50/p99/max, angle p50/max, error p50/p99/max, counts >1/5/20% |
| if truncated | also record the trend; fix at >1% of blocks past 20% |
| before trusting a long run | confirm the curve is not growing linearly without bound |
