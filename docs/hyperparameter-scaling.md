# Whiteboard: optimal training parameters for ViT-B, extrapolated to ViT-L

Written 2026-10-21, before spending any more GPU. Tags: M = measured here, I = inferred from a scaling rule, U = unknown / never bracketed

## 1. What "optimal" can even mean here

Objective (AGENTS.md): Macro@0.40 and Micro@0.40 against `convnextv2_huge.dbv4-full`, same pixels; best checkpoint vs the bar on the 50000-image nested cache at 0.40

| metric | ours (86241) | bar | ratio |
|---|---|---|---|
| Macro F1 | 0.2815 | 0.5579 | 50.4% |
| Macro MCC / P / R | 0.3062 / 0.4608 / 0.2266 | 0.5658 / 0.6201 / 0.5333 | 54 / 74 / 43% |
| Micro F1 | 0.6133 | 0.6886 | 89.1% |
| Micro MCC / P / R | 0.6179 / 0.7068 / 0.5417 | 0.6880 / 0.7033 / 0.6746 | 90 / 100.5% / 80% |

Bar: 97% of final macro in epoch 1 (0.5634 → 0.5792 over 96 more epochs, its tfevents) M. Ours: 0.2815 macro on the same single pass over the same 5.32M images M. Gap = per-sample efficiency at equal budget, so more steps or a better cosine cannot halve it. Bar's edge: capacity (692.6M params vs our 123.5M trainable) and/or head design (3072-d pooled vector → 12476 labels). Which is the wall is U; it decides whether ViT-L is worth 45 hours

## 2. ViT-B: what we know per knob

Config of the current run (`runs/4a13a120/config.json`)

| knob | value | evidence | tag |
|---|---|---|---|
| `batch_size` | 48 | VRAM-limited (~21 of 24.5 GB), not a free parameter. | M / constraint |
| `lr_oft` | 5e-4 | Bracketed both sides: 0 (frozen) → 0.3279 micro; 5e-4 → 0.4142; 2e-3 → 0.3804; 2e-2 → 0.1427 (destroyed). Old head LR. | M (old regime) |
| `lr_head` | 1.2e-4 | 4-point bracket: 4e-3 → 0.3888, 2.5e-4 → 0.4048, 1.2e-4 → 0.4142, 6e-5 → 0.4068. The 0.004 default was 33x too high; old regime. | M (old regime) |
| `lr_unfreeze` | 3e-5 | Never bracketed. Highest uncertainty, and the knob that matters most for ViT-L. | U |
| `momentum` | 0.95 | 0.95 > 0.99; 0.998 catastrophic. | M |
| `weight_decay` | 0.0 | 1 pass, underfit; the bar used 1e-3 with mixup 0.6. | I |
| `warmup_steps` | 50 | 0.045% of 110870 steps — essentially none; schedules want 0.5-2%; we have been bitten by NaN from unstable early steps. | U (suspicious) |
| `block_size` | 128 | Sets the rotation block; exact Cayley at `cayley_terms=0`. | M |
| `cayley_terms` | 0 | Neumann truncation diverges once ‖Q‖ > 1; exact solve is required. | M |
| `unfreeze_blocks` | 12 (all) | No bracket against 6/9. | U |
| `steps` | 110870 (1 pass) | Converged: last four checkpoints flat (0.2563 / 0.2601 / 0.2594 macro). | M |
| `compile_model` | default | 116.3 img/s (`bench_ref.py`, reference config) vs 92.8 eager = 1.25x; the old 150-170 vs 89 = 1.7x figure was batch-32/block-32 `cls_mean`, not this model. | M |
| `grad_clip` | 1.0 | | — |
| geometry | squash | Settled and not a lever: pad 0.4078 vs squash 0.4142/0.4186; per-image F1 vs distortion r = +0.047. | M |

Verdict: locally optimal in every tested knob; two defects — a 50-step warmup too short, and `lr_unfreeze` never tested. Neither explains a 2x macro gap. v2 if warranted: `warmup_steps 2000`, bracket {1e-5, 3e-5, 1e-4}, else held — single-digit % on micro, not macro

## 3. The extrapolation itself

### Scaling factors from ViT-B (12 layers, d=768, batch 48) to ViT-L (24 layers, d=1024, batch ~20)

| factor | rule | value | why |
|---|---|---|---|
| depth 12→24 | LR x 1/sqrt(2) = 0.707 | 0.71 | 2x residual depth compounds per-layer rotation error; deeper nets want lower LR |
| width 768→1024 | LR x 1/sqrt(1.33) = 0.866 | 0.87 | per-parameter step held roughly constant |
| batch 48→20 | LR x sqrt(20/48) = 0.645 | 0.65 | sqrt, not linear: SinkSGD_ADV is normalized (Adam/Muon-family), its step size is ~lr regardless of gradient magnitude, so linear scaling overshoots |
| combined | 0.71 x 0.87 x 0.65 | 0.40 | |

### Proposed ViT-L config (bracket centres to test, not laws)

| knob | ViT-B | ViT-L | arithmetic |
|---|---|---|---|
| `lr_oft` | 5e-4 | 2.0e-4 | 5e-4 x 0.40; corroborated: the abandoned ViT-L probe NaN'd at 5e-4 (theta grew until the Cayley rotation saturated), so 5e-4 is a hard upper bound and 2e-4 sits 2.5x below it. |
| `lr_head` | 1.2e-4 | 7e-5 | 1.2e-4 x 0.87 (width, through the 4096-d pool) x 0.65 (batch); one linear layer, so weaker depth dependence than the adapter. |
| `lr_unfreeze` | 3e-5 | 1.2e-5 | 3e-5 x 0.40; highest uncertainty — scale by bracket, not formula. |
| `batch_size` | 48 | 20 | VRAM: measured 19.3 GB at batch 20. Ceiling, not optimum. |
| `momentum` | 0.95 | 0.95 | Optimizer property, dimension-independent. |
| `weight_decay` | 0.0 | 0.0 | Same single-pass reasoning. |
| `warmup_steps` | 50 | 4000 | 1.5% of 266086 steps; bigger model + lower LR + no warmup experience at this depth. |
| `block_size` | 128 | 128 | d=1024 divides into 8 blocks of 128. |
| `cayley_terms` | 0 | 0 | Exact solve only. |
| `unfreeze_blocks` | 12 | 24 (all) | Or bracket 12/18/24 — unfreezing 24 blocks at 512px is the dominant VRAM cost. |
| `steps` (1 pass) | 110870 | 266086 | 5321713 / 20; steps define the schedule, so the cosine stretches 2.4x. |
| `compile_model` | max-autotune | max-autotune | |
| OFT module count | 73 | ~145 | Doubling this doubles the per-tensor optimizer cost, see below. |

Bracket to run for `lr_oft`: {1.5e-4, 2.0e-4, 3.0e-4} — failure mode is silent divergence, not a plateau: verify, do not assume

## 4. The feasibility wall

Probe: ViT-L, batch 20, 512px, 19.3 GB, ~33 img/s. ViT-B same geometry: 116.3 img/s (reference config, exact Cayley, compiled) = 3.5x slower per image, not the 4.9x quoted against a stale 150-170 (old batch-32); live ViT-B ~109 img/s → 3.3x

| run | steps | wall clock, 1 pass | wall clock, 1M-image subset |
|---|---|---|---|
| ViT-B, batch 48 | 110870 | ~12.7 h (at 116.3 img/s) | ~2.4 h |
| ViT-L, batch 20 | 266086 | ~44.8 h | ~8.4 h |

Earlier ~9.2 h ViT-B was 38% optimistic (160 img/s, never measured; `docs/benchmark.md` traps). Optimizer tax: ViT-L ~145 OFT modules x ~1.2 ms/tensor eager = ~175 ms/step against ~600 ms = 29% (ViT-B ~9%); `compiled_optimizer=True` → ~28 ms/73 tensors, keep it on. Full-corpus ViT-L = ~2-day single pass, and nothing shows one pass suffices (the bar needed one, but it is convolutional, 5.6x our parameters, different loss)

Cheaper ways down:
384px: tokens 1024→576, ~1.78x faster → 25 h/pass; costs accuracy vs a 512px bar
Subset (1M): 8.4 h/pass, drops 81% of the corpus, reopening the data-coverage wall in `docs/label-integrity-audit.md` (labels at 1-9x scored 0.0008) — macro suffers most
Activation checkpointing: larger batch, amortises the optimizer tax, but `torch.compile` x checkpointing produced NaN backbone gradients here

## 5. Recommendation: buy information before buying compute

The discriminating experiment costs ~5 h and is image-matched (`stream_images` + `seed`: same seed, `pass_index`, first N images); a 45-hour run bets capacity is the wall without it

| arm | config | images | est. time |
|---|---|---|---|
| A | ViT-B, current best config, `stream_images=500000` | 500k | ~0.9 h |
| B | ViT-L, table above, `stream_images=500000` | 500k | ~4.2 h |

Both scored at 0.40 on the same 50k cache. Capacity elasticity d(macro)/d(log params): B near the bar's ~0.55 macro on 500k → capacity is the wall, full run justified; B near A → not, save 45 h, fix the head/adapter. Either beats a fifth ViT-B checkpoint

## 6. Open items that are not hyperparameters

Test split: AGENTS.md says "with validation test"; our nested cache is the val split, the bar publishes Val and Test. A 50k nested test cache is the same pipeline (12 tars / 295926 images, ~39.3 GB, ~1 h) and would show whether we overfit the val split we tune on
Macro/Micro want opposite thresholds (macro-optimal 0.05, micro-optimal ~0.28) — irrelevant at fixed 0.40, but it explains our precision-heavy look (macro P 0.46 / R 0.23)
`lr_head` / `lr_oft` were bracketed in the old regime (frozen backbone, 10k images, head 0.4142 micro); now full corpus, all blocks unfrozen, 0.6133 micro. Re-bracket `lr_head` — the only knob with a proven history of being 33x wrong

## 7. THE ARM TAXONOMY HAS A BUG

Run `4a13a120` has `unfreeze_blocks=12` and `freeze_oft=false`: backbone and OFT rotations both train. Pathological cross-term: with a layer's weights free, `R @ W` spans every matrix, so the 5.32M adapter parameters are redundant with the 86M backbone. Arm C (whole backbone + frozen OFT + head) could not run:

```python
if cfg.freeze_oft:
    for p in oft_params: p.requires_grad_(False)
    groups = [{"params": head_params, "lr": cfg.lr_head}]   # blocks silently dropped
```

`block_params` never reached `groups` and `bases` matched, so `freeze_oft=True` + `unfreeze_blocks>0` trained the head only — a no-op that looked configurable. Fixed in both places: with `freeze_oft=True` and unfrozen blocks the optimizer now gets `[block_params @ lr_unfreeze, head_params @ lr_head]`. Consequences: `freeze_oft=True` + `init_from=<converged ckpt>` freezes learned rotations, not identity, so arm C needs a fresh run, not a probe from 86241; no config hash changes (past runs had `freeze_oft=False` or `unfreeze_blocks=0`)

## 8. Plan: perfect ViT-B, then move the settings to ViT-L with a 5k-step test

`--init-from <dir>` warm-starts weights+head from that dir's highest checkpoint (`runs/4a13a120` → 86241), fresh optimizer state and schedule; `--stream-images 240000` at batch 48 = exactly `total_steps = 5000`. One probe: 240k images at ~160 img/s = 25 min

### Stage A — four 5k-step probes from 86241 (~2 h total)

All image-matched, scored at 0.40 from cached probabilities (~5 s each):

| arm | change | what it answers |
|---|---|---|
| A1 | none (control) | how much macro does 5k more steps buy at all? |
| A2 | `lr_unfreeze` 3e-5 → 1e-5 | drift says 3e-5 is hot: 11.18% in 20k steps, `o_proj` block 0 at 40.5%, extrapolating to ~15% by 110k |
| A3 | `lr_unfreeze` 3e-5 → 1e-4 | is 3e-5 far below the optimum? |
| A4 | `freeze_oft=true` | is the adapter redundant now that the backbone trains? (the taxonomy cross-term) |

### Stage B — one full run at the winning settings (~9.2 h)

110870 steps, 1 pass, `max-autotune`. If A2/A3 show an effect this confirms it; if all four arms are within noise, ViT-B is converged and only the capacity argument (§5) remains

### Stage C — ViT-L, 5k steps, identical settings (~1 h)

ViT-B's LRs, unscaled. `stream_images = 100000` at batch 20 = exactly 5000 steps = 50 min at 33 img/s. Pass: finite loss, theta bounded (no Cayley saturation), macro@0.40 at 5k within reach of ViT-B's A1/A2 probe. The earlier ViT-L probe NaN'd by step 500 at `lr_oft` 5e-4 (theta growth making `(I - Q)` near-singular); fallback: §3's 0.40x scaling (`lr_oft` 2e-4, `lr_head` 7e-5, `lr_unfreeze` 1.2e-5) plus one more 5k probe, 1 more hour

Honest limitation: checkpoint-to-checkpoint spread over 2257 steps was under the seed noise floor, so a 5k probe resolves a 10x LR difference but not a 2x one. Stage A ranks large effects and detects NaN; it cannot pick between 6e-5 and 1.2e-4
