# Arm taxonomy — what stacks, what doesn't

`h1..h13` are experiment labels, not variants. Each one changes a single factor against the control and holds everything else fixed. They look mutually exclusive because as arms they are — you can only train one at a time — but the factors themselves are orthogonal and stack freely. h3 is "cls pooling instead of CMSS, BCE, 12 blocks". It is not "a head that replaces ASL".

## The four axes

| axis | symbol | values | mutually exclusive? |
|---|---|---|---|
| backbone freeze depth | `u{N}` | N = 0..12 unfrozen blocks (0 = frozen) | yes — one number per run |
| pooling | `P` | any subset of `C`ls / `M`ean / `X`max / `S`td, in that order | yes — one subset per run |
| head projection | `H` | `L` = linear (no hidden), or `H{n}` = MLP with hidden width n | yes — linear XOR MLP |
| loss | `loss` | `bce`, `asl(g)`, `ntr(l)`, `pw`, `csl(k)`, `splc` | one base loss, but see below |

Across axes: everything stacks. Any `(N, P, H, loss)` tuple is a legal arm. Freeze depth, pooling and projection do not conflict with the loss; the loss never conflicts with the head.

Within an axis: choices, not additions.
pooling: you pick a subset (`C`, `CM`, `CMSS`, `M`, …). Adding `X`/`S` is additive by design — that's a wider vector, not two competing mechanisms — but `CMSS` and `C` are rival arms
projection: `L` and `H{n}` are rivals. An MLP with a hidden layer is not "linear plus extras"
base loss: one at a time, with one real exception — see the half-losses below

## The half-loss idea (this is the part that was ambiguous)

ASL and NTR are not full rivals. They act on different branches:

| component | branch it shapes | composes with |
|---|---|---|
| ASL focusing + probability clip | positive (and easy-negative suppression) | NTR, CSL, SPLC |
| NTR λ negative tolerance | negative only | ASL, CSL, SPLC |
| CSL top-K unannotated dropping | negative only | ASL, NTR, SPLC |
| SPLC / Hill | negative only | ASL, NTR, CSL |
| `pos_weight` (per-class weighting) | whole loss, per class | nothing — measured 0.0097, and it is the family the literature rejects |

So `asl+ntr` is a legitimate, untested composition (our own — it appears in no paper). Whereas `pw` alongside anything is a bad idea, measured rather than assumed.

One safety rule: three negative-side relaxations stacked (NTR + CSL + SPLC) will inflate scores until macro precision collapses. The research is explicit that macro-P drops before macro-F1 follows it. Stack at most two, and log macro-P on every eval.

## Notation

```
u{N}-{pool}-{projection}-{loss}[+mod][-{tower}][seed]
```

Examples: `u12-CMSS-L-bce` · `u12-C-L-asl(g2.0)` · `u6-CMSS-L-asl(g2.0)+ntr(l2.0)` · `u1-C-L-asl(g2.0)-siglip2b`

The tower suffix is omitted for DINOv3-B/16, which was the only tower until 2026-10-22, so every name written before that date still means what it said. `-siglip2b` is SigLIP2-B/16-512 as the adapted tower. `+siglipb` inside the modifier slot is a different thing and must not be confused with it: it means a second frozen tower bolted beside a DINOv3 backbone, one `siglip_dir` field rather than a `backbone_dir` swap. Disambiguating rule: a tower suffix after the loss changes what is being adapted, a `+tower` modifier adds a frozen one that is never adapted.

This gives a fifth axis, and it is not orthogonal to the other four in cost terms: the tower determines the FLOPs per image and therefore what the freeze depth and batch can afford.

## Mapping of everything run or queued

| arm | notation | one-line meaning |
|---|---|---|
| h1 control | `u12-CMSS-L-bce` | baseline |
| h2 | `u12-CM-L-bce` | drop max+std |
| h3 | `u12-C-L-bce` | cls only (768-d, 9.58M head) |
| h4 | `u12-CMSS-H1024-bce` | MLP 1024 |
| h5 | `u12-CMSS-H2048-bce` | MLP 2048 |
| h7 | `u12-CMSS-H512-bce` | MLP 512 |
| h8 | `u12-CMSS-H4096-bce` | MLP 4096 |
| h6 | `u12-CMSS-L-bce[seed2025]` | noise ruler |
| h9 | `u12-CMSS-L-ntr(l2.0)` | negative tolerance only |
| h10 | `u12-CMSS-L-asl(g2.0)` | current champion |
| h11 | `u12-CMSS-L-pw` | class weighting (catastrophic) |
| h12 (queued) | `u12-M-L-bce` | the convnext bar's exact head |
| h13 (queued) | `u12-C-L-asl(g2.0)` | ASL × the nominal best head |
| h3 replicate (queued) | `u12-C-L-bce[seed2025]` | is h3's +4.8% real? |
| phase3 asl_u0/4/6 | `u{0,4,6}-CMSS-L-asl(g4.0)` | ASL × freeze depth |

Gamma is no longer a dial. The phase-3 g1/g3/g4 ladder was dropped when `loss_gamma` turned out never to have reached the loss: every ASL arm had been running at the function default 4.0 while its config claimed otherwise. Once the kwargs were wired, 4.0 stayed the setting (it is what the champion arms actually ran) and the ladder would have been three near-identical arms.

## The combinations that exist but were never run

The sweep was one-factor-at-a-time, so every cross is unmeasured:

ASL × best head → h13, queued as phase-3 arm `01_u12-C-L-asl-g4.0`
ASL × freeze depth → phase3 `asl_u{0,4,6}` (queued)
ASL × NTR → implemented as `asl_ntr` in `wdtag/losses.py` and unit-checked against `asl` (identical at λ=1), but no arm has been run: the composition is ours, not a paper's
ASL × the bar's mean pooling → `u12-M-L-asl` (not queued)
Best head × best loss × best depth → the actual candidate for the long run, and we have no evidence for any of its three coordinates interacting
