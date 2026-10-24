# Precision

bf16 autocast, no GradScaler (`train.py:688-702`: "its exponent range matches fp32, so it needs no GradScaler"; default `amp="off"` at `241`, real runs pass `bf16`). → Same as yours
The fp32 island (`oft.py:232-239`, `243-260`): autocast suspended around the Cayley solve and the rotation einsum, because "casting the tensors alone is not enough, the autocast policy would downcast the matmuls right back". `oft_weight_dtype="float32"` — "only ~1.3M params … costs essentially nothing" (`train.py:182-185`), the same 1.3M you have. → High, correctness+speed pair. The einsum left in fp32 is the apply — lowest arithmetic intensity
Loss kept fp32: forward under autocast, `loss_sum.float()` before mean/backward so "the scale of the update never depends on the autocast dtype" (`train.py:1585-1588`). Low, free
Frozen tower loaded in `float32`, cast per-op by autocast (`train.py:944`, `1095`; `backbones.py:521`); bf16 loading never tried. → Gap: a frozen 87M backbone in bf16 removes a per-op weight cast and halves resident weights; worth one A/B
No TF32/cudnn flags in `src/` (zero grep hits) despite their A/B calling it the biggest flag lever (+14.1%, `docs/startupflags-costs.md:10-11`) — they leave fp32 matmuls on the table. → Medium for your fp32 islands (optimizer math, Cayley/rotation apply, fp32 loss path); one env var, one A/B
Channel-last: never used, never measured. Correct for a ViT (one patch-embed conv)
