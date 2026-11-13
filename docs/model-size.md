# Model size — what is inside DINOv3 ViT-B/16, block by block

Every freeze-depth decision needs to know what one block costs. Read from the safetensors header of `models\dinov3-vitb16-pretrain-lvd1689m\model.safetensors` (211 tensors, names and shapes only, no weights loaded), and cross-checked against the trainer's own banner and against the optimizer state in `runs/83231ec3/ckpt_step001000.pt`.

## The whole backbone: 85660416 parameters

| part | params |
|---|---:|
| patch embed, conv 3x16x16 to 768 | 590592 |
| patch embed bias | 768 |
| cls_token | 768 |
| mask_token | 768 |
| register_tokens, 4 of them | 3072 |
| 12 transformer blocks | 85063680 |
| final norm weight and bias | 1536 |
| total | 85660416 |

## One block, and all twelve are identical: 7088640

| part | params | share |
|---|---:|---:|
| q_proj 768 to 768 | 589824 | |
| k_proj 768 to 768, no bias | 589824 | |
| v_proj 768 to 768 | 589824 | |
| o_proj 768 to 768 | 589824 | |
| q, v and o biases | 2304 | |
| attention subtotal | 2361600 | 33.3% |
| mlp up 768 to 3072 plus bias | 2362368 | |
| mlp down 3072 to 768 plus bias | 2360064 | |
| mlp subtotal | 4722432 | 66.6% |
| norm1 weight and bias | 1536 | |
| norm2 weight and bias | 1536 | |
| layer_scale1 and layer_scale2 | 1536 | |
| block subtotal | 7088640 | 100% |

Three details separate this tower from a textbook ViT-B, and they matter when wrapping adapters: the projections are separate q, k, v rather than one fused qkv, and k carries no bias; the MLP is two matrices with no gate, so the 3072 width sits in the up projection; every block carries two LayerScale vectors beside its two LayerNorms. That is 211 tensors and 85660416, not a round 86M.

## What the OFTv2 adapters add

At `block_size` 128 the rotation stores one free parameter per upper-triangle entry, 8128 per 128-row block, so a module's cost is its input width divided by 128, times 8128.

| module | params each | count | total |
|---|---:|---:|---:|
| q, k, v, o_proj (in 768) | 48768 | 48 | 2340864 |
| mlp.up_proj (in 768) | 48768 | 12 | 585216 |
| mlp.down_proj (in 3072) | 195072 | 12 | 2340864 |
| patch embed conv (in 768) | 48768 | 1 | 48768 |
| total, 73 modules | | | 5315712 |

One block's adapters are 438912, which is 6.2% of that block's own 7088640. All of them together are 0.75 of one block.

## Live parameters per freeze depth

Only the rotations outside the unfrozen blocks receive a gradient: `unfreeze_last_blocks` freezes `oft.weight` inside every block it unfreezes, because a rotation beside an already-trainable weight is redundant. So the adapter's live share shrinks as blocks are unfrozen.

| arm | blocks | OFT live | head | live total |
|---|---:|---:|---:|---:|
| u12 | 85063680 | 48768 | 9595580 | 94708028 |
| u6 | 42531840 | 2682240 | 9595580 | 54809660 |
| u4 | 28354560 | 3560064 | 9595580 | 41510204 |
| u0 | 0 | 5315712 | 9595580 | 14911292 |

`u12` is the arm that just ran as `runs/83231ec3`, and its optimizer state confirms the arithmetic: group 0 holds exactly one tensor of 48768 parameters, the patch-embed rotation, while the other 5266944 rotation parameters sit on disk with no optimizer entry at all.

## Scale, in units that make the trade visible

One block is 7.09M, which is 74% of the cls head (9.60M) and 18% of the cls_mean_max_std head (38.34M including its LayerNorm).

Frozen blocks still cost full forward FLOPs, so speed does not scale with live parameters. Going from 12 unfrozen blocks to none removes the backward pass and the optimizer state of those blocks, which is the measured 109-116 to 160 img/s, not the 6.4x that 94.7M against 14.9M would suggest. Use `docs/benchmark.md` for throughput and this file only for parameter counts.
