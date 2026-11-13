# The ensemble we actually want: Dino adapted, cached trained-SigLIP features frozen

Written 2026-10-26 because an assistant built the wrong version twice and the user had to say it again. This file is the intent of record. If a run disagrees with this text, the run is wrong.

## What the user asked for

Train the ensemble of a DINOv3 tower whose last block is unfrozen and whose blocks carry OFT adapters, using the cached SigLIP2-plus-OFT embeddings as the second view. The cached embeddings, not a recomputed tower.

## The two sides

Adapted side: DINOv3-B or DINOv3-L loaded from `models\dinov3-*`, OFTv2 adapters in every transformer block including the patch-embed convolution, block 128, exact Cayley, and `unfreeze_blocks: 1` so the last block's base weights train at `lr_unfreeze` while its rotation stays at identity. This is the production recipe's front tower.

Frozen side: the cls vector of the trained SigLIP2-B trunk from `runs/459e2a02` at step 16000, meaning the base SigLIP2 weights with the 76 trained OFT rotations baked in, pooled as cls only, 768 wide, float16. It exists on disk as 213 per-tar shards at `cache/wd-tagger-trainer\feat\siglip2-cmss-shards`, 5321713 rows, 32.7 GB, built at seed 1 and pass index 0. Those shards hold four pools in the order cls then mean then max then std, each 768 wide, so cls is columns 0 to 768 and the other three are ignored. `the private experiment log` is why: cls wins or ties at every depth and loss where pooling was varied, including a four-decimal tie at production scale, and `docs/best-settings.md` prescribes `head_pool: cls`.

## Why the cache and not a live tower

The features already exist for every training image, so the training step reads them instead of paying a second tower's forward pass, which was about 30 percent of step time in the live version. The cache also carries four pools rather than the single 768-wide pooler output, so the head sees 3072 static dimensions plus Dino's own. And it is the exact feature set that already scored 0.3840 macro on the 50k protocol as a head, which is the strongest frozen view this project owns.

Two live versions were built and both are wrong for this intent. The first loaded the raw pretrained SigLIP2 with no adapter, so the second view was a base model whose features were never trained on our labels. The second baked the trained checkpoint into a live frozen tower, which produces the right numbers but pays the forward and still only offers cls. Neither is what was asked for.

## What has to hold for the cache to be usable

The cache must be in the same order as the training stream: seed 1, pass index 0, tar order from `manifest.order`, file order inside each tar. The shard metadata records `seed: 1` and `pass_index: 0` and a `source_start` per shard, so the concatenation of shards is the stream's order, one row per image. Any mismatch silently trains the head on another image's features, which is worse than a crash because it still looks like learning.

The alignment must therefore be proven, not assumed: for a handful of known positions, compare the cached row against a live forward of the same trained trunk on the same image. The first 768 columns must equal the trunk's cls pooler output to float16 rounding.

Resume must seek the cache to `n_consumed`, which is exact because one streamed image is one row.

## Evidence this is worth doing

At 261 steps and 100k images, on the 50k protocol at fixed 0.40, the live ensemble of DinoV3-B with the raw base SigLIP2 second tower scored macro 0.1011, against 0.0493 for SigLIP2-B alone and 0.0327 for DinoV3-B alone. The gain is 2.05 times the best single tower and it lands in narrow tags, which is the column our 0.1739 macro gap to the bar lives in.

A head on the frozen production features alone, four pools, scores 0.3840 macro on the same 50k protocol. That is the frozen view being handed to the ensemble, and it is a far stronger starting point than the base model's.

## Open questions this does not settle

Whether the frozen contribution decays as the adapted tower learns. Every earlier frozen-tower probe did decay, and the record's conclusion is that adaptation beats a second tower's features. The live ensemble's lead at 261 steps is therefore a starting position, not a forecast, and the run's own validation curve is the only honest test.

Whether the adapted side should be DinoV3-L. L measures 34 img/s at 512 against 88 for B, so an L ensemble is a two-day arm at the same schedule, and L has no pooling or ensemble measurement of its own.
