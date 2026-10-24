# Query / attention decoders

Query2Label ([2107.10834](https://arxiv.org/abs/2107.10834)): 91.3 mAP COCO; evaluated on COCO / VOC / NUS-WIDE / Visual Genome — tens to a few hundred labels. Zero evidence at 10k+, and per-label cross-attention over 1k tokens costs O(labels × tokens). It is the paper ML-Decoder was designed to replace
ML-Decoder: remove self-attention, use non-learnable queries, and group decoding (share one query across N/K labels) to keep parameters below a full FC layer. At 9.6k: +0.5 to +1.8 GFLOPs over a 5.8 GFLOP model (+10–30%) for +0.8 mAP
Successors: the 2024 line (e.g. CLIP-Decoder, [2406.14830](https://arxiv.org/abs/2406.14830)) bolts the ML-Decoder head onto CLIP text embeddings and reports +3.9 zero-shot / +2.3 generalized-zero-shot mAP — for unseen labels, nothing for a fully supervised 12k head

Verdict. Real but small, mAP-shaped, label-count-flat. Costs 6–10 h (per-group output projection, 100–400 queries, group decoding); expected effect at 0.40 is a macro shift in the +0.01 range against a 0.17 score-quality gap. Ranked last of things worth doing at all.
