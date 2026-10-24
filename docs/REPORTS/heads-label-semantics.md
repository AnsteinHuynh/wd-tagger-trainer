# Label-semantics heads (text embeddings, co-occurrence, hierarchy)

DECAF ([2108.00368](https://arxiv.org/abs/2108.00368), WSDM 2021) is the strongest evidence for label features: 1-vs-all classifiers whose weights are built from label text. "Up to 2–6% more accurate than leading extreme classifiers" on LF-AmazonTitles-1.3M and friends; the gain is concentrated in data-scarce tail bins — the "shared semantic prior instead of per-label estimation" mechanism
CLIP-Decoder ([2406.14830](https://arxiv.org/abs/2406.14830), ICCVW 2024) + ML-Decoder with CLIP text queries: +3.9 zero-shot, +2.3 generalized zero-shot mAP
The catch for us: both work where labels have few or zero examples. Our median tag has thousands of positives, so the text prior solves a problem we mostly do not have — and it must be mapped into DINOv3's feature space by a learned linear map, which itself needs the data the scarce labels lack. Plausible, unproven, probably small: rank 3. Cheap to try: `I:\Models\siglip2-base-patch16-224` and
`metaclip-2-worldwide-b16` are on disk; encoding 12476 tag strings is minutes.
Label co-occurrence graphs / GCN heads: dominated. ML-GCN scores 83.0 mAP on COCO against ASL's 86.6 in ASL's own table — graph machinery loses to a loss function, and at 12k the graph is a 12476² object
Hierarchical label structure: real (danbooru has a taxonomy) but no measured vision result at 10k+ showing macro-F1@fixed-threshold gains over a well-trained flat head. Plausible, unproven, medium effort. Do not start here
