# Anything newer (2024–2026) worth knowing

Sparse / MoE heads: no credible measured evidence for vision tagging at 12k labels. The multi-label MoE literature is either old (2014) or text prompt-learning for long-tailed MLC with small label sets. Unproven; a MoE head also multiplies parameters exactly where we have shown capacity is not the bottleneck
Per-label adapters: same verdict — parameter count where capacity is demonstrably not the differentiator
"Any-label" / zero-shot heads: the honest version is "label-text classifier + linear", i.e. DECAF/CLIP-Decoder above. It pays where labels are unseen; we are fully supervised on all 12476. It could pay as a tie-breaker for the ~1–2k tags that have handfuls of positives — worth measuring, not worth betting the run on
What actually moved the needle 2024–2026 in this task family: bigger/better backbones and cleaner data, not heads. Our own docs already conclude the gap is per-sample efficiency
