# Limitations of this review

No published head/loss comparison exists on long-tailed macro F1 at a fixed threshold with ≥10k labels and a frozen ViT backbone. ML-Decoder's 9.6k table (mAP) and BALMS's 18.2k table (macro F1, text) are the two closest anchors, and both require extrapolation
BALMS's threshold is micro-F1-optimal on val, not 0.40; their absolute numbers are not transferable, only the ordering of losses
The Decoupling/Menon evidence is single-label softmax; the multi-label BCE analogue is a mechanistic argument, not a measured result
Local data anomaly worth a look: for ckpt 086241 / 50k, `runs/4a13a120/validation.json` reports micro F1 0.6133 while `runs/_sweep_last3.json` reports 0.6235 at the same threshold (macro agrees to 4 decimals, 0.2814/0.2815). One path is scoring slightly different rows or labels; it changes the headline micro number by 1.6% relative
Out of scope but cheap: `docs/oft-saturation.md` exists, and an over-rotating adapter would corrupt features silently. It would hurt head classes as much as tail, so unlikely to explain a tail-specific deficit — but a 15-minute check on an existing checkpoint
