# Things that are actively WRONG under a fixed global 0.40

1. Per-class / per-category threshold tuning, including the bar's own trick (general 0.38, character 0.51, rating 0.24 → macro 0.611 vs 0.580). Our per-tag oracle is 0.3879; the bar's tuned 0.611 is not comparable to our 0.2815 and must never be quoted as such.
2. P=R threshold selection, how wd-eva02/wd-swinv2 report (0.4772 / 0.4541): whatever threshold equalises P and R, typically far below 0.40, overstating fixed-threshold macro by ~30–40% relative.
3. Threshold-free ranking losses (ZLPR, LSEP, AUC-style): an ordinal score means a 0.40 sigmoid cut is not a calibrated point; adopting one silently invalidates the comparison.
4. Methods whose only claimed wins are mAP/AP (ASL's headline, query decoders, DBL): +0.8 mAP at 9.6k labels does not imply a macro-F1 gain at a fixed threshold.
5. Label smoothing for BCE: destroys calibration in the recall-reducing direction.
6. Full class-balanced resampling / uniform-prior cRT: assumes a balanced test distribution. Our val set has the natural prior, so a hard-balanced classifier over-predicts tail at 0.40 and craters micro precision — the axis where we lead the bar.
