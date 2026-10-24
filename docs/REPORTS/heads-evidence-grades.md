# Evidence grades

(a) Well-evidenced (measured, multiple independent sources, mechanism understood)
Class-frequency loss weighting + focal-style negative down-weighting: BALMS@18k labels, DBL@COCO-MLT, ASL@Open Images, and +0.013 macro on the same task/scale in wd-tagger v3 ([tail losses](heads-tail-losses.md))
Negative-tolerant regularization (BALMS's biggest ingredient: λ applied twice on the negative branch). Its bias is fixed from the prior, not learned, and does not lower tail operating points — the gain is λ's ([tail losses](heads-tail-losses.md) point 1)
Per-class logit/prior correction as a training-time intervention (Menon: consistent for balanced error; dominates weight normalization — [bias fixes](heads-bias-fixes.md))
A global-pooled linear head is competitive at 9.6k labels (ML-Decoder's Table 2: 0.8 mAP — [single linear head](heads-single-linear-head.md))
Classifier rebalancing trades head for tail in measurable amounts (Decoupling: +7.6 few-shot / −4.6 many-shot)

(b) Plausible but unproven for our regime
Label-text/label-feature heads for labels that already have thousands of positives (DECAF and CLIP-Decoder evidence is zero/few-shot)
cRT-style head-only retraining on balanced batches: strong single-label softmax evidence, no multi-label BCE evidence, and it assumes a balanced test prior we do not have
Query/attention decoders at 12k for macro F1@0.40 (one mAP table at 9.6k; nothing on macro)
Mixup (which the bar used at 0.6) as a calibration lever: raises recall at 0.40, costs precision; the trade is unmeasured for our metric
Hierarchical label structure; label co-occurrence as a regularizer at 12k
ZLPR / ranking losses — real idea, mixed results, protocol-hostile

(c) Likely hype or actively wrong here
Label co-occurrence GCN heads: ML-GCN 83.0 vs ASL 86.6 mAP on COCO. Dominated
MoE / sparse heads, per-label adapters at 12k: no credible vision evidence, and they add capacity where capacity is provably not the differentiator
Adding an MLP hidden layer: our head is already 38.33M params vs the bar's 35.15M and loses
Cosine classifier as a post-hoc rescale: inside the 0.3879 ceiling, and Menon measured it making balanced error worse on iNat (48.05 vs 38.66 ERM) when τ is wrong
Raising classifier weight decay expecting tail recall
"Any-label"/zero-shot heads for a fully supervised 12k-label task
