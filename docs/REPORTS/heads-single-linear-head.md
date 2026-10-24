# Is a single Linear head still competitive at 10k+ labels?

Yes, and the direct evidence is the ablation inside the paper that proposed the alternative. ML-Decoder ([2111.12933](https://arxiv.org/abs/2111.12933), WACV 2023), Table 2, Open Images (9600
labels), TResNet-M backbone:

| head | GFLOPs | mAP |
|---|---|---|
| global average pooling + FC (GAP) | 5.8 | 86.0 |
| ML-Decoder, 100 queries (group decoding 48×) | 6.3 | 86.7 |
| ML-Decoder, 200 queries | 6.7 | 86.8 |
| ML-Decoder, 400 queries | 7.6 | 86.8 |
| full transformer decoder, 9600 queries | 178.6 | OOM even at batch 1, 224px |

On MS-COCO (80 labels), Table 1: GAP 87.0 → ML-Decoder 88.1 mAP. The gain is ~+0.8 mAP at 9600 labels and ~+1.1 at 80 — it does not grow with label count. The advantage is spatial attention (per-label pooling instead of one global vector), roughly constant, and already partly captured by our max/std pooling over patches.

State of the practice at our scale. The strongest public danbooru taggers, same ~12k label space, are GAP/CLS + one Linear layer: [wd-eva02-large-tagger-v3](https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3)
(macro F1 0.4772 at its P=R threshold 0.5296) and [wd-swinv2-tagger-v3](https://huggingface.co/SmilingWolf/wd-swinv2-tagger-v3)
(0.4541). Our bar `convnextv2_huge.dbv4-full` is the same shape (`avgpool → LN → Dropout(0.2) → Linear`); nobody ships a query decoder at 12k labels.

Evidence gap: no published study compares head architectures on long-tailed macro F1 at a fixed threshold with ≥10k labels. The literature is all mAP at ≤200 labels, plus ML-Decoder's 9.6k table. Treat "query heads are better at scale" as unproven rather than false.
