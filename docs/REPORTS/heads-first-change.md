# The single change I would try first

CB-NTR in the loss (item 1), gated behind the 30-minute post-hoc prior-offset probe (item 2) and the 30-minute frequency-binned error analysis (item 4).

Why: it is the only intervention that (i) has measured macro/tail F1 evidence at a label count larger than ours (18211), (ii) moves class operating points inside the trained scores, so it is not capped by the 0.3879 ceiling, (iii) costs one line plus config, and (iv) has a same-task, same-scale supporting datapoint (+0.013 macro from frequency-based loss scaling in wd-tagger v3) that keeps expectations honest.

Sequence: probe → binned analysis → 5–10k-step arm with CB-NTR → full 50k eval. Total ~2 days of wall clock; the two probes can kill the idea before you spend 9 h on it.
