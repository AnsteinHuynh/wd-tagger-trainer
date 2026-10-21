"""Which quality/negative-prompt tags exist in our label space, how common are
they, and can the 5.3M-image baseline already spot them?

A "defect detector" is a far easier problem than 12,476-label tagging: a few
dozen tags with real prevalence, learnable from the same 10,000 images.
"""
import numpy as np
from wdtag.data import load_targets
from wdtag.labels import load_label_space
from wdtag.metrics import compute_metrics

space = load_label_space()
names = list(space.names)
N = len(names)
idx = {n: i for i, n in enumerate(names)}

WANT = ["lowres", "highres", "absurdres", "bad anatomy", "bad hands", "bad feet",
        "worst quality", "low quality", "jpeg artifacts", "blurry", "watermark",
        "signature", "artist name", "text", "english text", "speech bubble",
        "translated", "monochrome", "greyscale", "multiple views", "cropped",
        "border", "comic", "sketch", "manga", "realistic", "photo", "3d",
        "screencap", "official art", "twitter username", "web address",
        "character request", "commentary", "commentary request", "bad id",
        "bad proportions", "extra digits", "missing fingers", "fused fingers"]

tr = load_targets("train", "squash", N).astype(bool)
te = load_targets("test", "squash", N).astype(bool)
base = np.load(r"runs\_baseline\convnextv2_huge\squash\test_probs.npy")

rows = []
for t in WANT:
    if t not in idx:
        continue
    i = idx[t]
    prev = tr[:, i].mean() * 100
    n_te = int(te[:, i].sum())
    f1 = b = float("nan")
    if n_te >= 5:
        m = compute_metrics(base[:, i][:, None] >= 0.5, te[:, i][:, None])
        f1 = m["macro_f1"]
        # best achievable threshold on this label, on test, as an upper reference
        best = max(((2 * ((base[:, i] >= th) & te[:, i]).sum()) /
                    max(1, (base[:, i] >= th).sum() + te[:, i].sum()))
                   for th in np.arange(0.05, 0.96, 0.05))
        b = best
    rows.append((t, prev, n_te, f1, b))

rows.sort(key=lambda r: -r[1])
print(f"{'tag':22s} {'train %':>8s} {'test n':>7s} {'baseline F1@.5':>15s} {'baseline best':>14s}")
for t, prev, n_te, f1, b in rows:
    f1s = f"{f1:.3f}" if f1 == f1 else "  -  "
    bs = f"{b:.3f}" if b == b else "  -  "
    print(f"{t:22s} {prev:8.2f} {n_te:7d} {f1s:>15s} {bs:>14s}")

print(f"\nmissing from the 12,476-label space: "
      f"{[t for t in WANT if t not in idx]}")
tot = np.mean([tr[:, idx[t]].any(1).mean() for t in WANT if t in idx])
print(f"prevalence of 'any quality tag' in train: {tot*100:.1f}% of images")
