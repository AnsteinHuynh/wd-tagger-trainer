import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
rows = [json.loads(l) for l in open(os.path.join(HERE, "geom_train_broad.jsonl"), encoding="utf-8")]
w = np.array([r["w"] for r in rows], float)
h = np.array([r["h"] for r in rows], float)
a = w / h
d = np.maximum(w, h) / np.minimum(w, h)
waste = 100.0 * (1.0 - (w * h) / np.maximum(w, h) ** 2)

fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.8), dpi=110)
fig.patch.set_facecolor("white")

ax[0].hist(np.clip(a, 0.2, 2.4), bins=np.arange(0.2, 2.42, 0.02), color="#4c72b0")
ax[0].set_yscale("log")
for x, lab in [(0.75, "0.75"), (1.0, "1.00"), (1.333, "1.33"), (1.5, "1.50"), (1.778, "1.78")]:
    ax[0].axvline(x, color="crimson", lw=0.8, ls="--")
    ax[0].text(x - 0.01, 1.0e4, lab, color="crimson", fontsize=7, rotation=90,
               ha="right", va="top")
ax[0].set_title("aspect = w/h  (n=40,000 train images)", fontsize=9)
ax[0].set_xlabel("aspect", fontsize=8)
ax[0].set_ylabel("images", fontsize=8)
ax[0].set_xlim(0.2, 2.4)
ax[0].tick_params(labelsize=7)
ax[0].text(0.02, 0.78, f"median {np.median(a):.2f}\np95 {np.percentile(a,95):.2f}",
           transform=ax[0].transAxes, fontsize=8)

ax[1].hist(np.clip(d, 1.0, 3.0), bins=np.arange(1.0, 3.02, 0.02), color="#dd8452")
for x, lab in [(1.25, "1.25"), (1.5, "1.5"), (1.75, "1.75")]:
    ax[1].axvline(x, color="crimson", lw=0.8, ls="--")
    ax[1].text(x, ax[1].get_ylim()[1] * 0.95, lab, color="crimson", fontsize=7, rotation=90,
               ha="right", va="top")
ax[1].set_title("distortion factor d = max(w,h)/min(w,h)", fontsize=9)
ax[1].set_xlabel("d (squash to 512x512)", fontsize=8)
ax[1].set_ylabel("images", fontsize=8)
ax[1].tick_params(labelsize=7)
ax[1].text(0.35, 0.72, f"d>1.25 {100*(d>1.25).mean():.1f}%\nd>1.5 {100*(d>1.5).mean():.1f}%\n"
                       f"d>1.75 {100*(d>1.75).mean():.1f}%", transform=ax[1].transAxes, fontsize=8)

ax[2].hist(waste, bins=np.arange(0, 100, 1), color="#55a868")
ax[2].set_title("letterbox-to-square: % of 512x512 canvas left blank", fontsize=9)
ax[2].set_xlabel("blank border (%)", fontsize=8)
ax[2].set_ylabel("images", fontsize=8)
ax[2].tick_params(labelsize=7)
ax[2].text(0.42, 0.72, f"mean {waste.mean():.1f}%\nmedian {np.median(waste):.1f}%",
           transform=ax[2].transAxes, fontsize=8)

for x in ax:
    x.grid(alpha=0.25, lw=0.5)
fig.tight_layout()
out = os.path.join(HERE, "for-user", "geom_aspect.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out)
print("wrote", out)
