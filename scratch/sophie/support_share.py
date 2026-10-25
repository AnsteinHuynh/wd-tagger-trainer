import numpy as np

z = np.load(r"I:\.cache\wd-tagger-trainer\valid_nested\labels.npz")
s = np.bincount(z["indices"], minlength=12476)
tot = s.sum()
o = np.sort(s)[::-1]
print(f"total label cells {int(tot)} over 50000 images")
for k in (4, 100, 1000, 5000):
    print(f"top {k:5d} tags hold {o[:k].sum() / tot * 100:5.1f} percent of all positives")
print(f"tags with under 10 positives: {int((o < 10).sum())} tags, "
      f"{o[o < 10].sum() / tot * 100:.2f} percent of positives")
print(f"tags with under 50 positives: {int((o < 50).sum())} tags, "
      f"{o[o < 50].sum() / tot * 100:.2f} percent of positives")
