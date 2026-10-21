"""Two extras: (1) visual contact sheet of the 4 most similar train/test pairs,
(2) label coverage -- which of the 12,476 labels have zero support in train, and
how many test positives belong to labels the train set never shows."""
from __future__ import annotations

import json
import os
import pickle

import numpy as np
from PIL import Image

from wdtag.cache import IMAGES_PER_SHARD, SPLITS, _shard_name
from wdtag.labels import load_label_space

WS = r"C:\Users\kerok\apps\wd-tagger-trainer"
OUT = os.path.join(WS, "scratch", "sophie")
FU = os.path.join(OUT, "for-user")
space = load_label_space()

sim = json.load(open(os.path.join(OUT, "similarity.json"), encoding="utf-8"))


def get(split: str, key: str) -> np.ndarray:
    with open(os.path.join(OUT, f"records_{split}.pkl"), "rb") as f:
        recs = pickle.load(f)
    g = next(i for i, r in enumerate(recs) if r["key"] == key)
    total = SPLITS[split]["n"]
    k = g // IMAGES_PER_SHARD
    s = __import__("torch").load(_shard_name(split, "squash", k, total),
                                 map_location="cpu", mmap=True, weights_only=False)
    return s["images"][g % IMAGES_PER_SHARD].numpy()


pairs = sim["train|test"][:4] + sim["train|valid"][:2]
tiles = []
for p in pairs:
    a_split = "train"
    b_split = "test" if p in sim["train|test"] else "valid"
    a = Image.fromarray(get(a_split, p["a_key"]).transpose(1, 2, 0))
    b = Image.fromarray(get(b_split, p["b_key"]).transpose(1, 2, 0))
    tiles.append((a, b))
H = 512
canvas = Image.new("RGB", (2 * H, H * len(tiles)), (0, 0, 0))
for i, (a, b) in enumerate(tiles):
    canvas.paste(a, (0, i * H))
    canvas.paste(b, (H, i * H))
canvas.save(os.path.join(FU, "closest-train-test-pairs.png"))
print("wrote", os.path.join(FU, "closest-train-test-pairs.png"), canvas.size, flush=True)
for i, p in enumerate(pairs):
    print(i, p["a_key"], p["a_tar"], "|", p["b_key"], p["b_tar"], "cos", round(p["cos"], 4), flush=True)

# ------------------------------------------------------------ label coverage
pos = {}
for split in SPLITS:
    with open(os.path.join(OUT, f"records_{split}.pkl"), "rb") as f:
        recs = pickle.load(f)
    from wdtag.labels import tag_indices
    c = np.zeros(len(space), dtype=np.int64)
    for r in recs:
        idx, _ = tag_indices({"general_tags": r["gtags"], "character_tags": r["ctags"],
                              "rating": r["rt"]}, space)
        c[np.array(idx, dtype=np.int64)] += 1
    pos[split] = c

tr, va, te = pos["train"], pos["valid"], pos["test"]
out = {}
for split, c in (("train", tr), ("valid", va), ("test", te)):
    out[split] = dict(
        labels_with_support=int((c > 0).sum()),
        labels_never=int((c == 0).sum()),
        labels_lt_10=int(((c > 0) & (c < 10)).sum()),
        label_occurrences=int(c.sum()),
    )
out["test_labels_absent_from_train"] = int(((te > 0) & (tr == 0)).sum())
out["test_positives_of_those_labels"] = int(te[(te > 0) & (tr == 0)].sum())
out["test_positives_total"] = int(te.sum())
out["valid_labels_absent_from_train"] = int(((va > 0) & (tr == 0)).sum())
out["valid_positives_of_those_labels"] = int(va[(va > 0) & (tr == 0)].sum())
out["valid_positives_total"] = int(va.sum())
out["labels_never_in_any_split"] = int(((tr + va + te) == 0).sum())
out["train_labels_never_in_test"] = int(((tr > 0) & (te == 0)).sum())
print(json.dumps(out, indent=1), flush=True)
with open(os.path.join(OUT, "coverage.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1)
