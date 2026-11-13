"""Label space for the WD-Tagger v4 tagger (12,476 tags).

The canonical order is the one baked into `convnextv2_huge.dbv4-full`:
`config.json["tags"]` (12,476 names, rating -> general -> character, i.e. exactly
the row order of `selected_tags.csv`).  We reuse it verbatim so that our scores
and the baseline's are directly comparable.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

BASELINE_DIR = os.environ.get("WDTAG_BASELINE_DIR", "<path/to/convnextv2_huge.dbv4-full>")

# categories in the dataset: 9 = rating, 0 = general, 4 = character
CAT_RATING, CAT_GENERAL, CAT_CHARACTER = 9, 0, 4


@dataclass
class LabelSpace:
    names: list[str]
    categories: list[int]
    name2idx: dict[str, int]

    def __len__(self) -> int:
        return len(self.names)

    @property
    def n_rating(self) -> int:
        return sum(c == CAT_RATING for c in self.categories)


def load_label_space(baseline_dir: str = BASELINE_DIR) -> LabelSpace:
    with open(os.path.join(baseline_dir, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    names = cfg["tags"]
    # selected_tags.csv is in the same order and carries the category column.
    import csv

    with open(os.path.join(baseline_dir, "selected_tags.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != len(names):
        raise RuntimeError(f"label space mismatch: {len(rows)} csv rows vs {len(names)} config tags")
    categories = [int(r["category"]) for r in rows]
    csv_names = [r["name"] for r in rows]
    if csv_names != names:
        raise RuntimeError("selected_tags.csv row order does not match config.json['tags']")
    return LabelSpace(names=names, categories=categories, name2idx={n: i for i, n in enumerate(names)})


def tag_indices(meta: dict, space: LabelSpace) -> tuple[list[int], list[str]]:
    """Map a webdataset json annotation to (sorted unique label indices, unknown names)."""
    raw: list[str] = []
    raw.extend(meta.get("general_tags") or [])
    raw.extend(meta.get("character_tags") or [])
    raw.extend(meta.get("rating") or [])
    idx: set[int] = set()
    unknown: list[str] = []
    for t in raw:
        i = space.name2idx.get(t)
        if i is None:
            unknown.append(t)
        else:
            idx.add(i)
    return sorted(idx), unknown
