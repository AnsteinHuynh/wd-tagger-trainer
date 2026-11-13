"""The corpus manifest: one durable table of what is in the tars, and the only place tar order lives.

The tars are 318 GB of read-only tar files holding 5.3 million image/annotation pairs. Every pass
over them, every cache, every shard split needs three facts: which tars exist, how many samples each
one holds, and in what order a given seed visits them. Deriving that from the tars means walking
headers, roughly ten seconds per tar, so it is recorded once here and read from then on.

Three rules keep this honest, each of them paid for:

The order is `random.Random(seed * 1000003 + pass_index)` shuffled over the sorted directory listing,
never over the key order of the stored table. A table written in the order counts were learned is
itself shuffled, so shuffling it again silently produces a different permutation from the one the
streamer used, and shards then carry the wrong tar names.

The counts are verified against headers on demand, by sample or in full, because a table that is
wrong is worse than no table: it makes a resume seek to the wrong offset without complaint.

File identity is recorded as size and modification time, so a replaced or truncated tar is caught
before a pass starts rather than as a mysterious decode error three hours in.

The file lives in runs/_manifests, which is durable, and is mirrored to the cache root for older
readers. Schema 2 keeps the schema-1 keys (`split`, `tars`, `total`) at the top level, so anything
written against the old shape keeps working.

Command line:
  python -m wdtag.manifest --split all --build          # stat every tar, record identity
  python -m wdtag.manifest --split all --check          # compare records against the filesystem
  python -m wdtag.manifest --split train --verify 3     # walk three tars and confirm their counts
  python -m wdtag.manifest --split train --ranges       # row ranges for seed 1, pass 0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import tarfile
import time

SCHEMA = 2
IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
COUNT_RULE = "complete image+json pair, matched by stem, headers only"


def _roots() -> tuple[str, str]:
    """(durable dir, cache root) without importing stream, which imports this module."""
    from .stream import CACHE_ROOT, RUNS_ROOT
    return os.path.join(RUNS_ROOT, "_manifests"), CACHE_ROOT


def split_root(split: str) -> str:
    from .stream import tar_dir
    return tar_dir(split)


def paths(split: str) -> list[str]:
    durable, cache = _roots()
    name = f"manifest_{split}.json"
    return [os.path.join(durable, name), os.path.join(cache, name)]


def save(split: str, man: dict) -> str:
    """Write to the durable copy first, then mirror. Atomic on each path."""
    ok = ""
    for path in paths(split):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(man, f, indent=1, sort_keys=False)
            os.replace(tmp, path)
            ok = ok or path
        except OSError as e:
            print(f"[manifest] could not write {path}: {e}", file=sys.stderr)
    if not ok:
        raise SystemExit("[manifest] every write location failed")
    return ok


def load(split: str) -> dict | None:
    for path in paths(split):
        try:
            man = json.load(open(path, encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if man.get("split") == split and man.get("tars"):
            return man
    return None


def disk_tars(split: str) -> list[str]:
    """The sorted directory listing. This list, not the table's key order, is what gets shuffled."""
    d = split_root(split)
    return sorted(f for f in os.listdir(d) if f.endswith(".tar"))


def order(split: str, seed: int = 1, pass_index: int = 0, names: list[str] | None = None) -> list[str]:
    """The one definition of tar order. Import this instead of writing the shuffle again."""
    tars = list(names if names is not None else disk_tars(split))
    random.Random(seed * 1000003 + pass_index).shuffle(tars)
    return tars


def counts(split: str) -> dict[str, int]:
    man = load(split)
    if not man:
        raise SystemExit(f"no manifest for {split}; run --build first")
    return {k: int(v) for k, v in man["tars"].items()}


def ranges(split: str, seed: int = 1, pass_index: int = 0) -> list[tuple[str, int, int]]:
    """[(tar, first_row, rows)] in the order a stream with this seed visits them."""
    c = counts(split)
    out, pos = [], 0
    for name in order(split, seed, pass_index):
        n = c[name]
        out.append((name, pos, n))
        pos += n
    return out


def _stat(path: str) -> dict:
    st = os.stat(path)
    return {"bytes": int(st.st_size), "mtime": int(st.st_mtime)}


def build(split: str, keep_unknown: bool = True) -> dict:
    """Record every tar's identity, carrying counts forward from an existing table.

    Counts are not re-derived here because that costs a header walk per tar. `verify` does that
    deliberately, on a sample or on everything.
    """
    root = split_root(split)
    names = disk_tars(split)
    old = load(split) or {}
    old_counts = {k: int(v) for k, v in (old.get("tars") or {}).items()}
    old_files = old.get("files") or {}

    missing = [n for n in names if n not in old_counts]
    files = {}
    for n in names:
        try:
            files[n] = _stat(os.path.join(root, n))
        except OSError:
            files[n] = {"bytes": 0, "mtime": 0}

    tars = {n: old_counts.get(n, 0) for n in names}
    man = {
        "schema": SCHEMA,
        "split": split,
        "root": root,
        "count_rule": COUNT_RULE,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total": int(sum(tars.values())),
        "tars": tars,
        "files": files,
        "unverified": [n for n in names if n not in old_counts],
    }
    if keep_unknown and missing:
        print(f"[manifest] {split}: {len(missing)} tars have no recorded count yet")
    return man


def check(split: str) -> dict:
    """Compare the recorded table against the filesystem without walking any tar."""
    man = load(split)
    if not man:
        return {"split": split, "error": "no manifest"}
    names = disk_tars(split)
    recorded = man.get("tars") or {}
    files = man.get("files") or {}
    missing = [n for n in recorded if n not in set(names)]
    added = [n for n in names if n not in recorded]
    changed = []
    for n in names:
        want = files.get(n)
        if not want:
            continue
        try:
            got = _stat(os.path.join(split_root(split), n))
        except OSError:
            continue
        if got["bytes"] != want.get("bytes") or got["mtime"] != want.get("mtime"):
            changed.append(n)
    zero = [n for n in names if int(recorded.get(n, 0)) <= 0]
    return {
        "split": split,
        "tars_on_disk": len(names),
        "tars_recorded": len(recorded),
        "total": int(sum(int(recorded.get(n, 0)) for n in names)),
        "missing_from_disk": missing,
        "added_since_build": added,
        "identity_changed": changed,
        "counts_unrecorded": zero,
        "ok": not (missing or added or changed or zero),
    }


def verify(split: str, sample: int | None = None, names: list[str] | None = None) -> dict:
    """Walk tar headers and compare the real sample count with the table.

    This is the slow, honest check. `sample` picks tars evenly spread across the split so a stale
    table cannot hide in one corner of the directory.
    """
    man = load(split)
    rec = {k: int(v) for k, v in ((man or {}).get("tars") or {}).items()}
    tars = names or disk_tars(split)
    if sample and not names:
        step = max(1, len(tars) // sample)
        tars = tars[::step][:sample]
    out = {"split": split, "checked": [], "mismatch": [], "seconds": 0.0}
    t0 = time.time()
    for n in tars:
        got = _count_samples(os.path.join(split_root(split), n))
        want = rec.get(n)
        row = {"tar": n, "counted": got, "recorded": want}
        out["checked"].append(row)
        if want is not None and got != want:
            out["mismatch"].append(row)
        print(f"  {n}: counted {got}, recorded {want}", flush=True)
    out["seconds"] = round(time.time() - t0, 1)
    out["ok"] = not out["mismatch"]
    return out


def _count_samples(tar_path: str) -> int:
    pending: set[str] = set()
    n = 0
    with tarfile.open(tar_path, "r:") as tf:
        for m in tf:
            if not m.isfile():
                continue
            stem, ext = os.path.splitext(os.path.basename(m.name))
            if ext in IMG_EXT:
                pending.add(stem)
            elif ext == ".json" and stem in pending:
                pending.discard(stem)
                n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description="durable tar manifest")
    ap.add_argument("--split", default="all", help="train, valid, test or all")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--verify", type=int, default=None, metavar="N",
                    help="walk N tars and confirm their counts (0 means every tar)")
    ap.add_argument("--ranges", action="store_true", help="print row ranges for the given seed")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--pass-index", type=int, default=0)
    args = ap.parse_args()

    splits = ["train", "valid", "test"] if args.split == "all" else [args.split]
    for split in splits:
        print(f"== {split}")
        if args.build:
            man = build(split)
            path = save(split, man)
            print(f"  wrote {path}: {len(man['tars'])} tars, total {man['total']}")
        if args.check:
            print("  " + json.dumps(check(split)))
        if args.verify is not None:
            n = None if args.verify == 0 else args.verify
            r = verify(split, sample=n)
            print("  " + json.dumps({k: v for k, v in r.items() if k != "checked"}))
        if args.ranges:
            c = counts(split)
            for name, start, n in ranges(split, args.seed, args.pass_index):
                print(f"  {name} rows {start}..{start + n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
