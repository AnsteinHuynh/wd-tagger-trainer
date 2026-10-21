"""Pin each existing run's directory name into its config.json (`_sha8`).

`TrainConfig.sha8()` hashes every dataclass field, and that hash is the run
directory name.  Adding or removing a field changes every hash, so resuming an
old run would silently land in a fresh directory and start from scratch.  A
`_sha8` key freezes the run's identity: `from_json()` loads it into the pin and
`sha8()`/`run_dir()` return it unchanged.

The script is idempotent and only touches `runs/<8hex>/config.json`.  It never
pins `configs/*.json`: those are templates, and every new run must hash to its
own fresh identity (train.py writes that run's pin when it creates the dir).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

from wdtag.config import RUNS_ROOT, TrainConfig

HEX8 = re.compile(r"^[0-9a-f]{8}$")


def iter_runs():
    for name in sorted(os.listdir(RUNS_ROOT)):
        path = os.path.join(RUNS_ROOT, name)
        cfg_path = os.path.join(path, "config.json")
        if HEX8.match(name) and os.path.isfile(cfg_path):
            yield name, path, cfg_path


def pin(check_only: bool) -> int:
    changed = 0
    for name, _path, cfg_path in iter_runs():
        with open(cfg_path, encoding="utf-8") as f:
            raw = json.load(f)
        existing = raw.get("_sha8")
        if existing == name:
            print(f"{name}: already pinned")
            continue
        if "_sha8" in raw:
            print(f"{name}: WARNING _sha8 is {existing!r}, not {name!r}; leaving it untouched")
            continue
        if check_only:
            print(f"{name}: would add _sha8 {name!r}")
            changed += 1
            continue
        # Preserve every existing key and its insertion order; append the pin.
        raw["_sha8"] = name
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(raw, f, indent=2)
        print(f"{name}: added _sha8 {name!r}")
        changed += 1
    print(f"{changed} run config(s) {'would change' if check_only else 'changed'}")
    return changed


def verify() -> bool:
    """Every run must resolve to itself, both by hash and by run_dir()."""
    ok = True
    print("\nverifying:")
    for name, path, cfg_path in iter_runs():
        cfg = TrainConfig.from_json(cfg_path)
        sha = cfg.sha8()
        rd = cfg.run_dir()
        rd_name = os.path.basename(os.path.normpath(rd))
        same = (
            sha == name
            and os.path.normcase(os.path.abspath(rd))
            == os.path.normcase(os.path.abspath(path))
        )
        ok = ok and same
        print(f"{'OK  ' if same else 'FAIL'} {name}: sha8()={sha} run_dir()={rd_name}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="report what would change, write nothing")
    ap.add_argument("--verify-only", action="store_true", help="do not pin; just verify")
    a = ap.parse_args()
    if not a.verify_only:
        pin(a.check)
        if a.check:
            return 0
    return 0 if verify() else 1


if __name__ == "__main__":
    sys.exit(main())
