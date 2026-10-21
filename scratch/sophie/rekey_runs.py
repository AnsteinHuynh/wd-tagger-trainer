"""Re-key run directories to the hash of their own config.

Adding a field to TrainConfig changes every sha8, so directories created before
the change no longer match `run_dir()` and a resume would silently start a new
run. This rewrites each run's config.json with the full current field set and
renames the directory to the hash that file now produces.
"""
import json
import os
import re
import shutil

from wdtag.config import RUNS_ROOT, TrainConfig

HEX8 = re.compile(r"^[0-9a-f]{8}$")

for d in sorted(os.listdir(RUNS_ROOT)):
    path = os.path.join(RUNS_ROOT, d)
    cfg_path = os.path.join(path, "config.json")
    if not HEX8.match(d) or not os.path.isfile(cfg_path):
        continue
    with open(cfg_path, encoding="utf-8") as f:
        raw = json.load(f)
    cfg = TrainConfig(**raw)             # fills any fields added since the run
    new_hash = cfg.sha8()
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, indent=2)
    if new_hash == d:
        print(f"{d}: unchanged")
        continue
    dst = os.path.join(RUNS_ROOT, new_hash)
    if os.path.exists(dst):
        print(f"{d}: target {new_hash} already exists, leaving alone")
        continue
    os.rename(path, dst)
    print(f"{d} -> {new_hash}")
