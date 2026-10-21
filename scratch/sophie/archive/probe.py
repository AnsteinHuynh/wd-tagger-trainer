import json, csv, os, sys, time
import torch

DS = r"X:\unique\datasets\animetimm-danbooru-wdtagger-v4-w640-ws-full\repo"
CN = r"C:\MyApps\ComfyUIRoot\convnextv2_huge.dbv4-full"

print("== selected_tags.csv ==")
rows = list(csv.DictReader(open(os.path.join(CN, "selected_tags.csv"), encoding="utf-8")))
print("n_labels:", len(rows))
from collections import Counter
print("categories:", Counter(r["category"] for r in rows))
print("first 3:", [(r["tag_id"], r["name"], r["category"]) for r in rows[:3]])
print("last 3:", [(r["tag_id"], r["name"], r["category"]) for r in rows[-3:]])

cfg = json.load(open(os.path.join(CN, "config.json")))
print("== baseline config keys ==", [k for k in cfg if k not in ("id2label",)])
print("architectures:", cfg.get("architectures"), "model_type:", cfg.get("model_type"))
i2l = cfg.get("id2label") or {}
print("n id2label:", len(i2l))
if i2l:
    print("id2label sample:", list(i2l.items())[:3], list(i2l.items())[-3:])

print("== tags.csv header ==")
with open(os.path.join(DS, "tags.csv"), encoding="utf-8") as f:
    hdr = f.readline().strip()
print(hdr[:300])

print("== meta.json keys ==")
meta = json.load(open(os.path.join(DS, "meta.json")))
print(list(meta.keys()))
print("n tags in meta:", len(meta.get("tags", [])))
print("tag sample:", meta["tags"][:3])
print("max_ids:", meta.get("max_ids"))

print("== preprocessor_config ==")
print(open(r"I:\Models\dinov3-vitb16-pretrain-lvd1689m\preprocessor_config.json").read())
