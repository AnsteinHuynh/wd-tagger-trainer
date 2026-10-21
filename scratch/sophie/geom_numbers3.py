import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
br = [json.loads(l) for l in open(os.path.join(HERE, "geom_train_broad.jsonl"), encoding="utf-8")]
def imgcount(pred):
    return sum(1 for r in br if pred({t for t in r["general"]}))
N = len(br)
core = {"border", "white_border", "outside_border"}
print("any *border:", imgcount(lambda s: any(t == "border" or t.endswith("_border") for t in s)))
print("border exact:", imgcount(lambda s: "border" in s))
print("white_border:", imgcount(lambda s: "white_border" in s))
print("outside_border:", imgcount(lambda s: "outside_border" in s))
print("other *_border (not core three):",
      imgcount(lambda s: any(t.endswith("_border") for t in s) and not (s & core)))
for t in ("black_border", "grey_border", "pink_border", "inset_border", "ornate_border"):
    print(" ", t, imgcount(lambda s, t=t: t in s))
print("silent_comic:", imgcount(lambda s: "silent_comic" in s))
print("segmented_comic:", imgcount(lambda s: "segmented_comic" in s))
print("right-to-left_comic:", imgcount(lambda s: "right-to-left_comic" in s))
print("comic family:", imgcount(lambda s: any("comic" in t for t in s)))
print("out-of-frame family:", imgcount(lambda s: any("out_of_frame" in t for t in s)))
print("cropped_* family:", imgcount(lambda s: any(t.startswith("cropped_") for t in s)))
print("2koma/4koma family:", imgcount(lambda s: any("koma" in t for t in s)))
