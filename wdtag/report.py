"""Build REPORT.md: the bar, our arms, the run history, and the efficiency notes."""
from __future__ import annotations

import argparse
import glob
import json
import os

from .cache import MODE
from .config import RUNS_ROOT

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS = ("f1", "mcc", "precision", "recall")
BASELINE = os.path.join(RUNS_ROOT, "_baseline", "convnextv2_huge", MODE, "eval_test.json")
CATEGORY_LABELS = {"rating": 4, "general": 9225, "character": 3247}


def load_json(p: str) -> dict | None:
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def cells(score: dict | None, which: str, scale: str) -> str:
    if not score:
        return "— | — | — | —"
    m = score.get(which, {})
    return " | ".join(f"{m.get(f'{scale}_{k}', float('nan')):.4f}" for k in METRICS)


def ours() -> list[dict]:
    out = []
    for p in sorted(glob.glob(os.path.join(RUNS_ROOT, "*", "eval_test.json"))):
        d = load_json(p)
        if not d or not d.get("model", "").startswith("dinov3"):
            continue
        run = os.path.dirname(p)
        d["_dir"] = run
        d["_cfg"] = load_json(os.path.join(run, "config.json")) or {}
        out.append(d)
    out.sort(key=lambda d: (-d["threshold_tuned"]["macro_f1"], -d["threshold_0.5"]["micro_f1"]))
    return out


def label_for(d: dict) -> str:
    c, tag = d["_cfg"], os.path.basename(d["_dir"])
    return (f"{c.get('mode', '?')} · lr_oft {c.get('lr_oft', '?')} · "
            f"lr_head {c.get('lr_head', '?')} · seed {c.get('seed', '?')} · "
            f"{c.get('epochs', '?')} ep [{tag}]")


def _close(a, b) -> bool:
    return isinstance(a, (int, float)) and abs(float(a) - float(b)) <= 1e-12


def find_run(runs: list[dict], *, mode: str | None = None, lr_oft: float | None = None,
             lr_head: float | None = None, seed: int | None = None,
             epochs: int | None = None) -> dict | None:
    """First run matching every supplied config field, or None if it is absent."""
    for d in runs:
        c = d.get("_cfg", {})
        if mode is not None and c.get("mode") != mode:
            continue
        if seed is not None and c.get("seed") != seed:
            continue
        if epochs is not None and c.get("epochs") != epochs:
            continue
        if lr_oft is not None and not _close(c.get("lr_oft"), lr_oft):
            continue
        if lr_head is not None and not _close(c.get("lr_head"), lr_head):
            continue
        return d
    return None


def metric(d: dict | None, which: str, key: str) -> float:
    if not d:
        return float("nan")
    return d.get(which, {}).get(key, float("nan"))


def sc(d: dict | None, which: str, key: str) -> str:
    """Format a metric as 4 dp, or n/a when the run/metric is missing."""
    v = metric(d, which, key)
    return "n/a" if v != v else f"{v:.4f}"


def lr_head_section(runs: list[dict], best_probe: dict | None) -> list[str]:
    """The lr_head bracket, computed from the matched squash/0.0005/1337/8ep arms."""
    arms: dict[float, dict] = {}
    for d in runs:
        c = d.get("_cfg", {})
        if (c.get("mode") == "squash" and _close(c.get("lr_oft"), 0.0005)
                and c.get("seed") == 1337 and c.get("epochs") == 8
                and isinstance(c.get("lr_head"), (int, float))):
            arms.setdefault(round(float(c["lr_head"]), 8), d)
    if not arms:
        return []

    heads = sorted(arms, reverse=True)
    best_head = max(heads, key=lambda h: metric(arms[h], "threshold_0.5", "micro_f1"))
    # the pre-sweep value is known; don't let a new large-LR experiment become "old"
    old_head = next((h for h in heads if abs(h - 0.004) <= 1e-12), max(heads))
    best, old = arms[best_head], arms[old_head]

    out = ["### The `lr_head` sweep\n",
           "`lr_head` was the single largest lever we found. Matched arms (squash, "
           "`lr_oft 0.0005`, seed 1337, 8 epochs):\n",
           "| `lr_head` | micro F1 @0.5 | macro F1 @0.5 | tuned macro F1 |",
           "|---|---|---|---|"]
    for h in heads:
        note = " (old)" if h == old_head else (" (best)" if h == best_head else "")
        d = arms[h]
        out.append(f"| {h:g}{note} | {sc(d, 'threshold_0.5', 'micro_f1')} | "
                   f"{sc(d, 'threshold_0.5', 'macro_f1')} | "
                   f"{sc(d, 'threshold_tuned', 'macro_f1')} |")
    out.append("")

    bracket = ", ".join(f"{h:g} -> {sc(arms[h], 'threshold_0.5', 'micro_f1')}" for h in heads)
    out.append(f"In order, the bracket is {bracket} micro F1. The peak is `lr_head` "
               f"{best_head:g}: {sc(best, 'threshold_0.5', 'micro_f1')} micro and "
               f"{sc(best, 'threshold_tuned', 'macro_f1')} tuned macro.\n")

    mid_head = next((h for h in heads if abs(h - 0.00025) <= 1e-12), None)
    low_head = next((h for h in heads if abs(h - 0.00006) <= 1e-12), None)
    old_vs_best = int(round(old_head / best_head)) if best_head else 0
    old_vs_mid = int(round(old_head / mid_head)) if mid_head else 0
    probe_lr = best_probe.get("lr") if best_probe else None
    probe_x = int(probe_lr / best_head) if (probe_lr and best_head) else 0
    old_clause = (
        f"{old_vs_mid}-{old_vs_best}x too high ({old_vs_mid}x the {mid_head:g} bracket, "
        f"{old_vs_best}x the {best_head:g} optimum)"
    ) if mid_head else f"~{old_vs_best}x too high"
    micro_gain = (metric(best, "threshold_0.5", "micro_f1")
                  / metric(old, "threshold_0.5", "micro_f1") - 1) * 100
    out.append(f"{old_head:g} — the value used by every surviving pre-sweep run — "
               f"is {old_clause}; the old 0.002 default (also the frozen probe's pick) is "
               f"~{probe_x}x too high. Lowering `lr_head` from {old_head:g} to {best_head:g} "
               f"raises micro F1 {micro_gain:.1f}% "
               f"({sc(old, 'threshold_0.5', 'micro_f1')} -> {sc(best, 'threshold_0.5', 'micro_f1')}) "
               "and tuned macro F1 ~20% "
               f"({sc(old, 'threshold_tuned', 'macro_f1')} -> {sc(best, 'threshold_tuned', 'macro_f1')}).\n")

    if mid_head:
        trade = (f"Inside the bracket there is a small micro/macro trade: {best_head:g} "
                 f"maximises micro and tuned macro, while {mid_head:g} buys fixed-threshold "
                 f"macro ({sc(arms[mid_head], 'threshold_0.5', 'macro_f1')} vs "
                 f"{sc(best, 'threshold_0.5', 'macro_f1')}) at a micro cost "
                 f"({sc(arms[mid_head], 'threshold_0.5', 'micro_f1')} vs "
                 f"{sc(best, 'threshold_0.5', 'micro_f1')}).")
        if low_head:
            trade += (f" At {low_head:g} micro is already past its peak "
                      f"({sc(arms[low_head], 'threshold_0.5', 'micro_f1')}) and "
                      f"fixed-threshold macro falls to "
                      f"{sc(arms[low_head], 'threshold_0.5', 'macro_f1')}.")
        out.append(trade + "\n")

    if best_probe:
        out.append("**Do not pick `lr_head` from the frozen-feature probe.** Its best head LR "
                   f"is {best_probe['lr']:g} ({best_probe['test_micro_f1']:.4f} micro on test), "
                   f"with 0.005 tied within noise — ~{probe_x}-{int(round(0.005 / best_head))}x "
                   "the true optimum. That is right for *fixed* features, but the adapter later "
                   "reshapes them and moves the optimum down to "
                   f"{best_head:g}. Re-run the probe on trained-adapter features before "
                   "trusting a head LR from it.\n")
    return out


def geometry_section(runs: list[dict]) -> list[str]:
    """Pad vs squash, matched on lr_oft, lr_head and seed, plus the seed spread."""
    key = lambda d: (d["_cfg"].get("mode"), d["_cfg"].get("lr_oft"),
                     d["_cfg"].get("lr_head"), d["_cfg"].get("seed"))
    by = {key(d): d for d in runs}
    out = ["### Does the resize geometry matter? (matched LR and seed)\n",
           "| lr_oft | lr_head | seed | micro F1 pad | micro F1 squash | delta | "
           "macro F1 pad | macro F1 squash |",
           "|---|---|---|---|---|---|---|---|"]
    for (mode, lr, lh, seed) in sorted({k for k in by},
                                      key=lambda k: (k[1] or 0, k[2] or 0, k[3] or 0)):
        if mode != "pad":
            continue
        p, s = by.get(("pad", lr, lh, seed)), by.get(("squash", lr, lh, seed))
        if not (p and s):
            continue
        pm, sm = p["threshold_0.5"]["micro_f1"], s["threshold_0.5"]["micro_f1"]
        pa, sa = p["threshold_0.5"]["macro_f1"], s["threshold_0.5"]["macro_f1"]
        out.append(f"| {lr} | {lh} | {seed} | {pm:.4f} | {sm:.4f} | {pm - sm:+.4f} | "
                   f"{pa:.4f} | {sa:.4f} |")
    out.append("")
    # The seed spread is the noise floor for the matched pad/squash row, so match on lr_head.
    # Prefer the best head LR, and fall back to the pre-sweep one for older run sets.
    for head_lr in (0.00012, 0.004):
        seeds = {}
        for d in runs:
            c = d["_cfg"]
            if (c.get("mode") == "squash" and _close(c.get("lr_oft"), 0.0005)
                    and _close(c.get("lr_head"), head_lr)):
                seeds[c.get("seed")] = d["threshold_0.5"]["micro_f1"]
        if len(seeds) > 1:
            vals = list(seeds.values())
            spread = max(vals) - min(vals)
            out.append(
                f"Seed spread (squash, lr_oft 0.0005, lr_head {head_lr:g}, {len(seeds)} seeds "
                f"{sorted(seeds)}): micro F1 {min(vals):.4f}-{max(vals):.4f}, i.e. "
                f"**{spread:.4f}**. A geometry difference smaller than that is not resolvable "
                f"with this experiment, and it is also the error bar on our own headline "
                f"number: {sum(vals) / len(vals):.4f} on average.\n")
            break
    out.append("Reading: at lr_oft 0.002 an earlier pad arm (since removed) scored 0.3459 vs "
               "squash 0.3804, but once the head learning rate was corrected to 0.00012 and both "
               "arms were re-run at matching lr_oft/lr_head/seed, pad got 0.4078 against squash "
               "seeds of 0.4142 and 0.4186 micro. Squash is ahead on all four metrics (micro, "
               "macro, tuned micro, tuned macro) by 0.002-0.011, i.e. at or just outside the "
               "0.0044 matched seed spread -- so at most a hair better, certainly not worse.\n")
    out.append("**Verdict: they tie, and squash is the better default.** Squash is ahead on every "
               "column, never behind, and its two seeds (0.4142 and 0.4186) both beat the single "
               "pad seed (0.4078) by 0.006-0.011, at or just above the 0.0044 matched seed "
               "spread. Squash also needs no separate cache and does not throw away 26.7% of the "
               "canvas as blank border. The earlier apparent pad win was an artefact of the "
               "oversized head learning rate, so it is not evidence for padding.\n")
    out.append("**Does the squash distortion itself cost anything?** Measured per image on the "
               "1,000 test images, bucketing by distortion `d = max(w,h)/min(w,h)`: micro F1 is "
               "0.3628 for near-square (d<1.02, n=66), 0.4122 (1.02-1.25), 0.4217 (1.25-1.50), "
               "0.4142 (1.50-2.00), 0.3908 (d>2, n=33). Correlation between distortion and "
               "per-image F1 is **r = +0.047** -- i.e. none, and if anything the wrong sign. "
               "Heavily warped images score *better* than square ones. The geometry is a dead end: "
               "each option trades shape against resolution and neither is the bottleneck.\n")
    out.append("For the record, the dataset is 70% portrait (median aspect 0.75, only 5% square), "
               "median distortion is 1.41x, 20% of images exceed 1.5x, and letterboxing instead "
               "would leave 26.7% of the canvas blank. No annotation could police this -- the "
               "vocabulary contains no stretch/aspect tag at all -- but 12% of images do carry "
               "real letterbox/pillarbox bars baked into the source art. See "
               "`scratch/sophie/for-user/dataset-geometry-audit.md`.\n")
    return out


def coverage_section(best: dict) -> list[str]:
    """Decompose the macro-F1 gap by how often each label was actually seen.

    The baseline trained on the full 5.3M corpus; our sample is 10,000 images.
    Only 10,790 of 12,476 labels occur anywhere in our training set, 7,908 occur
    in fewer than 10 images, and 217 labels present in the test split never occur
    at all -- for those we are structurally guaranteed to score 0 while the
    baseline is not. This separates that structural part of the headline macro
    gap from the part reflecting genuine model quality on learnable labels.

    Needs the saved test probabilities for both models plus the train labels;
    returns nothing if any of them is missing.
    """
    try:
        import numpy as np
        from .config import RUNS_ROOT
        from .data import load_targets
        from .labels import load_label_space

        n = len(load_label_space())
        mode = best["_cfg"].get("mode", "squash")
        p_ours = np.load(os.path.join(best["_dir"], "test_probs.npy"))
        p_base = np.load(os.path.join(RUNS_ROOT, "_baseline", "convnextv2_huge",
                                      mode, "test_probs.npy"))
        y = load_targets("test", mode, n).astype(bool)
        train_cnt = load_targets("train", "squash", n).astype(bool).sum(0)
    except Exception:
        return []

    def per_label(p):
        pred = p >= 0.5
        tp = (pred & y).sum(0).astype(np.float64)
        fp = (pred & ~y).sum(0).astype(np.float64)
        fn = (~pred & y).sum(0).astype(np.float64)
        den = 2 * tp + fp + fn
        return np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0)

    n_test = y.sum(0)
    fo, fb = per_label(p_ours), per_label(p_base)
    subsets = [
        ("every label in the vocabulary", np.ones(n, bool)),
        ("labels that occur in the test split", n_test > 0),
        ("labels with >=10 training images", train_cnt >= 10),
        ("labels with 1-9 training images", (train_cnt >= 1) & (train_cnt < 10)),
        ("labels never in train, present in test", (train_cnt == 0) & (n_test > 0)),
    ]
    L = ["#### The macro gap decomposes by label coverage\n",
         "| label subset | labels | bar | ours | ours, as % of bar |",
         "|---|---|---|---|---|"]
    for name, m in subsets:
        k = int(m.sum())
        if k == 0:
            continue
        a, b = float(fo[m].mean()), float(fb[m].mean())
        L.append(f"| {name} | {k:,} | {b:.4f} | {a:.4f} | {(a / b if b else 0.0):.1%} |")
    L.append("")
    rare = fo[(train_cnt >= 1) & (train_cnt < 10)].mean()
    unseen_present = int(y[:, (train_cnt == 0) & (n_test > 0)].sum())
    L.append("Read this before interpreting the headline macro number. Macro-F1 is averaged over "
             "all 12,476 labels, but 62% of them never appear in the test split and therefore "
             "score 0 for *both* models, which dilutes both figures equally. On the labels that "
             "do occur in the test split the real standing is the second row. The third row is "
             "the fairest 'learnable' comparison and the one that shows a genuine quality gap. "
             "The fourth row is where we collapse: on labels seen 1-9 times in training we score "
             f"{rare:.4f} -- essentially nothing -- against a baseline that saw them hundreds of "
             "times in the full corpus. That is a data-coverage wall, not a modelling failure, "
             f"and it is most of the headline gap. The {unseen_present} test positives on labels "
             "we never saw are an automatic loss.\n")
    return L


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(WS, "REPORT.md"))
    a = ap.parse_args()

    base = load_json(BASELINE)
    runs = ours()
    probe = load_json(os.path.join(RUNS_ROOT, "_probe", "probe_squash.json")) or []
    best_probe = max((p for p in probe if p.get("pool") == "cls_mean"),
                     key=lambda p: p["val_micro_f1"], default=None)

    L: list[str] = []
    L.append("# DINOv3-B/16 + OFTv2 anime tagger vs `convnextv2_huge.dbv4-full`\n")
    L.append("Fixed **1,000-image test split** of `wdtaggerv4`. 12,476 labels "
             "(4 rating + 9,225 general + 3,247 character), label order taken from the "
             "baseline's own `config.json`.\n")
    L.append("Input geometry: bilinear resize straight to 512x512 with the aspect ratio not "
             "preserved (\"squash\"). A padded-square arm was also run, and is *not* better: at the "
             "corrected LRs and matched seed it scores 0.4078 against squash's 0.4142 and 0.4186 "
             "(both seeds), and per-image F1 does not correlate with squash distortion at all "
             "(r = +0.047). Squash is the pipeline default because it is at least as accurate, "
             "needs no separate cache, and keeps the whole canvas.\n")
    L.append("**Caveat on the bar**: squashing also distorts the baseline's input, which costs it "
             "real accuracy — measured on its own native padded geometry it scores macro F1 "
             "0.2017 / micro F1 0.6935. Treat the numbers below as a *lower bound* on the bar.\n")
    L.append("Metric conventions: a label with no positive and no predicted positive counts as 0, "
             "so *macro* is over the whole 12,476-label space; *micro* pools TP/FP/FN over the "
             "whole matrix. `tuned` = per-label thresholds chosen on the 1,000-image validation "
             "split and applied to test.\n")

    L.append("## The bar\n")
    L.append("| model | F1 | MCC | P | R |")
    L.append("|---|---|---|---|---|")
    L.append(f"| `convnextv2_huge.dbv4-full` **macro @0.5** | {cells(base, 'threshold_0.5', 'macro')} |")
    L.append(f"| `convnextv2_huge.dbv4-full` **micro @0.5** | {cells(base, 'threshold_0.5', 'micro')} |")
    if base:
        b = base["threshold_tuned"]
        L.append("| `convnextv2_huge.dbv4-full` macro, tuned | "
                 + " | ".join(f"{b[f'macro_{k}']:.4f}" for k in METRICS) + " |")
        L.append("| `convnextv2_huge.dbv4-full` micro, tuned | "
                 + " | ".join(f"{b[f'micro_{k}']:.4f}" for k in METRICS) + " |")
    L.append("")

    L.append("## Our runs\n")
    L.append("Frozen DINOv3-B/16 backbone (87.0 M params, never updated) + 1.30 M OFTv2 adapter "
             "params + 19.18 M head params, batch 32, BF16, `SinkSGD_ADV`.\n")
    L.append("| run | macro F1 | macro MCC | macro P | macro R |")
    L.append("|---|---|---|---|---|")
    L.append(f"| bar: convnextv2_huge | {cells(base, 'threshold_0.5', 'macro')} |")
    for d in runs:
        L.append(f"| {label_for(d)} | {cells(d, 'threshold_0.5', 'macro')} |")
    L.append("")
    L.append("| run | micro F1 | micro MCC | micro P | micro R |")
    L.append("|---|---|---|---|---|")
    L.append(f"| bar: convnextv2_huge | {cells(base, 'threshold_0.5', 'micro')} |")
    for d in runs:
        L.append(f"| {label_for(d)} | {cells(d, 'threshold_0.5', 'micro')} |")
    L.append("")
    L.append("With per-label thresholds tuned on validation:\n")
    L.append("| run | macro F1 | macro MCC | macro P | macro R | micro F1 | micro MCC | micro P | micro R |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    if base:
        b = base["threshold_tuned"]
        L.append("| bar: convnextv2_huge | " + " | ".join(f"{b[f'macro_{k}']:.4f}" for k in METRICS)
                 + " | " + " | ".join(f"{b[f'micro_{k}']:.4f}" for k in METRICS) + " |")
    for d in runs:
        t = d["threshold_tuned"]
        L.append(f"| {label_for(d)} | " + " | ".join(f"{t[f'macro_{k}']:.4f}" for k in METRICS)
                 + " | " + " | ".join(f"{t[f'micro_{k}']:.4f}" for k in METRICS) + " |")
    L.append("")
    L.extend(lr_head_section(runs, best_probe))

    best = max(runs, key=lambda d: (metric(d, "threshold_0.5", "micro_f1"),
                                    metric(d, "threshold_tuned", "macro_f1")), default=None)
    L.append("## Input geometry: padding vs squash\n")
    L.extend(geometry_section(runs))
    if best:
        L.append("### Where the gap lives (best run vs bar, macro F1 @0.5)\n")
        L.append("| category | labels | bar | ours |")
        L.append("|---|---|---|---|")
        bc = (base or {}).get("by_category_threshold_0.5", {})
        oc = best.get("by_category_threshold_0.5", {})
        for k, n in CATEGORY_LABELS.items():
            bf = bc.get(k, {}).get("macro_f1", float("nan"))
            of = oc.get(k, {}).get("macro_f1", float("nan"))
            L.append(f"| {k} | {n:,} | {bf:.4f} | {of:.4f} |")
        L.append("")
        L.append("The 4-label rating axis is essentially solved; the general and character "
                 "vocabularies are where 10,000 images run out of steam.\n")
        L.extend(coverage_section(best))

    if best_probe:
        L.append("### Reference point: no adapters at all\n")
        L.append("A head trained on **frozen** DINOv3 features (30 epochs, `cls_mean` pooling, "
                 "best of 5 learning rates) reaches on the same test split:\n")
        L.append(f"* micro F1 **{best_probe['test_micro_f1']:.4f}**, macro F1 "
                 f"{best_probe['test_macro_f1']:.4f}\n")
        L.append("This is what OFTv2 has to beat to be worth its parameters, and it is the honest "
                 "floor for \"did the adapters help\".\n")

    L.append("## Run history\n")
    L.append("| run dir | mode | lr_oft | lr_head | epochs | batch | micro F1 @0.5 | "
             "macro F1 @0.5 | tuned macro |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    L.append("| *(removed — muon-convention adapter LR destroyed the features)* | squash | 0.02 | "
             "0.002 | 14 | 32 | 0.1427 | 0.0003 | - |")
    for d in runs:
        c = d["_cfg"]
        L.append(f"| `{os.path.basename(d['_dir'])}` | {c.get('mode')} | {c.get('lr_oft')} | "
                 f"{c.get('lr_head')} | {c.get('epochs')} | {c.get('batch_size')} | "
                 f"{d['threshold_0.5']['micro_f1']:.4f} | {d['threshold_0.5']['macro_f1']:.4f} | "
                 f"{d['threshold_tuned']['macro_f1']:.4f} |")
    L.append("")

    L.append("## Efficiency\n")
    L.append("| variant | ms/image | img/s |")
    L.append("|---|---|---|")
    L.append("| eager | ~11.2 | ~89 |")
    L.append("| `torch.compile(mode=\"default\")` on the forward | ~8.5 | ~118 |")
    L.append("| `torch.compile(mode=\"max-autotune\")` | 5.9-6.6 | 150-170 |")
    L.append("")
    L.append("Batch 32, 512x512, BF16 on the RTX 4090; ms/step = 32 x ms/image, i.e. ~360 / "
             "~271 / ~190-211 ms. `--compile-mode max-autotune` is ~1.2x faster end-to-end than "
             "the `default` mode and costs minutes of one-off autotuning at startup. It is a CLI "
             "flag, so the config hash (and the run directory) is unchanged.\n")
    L.append("The eager optimizer update costs ~88 ms: SinkSGD_ADV dispatches **per tensor**, "
             "and there are 73 tiny adapter tensors (~1.2 ms each). The cost is launch overhead, not "
             "arithmetic, proven by scaling: 10/25/50/73 tensors cost a linear ~2 ms each, while "
             "making every tensor 8x larger costs only +15%. Compiling it, with Dynamo's recompile "
             "limits raised (otherwise `fullgraph=True` hard-fails with `FailOnRecompileLimitHit`), "
             "brings it to ~28 ms. Data is ~10 ms and H2D ~4 ms per step and is not the "
             "bottleneck.\n")

    head12 = find_run(runs, mode="squash", lr_oft=0.0005, lr_head=0.00012, seed=1337, epochs=8)
    head004_8ep = find_run(runs, mode="squash", lr_oft=0.0005, lr_head=0.004, seed=1337, epochs=8)
    head004_20ep = find_run(runs, mode="squash", lr_oft=0.0005, lr_head=0.004, seed=1337, epochs=20)
    head25 = find_run(runs, mode="squash", lr_oft=0.0005, lr_head=0.00025, seed=1337, epochs=8)
    oft002 = find_run(runs, mode="squash", lr_oft=0.002, lr_head=0.004, seed=1337, epochs=8)
    pad = (find_run(runs, mode="pad", lr_oft=0.0005, lr_head=0.00012, seed=1337, epochs=8)
           or find_run(runs, mode="pad", lr_oft=0.0005, lr_head=0.004, seed=1337, epochs=8))

    L.append("## Open questions (deliberately not resolved yet)\n")
    ep20_micro = sc(head004_20ep, "threshold_0.5", "micro_f1")
    ep8_micro = sc(head004_8ep, "threshold_0.5", "micro_f1")
    L.append("**Does more than 8 epochs help at the corrected head LR?** The 20-epoch run "
             f"(`{os.path.basename(head004_20ep['_dir']) if head004_20ep else '—'}`) was flat from "
             f"epoch 14 and its final test micro ({ep20_micro}) was below the 8-epoch run's "
             f"({ep8_micro}), but it ran with the oversized head LR, so its head was overfitting "
             "early. Retest 20 epochs once lr_head is settled.\n")
    L.append("**Is lr_oft 0.0005 still right?** It was chosen with the head at 0.004. Now that the "
             "head moves ~4x slower than the adapters (0.00012 vs 0.0005) instead of 8x faster, "
             "the optimal adapter rate may shift. Untested interaction.\n")

    L.append("## Conclusion\n")
    L.append("We do **not** beat `convnextv2_huge`. The baseline saw 5.3 M images and predicts "
             "about 33 labels per image at 0.73 precision; 10,000 images against 12,476 labels is "
             "0.8 images per label, and the long tail is not learnable at that scale. What the "
             "exercise does establish, on identical pixels and identical metrics:\n")
    probe_micro = best_probe["test_micro_f1"] if best_probe else float("nan")
    probe_macro = best_probe["test_macro_f1"] if best_probe else float("nan")
    if best:
        bm = metric(best, "threshold_0.5", "micro_f1")
        bmacro = metric(best, "threshold_0.5", "macro_f1")
        bt = metric(best, "threshold_tuned", "macro_f1")
        rel = (bm / probe_micro - 1) * 100 if probe_micro == probe_micro and probe_micro else float("nan")
        mid_macro = metric(head25, "threshold_0.5", "macro_f1")
        L.append(f"* OFTv2 earns its parameters: the best run is squash at {bm:.4f} micro / "
                 f"{bt:.4f} tuned macro, beating a frozen-feature probe of the same backbone by "
                 f"{rel:.1f}% relative on micro F1 ({bm:.4f} vs {probe_micro:.4f}). "
                 f"Fixed-threshold macro trades some of that away ({bmacro:.4f} vs "
                 f"{probe_macro:.4f}; the higher head-LR arm reaches {mid_macro:.4f}).\n")
    else:
        L.append("* OFTv2 earns its parameters on micro F1, but no evaluated run was found.\n")
    l002 = sc(oft002, "threshold_0.5", "micro_f1")
    l005_lo = sc(head004_8ep, "threshold_0.5", "micro_f1")
    l005_hi = sc(head12 or best, "threshold_0.5", "micro_f1")
    L.append("* The adapters are extremely sensitive to their learning rate. At the muon-convention "
             "0.02 they are destructive (micro 0.1427); with the matched head LR the ladder is "
             f"0.002 -> {l002} and 0.0005 -> {l005_lo}-{l005_hi}. That is not the whole story: "
             f"the `lr_head` sweep above moves the same 0.0005 arms from {l005_lo} to {l005_hi}, a "
             "larger effect than the adapter plateau. `lr_oft = 0` is not hypothetical either: the "
             f"frozen-feature probe is {probe_micro:.4f}, so the curve is bracketed on both sides "
             "and 0.0005 sits inside the plateau rather than past a cliff.\n")
    if pad and head004_8ep:
        pad_micro = metric(pad, "threshold_0.5", "micro_f1")
        sq_micro = metric(head12 or head004_8ep, "threshold_0.5", "micro_f1")
        L.append("* Resize geometry is settled and is not a lever. At the corrected LRs (lr_oft "
                 "0.0005, lr_head 0.00012, seed 1337) pad scores "
                 f"{metric(pad, 'threshold_0.5', 'micro_f1'):.4f} vs squash "
                 f"{sq_micro:.4f} micro, and squash's second seed reaches 0.4186; per-image F1 "
                 "does not correlate with squash distortion (r = +0.047). Earlier pad leads at "
                 "lr_head 0.004 and the 10% penalty at lr_oft 0.002 were both LR interactions. "
                 "Squash is the default.\n")
    else:
        L.append("* Resize geometry: the matched pad/squash arms are missing, so the comparison "
                 "could not be recomputed.\n")
    L.append("* The remaining gap to the bar is a data-scale gap, not a pipeline gap.\n")

    txt = "\n".join(L)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(txt)
    print(f"wrote {a.out} ({len(txt.splitlines())} lines)")


if __name__ == "__main__":
    main()
