"""Standalone 50k validation, separate from training: a step holds ~21/24.5 GB, so 256@512px batches cannot coexist. RTX 4090 measured: 256=246 img/s @7.5 GB, 384=258 @11.0 GB, 512 maxes VRAM; 50k ~3.4 min.

1k/5k/10k/50k are prefixes of ONE array, so one pass yields all; batches straddle boundaries, so each subset counts only rows below its size. Reused buffer, not a long-lived memmap: a 39.3 GB map showed 35.9 GB working set (6.7 GB private), starving a 47.9 GB machine; .npy's 128-byte header is skipped by np.load, not a raw read.

Exact Macro/Micro F1/MCC/P/R from per-(threshold,label) TP/FP/FN/TN counts, no 50k x 12476 matrix.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

from .cache import CACHE_ROOT
from .labels import load_label_space

NESTED = os.path.join(CACHE_ROOT, "valid_nested")
PROBS_DIR = os.path.join(CACHE_ROOT, "probs")
DEFAULT_SIZES = "1000,5000,10000,50000"


class ImageChunkReader:
    """Sequential reader with a single reused buffer (bounded RAM, no long mapping)."""

    def __init__(self, path: str, shape: tuple, itemsize: int = 1):
        self.f = open(path, "rb")
        version = np.lib.format.read_magic(self.f)
        if version == (1, 0):
            _, _, dtype = np.lib.format.read_array_header_1_0(self.f)
        elif version == (2, 0):
            _, _, dtype = np.lib.format.read_array_header_2_0(self.f)
        else:
            raise ValueError(f"unsupported .npy version {version}")
        if dtype != np.uint8:
            raise ValueError(f"expected uint8 cache, found {dtype}")
        self.offset = self.f.tell()
        self.row = int(np.prod(shape[1:])) * itemsize
        self.tail = tuple(shape[1:])
        self.buf = bytearray(self.row * 8)

    def slice(self, start: int, end: int) -> np.ndarray:
        n = end - start
        need = n * self.row
        if len(self.buf) < need:
            self.buf = bytearray(need)
        mv = memoryview(self.buf)[:need]
        self.f.seek(self.offset + start * self.row)
        got = 0
        while got < need:
            k = self.f.readinto(mv[got:])
            if not k:
                raise IOError(f"short read at {start}: {got}/{need}")
            got += k
        return np.frombuffer(mv, dtype=np.uint8).reshape((n,) + self.tail)

    def close(self):
        self.f.close()


def metrics_from_counts(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray,
                        n_images: int) -> dict:
    """Exact Macro/Micro F1/MCC/P/R from per-(threshold, label) counts."""
    tn = n_images - tp - fp - fn
    eps = 1e-12
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, eps)
    prec = tp / np.maximum(tp + fp, eps)
    rec = tp / np.maximum(tp + fn, eps)
    den = np.sqrt(np.maximum(tp + fp, 0) * np.maximum(tp + fn, 0)
                  * np.maximum(tn + fp, 0) * np.maximum(tn + fn, 0))
    mcc = np.where(den > 0, (tp * tn - fp * fn) / np.maximum(den, eps), 0.0)

    TP, FP, FN, TN = tp.sum(1), fp.sum(1), fn.sum(1), tn.sum(1)
    P, R = TP / np.maximum(TP + FP, eps), TP / np.maximum(TP + FN, eps)
    mden = np.sqrt(np.maximum(TP + FP, 0) * np.maximum(TP + FN, 0)
                   * np.maximum(TN + FP, 0) * np.maximum(TN + FN, 0))
    return {
        "macro_f1": f1.mean(1), "macro_mcc": mcc.mean(1),
        "macro_precision": prec.mean(1), "macro_recall": rec.mean(1),
        "micro_f1": 2 * P * R / np.maximum(P + R, eps),
        "micro_mcc": np.where(mden > 0, (TP * TN - FP * FN) / np.maximum(mden, eps), 0.0),
        "micro_precision": P, "micro_recall": R,
        "macro_f1_per_tag_best": float(f1.max(0).mean()),
        "n_supported_labels": int((tp.max(0) + fn.max(0) > 0).sum()),
    }


def load_run_model(run_dir: str, ckpt: str, n_labels: int, device: str):
    from .config import TrainConfig
    from .model import DinoTagger

    cfg_path = os.path.join(run_dir, "config.json")
    cfg = TrainConfig.from_json(cfg_path)
    if not ckpt:
        found = sorted(glob.glob(os.path.join(run_dir, "ckpt_step*.pt")),
                       key=lambda p: int(os.path.basename(p).split("step")[1].split(".")[0]))
        if not found:
            raise SystemExit(f"no checkpoints in {run_dir}")
        ckpt = os.path.basename(found[-1])
    path = ckpt if os.path.isabs(ckpt) else os.path.join(run_dir, ckpt)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = DinoTagger(cfg, n_labels).to(device)
    model.load_adapter_state_dict(ck["adapter"], strict=False)
    return model.eval(), path, cfg


def main() -> int:
    ap = argparse.ArgumentParser(description="standalone nested-cache validation")
    ap.add_argument("--run", default="", help="run directory (resolve by DIR, not config)")
    ap.add_argument("--ckpt", default="", help="checkpoint filename or absolute path")
    ap.add_argument("--bar", action="store_true", help="evaluate the convnextv2_huge bar")
    ap.add_argument("--sizes", default=DEFAULT_SIZES)
    # Default batch is model-dependent (5x size): our 129M=7.5 GB at 256; the 692.6M bar
    # needs 23.7 GB at 128 (96% of card), 17.0 GB at 64. One default maxed VRAM for no gain.
    ap.add_argument("--batch-size", type=int, default=None)
    # The bar reports one fixed point, Macro@0.40/Micro@0.40, the only comparison that
    # matters. lo:hi:step is diagnostic only; grid-search is not the protocol.
    ap.add_argument("--thr", default="0.40")
    ap.add_argument("--cal-bias", default="", help="npy of per-tag logit biases to add")
    ap.add_argument("--probs-dir", default=PROBS_DIR)
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--out", default="", help="defaults to <run>/validation.json")
    a = ap.parse_args()
    if not a.run and not a.bar:
        ap.error("--run is required unless --bar is given")
    if a.batch_size is None:
        a.batch_size = 64 if a.bar else 256

    n_labels = len(load_label_space())
    device = "cuda"
    meta = json.load(open(os.path.join(NESTED, "meta.json"), encoding="utf-8"))
    total = int(meta["count"])
    sizes = sorted(int(s) for s in a.sizes.split(",") if int(s) <= total)
    lab = np.load(os.path.join(NESTED, "labels.npz"))
    idx_all, offs = lab["indices"], lab["offsets"]
    if ":" in a.thr:
        lo, hi, st = (float(x) for x in a.thr.split(":"))
        thresholds = np.arange(lo, hi + 1e-9, st)
    else:
        thresholds = np.array([float(a.thr)])
    T = len(thresholds)

    if a.bar:
        from .baseline import MEAN, STD, load_baseline
        model = load_baseline(device)
        ckpt_path = "convnextv2_huge.dbv4-full"
        tag = "bar_convnextv2_huge"
    else:
        model, ckpt_path, cfg = load_run_model(a.run, a.ckpt, n_labels, device)
        tag = os.path.basename(a.run).rstrip("\\/") + "_" + \
              os.path.basename(ckpt_path).replace("ckpt_step", "").replace(".pt", "")
    # an ensemble run trained against a feature cache must score against the same frozen
    # view: pair each image row with its cached second-tower feature
    vf = None
    if getattr(cfg, "feat_cache", "") and getattr(cfg, "feat_val_cache", ""):
        vf = np.load(cfg.feat_val_cache, mmap_mode="r")
        print(f"[val] frozen view from {cfg.feat_val_cache}: {len(vf)} rows", flush=True)
    print(f"checkpoint {ckpt_path}\nrun {a.run}  sizes {sizes}  "
          f"batch {a.batch_size}  thresholds {T} ({thresholds[0]:.2f}..{thresholds[-1]:.2f})",
          flush=True)

    os.makedirs(a.probs_dir, exist_ok=True)
    pf = os.path.join(a.probs_dir, f"{tag}.npy")
    counts = {s: {k: torch.zeros((T, n_labels), dtype=torch.float64, device=device)
                  for k in ("tp", "fp", "fn")} for s in sizes}

    def batch_targets(start: int, end: int) -> torch.Tensor:
        y = np.zeros((end - start, n_labels), dtype=np.float32)
        for i in range(start, end):
            lo_i, hi_i = int(offs[i]), int(offs[i + 1])
            y[i - start, idx_all[lo_i:hi_i]] = 1.0
        return torch.from_numpy(y).to(device) > 0.5

    def accumulate(probs, true, start, end):
        si = 0
        while si < len(sizes) and sizes[si] <= start:
            si += 1
        for s in sizes[si:]:
            lim = min(end, s) - start
            if lim <= 0:
                continue
            pr, tr = probs[:lim], true[:lim]
            for ti, thr in enumerate(thresholds):
                pred = pr >= thr
                counts[s]["tp"][ti] += (pred & tr).sum(0)
                counts[s]["fp"][ti] += (pred & ~tr).sum(0)
                counts[s]["fn"][ti] += (~pred & tr).sum(0)

    t0 = time.time()
    # The probability file is allocated at the full count, but only rows below sizes[-1]
    # are written, so a sidecar records valid rows: without it a 5k cache was reused for
    # 50k and scored 45000 zero rows (macro 0.0371 instead of 0.2594).
    sidecar = pf + ".json"
    cached_rows = 0
    if os.path.exists(pf) and os.path.exists(sidecar):
        try:
            cached_rows = int(json.load(open(sidecar, encoding="utf-8"))["rows"])
        except Exception:
            cached_rows = 0
    if a.recompute or cached_rows < sizes[-1]:
        reader = ImageChunkReader(os.path.join(NESTED, "images.npy"),
                                  (total, 3, meta["size"], meta["size"]))
        pm = np.lib.format.open_memmap(pf, mode="w+", dtype=np.float16,
                                       shape=(total, n_labels))
        if a.bar:
            mean_t = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
            std_t = torch.tensor(STD, device=device).view(1, 3, 1, 1)
        cal_bias = None
        if a.cal_bias:
            cal_bias = torch.tensor(np.load(a.cal_bias), dtype=torch.float32)
            if cal_bias.numel() != n_labels:
                raise SystemExit(f"--cal-bias has {cal_bias.numel()} values, expected {n_labels}")
            print(f"calibration bias from {a.cal_bias}, median {float(cal_bias.median()):+.3f}",
                  flush=True)
        with torch.no_grad():
            for start in range(0, sizes[-1], a.batch_size):
                end = min(start + a.batch_size, sizes[-1])
                xb = torch.from_numpy(
                    np.ascontiguousarray(reader.slice(start, end))).to(device)
                if a.bar:
                    xf = xb.float().div_(255.0)
                    logits = model(((xf - mean_t) / std_t).to(torch.bfloat16))
                elif vf is not None:
                    fb = torch.from_numpy(np.ascontiguousarray(vf[start:end])).to(device)
                    logits = model(xb, fb)
                else:
                    logits = model(xb)
                # A run trained with per-tag rate matching saves that bias beside the checkpoint;
                # without it the checkpoint alone over-predicts wildly, because the model's weights
                # drifted to compensate for a bias applied during training.
                if cal_bias is not None:
                    logits = logits.float() + cal_bias.to(device)[None, :]
                probs = torch.sigmoid(logits.float())
                pm[start:end] = probs.half().cpu().numpy()
                accumulate(probs, batch_targets(start, end), start, end)
        pm.flush()
        del pm
        reader.close()
        with open(sidecar, "w", encoding="utf-8") as fh:
            json.dump({"rows": sizes[-1], "total": total, "labels": n_labels}, fh)
    else:
        print(f"reusing cached probabilities {pf} ({cached_rows:,} rows)", flush=True)
        pm = np.load(pf, mmap_mode="r")
        cal_bias = None
        if a.cal_bias:
            cal_bias = torch.tensor(np.load(a.cal_bias), dtype=torch.float32)
            if cal_bias.numel() != n_labels:
                raise SystemExit(f"--cal-bias has {cal_bias.numel()} values, expected {n_labels}")
            print(f"calibration bias from {a.cal_bias}, median {float(cal_bias.median()):+.3f}",
                  flush=True)
        with torch.no_grad():
            for start in range(0, sizes[-1], a.batch_size):
                end = min(start + a.batch_size, sizes[-1])
                pr = torch.from_numpy(
                    np.ascontiguousarray(pm[start:end])).to(device).float()
                accumulate(pr, batch_targets(start, end), start, end)
        del pm
    elapsed = time.time() - t0
    print(f"scored {sizes[-1]:,} images in {elapsed:.1f}s "
          f"({sizes[-1]/elapsed:.0f} img/s)", flush=True)

    report = {"checkpoint": ckpt_path, "run": a.run, "batch_size": a.batch_size,
              "seconds": round(elapsed, 1), "thresholds": thresholds.tolist(),
              "subsets": {}}
    for s in sizes:
        m = metrics_from_counts(counts[s]["tp"].cpu().numpy(),
                                counts[s]["fp"].cpu().numpy(),
                                counts[s]["fn"].cpu().numpy(), s)
        report["subsets"][str(s)] = {
            k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in m.items()}
        bi = int(np.argmax(m["macro_f1"]))
        i50 = int(np.argmin(np.abs(thresholds - 0.50)))
        print(f"\n=== {s:,} images ({m['n_supported_labels']:,} supported labels) ===")
        print(f"  at {thresholds[i50]:.2f}: macro F1 {m['macro_f1'][i50]:.4f} MCC {m['macro_mcc'][i50]:.4f} "
              f"P {m['macro_precision'][i50]:.4f} R {m['macro_recall'][i50]:.4f} | "
              f"micro F1 {m['micro_f1'][i50]:.4f} MCC {m['micro_mcc'][i50]:.4f} "
              f"P {m['micro_precision'][i50]:.4f} R {m['micro_recall'][i50]:.4f}")
        if T > 1:
            print(f"  best macro @thr {thresholds[bi]:.2f}: macro F1 {m['macro_f1'][bi]:.4f} "
                  f"MCC {m['macro_mcc'][bi]:.4f} | micro F1 {m['micro_f1'][bi]:.4f} "
                  f"MCC {m['micro_mcc'][bi]:.4f}")
            print(f"  macro F1 with per-tag best thresholds: {m['macro_f1_per_tag_best']:.4f}")

    out = a.out or (os.path.join(a.run, "validation.json") if a.run
                    else os.path.join(os.path.dirname(NESTED), "bar_validation.json"))
    report["review"] = {"note": "macro/micro F1/MCC/P/R at each threshold; "
                                "macro_f1_per_tag_best tunes a threshold per tag"}
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

