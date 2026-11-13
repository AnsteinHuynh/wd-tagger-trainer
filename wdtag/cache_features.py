"""Cache pooled features from a checkpoint so head experiments stop costing GPU hours.

The production model is a frozen SigLIP2 tower plus OFT adapters, one unfrozen block and a
linear head, at 90 to 120 images per second for the whole training step. Almost all of that
time is the backbone. Once a checkpoint is chosen, its features are fixed, and every head,
loss, calibration and ensembling experiment can be run against a memmap instead of a tar.

This writes, for one split:
  feats.npy   float16 memmap, (n, dim), the pooled feature vector before the head
  labels.npz  indices/offsets, the same CSR format the validation cache uses
  meta.json   n_images, dim, run, checkpoint, seed, pass, image size, split

Order is a pure function of (seed, pass_index, split), and rows are written in that order, so
row i of feats.npy always belongs to row i of labels.npz. A partial cache resumes by skipping
the samples already written.

Usage:
  python -m wdtag.cache_features --run runs/459e2a02 --split valid --out I:/.cache/feat/459e2a02
  python -m wdtag.cache_features --run runs/459e2a02 --split train --limit 50000 --out ...
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np
import torch

CACHE_ROOT = os.environ.get("WDTAG_CACHE_ROOT", os.path.join(".cache", "wd-tagger-trainer"))


def parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="cache pooled features from a run checkpoint")
    ap.add_argument("--run", required=True, help="run directory")
    ap.add_argument("--ckpt", default="", help="checkpoint filename; default the newest")
    ap.add_argument("--split", default="train", choices=("train", "valid", "test"))
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--limit", type=int, default=0, help="stop after this many images")
    ap.add_argument("--batch", type=int, default=64, help="forward batch, no gradients")
    ap.add_argument("--compute-dtype", default="bf16", choices=("bf16", "fp32"),
                    help="arithmetic precision of the pass; fp32 is slower and buys no measured accuracy")
    ap.add_argument("--store-dtype", default="fp16", choices=("fp16", "fp32"),
                    help="container on disk; fp16 is exact for bf16 output")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--pass-index", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--prefetch", type=int, default=8)
    ap.add_argument("--log-every", type=int, default=200)
    return ap.parse_args()


def newest_ckpt(run_dir: str) -> str:
    found = sorted(glob.glob(os.path.join(run_dir, "ckpt_step*.pt")),
                   key=lambda p: int(os.path.basename(p).split("step")[1].split(".")[0]))
    if not found:
        raise SystemExit(f"no checkpoints in {run_dir}")
    return os.path.basename(found[-1])


def checkpoint_labels(path_dir: str, row: int, **arrays):
    """Write labels to a new file named for its row and return that name.

    Never renames over an existing file: a rename onto labels.npz raised WinError 5 once at row
    418560 and killed a six-hour run, because Windows can hold the target open for a moment after a
    large write. A fresh name cannot collide, so the checkpoint either appears whole or not at all,
    and meta.json is the pointer that decides which one a resume trusts.
    """
    name = f"labels_{row:07d}.npz"
    np.savez(os.path.join(path_dir, name), **arrays)
    for old in sorted(glob.glob(os.path.join(path_dir, "labels_*.npz")))[:-2]:
        try:
            os.remove(old)
        except OSError:
            pass
    return name


def atomic_json(path: str, obj) -> None:
    tmp = path + ".tmp"
    json.dump(obj, open(tmp, "w"), indent=1)
    os.replace(tmp, path)


def main() -> int:
    a = parse()
    from .config import TrainConfig
    from .model import DinoTagger

    device = "cuda"
    cfg = TrainConfig.from_json(os.path.join(a.run, "config.json"))
    n_labels = int(cfg.n_labels) if hasattr(cfg, "n_labels") else 12476
    # A raw backbone has no checkpoint: identity adapters, nothing to load, still a valid cache.
    try:
        ckpt = a.ckpt or newest_ckpt(a.run)
    except SystemExit:
        ckpt = ""
    compute = torch.float32 if a.compute_dtype == "fp32" else torch.bfloat16
    np_store = np.float32 if a.store_dtype == "fp32" else np.float16
    torch_store = torch.float32 if a.store_dtype == "fp32" else torch.float16
    model = DinoTagger(cfg, n_labels, dtype=compute).to(device)
    if ckpt:
        ck = torch.load(os.path.join(a.run, ckpt), map_location="cpu", weights_only=False)
        try:
            model.load_adapter_state_dict(ck["adapter"], strict=False)
        except RuntimeError as e:
            # A wider pooling set changes the width of norm and the head, so the loader refuses even
            # though every backbone and adapter tensor matched and was copied before it raised. That
            # is exactly the case here: the head we are about to train is new, so only the trunk
            # matters. Say so loudly rather than hiding it.
            print(f"[load] head and norm left at init, trunk and adapters copied: {e}", flush=True)
        print(f"loaded {ckpt}", flush=True)
    else:
        print("no checkpoint in the run directory: caching the raw backbone", flush=True)
    model.eval()

    os.makedirs(a.out, exist_ok=True)
    meta_path = os.path.join(a.out, "meta.json")
    feats_path = os.path.join(a.out, "feats.npy")
    labels_path = os.path.join(a.out, "labels.npz")

    if a.split == "valid":
        total = int(a.limit or 50000)
        src = os.path.join(CACHE_ROOT, "valid_nested", "images.npy")
        images = np.load(src, mmap_mode="r")
        lab = np.load(os.path.join(CACHE_ROOT, "valid_nested", "labels.npz"))
        n_rows = min(total, images.shape[0])

        dim = None
        start = 0
        if os.path.isfile(feats_path) and os.path.isfile(meta_path):
            meta = json.load(open(meta_path))
            prev = np.load(feats_path, mmap_mode="r")
            start, dim = prev.shape[0], prev.shape[1]
            print(f"resuming validation cache at {start} rows of dim {dim}", flush=True)
        if start == 0:
            with torch.no_grad():
                probe = torch.from_numpy(np.ascontiguousarray(images[:1])).to(device)
                dim = int(model.forward_features(model._prep(probe)).shape[-1])
            np.lib.format.open_memmap(feats_path, mode="w+", dtype=np_store,
                                      shape=(n_rows, dim))
            start = 0
            print(f"validation cache: {n_rows} rows of dim {dim}", flush=True)
        fm = np.lib.format.open_memmap(feats_path, mode="r+")
        t0 = time.time()
        done = start
        with torch.no_grad():
            for s in range(start, n_rows, a.batch):
                e = min(s + a.batch, n_rows)
                xb = torch.from_numpy(np.ascontiguousarray(images[s:e])).to(device)
                f = model.forward_features(model._prep(xb))
                fm[s:e] = f.to(torch_store).float().cpu().numpy()
                done = e
                if done % 5000 < a.batch or done == n_rows:
                    rate = done / max(1e-9, time.time() - t0)
                    print(f"{done}/{n_rows} rows, {rate:.0f} img/s", flush=True)
        fm.flush()
        # CSR prefix only: the full indices array covers all 50k rows, and a shorter cache must
        # not carry labels for images it never wrote.
        cut = int(lab["offsets"][n_rows])
        np.savez(labels_path, indices=lab["indices"][:cut], offsets=lab["offsets"][:n_rows + 1])
        json.dump({"split": "valid", "n_images": n_rows, "dim": dim, "run": a.run,
                   "ckpt": ckpt, "source": src}, open(meta_path, "w"), indent=1)
        print("valid cache done", flush=True)
        return 0

    from .stream import TarStream, KNOWN_TOTAL

    total = int(a.limit or KNOWN_TOTAL[a.split])

    # A memmap is preallocated to the full row count, so its shape says nothing about how far the
    # cache got. Progress lives in meta.json, written at the start and rewritten as rows land; a
    # thirteen-hour job that cannot resume is a job that loses thirteen hours.
    meta = {"split": a.split, "n_images": total, "dim": None, "run": a.run, "ckpt": ckpt,
            "seed": a.seed, "pass_index": a.pass_index, "image_size": int(cfg.image_size),
            "written": 0}
    start = 0
    if os.path.isfile(feats_path) and os.path.isfile(meta_path):
        old = json.load(open(meta_path))
        written = int(old.get("written", 0))
        if written > 0:
            meta["dim"] = old.get("dim")
            # The pointer to the labels file must come back too. Restoring the row count without it
            # silently loaded an older labels file, 512 rows short, and the offsets then did not line
            # up with the row counter.
            meta["labels"] = old.get("labels")
            start = written
            print(f"resuming at {start} of {total} rows", flush=True)

    dim = meta.get("dim")
    if not dim:
        with torch.no_grad():
            probe = torch.zeros(1, 3, int(cfg.image_size), int(cfg.image_size), device=device)
            dim = int(model.forward_features(model._prep(probe)).shape[-1])
        np.lib.format.open_memmap(feats_path, mode="w+", dtype=np_store, shape=(total, dim))
        print(f"train cache: {total} rows of dim {dim}", flush=True)
    meta["dim"] = dim
    meta["written"] = start
    json.dump(meta, open(meta_path, "w"), indent=1)

    fm = np.lib.format.open_memmap(feats_path, mode="r+")
    if fm.shape[1] != dim:
        raise SystemExit(f"cache at {feats_path} is dim {fm.shape[1]}, the model says {dim}; "
                         "refusing to append features of a different width")
    if fm.shape[0] != total:
        # A memmap cannot be resized, so a cache begun with a small --limit has to be copied into a
        # larger one before it can be extended. Chunked so a 32 GB cache does not need 32 GB of RAM.
        if fm.shape[0] > total:
            total = int(fm.shape[0])
            print(f"cache already holds {total} rows, keeping that size", flush=True)
        else:
            tmp = feats_path + ".grow"
            grown = np.lib.format.open_memmap(tmp, mode="w+", dtype=np_store,
                                              shape=(total, dim))
            for s in range(0, fm.shape[0], 262144):
                e = min(s + 262144, fm.shape[0])
                grown[s:e] = fm[s:e]
            grown.flush()
            del grown, fm
            os.replace(tmp, feats_path)
            fm = np.lib.format.open_memmap(feats_path, mode="r+")
            print(f"grew the cache from {start} to {total} rows", flush=True)
    meta["n_images"] = int(total)
    # Labels are append-only raw arrays, one int32 per tag and one int64 cumulative offset per row.
    # They were rewritten in full at every checkpoint, which at 1.5 million rows meant a 198 MB file
    # every four seconds and a cost that grew with the cache; that is why throughput fell from 195 to
    # 147 images per second as the run went on. Appending makes the total label I/O equal to the
    # labels themselves.
    idx_path = os.path.join(os.path.dirname(feats_path), "labels_indices.i32")
    off_path = os.path.join(os.path.dirname(feats_path), "labels_offsets.i64")

    have_idx = 0
    if start > 0 and os.path.isfile(off_path) and os.path.isfile(idx_path):
        offs = np.fromfile(off_path, dtype=np.int64)
        if offs.size < start:
            raise SystemExit(f"{off_path} holds {offs.size} offsets but the cache claims {start}")
        offs = offs[:start]
        have_idx = int(offs[-1]) if start else 0
        # Truncate whatever a crash left half-appended, so the files match the counter exactly.
        with open(idx_path, "r+b") as fh:
            fh.truncate(have_idx * 4)
        with open(off_path, "r+b") as fh:
            fh.truncate(start * 8)
        print(f"resumed {start} label rows, {have_idx} tag indices", flush=True)
    elif start > 0:
        # Migrate a cumulative npz written before labels became append-only.
        ptr = meta.get("labels") or f"labels_{start:07d}.npz"
        src = os.path.join(os.path.dirname(feats_path), ptr)
        if not os.path.isfile(src):
            raise SystemExit(f"no labels for {start} rows: neither {off_path} nor {src}")
        z = np.load(src)
        if len(z["offsets"]) - 1 != start:
            raise SystemExit(f"{src} covers {len(z['offsets']) - 1} rows but the cache claims {start}")
        counts = np.diff(z["offsets"]).astype(np.int64)
        offs = np.cumsum(counts)
        z["indices"].astype(np.int32).tofile(idx_path)
        offs.tofile(off_path)
        z.close()  # np.load is lazy and holds the zip open until this is called
        have_idx = int(offs[-1]) if start else 0
        print(f"migrated {start} label rows from {os.path.basename(src)}", flush=True)
    else:
        open(idx_path, "wb").close()
        open(off_path, "wb").close()

    fi = open(idx_path, "ab")
    fo = open(off_path, "ab")
    pending_idx: list[np.ndarray] = []
    pending_counts: list[np.ndarray] = []
    n_pending = 0

    def flush_labels(row_now: int) -> None:
        """Append what is buffered, then advance the counter. Called with the row it reaches."""
        nonlocal have_idx, n_pending
        if pending_counts:
            counts = np.concatenate(pending_counts).astype(np.int64)
            idx = (np.concatenate(pending_idx).astype(np.int32) if pending_idx
                   else np.zeros(0, dtype=np.int32))
            cum = (np.cumsum(counts) + have_idx).astype(np.int64)
            idx.tofile(fi)
            cum.tofile(fo)
            fi.flush()
            fo.flush()
            have_idx = int(cum[-1]) if cum.size else have_idx
            pending_idx.clear()
            pending_counts.clear()
            n_pending = 0
        meta["labels_format"] = "raw"
        meta["written"] = int(row_now)

    row = start
    t0 = time.time()
    st = TarStream(a.split, seed=a.seed, pass_index=a.pass_index, batch_size=a.batch,
                   n_labels=n_labels, workers=a.workers, prefetch=a.prefetch,
                   skip=start, max_samples=total)
    with torch.no_grad():
        for item in st.batches():
            images, labels = item[0], item[1]
            take = min(len(images), total - row)
            if take <= 0:
                break
            xb = torch.from_numpy(np.ascontiguousarray(images[:take])).to(device)
            f = model.forward_features(model._prep(xb))
            fm[row:row + take] = f.to(torch_store).float().cpu().numpy()
            y = labels[:take]
            y = y.detach().cpu().numpy() if torch.is_tensor(y) else np.asarray(y)
            nz = np.nonzero(y > 0.5)
            pending_idx.append(nz[1].astype(np.int32))
            pending_counts.append(np.bincount(nz[0], minlength=take).astype(np.int64))
            n_pending += take
            row += take
            if row % a.log_every < a.batch:
                rate = (row - start) / max(1e-9, time.time() - t0)
                print(f"{row}/{total} rows, {rate:.0f} img/s", flush=True)
                # The features are flushed before the counter moves, so a resume never trusts a row
                # whose page never reached disk.
                fm.flush()
                flush_labels(int(row))
                try:
                    atomic_json(meta_path, meta)
                except OSError as e:
                    print(f"[ckpt] meta not advanced ({e}); the previous row stands", flush=True)
            if row >= total:
                break
    fm.flush()
    flush_labels(int(row))
    meta["n_images"] = int(row)
    meta["labels_indices"] = have_idx
    atomic_json(meta_path, meta)
    fi.close()
    fo.close()
    print(f"done, {row} rows, dim {dim}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
















