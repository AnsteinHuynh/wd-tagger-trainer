"""Per-tar fp32 cls-only feature cache from a trained trunk. Extendable and resumable.

One shard per tar, written atomically at tar completion, so a crash loses only the tar in
flight and resume is: read meta.json, skip tars already recorded, cache the rest.

  <stem>.npy   (rows, dim) in the store dtype, cls pool of the trained trunk
  <stem>.npz   indices int32, offsets int64 (CSR, cumulative ends, len = rows + 1)
  meta.json    run, ckpt, pool, dtypes, and per-tar {rows, dim, blake2b} records

Usage:
  python -m wdtag.cache_fp32_cls --tars 00001
  python -m wdtag.cache_fp32_cls --tars 00002,00003            # extend
  python -m wdtag.cache_fp32_cls --tars all                    # every tar in the split
  python -m wdtag.cache_fp32_cls --tars 00001 --compute-dtype bf16 --out <other dir>

Speed options, off by default so the shipped behaviour is unchanged, measured in
scratch/sophie/cache-throughput.md (fp32, batch 192: 78.5 -> 110.8 img/s with both):
  --overlap     decode into a pinned double buffer on a producer thread, so the 2.5-3.5 ms/img
                of PIL decode and the tar read happen under the forward instead of before it
  --cache-rot   build every OFT Cayley rotation once at load. Valid only because this pass is
                eval + no_grad and the weights never change: the rotation is then a constant,
                and the value is bit-identical to rebuilding it per forward (verified by
                comparing cached shards byte for byte)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

DEFAULT_RUN = "runs/459e2a02"


def parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="per-tar fp32 cls-only feature cache")
    ap.add_argument("--run", default=DEFAULT_RUN, help="run directory of the trained trunk")
    ap.add_argument("--ckpt", default="ckpt_step016000.pt",
                    help="checkpoint holding the trained trunk and adapters")
    ap.add_argument("--tars", default="00001",
                    help="comma list of tar stems ('00001,00002') or 'all'")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", default=os.path.join(".cache", "wd-tagger-trainer", "feat", "fp32-cls"))
    ap.add_argument("--batch", type=int, default=192, help="images per GPU forward")
    ap.add_argument("--compute-dtype", default="fp32", choices=("fp32", "bf16"))
    ap.add_argument("--store-dtype", default="fp32", choices=("fp32", "fp16", "bf16"),
                    help="fp32 for a lossless container; bf16 stores exact bfloat16 bits in a "
                         "uint16 container, because numpy has no bfloat16 dtype and .npy cannot "
                         "round-trip an extension one")
    ap.add_argument("--workers", type=int, default=12, help="CPU decode threads")
    ap.add_argument("--overlap", action="store_true",
                    help="pinned double buffer on a producer thread; decode and tar read happen "
                         "under the previous forward instead of before it (same rows/order)")
    ap.add_argument("--cache-rot", action="store_true",
                    help="build each OFT Cayley rotation once at load instead of per forward; "
                         "bit-identical because the weights are frozen for the whole pass")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N images of the first tar (benchmarking only); the shard is "
                         "written but recorded as limited, so a later invocation redoes that tar")
    return ap.parse_args()


def range_report(feats: np.ndarray) -> dict:
    """Count the values the chosen container cannot hold, so that choice is checked per tar.

    fp16 max is 65504 and its smallest subnormal is 2^-24; anything past either is lost as inf
    or zero. bfloat16 spans fp32's exponent range, so it needs its own bound (~3.39e38).
    Pooled LayerNorm-terminated features are bounded in practice, but "in practice" is exactly
    the assertion this replaces: with a lossy container a violation refuses the tar.
    """
    finite = np.isfinite(feats)
    mag = np.abs(feats[finite]) if finite.any() else np.zeros(1, np.float32)
    return {
        "nonfinite": int((~finite).sum()),
        "over_fp16_max": int((mag > 65504).sum()),
        "below_fp16_floor": int(((mag < 2**-24) & (mag > 0)).sum()),
        "over_bf16_max": int((mag > 3.3895314e38).sum()),
        "max_abs": float(mag.max()) if mag.size else 0.0,
    }


def encode_bf16_bits(feats: np.ndarray) -> np.ndarray:
    """float32 array -> uint16 array holding the bfloat16 bits, truncation-free.

    bf16 in a uint16 container is the only bit-faithful bf16 on-disk format available: numpy has
    no bfloat16 dtype, np.save/np.load demote ml_dtypes.bfloat16 to |V2 (measured), and torch
    refuses an ml_dtypes array. The reader shifts these bits back: (u.astype(u32) << 16).
    """
    import torch as _t
    t = _t.from_numpy(np.ascontiguousarray(feats)).to(_t.bfloat16)
    return t.contiguous().view(_t.int16).numpy().view(np.uint16)


def decode_bf16_bits(bits: np.ndarray) -> np.ndarray:
    """uint16 array of bfloat16 bits -> float32 array (exact, the bits are just shifted up)."""
    return (bits.astype(np.uint32) << 16).view(np.float32)


def atomic_json(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def _expected_rows(split: str, stem: str) -> int:
    """Manifest count for a tar, 0 when unknown (drives the ETA display)."""
    from .stream import _manifest_candidates
    for path in _manifest_candidates(split):
        try:
            man = json.load(open(path, encoding="utf-8"))
            if man.get("split") == split:
                c = man.get("tars", {}).get(stem + ".tar")
                return int(c) if c else 0
        except Exception:
            continue
    return 0


def main() -> int:
    a = parse()
    from .config import TrainConfig
    from .model import DinoTagger
    from .stream import _iter_tar, _decode_pair, tar_dir
    from .labels import load_label_space

    device = "cuda"
    cfg = TrainConfig.from_json(os.path.join(a.run, "config.json"))
    if cfg.head_pool != "cls":
        raise SystemExit(f"this cacher is cls-only but the run pools {cfg.head_pool!r}; "
                         "refusing to mix pools in one cache")
    compute = torch.float32 if a.compute_dtype == "fp32" else torch.bfloat16
    np_store = {"fp32": np.float32, "fp16": np.float16, "bf16": np.uint16}[a.store_dtype]
    model = DinoTagger(cfg, 12476, dtype=compute).to(device)
    ck = torch.load(os.path.join(a.run, a.ckpt), map_location="cpu", weights_only=False)
    try:
        model.load_adapter_state_dict(ck["adapter"], strict=False)
    except RuntimeError as e:
        print(f"[load] head/norm left at init, trunk and adapters copied: {e}", flush=True)
    model.eval()
    if a.cache_rot:
        from .oft import iter_oft_modules
        for m in iter_oft_modules(model.backbone):
            r = m.oft._cayley_batch().detach()  # frozen for the pass: eval + no_grad
            object.__setattr__(m.oft, "_cayley_batch", (lambda r=r: r))
        print("[rot] OFT Cayley rotations built once", flush=True)

    os.makedirs(a.out, exist_ok=True)
    meta_path = os.path.join(a.out, "meta.json")
    meta: dict = {}
    if os.path.isfile(meta_path):
        meta = json.load(open(meta_path, encoding="utf-8"))
        # refuse to mix provenance in one directory: a shard from another trunk or dtype
        # beside these would silently be treated as interchangeable
        for key, now in (("run", a.run), ("ckpt", a.ckpt), ("pool", cfg.head_pool),
                         ("compute_dtype", a.compute_dtype), ("store_dtype", a.store_dtype)):
            if meta.get(key) and meta[key] != now:
                raise SystemExit(f"meta.json {key}={meta[key]!r} but this invocation says "
                                 f"{now!r}; use a fresh --out")
    meta.update({"run": a.run, "ckpt": a.ckpt, "pool": cfg.head_pool, "split": a.split,
                 "compute_dtype": a.compute_dtype, "store_dtype": a.store_dtype, "tars": {}})
    done: dict = meta["tars"]

    all_tars = sorted(f[:-4] for f in os.listdir(tar_dir(a.split)) if f.endswith(".tar"))
    if a.tars == "all":
        want = all_tars
    else:
        want = [s.strip() for s in a.tars.split(",") if s.strip()]
    bad = [s for s in want if s not in all_tars]
    if bad:
        raise SystemExit(f"unknown tars {bad}; stems look like {all_tars[:3]}")

    todo = [s for s in want if s not in done or done[s].get("limited")]
    print(f"{len(done)} tar(s) already cached, {len(todo)} to do: {todo[:8]}"
          f"{' ...' if len(todo) > 8 else ''}", flush=True)

    space = load_label_space()
    n_labels = len(space)

    def forward_chunk_tensor(x: torch.Tensor) -> np.ndarray:
        f = model.forward_features(model._prep(x))
        # keep the model's full output; the container is chosen once at write time
        return f.float().cpu().numpy()

    def forward_chunk(dec) -> np.ndarray:
        return forward_chunk_tensor(torch.from_numpy(np.stack([d[0] for d in dec])).to(device))

    ring = None
    if a.overlap:
        from .cache import SIZE
        ring = [torch.empty((a.batch, 3, SIZE, SIZE), dtype=torch.uint8, pin_memory=True)
                for _ in range(2)]
        print(f"[overlap] pinned double buffer of {a.batch} images "
              f"({ring[0].numel() / 1e6:.0f} MB per slot)", flush=True)

    with torch.no_grad(), ThreadPoolExecutor(max_workers=a.workers) as pool:
        for stem in todo:
            t0 = time.time()
            path = os.path.join(tar_dir(a.split), stem + ".tar")
            expected = _expected_rows(a.split, stem)
            if a.limit:
                expected = min(expected, a.limit) if expected else a.limit
            feats_l, idx_l, cnt_l = [], [], []
            buf: list = []
            shown = 0

            def drain() -> None:
                if not buf:
                    return
                dec = list(pool.map(lambda it: _decode_pair((it[1], it[2]), space, n_labels),
                                    buf))
                feats_l.append(forward_chunk(dec))
                for _arr, idx in dec:
                    idx_l.append(np.asarray(idx, np.int32))
                    cnt_l.append(len(idx))
                buf.clear()

            def maybe_log() -> None:
                nonlocal shown
                rows_now = len(cnt_l)
                if rows_now - shown >= 2000 or rows_now == expected:
                    shown = rows_now
                    el = time.time() - t0
                    rate = rows_now / max(1e-9, el)
                    if expected:
                        eta = (expected - rows_now) / max(rate, 1e-9)
                        print(f"{stem}: {rows_now}/{expected} rows, {rate:.0f} img/s, "
                              f"eta {eta / 60:.1f} min", flush=True)
                    else:
                        print(f"{stem}: {rows_now} rows, {rate:.0f} img/s, eta unknown "
                              f"(no manifest count)", flush=True)

            n_seen = 0

            def overlap_tar(tar_path: str) -> None:
                """Decode into a pinned slot on a producer thread while the GPU runs the last one.

                Same rows, same order, same batch boundaries as the serial loop: the producer
                only moves the tar read and the PIL decode off the critical path, and the slot
                is copied with non_blocking into the same uint8 tensor the forward always got.
                """
                import queue
                import threading

                ready: queue.Queue = queue.Queue()
                free: queue.Queue = queue.Queue()
                for i in range(len(ring)):
                    free.put(i)
                seen = [0]

                def produce() -> None:
                    it = _iter_tar(tar_path)
                    while True:
                        si = free.get()
                        items = []
                        while len(items) < a.batch:
                            if a.limit and seen[0] >= a.limit:
                                break
                            item = next(it, None)
                            if item is None:
                                break
                            items.append(item)
                            seen[0] += 1
                        if not items:
                            free.put(si)
                            break
                        dec = list(pool.map(
                            lambda it2: _decode_pair((it2[1], it2[2]), space, n_labels), items))
                        slot = ring[si].numpy()
                        for k, (arr, _idx) in enumerate(dec):
                            slot[k] = arr
                        ready.put((si, len(dec), [d[1] for d in dec]))
                    ready.put(None)

                th = threading.Thread(target=produce, name="cache-decode", daemon=True)
                th.start()
                try:
                    while True:
                        job = ready.get()
                        if job is None:
                            break
                        si, n, idxs = job
                        x = ring[si][:n].to(device, non_blocking=True)
                        feats_l.append(forward_chunk_tensor(x))
                        for idx in idxs:
                            idx_l.append(np.asarray(idx, np.int32))
                            cnt_l.append(len(idx))
                        free.put(si)
                        maybe_log()
                finally:
                    th.join(timeout=1.0)

            if a.overlap:
                overlap_tar(path)
            else:
                for item in _iter_tar(path):
                    buf.append(item)
                    n_seen += 1
                    if len(buf) >= a.batch:
                        drain()
                        maybe_log()
                    if a.limit and n_seen >= a.limit:
                        break
                drain()
            maybe_log()

            rows = len(cnt_l)
            feats = (np.concatenate(feats_l, axis=0) if feats_l
                     else np.zeros((0, feats_l[0].shape[1] if feats_l else 768), np.float32))
            assert feats.shape[0] == rows, f"{feats.shape[0]} features for {rows} samples"
            indices = np.concatenate(idx_l) if idx_l else np.zeros(0, np.int32)
            offsets = np.zeros(rows + 1, np.int64)
            if rows:
                np.cumsum(np.asarray(cnt_l, np.int64), out=offsets[1:])
            rng = range_report(feats)
            if a.store_dtype == "fp16":
                bad = rng["nonfinite"] + rng["over_fp16_max"] + rng["below_fp16_floor"]
                if bad:
                    raise SystemExit(
                        f"{stem}: {bad} value(s) outside fp16's representable range "
                        f"({rng}); refusing to store fp16 for this tar. Use "
                        "--store-dtype bf16, or investigate the outliers before deciding.")
            if a.store_dtype == "bf16" and (rng["nonfinite"] + rng["over_bf16_max"]):
                raise SystemExit(f"{stem}: non-finite or out-of-bf16-range values ({rng}); "
                                 "refusing to store")
            if a.store_dtype == "bf16":
                store = encode_bf16_bits(feats)
                # prove the container is lossless before committing: decode and compare bits
                back = decode_bf16_bits(store)
                same = int(np.array_equal(encode_bf16_bits(back), store))
                if not same:
                    raise SystemExit(f"{stem}: bf16 container failed its own round trip")
                print(f"{stem}: bf16 container round trip exact", flush=True)
            else:
                store = feats.astype(np_store)

            # atomic per tar: write tmp files, replace, then record in meta
            # np.save/np.savez append their suffix when it is missing, so hand them open handles
            npy_tmp = os.path.join(a.out, f"{stem}.npy.tmp")
            with open(npy_tmp, "wb") as fh:
                np.save(fh, store)
            npz_tmp = os.path.join(a.out, f"{stem}.npz.tmp")
            with open(npz_tmp, "wb") as fh:
                np.savez(fh, indices=indices, offsets=offsets)
            h = hashlib.blake2b(store.tobytes(), digest_size=16).hexdigest()
            os.replace(npy_tmp, os.path.join(a.out, f"{stem}.npy"))
            os.replace(npz_tmp, os.path.join(a.out, f"{stem}.npz"))
            done[stem] = {"rows": int(rows), "dim": int(store.shape[1]), "blake2b": h,
                          "range": rng}
            if a.limit:
                # a partial shard must never be mistaken for a completed tar on resume
                done[stem]["limited"] = True
            atomic_json(meta_path, meta)
            print(f"{stem}: {rows} rows x {store.shape[1]}, "
                  f"{rows / max(1e-9, time.time() - t0):.0f} img/s, blake2b {h[:12]}"
                  f"{' LIMITED ' + str(a.limit) if a.limit else ''}", flush=True)
    print(f"done: {len(done)} tar(s) cached in {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
