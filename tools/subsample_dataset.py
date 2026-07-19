"""
subsample_dataset.py

Write a physically smaller copy of a sharded FireDataset by keeping a deterministic random
fraction of the samples in every shard. Purpose: a big fp16 observation dataset (~448 GiB) does not
fit a RAM-tight node's page cache, and training with a load-time `--subsample-frac` has to disable
prewarm (which streams whole shards = the full set). A physical half-size dataset restores prewarm
(sequential warm of only the kept bytes) AND bounds the cache, so training is fast at any packing.

AE-ONLY: this thins frames, which is fine for per-frame autoencoder reconstruction but NOT for the
dynamics head (which needs consecutive frames). Never point the dynamics trainer at a subsampled set.

Handles both dataset layouts FireDataset supports:
  * flat   -- shards + meta.json directly under the root
  * nested -- one subdir per source (e.g. w0..w3), each with its own shards + meta.json
The per-dataset stats.json (normalization) is copied verbatim -- min-max stats are unchanged by
dropping samples, and the shards are already normalized, so `normalized` carries over.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

_META = "meta.json"


def _shard_dirs(root: Path) -> list[Path]:
    """Return the directories that actually hold shards: [root] if flat, else the source subdirs."""
    if (root / _META).exists():
        return [root]
    subs = sorted(p.parent for p in root.glob(f"*/{_META}"))
    if not subs:
        raise SystemExit(f"no {_META} under {root} (flat) or {root}/*/{_META} (nested)")
    return subs


def _subsample_dir(src_dir: Path, dst_dir: Path, frac: float, seed: int) -> tuple[int, int]:
    """Copy one shard-holding dir, keeping a deterministic random `frac` of each shard's samples."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    meta = json.loads((src_dir / _META).read_text())
    shards = sorted(src_dir.glob("shard_*.npy"))
    if not shards:
        raise SystemExit(f"no shard_*.npy in {src_dir}")
    kept_total = 0
    per_shard_out = None
    for i, sp in enumerate(shards):
        arr = np.load(sp, mmap_mode="r")                 # (n, C, H, W), no full load
        n = arr.shape[0]
        n_keep = max(1, int(round(n * frac)))
        # Deterministic per-shard selection (seed folds in the shard index) so the run is
        # reproducible and every shard is thinned independently.
        idx = np.sort(np.random.default_rng([seed, i]).permutation(n)[:n_keep])
        out = np.ascontiguousarray(arr[idx])             # materialize only the kept samples
        np.save(dst_dir / sp.name, out)
        kept_total += n_keep
        per_shard_out = out.shape[0]
        print(f"  {src_dir.name}/{sp.name}: {n} -> {n_keep}  ({out.shape}, {out.dtype})", flush=True)
        del arr, out
    # Updated meta: same channels/grid/dtype/normalized, new per-shard + total sample counts.
    meta["samples_per_shard"] = per_shard_out
    meta["total_samples"] = kept_total
    meta["subsampled_from"] = str(src_dir)
    meta["subsample_frac"] = frac
    meta["subsample_seed"] = seed
    (dst_dir / _META).write_text(json.dumps(meta, indent=2))
    return kept_total, len(shards)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, help="source dataset dir (flat or nested w*/)")
    ap.add_argument("--dst", required=True, help="output dir for the subsampled dataset")
    ap.add_argument("--frac", type=float, default=0.5, help="fraction of samples to keep (default 0.5)")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for the deterministic selection")
    args = ap.parse_args()
    if not 0.0 < args.frac < 1.0:
        raise SystemExit(f"--frac must be in (0, 1), got {args.frac}")

    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    dirs = _shard_dirs(src)
    print(f"subsample {args.frac} of {src} -> {dst}  ({len(dirs)} source dir(s))", flush=True)

    grand = 0
    for sd in dirs:
        rel = "." if sd == src else sd.name
        kept, nsh = _subsample_dir(sd, dst if sd == src else dst / sd.name, args.frac, args.seed)
        print(f"[{rel}] kept {kept} samples across {nsh} shards", flush=True)
        grand += kept

    # Copy the normalization stats (unchanged by dropping samples) so the set stays self-describing.
    for extra in src.glob("*.json"):
        if extra.name != _META:
            shutil.copy2(extra, dst / extra.name)
            print(f"copied {extra.name}", flush=True)

    print(f"DONE: {grand} samples written to {dst}", flush=True)


if __name__ == "__main__":
    main()
