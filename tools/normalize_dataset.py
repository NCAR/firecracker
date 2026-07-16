"""
normalize_dataset.py

Normalize a built Firecracker dataset *on disk*: rewrite every shard to the per-channel min-max
scale (x - min) / (max - min) so each channel lands in [0, 1]. Most channels aren't normally
distributed, so min-max is preferred over z-scoring. Signed channels with a meaningful zero (the
wind components) are instead scaled symmetrically to [-1, 1] so calm wind stays at 0 and the sign
is preserved (see rollout.SIGNED_CHANNELS / stats_affine). Because each shard is read hundreds of times
over a training run, baking the scale in once here keeps the training hot path a plain memmap copy
(FireDataset applies no per-read normalization).

The per-channel min/max are measured over the whole dataset by default (see tools/compute_stats.py;
min/max from a subset would under-estimate the true range) and cached to <data>/stats.json — that
file records the exact transform that was baked in, for inference/de-normalization. Each source's
meta.json is flagged "normalized": true on success.

Idempotent and resumable: already-flagged sources are skipped, and each source is baked into a
sibling temp dir built from the pristine originals then swapped in, so an interrupted run never
leaves half-scaled shards live. This is I/O-bound (read + rewrite the whole set), not compute-bound
— no GPU needed; run it on a normal node, optionally over a scratch copy.

Usage:
    python tools/normalize_dataset.py --data "$SCRATCH/firecracker/stage/fire" --workers 4
    python tools/normalize_dataset.py --data data/fire --keep-backup      # keep <src>.raw.bak
"""

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from rollout import bake_normalization  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bake per-channel min-max normalization to [0, 1] into a dataset's shards."
    )
    parser.add_argument("--data", required=True, help="dataset dir (flat or nested per-worker)")
    parser.add_argument("--workers", type=int, default=1,
                        help="shards to bake in parallel across processes; scale to the node's cpus")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="random observations to measure min/max from (0 = whole dataset, the "
                             "default; a subset under-estimates the true range)")
    parser.add_argument("--seed", type=int, default=0, help="seed for the stats sample")
    parser.add_argument("--chunk", type=int, default=256,
                        help="samples per accumulation/rewrite slice (memory/speed knob)")
    parser.add_argument("--keep-backup", action="store_true",
                        help="keep each source's pre-bake copy as <src>.raw.bak instead of deleting it")
    args = parser.parse_args()

    summary = bake_normalization(
        args.data, max_samples=args.max_samples or None, seed=args.seed, chunk=args.chunk,
        workers=args.workers, keep_backup=args.keep_backup,
    )
    print(f"done: baked {summary['baked']} / {summary['sources']} source(s), "
          f"{summary.get('shards', 0)} shard(s) / {summary.get('samples', 0)} samples rewritten; "
          f"skipped {summary['skipped']} already-normalized.")


if __name__ == "__main__":
    main()
