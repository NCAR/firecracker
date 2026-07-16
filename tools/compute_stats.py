"""
compute_stats.py

Find per-channel min-max normalization statistics (min/max) for a built Firecracker dataset and
write them to <data>/stats.json. normalize_dataset.py then bakes the min-max scale (x-min)/(max-min)
into the shards so each channel leaves the dataset in [0, 1] — most channels aren't normally
distributed, so min-max is preferred over z-scoring (and genuine outliers ride above 1 rather than
being flattened by a std). Signed channels (the wind components) are instead scaled symmetrically to
[-1, 1] about their zero; stats.json records that subset under "signed_channels". This step only
writes stats.json; the shards are scaled by the bake.

Unlike a mean/std, min and max are extreme order statistics: a subset of shards *under*-estimates
the true range, so the default reads the whole dataset (--max-samples 0). Pass a positive
--max-samples to sample whole shards instead, trading exactness for speed. This is the same
computation normalize_dataset triggers when stats.json is missing — run it here to prepare a dataset
up front, inspect the numbers, or refresh them.

Usage:
    python tools/compute_stats.py --data /path/to/data/fire
    python tools/compute_stats.py --data data/fire --max-samples 20000 --recompute
"""

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from rollout import ensure_channel_stats  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find per-channel min/max for a Firecracker dataset (writes stats.json)."
    )
    parser.add_argument("--data", required=True, help="dataset dir (flat or nested per-worker)")
    parser.add_argument("--recompute", action="store_true",
                        help="overwrite an existing stats.json instead of reusing it")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="random observations to measure from (0 = whole dataset; the default, "
                             "since min/max from a subset under-estimate the true range)")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed for the random sample of observations")
    parser.add_argument("--chunk", type=int, default=256,
                        help="samples per accumulation slice (memory/speed knob)")
    args = parser.parse_args()

    ensure_channel_stats(
        args.data, recompute=args.recompute,
        max_samples=args.max_samples or None, seed=args.seed, chunk=args.chunk,
    )
    print(f"wrote {Path(args.data) / 'stats.json'}")


if __name__ == "__main__":
    main()
