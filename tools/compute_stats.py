"""
compute_stats.py

Estimate per-channel standardisation statistics (mean/std) for a built Firecracker dataset and
write them to <data>/stats.json. FireDataset(<data>) then z-scores every sample it serves so each
channel leaves the dataset with a mean of zero and a standard deviation of one — the scale the
autoencoder trains best on. The on-disk shards are untouched; the scaling is applied on read.

The full set is hundreds of GB, so the stats are estimated from a random subset (10k observations
by default — even that gives each channel ~6.5e8 values on a 256x256 grid, plenty for tight
estimates). This is the same computation train_autoencoder triggers automatically when stats.json
is missing — run it here to prepare a dataset up front, inspect the numbers, or refresh them.

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
        description="Compute per-channel mean/std for a Firecracker dataset (writes stats.json)."
    )
    parser.add_argument("--data", required=True, help="dataset dir (flat or nested per-worker)")
    parser.add_argument("--recompute", action="store_true",
                        help="overwrite an existing stats.json instead of reusing it")
    parser.add_argument("--max-samples", type=int, default=10_000,
                        help="random observations to estimate from (0 = use the whole dataset)")
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
