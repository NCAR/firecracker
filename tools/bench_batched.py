"""
bench_batched.py

Batched-rollout throughput: step B worlds in parallel through the shared
step_fields and measure how per-env tick rate scales with batch size. Same
subsystems as bench.py (wind + radiation + combustion + channeling), float64
on the default device. Reports, per batch size B:

    aggregate env-steps/s   = B * steps / elapsed   (total worlds advanced/s)
    per-env steps/s         = steps / elapsed        (tick rate each world sees)

CUDA timing is bracketed with torch.cuda.synchronize() -- the direct step loop
has no host round-trip to force a sync, unlike the single-env env.step path.

Usage:
    python tools/bench_batched.py --maps-dir maps --batches 1,4,16,64,128 --steps 100
"""

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import torch  # noqa: E402

from config import load_config       # noqa: E402
from rollout import BatchedRollout    # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Batched rollout throughput / per-env scaling")
    p.add_argument("--maps-dir", default="maps", help="directory of generated maps")
    p.add_argument("--grid", type=int, default=256, help="grid size (default: 256)")
    p.add_argument("--batches", default="1,4,16,64,128",
                   help="comma-separated num_envs values to sweep")
    p.add_argument("--steps", type=int, default=100, help="timed steps (default: 100)")
    p.add_argument("--warmup", type=int, default=20, help="untimed warmup steps (default: 20)")
    args = p.parse_args()

    cfg = load_config()
    cfg.setdefault("environment", {})["grid_size"] = args.grid
    cfg.setdefault("fire", {})["enabled"] = True
    cfg["fire"]["spawn_fire"] = True
    cfg.setdefault("radiation", {})["enabled"] = True
    cfg.setdefault("channeling", {})["enabled"] = True

    batches = [int(b) for b in args.batches.split(",") if b]
    dev = None
    print(f"grid {args.grid}x{args.grid}, float64, {args.steps} timed steps "
          f"(warmup {args.warmup}), all subsystems on")
    print(f"{'B':>5} {'agg env-steps/s':>16} {'per-env steps/s':>16} {'ms/step':>9}")

    for B in batches:
        roll = BatchedRollout(cfg, num_envs=B, maps_dir=args.maps_dir,
                              spawn_fire=True, seed=0)
        sim, state = roll._sim, roll._state
        dev = sim.device
        cuda = dev.type == "cuda"

        for _ in range(args.warmup):
            sim.step_fields(state)
        if cuda:
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(args.steps):
            sim.step_fields(state)
        if cuda:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        agg = B * args.steps / elapsed
        per_env = args.steps / elapsed
        print(f"{B:>5} {agg:>16.1f} {per_env:>16.1f} {1000 * elapsed / args.steps:>9.1f}")

    print(f"device: {dev}")


if __name__ == "__main__":
    main()
