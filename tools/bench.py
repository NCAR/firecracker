"""
bench.py

Measure step_fields throughput at the production grid with every subsystem on
(boundary-layer wind + radiation + combustion + radiant transfer + channeling).
Reports steps/s and ms/step for the configured grid; float64 on the default device.

Usage:
    python tools/bench.py                 # 256^2, 50 timed steps
    python tools/bench.py --grid 128 --steps 100
"""

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

# Headless: no display needed (the env never renders here, but keep SDL quiet).
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import numpy as np  # noqa: E402

from config import load_config              # noqa: E402
from gen_maps import MapGenerator, generate_map  # noqa: E402
from map_loader import save_map             # noqa: E402
from firecracker_env import FirecrackerEnv  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark step_fields throughput")
    parser.add_argument("--grid", type=int, default=256, help="grid size (default: 256)")
    parser.add_argument("--steps", type=int, default=50, help="timed steps (default: 50)")
    parser.add_argument("--warmup", type=int, default=5, help="untimed warmup steps (default: 5)")
    args = parser.parse_args()

    cfg = load_config()
    cfg.setdefault("environment", {})["grid_size"] = args.grid
    cfg.setdefault("fire", {})["enabled"] = True
    cfg["fire"]["spawn_fire"] = True                 # ignite so combustion runs
    cfg.setdefault("radiation", {})["enabled"] = True
    cfg.setdefault("channeling", {})["enabled"] = True

    gen = MapGenerator(cfg)
    m = generate_map(gen, args.grid, np.random.default_rng(0), seed=0)

    with tempfile.TemporaryDirectory() as d:
        save_map(Path(d) / "bench.npz", m)
        env = FirecrackerEnv(config=cfg, render_mode=None, maps_dir=d, map_name="bench")
        env.reset(seed=0)

        for _ in range(args.warmup):
            env.step(0)

        t0 = time.perf_counter()
        for _ in range(args.steps):
            env.step(0)
        elapsed = time.perf_counter() - t0

    dev = env._sim.device
    rate = args.steps / elapsed
    print(f"{args.grid}x{args.grid} float64 on {dev}, all subsystems on: "
          f"{rate:.1f} steps/s ({1000 * elapsed / args.steps:.1f} ms/step)")


if __name__ == "__main__":
    main()
