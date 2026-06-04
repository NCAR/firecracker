"""
main.py

Interactive runner for the Firecracker gymnasium environment.
Advances the simulation at SIMULATION_STEPS_PER_SECOND and renders at TARGET_FPS.
"""

import sys
import time

from firecracker_env import FirecrackerEnv, DEFAULT_GRID_SIZE, DEFAULT_WINDOW_SIZE
from simulation import SIMULATION_STEPS_PER_SECOND, MS_PER_STEP

INFO_SEPARATOR: str = "-" * 36
_STEP_INTERVAL: float = 1.0 / SIMULATION_STEPS_PER_SECOND


def print_environment_info(info: dict, label: str, grid_size: int, window_size: int) -> None:
    print(INFO_SEPARATOR)
    print(f" {label}")
    print(INFO_SEPARATOR)
    print(f"  Grid size   : {grid_size} x {grid_size}  ({grid_size * grid_size} cells)")
    print(f"  Window size : {window_size} x {window_size} px")
    print(f"  Sim rate    : {SIMULATION_STEPS_PER_SECOND} steps/sec")
    print(f"  Temperature : {info['temperature']:.6f}")
    print(f"  Temp min    : {info['temp_min']:.6f}")
    print(f"  Temp max    : {info['temp_max']:.6f}")
    print(f"  Temp std    : {info['temp_std']:.6f}")
    print(f"  Wind mean   : {info['wind_mean']:.6f}")
    print(f"  Wind max    : {info['wind_max']:.6f}")
    print(f"  Wind std    : {info['wind_std']:.6f}")
    print(INFO_SEPARATOR)


def run(grid_size: int = DEFAULT_GRID_SIZE, window_size: int = DEFAULT_WINDOW_SIZE) -> None:
    env = FirecrackerEnv(grid_size=grid_size, window_size=window_size, render_mode="human")
    obs, info = env.reset()
    print_environment_info(info, "Startup", grid_size, window_size)

    last_step_time = time.monotonic()

    while env._running:
        now = time.monotonic()
        if env._step_once:
            env._step_once = False
            obs, reward, terminated, truncated, info = env.step(0)
            last_step_time = now
            if terminated or truncated:
                break
        if not env._paused and now - last_step_time >= _STEP_INTERVAL:
            step_start = time.monotonic()
            obs, reward, terminated, truncated, info = env.step(0)
            last_step_time = now
            step_elapsed_ms = int((time.monotonic() - step_start) * 1000)
            if step_elapsed_ms > MS_PER_STEP:
                print(f"WARNING: Step took {step_elapsed_ms}ms (budget: {MS_PER_STEP}ms)")
            if terminated or truncated:
                break

        if env._reset_requested:
            obs, info = env.reset()
            last_step_time = time.monotonic()

        env.render()

    print_environment_info(info, "Shutdown", grid_size, window_size)
    env.close()


if __name__ == "__main__":
    try:
        n = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_GRID_SIZE
        w = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_WINDOW_SIZE
    except ValueError:
        print("Usage: python main.py [N [window_size]]  (both must be positive integers)")
        sys.exit(1)
    run(grid_size=n, window_size=w)
