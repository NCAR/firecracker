"""
main.py

Interactive runner for the Firecracker gymnasium environment.
Advances the simulation at steps_per_second (from config) and renders at TARGET_FPS.
"""

import argparse
import time

from firecracker_env import FirecrackerEnv
from config import load_config

INFO_SEPARATOR: str = "-" * 40


def print_config_info(env: FirecrackerEnv) -> None:
    sim = env._sim
    print(INFO_SEPARATOR)
    print(" Configuration")
    print(INFO_SEPARATOR)
    print("  [environment]")
    print(f"    device                 : {sim.device}")
    print(f"    grid_size              : {env.grid_size} x {env.grid_size}  ({env.grid_size ** 2} cells)")
    print(f"    window_size            : {env.window_size} x {env.window_size} px")
    print(f"    max_steps              : {env.max_steps}")
    print("  [simulation]")
    print(f"    steps_per_second       : {sim.simulation_steps_per_second}")
    print(f"    blur_sigma             : {sim.blur_sigma}")
    print("  [convection]")
    print(f"    thermal_expansion       : {sim.thermal_expansion}")
    print(f"    terrain_height_scale    : {sim.terrain_height_scale}")
    print(f"    buoyancy_transport_rate : {sim.buoyancy_transport_rate}")
    print(f"    pressure_transport_rate : {sim.pressure_transport_rate}")
    print("  [wind]")
    print(f"    smooth_sigma           : {sim.wind_smooth_sigma}")
    print(f"    temporal_smoothing     : {sim.wind_temporal_smoothing}")
    print("  [oxygen]")
    print(f"    diffusion_sigma        : {sim.oxygen_diffusion_sigma}")
    print(f"    advection_strength     : {sim.oxygen_advection_strength}")
    print("  [maps]")
    print(f"    dir                    : {env._maps_dir}")
    print(f"    name                   : {env._map_name or '(random)'}")
    print("  [relaxation]")
    print(f"    enabled                : {sim.relaxation_enabled}")
    print(f"    temperature_rate_low   : {sim.temperature_rate_low}")
    print(f"    temperature_rate_high  : {sim.temperature_rate_high}")
    print(f"    oxygen_rate            : {sim.oxygen_rate}")
    print("  [fire]")
    print(f"    enabled                     : {sim.fire_enabled}")
    print(f"    spawn_fire                  : {env._spawn_fire}")
    print(f"    spawn_radius                : {env._fire_spawn_radius}")
    print(f"    oxygen_consumption_rate     : {sim.oxygen_consumption_rate}")
    print(f"    burnt_threshold             : {sim.fuel_burnt_threshold}")
    print(f"    oxygen_extinction_threshold : {sim.oxygen_extinction_threshold}")
    print(f"    burn_heat_fuel_fraction     : {sim.burn_heat_fuel_fraction}")
    print("  [fuel_types]")
    for i, name in enumerate(sim.fuel_type_names):
        print(f"    [{name}]")
        print(f"      heat_capacity          : {sim.heat_capacities[i]}")
        print(f"      ignition_threshold     : {sim.ignition_thresholds[i]}")
        print(f"      consumption_rate       : {sim.consumption_rates[i]}")
        print(f"      burn_heat_scale        : {sim.burn_heat_scales[i]}")
        print(f"      fuel_air_transfer_rate : {sim.fuel_transfer_rates[i]}")
    print("  [radiant_heat]")
    print(f"    kernel_radius          : {sim.radiant_kernel_radius}")
    print(f"    emission_scale         : {sim.radiant_emission_scale}")
    print(f"    emission_exponent      : {sim.radiant_emission_exponent}")
    print(INFO_SEPARATOR)


def print_episode_info(info: dict, label: str) -> None:
    print(INFO_SEPARATOR)
    print(f" {label}")
    print(INFO_SEPARATOR)
    print(f"  Step          : {info['step']}")
    print(f"  Air temp mean : {info['air_temperature']:.6f}")
    print(f"  Air temp min  : {info['air_temp_min']:.6f}")
    print(f"  Air temp max  : {info['air_temp_max']:.6f}")
    print(f"  Air temp std  : {info['air_temp_std']:.6f}")
    print(f"  Wind mean     : {info['wind_mean']:.6f}")
    print(f"  Wind max      : {info['wind_max']:.6f}")
    print(f"  Wind std      : {info['wind_std']:.6f}")
    print(f"  Cells burning : {info['cells_burning']}")
    print(f"  Oxygen mean   : {info['oxygen_mean']:.6f}")
    print(f"  Oxygen min    : {info['oxygen_min']:.6f}")
    print(f"  Terrain mean  : {info['terrain_mean']:.6f}")
    print(f"  Terrain min   : {info['terrain_min']:.6f}")
    print(f"  Terrain max   : {info['terrain_max']:.6f}")
    for key, val in info.items():
        if key.startswith("fuel_") or key.startswith("cells_burning_"):
            print(f"  {key:<30}: {val:.6f}" if isinstance(val, float) else f"  {key:<30}: {val}")
    print(INFO_SEPARATOR)


def run_headless(config: dict, map_name: str | None = None, maps_dir: str | None = None) -> None:
    """Run with no window and no real-time throttle: step as fast as the device allows.

    There is no keyboard/mouse, so the episode runs to truncation (max_steps) or until
    interrupted (Ctrl-C). Set [fire].spawn_fire = true in the config to ignite, since
    there is no window to click. Throughput is printed periodically.
    """
    env = FirecrackerEnv(
        config=config, render_mode=None, map_name=map_name, maps_dir=maps_dir
    )
    print_config_info(env)
    log_episodes = bool((config or {}).get("environment", {}).get("print_episode_info", True))

    obs, info = env.reset()
    if log_episodes:
        print_episode_info(info, "Episode 1 — start")

    log_interval = max(1, env.max_steps // 20)   # ~20 progress lines per episode
    start = time.monotonic()
    window_start, window_steps = start, 0
    try:
        while True:
            obs, reward, terminated, truncated, info = env.step(0)
            window_steps += 1
            if info["step"] % log_interval == 0:
                now = time.monotonic()
                rate = window_steps / (now - window_start)
                print(f"  step {info['step']:>7d} / {env.max_steps}   {rate:8.1f} ticks/s   "
                      f"cells_burning={info['cells_burning']}")
                window_start, window_steps = now, 0
            if terminated or truncated:
                break
    except KeyboardInterrupt:
        print("\nInterrupted.")

    elapsed = time.monotonic() - start
    if log_episodes:
        print_episode_info(info, "Episode 1 — end")
    print(f"Ran {info['step']} steps in {elapsed:.2f}s  ({info['step'] / elapsed:.1f} ticks/s average).")
    env.close()


def run(config: dict, map_name: str | None = None, maps_dir: str | None = None) -> None:
    env = FirecrackerEnv(
        config=config, render_mode="human", map_name=map_name, maps_dir=maps_dir
    )

    step_interval = 1.0 / env._sim.simulation_steps_per_second
    ms_per_step   = env._sim.ms_per_step

    print_config_info(env)

    log_episodes = bool((config or {}).get("environment", {}).get("print_episode_info", True))

    episode = 0
    obs, info = env.reset()
    episode += 1
    if log_episodes:
        print_episode_info(info, f"Episode {episode} — start")

    last_step_time = time.monotonic()
    episode_ended = False

    while env._running:
        now = time.monotonic()

        if env._step_once:
            env._step_once = False
            obs, reward, terminated, truncated, info = env.step(0)
            last_step_time = now
            if terminated or truncated:
                if log_episodes:
                    print_episode_info(info, f"Episode {episode} — end")
                episode_ended = True
                break

        if not env._paused and now - last_step_time >= step_interval:
            step_start = time.monotonic()
            obs, reward, terminated, truncated, info = env.step(0)
            last_step_time = now
            step_elapsed_ms = int((time.monotonic() - step_start) * 1000)
            if step_elapsed_ms > ms_per_step:
                print(f"WARNING: Step took {step_elapsed_ms}ms (budget: {ms_per_step}ms)")
            if terminated or truncated:
                if log_episodes:
                    print_episode_info(info, f"Episode {episode} — end")
                episode_ended = True
                break

        if env._reset_requested:
            if log_episodes:
                print_episode_info(info, f"Episode {episode} — end")
            obs, info = env.reset()
            episode += 1
            if log_episodes:
                print_episode_info(info, f"Episode {episode} — start")
            last_step_time = time.monotonic()
            episode_ended = False

        env.render()

    if not episode_ended and log_episodes:
        print_episode_info(info, f"Episode {episode} — end")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Firecracker interactive runner")
    parser.add_argument(
        "--config", metavar="PATH",
        help="path to a TOML config file (default: cfg/default.toml)",
    )
    parser.add_argument(
        "--map", metavar="NAME",
        help="specific map to load from the maps dir (default: random)",
    )
    parser.add_argument(
        "--maps-dir", metavar="PATH",
        help="directory to load maps from (overrides config)",
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="run with no window and no real-time throttle (fastest; runs to max_steps)",
    )
    args = parser.parse_args()
    runner = run_headless if args.headless else run
    runner(load_config(args.config), map_name=args.map, maps_dir=args.maps_dir)
