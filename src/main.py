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
    print(f"    steps_per_second       : {sim.simulation_steps_per_second} Hz  (display throttle)")
    print("  [units]")
    print(f"    cell_size_m            : {sim.cell_size_m} m")
    print(f"    seconds_per_tick (dt)  : {sim.dt:.3g} s  ({sim.dt * sim.simulation_steps_per_second:.3g}x real time at {sim.simulation_steps_per_second} tps)")
    print("  [momentum]")
    print(f"    reduced_gravity        : {sim.reduced_gravity} m/s^2")
    print(f"    drag_coeff             : {sim.drag_coeff} 1/s")
    print(f"    viscosity              : {sim.viscosity} m^2/s")
    print(f"    cfl_target             : {sim.cfl_target}")
    print("  [channeling]")
    print(f"    enabled                : {sim.channeling_enabled}")
    print(f"    radius                 : {sim.channel_radius} cells")
    print(f"    height_scale_m         : {sim.channel_height_scale_m} m")
    print(f"    gain_max               : {sim.channel_gain_max}")
    print(f"    smooth_sigma           : {sim.channel_smooth_sigma} cells")
    print("  [oxygen]")
    print(f"    diffusion_sigma        : {sim.oxygen_diffusion_sigma} cells")
    print("  [maps]")
    print(f"    dir                    : {env._maps_dir}")
    print(f"    name                   : {env._map_name or '(random)'}")
    print("  [radiation]")
    print(f"    enabled                : {sim.radiation_enabled}")
    print(f"    emissivity             : {sim.emissivity}")
    print(f"    ground_heat_capacity   : {sim.ground_heat_capacity} J/m^2/K")
    print(f"    ground_air_exchange    : {sim.ground_air_exchange} W/m^2/K")
    print(f"    cover_fuel_scale       : {sim.cover_fuel_scale} kg/m^2")
    print("  [relaxation]")
    print(f"    enabled                : {sim.relaxation_enabled}")
    print(f"    oxygen_rate            : {sim.oxygen_rate}")
    print("  [fire]")
    print(f"    enabled                     : {sim.fire_enabled}")
    print(f"    spawn_fire                  : {env._spawn_fire}")
    print(f"    spawn_radius                : {env._fire_spawn_radius}")
    print(f"    burnt_threshold             : {sim.fuel_burnt_threshold} kg/m^2")
    print(f"    burn_heat_fuel_fraction     : {sim.burn_heat_fuel_fraction}")
    print(f"    combustion_mixing_depth     : {sim.combustion_mixing_depth} m")
    print(f"    surface_mass_transfer       : {sim.surface_mass_transfer} kg/m^2/s")
    print(f"    plume_mixing_depth          : {sim.plume_mixing_depth} m")
    print(f"    convective_fraction         : {sim.convective_fraction}")
    print(f"    flame_gate_temperature      : {sim.flame_gate_temperature} K")
    print(f"    convective_radius           : {sim.convective_radius} cells")
    print(f"    convective_wind_bias        : {sim.convective_wind_bias}")
    print(f"    convective_wind_ref         : {sim.convective_wind_ref} m/s")
    print(f"    convective_slope_bias       : {sim.convective_slope_bias}")
    print("  [fuel_types]")
    for i, name in enumerate(sim.fuel_type_names):
        print(f"    [{name}]")
        print(f"      specific_heat          : {sim.fuel_specific_heat[i].item()}")
        print(f"      ignition_threshold     : {sim.ignition_thresholds[i].item()}")
        print(f"      heat_of_combustion     : {sim.heat_of_combustion[i].item()}")
        print(f"      stoich_oxygen          : {sim.stoich_oxygen[i].item()}")
        print(f"      arrhenius_pre (B)      : {sim.arrhenius_pre[i].item()}")
        print(f"      activation_energy      : {sim.activation_energy[i].item()}")
        print(f"      specific_surface (a_s) : {sim.fuel_specific_surface[i].item()}")
        print(f"      fuel_air_transfer_rate : {sim.fuel_transfer_rates[i].item()}")
    print("  [radiant_heat]")
    print(f"    enabled                : {sim.radiant_heat_enabled}")
    print(f"    kernel_radius (cells)  : {sim.radiant_kernel_radius}")
    print(f"    sky_escape_fraction    : {sim.sky_escape_fraction}")
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
            obs, reward, terminated, truncated, info = env.step(None)
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


def run_comparison(
    config: dict,
    model_specs: list[str],
    map_name: str | None = None,
    maps_dir: str | None = None,
) -> None:
    """Run the world alongside one or more trained autoencoders' reconstructions of it, in a
    composited window (physics | model [| model ...]). The env is driven in rgb_array mode so the
    ComparisonViewer owns the single window; the run loop mirrors run() otherwise.
    """
    # Imported lazily so the normal viewer doesn't pull in torch/the model stack unless --model.
    from model_viewer import ComparisonViewer, load_model, resolve_model_path

    env = FirecrackerEnv(
        config=config, render_mode="rgb_array", map_name=map_name, maps_dir=maps_dir
    )

    models: list[tuple] = []
    for model_spec in model_specs:
        model_path = resolve_model_path(model_spec)
        model, model_config = load_model(model_path, env._sim.device)
        # Each model reconstructs a fixed observation shape; refuse a config whose grid or
        # fuel-derived channel count doesn't match what it was trained on rather than crash inside
        # the forward pass.
        if model_config["grid_size"] != env.grid_size:
            raise SystemExit(
                f"model '{model_path}' was trained on grid_size={model_config['grid_size']} but "
                f"this config uses {env.grid_size}; pass a matching --config."
            )
        if model_config["in_channels"] != len(env.obs_channels):
            raise SystemExit(
                f"model '{model_path}' expects {model_config['in_channels']} observation channels "
                f"but this config's fuel types give {len(env.obs_channels)}; the fuel_types must "
                f"match the trained model."
            )
        normalized = "channel_min" in model_config or "channel_mean" in model_config
        print(f"Loaded model: {model_path}  (arch={model_config.get('arch', 'pooled')}, "
              f"channels={tuple(model_config['channels'])}, latent_dim={model_config['latent_dim']}, "
              f"normalized={normalized})")
        models.append((model, model_config, model_path.stem))

    viewer = ComparisonViewer(env, models)

    step_interval = 1.0 / env._sim.simulation_steps_per_second

    print_config_info(env)

    log_episodes = bool((config or {}).get("environment", {}).get("print_episode_info", True))

    episode = 0
    obs, info = env.reset()
    episode += 1
    if log_episodes:
        print_episode_info(info, f"Episode {episode} — start")

    last_step_time = time.monotonic()

    while env._running:
        now = time.monotonic()

        if env._step_once:
            env._step_once = False
            obs, reward, terminated, truncated, info = env.step(None)
            last_step_time = now
            if terminated or truncated:
                if log_episodes:
                    print_episode_info(info, f"Episode {episode} — end")
                break

        if not env._paused and now - last_step_time >= step_interval:
            obs, reward, terminated, truncated, info = env.step(None)
            last_step_time = now
            if terminated or truncated:
                if log_episodes:
                    print_episode_info(info, f"Episode {episode} — end")
                break

        if env._reset_requested:
            obs, info = env.reset()
            episode += 1
            last_step_time = time.monotonic()

        viewer.render()

    viewer.close()
    env.close()


def parse_views(spec: str) -> tuple[int, int]:
    """Parse a --views 'RxC' argument (e.g. '2x2', '3x3') into a (rows, cols) grid, 1..3 each."""
    try:
        rows_s, cols_s = spec.lower().split("x")
        rows, cols = int(rows_s), int(cols_s)
    except ValueError:
        raise SystemExit(f"--views must look like RxC, e.g. 2x2 or 3x3; got {spec!r}")
    if not (1 <= rows <= 3 and 1 <= cols <= 3):
        raise SystemExit(f"--views rows and cols must each be between 1 and 3; got {spec!r}")
    return rows, cols


def run(
    config: dict,
    map_name: str | None = None,
    maps_dir: str | None = None,
    view_grid: tuple[int, int] | None = None,
) -> None:
    env = FirecrackerEnv(
        config=config, render_mode="human", map_name=map_name, maps_dir=maps_dir,
        view_grid=view_grid,
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
            obs, reward, terminated, truncated, info = env.step(None)
            last_step_time = now
            if terminated or truncated:
                if log_episodes:
                    print_episode_info(info, f"Episode {episode} — end")
                episode_ended = True
                break

        if not env._paused and now - last_step_time >= step_interval:
            step_start = time.monotonic()
            obs, reward, terminated, truncated, info = env.step(None)
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
    parser.add_argument(
        "--views", metavar="RxC",
        help="show a grid of independent view panes instead of one view, e.g. 2x2 or 3x3. "
             "SHIFT+arrow keys move the active pane; the letter keys / TAB menu set the active "
             "pane's view. (window mode only)",
    )
    parser.add_argument(
        "--steps", type=int, metavar="N",
        help="run for N steps then truncate (overrides [environment].max_steps)",
    )
    parser.add_argument(
        "--model", metavar="PATH", nargs="+",
        help="load one or more trained autoencoder checkpoints (each a .pt file, a directory to "
             "take best.pt from, or a name resolved under models/) and show their reconstructions "
             "beside the physics view, one panel per model; press TAB to flip the model panels to a "
             "per-cell error heatmap",
    )
    args = parser.parse_args()

    if args.model and args.headless:
        parser.error("--model needs a window to draw into; it can't be combined with --headless.")
    if args.views and (args.headless or args.model):
        parser.error("--views is a windowed single-world layout; it can't be combined with "
                     "--headless or --model.")

    view_grid = parse_views(args.views) if args.views else None

    config = load_config(args.config)
    if args.steps is not None:
        if args.steps < 1:
            parser.error(f"--steps must be >= 1, got {args.steps}")
        config.setdefault("environment", {})["max_steps"] = args.steps
    if args.model:
        run_comparison(config, args.model, map_name=args.map, maps_dir=args.maps_dir)
    elif args.headless:
        run_headless(config, map_name=args.map, maps_dir=args.maps_dir)
    else:
        run(config, map_name=args.map, maps_dir=args.maps_dir, view_grid=view_grid)
