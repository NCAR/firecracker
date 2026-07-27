"""
record_views.py

Record a point-fire spread as a sped-up GIF of the interactive GUI's multi-view grid (the same
rows x cols panes you get from `python src/main.py --views RxC`). Runs headless (SDL's dummy video
driver) so no window is needed: it composites each frame offscreen, captures every Nth simulation
tick, and encodes the frames with one shared color palette so static regions (labels, legend-free
panes) stay steady instead of shimmering.

With no arguments it reproduces the layered-fuel demo: a 3x2 grid showing grass+shrub, tree-bole, and
canopy fuel/fire alongside air temperature, terrain, and wind, with a strengthened wind and a fixed
air-temperature color scale.

Usage:
    python tools/record_views.py                          # the 3x2 layered-fuel demo -> fire_grid.gif
    python tools/record_views.py --views 2x2 --preset auto # 2x2 of the default view assignment
    python tools/record_views.py --wind map --air-ceiling none  # map's own wind, per-frame auto-range
    python tools/record_views.py --steps 4000 --duration 30 --out /tmp/big.gif

Notes:
  * --preset layers requires a 3x2 grid (the six layered views); --preset auto fills any RxC from the
    env's default view list.
  * --wind is m/s (the whole field is scaled to that ambient speed, direction kept) or "map" to keep
    the map's own wind. --air-ceiling is a fixed kelvin ceiling for the air-temp view, "auto" to pin
    it to the measured whole-episode peak (adds a no-render measurement pass), or "none" to auto-range
    per frame (the original, flickering behavior).
  * Needs Pillow (pip install Pillow) for GIF encoding.
"""

import argparse
import os
import sys
from pathlib import Path

# Headless: composite offscreen, no window required.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import numpy as np      # noqa: E402
import pygame           # noqa: E402
import torch            # noqa: E402
from PIL import Image   # noqa: E402

from firecracker_env import FirecrackerEnv, ViewMode  # noqa: E402
from config import load_config                        # noqa: E402


def resolve_layout(preset: str, view_grid: tuple[int, int], fuel_names: list[str]):
    """Return (panes, pane_fuel, pane_labels) for the chosen preset.

    - "layers": the 3x2 layered-fuel demo (three Fire & Fuel panes filtered to grass+shrub, the tree
      boles, and the canopy, plus air temperature, terrain, and wind).
    - "auto": fill the grid row-major from the env's default view assignment; no fuel filtering.
    """
    rows, cols = view_grid
    if preset == "auto":
        from firecracker_env import DEFAULT_PANE_VIEWS
        n = rows * cols
        panes = [DEFAULT_PANE_VIEWS[i % len(DEFAULT_PANE_VIEWS)] for i in range(n)]
        return panes, {}, {}

    # preset == "layers"
    if view_grid != (3, 2):
        raise SystemExit("--preset layers needs a 3x2 grid (got "
                         f"{rows}x{cols}); use --views 3x2 or --preset auto.")

    def idx(*names: str) -> set[int]:
        missing = [n for n in names if n not in fuel_names]
        if missing:
            raise SystemExit(f"--preset layers expects fuel types {missing} which this config lacks; "
                             "use --preset auto for this fuel set.")
        return {fuel_names.index(n) for n in names}

    panes = [ViewMode.FUEL, ViewMode.FUEL, ViewMode.FUEL,
             ViewMode.TEMPERATURE, ViewMode.TERRAIN, ViewMode.WIND]
    pane_fuel = {
        0: idx("grass", "shrub"),
        1: idx("tree_bole_surface", "tree_bole_canopy"),
        2: idx("tree_canopy"),
    }
    pane_labels = {
        0: "Grass + Shrub  —  fuel & fire",
        1: "Bole (surface + canopy)  —  fuel & fire",
        2: "Canopy  —  fuel & fire",
    }
    return panes, pane_fuel, pane_labels


def build_and_ignite(args, view_grid, panes, pane_fuel, pane_labels) -> FirecrackerEnv:
    """Create the grid env, configure the panes, apply the wind override, and ignite a single-cell
    point fire at the fuel-bearing cell nearest the grid center -- deterministically, so a measure
    pass and a render pass produce the identical episode."""
    env = FirecrackerEnv(config=load_config(args.config), render_mode="human",
                         map_name=args.map, view_grid=view_grid, panes=panes)
    env.reset(seed=args.seed)
    for i, sel in pane_fuel.items():
        env._pane_fuel_selected[i] = set(sel)
    for i, lbl in pane_labels.items():
        env._pane_labels[i] = lbl
    env._active_pane = 0

    if args.wind != "map":
        target = float(args.wind)
        cur = float((env._u_amb_x ** 2 + env._u_amb_y ** 2) ** 0.5)
        scale = target / cur if cur > 0 else 0.0
        for a in ("_u_amb_x", "_u_amb_y", "_x_wind_vel", "_y_wind_vel", "_x_wind_fire", "_y_wind_fire"):
            setattr(env, a, getattr(env, a) * scale)

    idx = env._sim.surface_ignitable_indices
    p, g = env._pad, env.grid_size
    fine = env._fuel[idx, p:p + g, p:p + g].sum(dim=0)
    receptive = torch.nonzero(fine > 0.3, as_tuple=False)
    if receptive.numel() == 0:
        raise SystemExit(f"map {args.map!r} has no fine surface fuel to ignite.")
    center = torch.tensor([g // 2, g // 2], device=receptive.device)
    nearest = receptive[((receptive - center) ** 2).sum(dim=1).argmin()]
    env._spawn_fire_patch(int(nearest[0]), int(nearest[1]))
    env._surfaces_dirty = True
    return env


def composite_grid_gui(env: FirecrackerEnv) -> np.ndarray:
    """Draw the full multi-view window (top bar + pane grid) into env._screen, exactly as the human
    render() path does, and return it as an (H, W, 3) image."""
    env._rebuild_surfaces_if_dirty()
    env._screen.fill((0, 0, 0))
    env._draw_top_bar()
    env._render_grid()
    return pygame.surfarray.array3d(env._screen).transpose(1, 0, 2).copy()


def measure_air_ceiling(args, view_grid, panes, pane_fuel, pane_labels) -> float:
    """No-render pass over the deterministic episode: the peak surface air temperature, for pinning
    the air-temp color ceiling."""
    env = build_and_ignite(args, view_grid, panes, pane_fuel, pane_labels)
    surface_air = lambda: env._crop(env._air_temperatures.select(-3, 0))
    peak = float(surface_air().max())
    for _ in range(args.steps):
        env.step(None)
        peak = max(peak, float(surface_air().max()))
    env.close()
    return peak


def encode_gif(stack: np.ndarray, out_path: str, frame_ms: int) -> None:
    """Encode the frame stack with ONE shared palette (built from a spatial subsample of every frame)
    so static regions stay color-steady frame to frame; dither=NONE keeps them from shimmering."""
    n, h, w, _ = stack.shape
    sample = stack[:, ::4, ::4, :].reshape(-1, 1, 3)
    pal = Image.fromarray(sample).convert("P", palette=Image.ADAPTIVE, colors=256)
    frames = [Image.fromarray(f).quantize(palette=pal, dither=Image.NONE) for f in stack]
    print(f"Encoding {n} frames at {w}x{h}  ({frame_ms}ms/frame -> {n * frame_ms / 1000:.1f}s)...")
    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=frame_ms,
                   loop=0, optimize=True, disposal=1)
    print(f"Wrote {out_path}  ({os.path.getsize(out_path) / 1e6:.1f} MB)")


def parse_views(spec: str) -> tuple[int, int]:
    try:
        rows, cols = (int(x) for x in spec.lower().split("x"))
    except ValueError:
        raise SystemExit(f"--views must look like RxC, e.g. 3x2; got {spec!r}")
    if not (1 <= rows <= 3 and 1 <= cols <= 3):
        raise SystemExit(f"--views rows and cols must each be 1..3; got {spec!r}")
    return rows, cols


def main() -> None:
    parser = argparse.ArgumentParser(description="Record a point-fire spread as a multi-view GUI GIF")
    parser.add_argument("--out", default="fire_grid.gif", help="output GIF path (default: fire_grid.gif)")
    parser.add_argument("--views", default="3x2", help="grid layout RxC, 1..3 each (default: 3x2)")
    parser.add_argument("--preset", choices=["layers", "auto"], default="layers",
                        help="'layers' = the 3x2 layered-fuel demo; 'auto' = the env's default views")
    parser.add_argument("--config", help="path to a TOML config (default: cfg/default.toml)")
    parser.add_argument("--map", default="map_0003", help="map to load (default: map_0003)")
    parser.add_argument("--seed", type=int, default=0, help="reset seed (default: 0)")
    parser.add_argument("--steps", type=int, default=2048, help="simulation ticks to run (default: 2048)")
    parser.add_argument("--warmup", type=int, default=48,
                        help="ticks to run before the GIF starts, so it opens with fire established (default: 48)")
    parser.add_argument("--wind", default="13", metavar="MS",
                        help="ambient wind speed in m/s (direction kept), or 'map' for the map's own (default: 13)")
    parser.add_argument("--air-ceiling", default="323.15", metavar="K", dest="air_ceiling",
                        help="fixed air-temp color ceiling in kelvin, 'auto' (episode peak), or 'none' "
                             "(per-frame auto-range) (default: 323.15 = 50 C)")
    parser.add_argument("--duration", type=float, default=20.0, help="GIF playback seconds (default: 20)")
    parser.add_argument("--fps", type=int, default=20, help="GIF frames per second (default: 20)")
    args = parser.parse_args()

    if args.warmup >= args.steps:
        parser.error(f"--warmup ({args.warmup}) must be less than --steps ({args.steps})")

    view_grid = parse_views(args.views)
    frame_ms = round(1000 / args.fps)
    n_frames = max(1, round(args.duration * 1000 / frame_ms))
    sample_every = max(1, (args.steps - args.warmup) // n_frames)

    # A probe env just to read the fuel-type names for the layout (cheap; no stepping).
    probe = FirecrackerEnv(config=load_config(args.config), render_mode=None, map_name=args.map)
    fuel_names = list(probe._sim.fuel_type_names)
    probe.close()
    panes, pane_fuel, pane_labels = resolve_layout(args.preset, view_grid, fuel_names)

    # Resolve the air-temp ceiling (may need a measurement pass).
    if args.air_ceiling == "none":
        ceiling = None
    elif args.air_ceiling == "auto":
        ceiling = measure_air_ceiling(args, view_grid, panes, pane_fuel, pane_labels)
        print(f"Measured episode peak air temperature = {ceiling:.1f} K ({ceiling - 273.15:.0f} C)")
    else:
        ceiling = float(args.air_ceiling)

    env = build_and_ignite(args, view_grid, panes, pane_fuel, pane_labels)
    env._fixed_air_temp_max = ceiling
    wind_desc = "map default" if args.wind == "map" else f"{float(args.wind):.0f} m/s"
    ceil_desc = "auto-range" if ceiling is None else f"{ceiling:.0f} K"
    print(f"{view_grid[0]}x{view_grid[1]} grid ({args.preset}), map {args.map}, wind {wind_desc}, "
          f"air ceiling {ceil_desc}; window {env._screen.get_size()}")

    for _ in range(args.warmup):
        env.step(None)
    print(f"Warmed up {args.warmup} steps; capturing {args.steps - args.warmup} steps every {sample_every}")

    frames = [composite_grid_gui(env)]
    for step in range(args.warmup + 1, args.steps + 1):
        env.step(None)
        if step % sample_every == 0:
            frames.append(composite_grid_gui(env))
        if step % 256 == 0:
            print(f"  step {step}/{args.steps}  captured {len(frames)} frames")
    env.close()

    encode_gif(np.stack(frames), args.out, frame_ms)


if __name__ == "__main__":
    main()
