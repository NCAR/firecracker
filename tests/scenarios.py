"""
Shared physics-test scenarios and helpers.

This module is deliberately pytest-free so it can be imported by both the test
suite and tools/visualize.py. Each scenario is a (config, MapData) pair built
from analytic fields, so the assertion tests and the visualizer operate on the
exact same states. VISUALS adds, per scenario, the view modes and timesteps
worth looking at and a one-line description.
"""

import sys
from pathlib import Path

import numpy as np
import torch

# The src/ modules import each other flatly (e.g. `from simulation import ...`),
# so put src/ on the path before importing them.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import physics_constants as pc          # noqa: E402
from map_loader import MapData          # noqa: E402
from firecracker_env import ViewMode    # noqa: E402
from gen_maps import MapGenerator       # noqa: E402

DEFAULT_GRID = 16


# ---------------------------------------------------------------------------
# Config / map builders
# ---------------------------------------------------------------------------

def make_config(
    grid_size: int = DEFAULT_GRID,
    fuel_type_names: tuple[str, ...] = ("grass",),
    *,
    fire: bool = False,
    relaxation: bool = False,
    radiation: bool = False,
    momentum: dict | None = None,
    channeling: dict | None = None,
) -> dict:
    """A minimal config with the major sinks/sources toggled for isolation.

    The momentum core uses its code defaults unless `momentum` overrides are given
    (e.g. {"sponge_strength": 0.0, "drag_coeff": 0.0} for a closed, conservative core).
    Channeling defaults to OFF here (a `channeling` override opts it in) so the isolation
    scenarios see the bare prognostic wind. Each named fuel type gets a subtable so
    Simulation parses matching fuel_type_names.
    """
    config = {
        "environment": {"grid_size": grid_size, "window_size": grid_size, "max_steps": 10_000},
        "fire": {"enabled": fire, "spawn_fire": False},
        "relaxation": {"enabled": relaxation},
        "radiation": {"enabled": radiation},
        "channeling": {"enabled": False},
        "fuel_types": {name: {"ignition_threshold": 1.5} for name in fuel_type_names},
    }
    if momentum is not None:
        config["momentum"] = dict(momentum)
    if channeling is not None:
        config["channeling"] = dict(channeling)
    return config


def build_map(
    grid_size: int = DEFAULT_GRID,
    fuel_type_names: tuple[str, ...] = ("grass",),
    *,
    terrain: np.ndarray | None = None,
    air: np.ndarray | float | None = None,
    mass: np.ndarray | float | None = None,
    oxygen: np.ndarray | float | None = None,
    fuel: np.ndarray | None = None,
    fuel_temperatures: np.ndarray | None = None,
    ambient_wind: tuple[float, float] = (0.0, 0.0),
) -> MapData:
    """Build a MapData from explicit fields; anything omitted defaults to uniform.

    Scalars for air/mass/oxygen are broadcast to a full field, so a test can write
    e.g. air=0.5 and override just the cells it cares about. ambient_wind is the
    per-map synoptic wind vector (x, y) [m/s].
    """
    n = len(fuel_type_names)
    shape = (grid_size, grid_size)

    def field(value, default):
        if value is None:
            return np.full(shape, default, dtype=np.float32)
        if np.isscalar(value):
            return np.full(shape, float(value), dtype=np.float32)
        return np.asarray(value, dtype=np.float32)

    terrain = field(terrain, 0.0)
    air     = field(air, 0.5)
    mass    = field(mass, 1.0)
    oxygen  = field(oxygen, 0.5)
    fuel = np.zeros((n, *shape), dtype=np.float32) if fuel is None else np.asarray(fuel, dtype=np.float32)
    fuel_temperatures = (
        np.tile(air[np.newaxis], (n, 1, 1)).astype(np.float32)
        if fuel_temperatures is None else np.asarray(fuel_temperatures, dtype=np.float32)
    )

    return MapData(
        terrain=terrain,
        air_temperatures=air,
        mass=mass,
        oxygen=oxygen,
        fuel=fuel,
        fuel_temperatures=fuel_temperatures,
        fuel_type_names=list(fuel_type_names),
        grid_size=grid_size,
        ambient_wind_x=float(ambient_wind[0]),
        ambient_wind_y=float(ambient_wind[1]),
    )


# ---------------------------------------------------------------------------
# Invariant probes
# ---------------------------------------------------------------------------

def to_numpy(x) -> np.ndarray:
    """Host numpy view of a sim field, which is now a (possibly GPU) torch tensor."""
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def total_mass(env) -> float:
    return float(env._mass.sum())


def total_air_energy(env) -> float:
    """Conserved transport quantity E = sum(m * T)."""
    return float((env._mass * env._air_temperatures).sum())


def total_oxygen(env) -> float:
    return float(env._oxygen.sum())


# ---------------------------------------------------------------------------
# Analytic field helpers
# ---------------------------------------------------------------------------

def disc(grid: int, row: int, col: int, radius: int, value: float, base: float) -> np.ndarray:
    rows, cols = np.ogrid[:grid, :grid]
    field = np.full((grid, grid), base, dtype=np.float32)
    field[(rows - row) ** 2 + (cols - col) ** 2 <= radius ** 2] = value
    return field


def ramp_terrain(grid: int) -> np.ndarray:
    """Linear elevation ramp across columns, normalised to [0, 1]."""
    return np.tile(np.linspace(0.0, 1.0, grid, dtype=np.float32), (grid, 1))


# ---------------------------------------------------------------------------
# Scenario builders -> (config, MapData)
# ---------------------------------------------------------------------------

def uniform():
    return make_config(), build_map(air=0.5, mass=1.0, oxygen=0.5)


def hot_blob():
    air = disc(DEFAULT_GRID, 8, 8, 3, value=5.0, base=0.5)
    return make_config(), build_map(air=air, mass=1.0)


def corner_blob():
    air = disc(DEFAULT_GRID, 1, 1, 3, value=8.0, base=0.5)
    return make_config(), build_map(air=air, mass=1.0)


def oxygen_saturation():
    mass = np.tile(np.linspace(1.2, 0.8, DEFAULT_GRID, dtype=np.float32), (DEFAULT_GRID, 1))
    oxygen = np.full((DEFAULT_GRID, DEFAULT_GRID), 0.4, dtype=np.float32)
    oxygen[:, DEFAULT_GRID // 2:] = 1.0
    return make_config(), build_map(mass=mass, oxygen=oxygen)


def mass_gradient():
    """Flat terrain with a left-high / right-low mass ramp; should equalize smoothly."""
    mass = np.tile(np.linspace(1.3, 0.7, DEFAULT_GRID, dtype=np.float32), (DEFAULT_GRID, 1))
    return make_config(), build_map(mass=mass)


def equilibrium():
    config = make_config()
    terrain = ramp_terrain(DEFAULT_GRID)
    air = np.full((DEFAULT_GRID, DEFAULT_GRID), 0.5, dtype=np.float32)
    mass = MapGenerator(config).equilibrium_mass(terrain, air)
    return config, build_map(terrain=terrain, air=air, mass=mass)


def off_equilibrium():
    terrain = ramp_terrain(DEFAULT_GRID)
    return make_config(), build_map(terrain=terrain, air=0.5, mass=1.0)


# ---------------------------------------------------------------------------
# SI scenario builders (Phase 2 shallow-water core) -> (config, MapData)
# ---------------------------------------------------------------------------

def _si_state(terrain_m: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SI air temperature [K], boundary-layer mass [kg/m^2] and oxygen [kg/m^3] for terrain [m]."""
    gen = MapGenerator(make_config(terrain_m.shape[0]))
    air = gen.air_temperature_profile(terrain_m)
    return air, gen.boundary_layer_mass(terrain_m), gen.oxygen_profile(terrain_m, air)


def si_flat(grid: int = DEFAULT_GRID, ambient: tuple[float, float] = (0.0, 0.0),
            elevation_m: float = 0.0):
    """Flat SI world at one elevation, with a uniform synoptic wind."""
    terrain = np.full((grid, grid), float(elevation_m), dtype=np.float32)
    air, mass, oxygen = _si_state(terrain)
    return make_config(grid), build_map(grid, terrain=terrain, air=air, mass=mass,
                                        oxygen=oxygen, ambient_wind=ambient)


def gaussian_hill(grid: int, peak_m: float) -> np.ndarray:
    """A single centred Gaussian hill of height peak_m [m]."""
    rows, cols = np.ogrid[:grid, :grid]
    c = grid / 2.0
    r2 = ((rows - c) ** 2 + (cols - c) ** 2) / (2.0 * (grid / 5.0) ** 2)
    return (peak_m * np.exp(-r2)).astype(np.float32)


def si_hill(grid: int = DEFAULT_GRID, ambient: tuple[float, float] = (12.0, 4.0),
            peak_m: float = 400.0):
    """SI world with a Gaussian hill (terrain in metres) and a synoptic wind."""
    terrain = gaussian_hill(grid, peak_m)
    air, mass, oxygen = _si_state(terrain)
    return make_config(grid), build_map(grid, terrain=terrain, air=air, mass=mass,
                                        oxygen=oxygen, ambient_wind=ambient)


def si_ridge(grid: int = DEFAULT_GRID, ambient: tuple[float, float] = (15.0, 0.0),
             peak_m: float = 400.0):
    """SI world with a ridge running across the x-wind (terrain varies in x only)."""
    cols = np.arange(grid)
    c = grid / 2.0
    profile = peak_m * np.exp(-((cols - c) ** 2) / (2.0 * (grid / 8.0) ** 2))
    terrain = np.tile(profile.astype(np.float32), (grid, 1))
    air, mass, oxygen = _si_state(terrain)
    return make_config(grid), build_map(grid, terrain=terrain, air=air, mass=mass,
                                        oxygen=oxygen, ambient_wind=ambient)


def si_channel(grid: int = DEFAULT_GRID, ambient: tuple[float, float] = (15.0, 0.0),
               peak_m: float = 800.0):
    """SI world with two high walls flanking a narrow low channel aligned with the x-wind.

    Terrain varies in y only (two Gaussian walls at y = c +/- grid/6, channel at the centre row),
    so the channel runs along the wind -- the classic gap/Venturi geometry. A real atmosphere
    funnels and accelerates wind through such a channel; the level-lid shallow-water core does not
    (see test_narrow_channel_does_not_speed_up_wind).
    """
    rows = np.arange(grid)[:, None] * np.ones((1, grid))
    c, off, sig = grid / 2.0, grid / 6.0, grid / 18.0
    wall = peak_m * (np.exp(-((rows - (c - off)) ** 2) / (2.0 * sig ** 2))
                     + np.exp(-((rows - (c + off)) ** 2) / (2.0 * sig ** 2)))
    terrain = wall.astype(np.float32)
    air, mass, oxygen = _si_state(terrain)
    return make_config(grid), build_map(grid, terrain=terrain, air=air, mass=mass,
                                        oxygen=oxygen, ambient_wind=ambient)


# name -> (builder, view modes to render, timesteps to capture, description)
VISUALS = {
    "uniform": (
        uniform, [ViewMode.TEMPERATURE], [0, 5],
        "Uniform field: must be identical at t=0 and t=5 (no spurious source/sink).",
    ),
    "hot_blob": (
        hot_blob, [ViewMode.TEMPERATURE, ViewMode.PRESSURE], [0, 8, 25],
        "Hot blob convects + diffuses; total mass and energy stay conserved.",
    ),
    "mass_gradient": (
        mass_gradient, [ViewMode.PRESSURE], [0, 100, 400],
        "Pressure ramp must equalize smoothly toward uniform (no checkerboard stripes).",
    ),
    "corner_blob": (
        corner_blob, [ViewMode.TEMPERATURE], [0, 20, 40],
        "Blob against the corner: no-flux walls, totals must not drop.",
    ),
    "oxygen_saturation": (
        oxygen_saturation, [ViewMode.OXYGEN, ViewMode.PRESSURE], [0, 10, 20],
        "Oxygen advects into full cells; total conserved and capped at 1.0.",
    ),
    "equilibrium": (
        equilibrium, [ViewMode.PRESSURE, ViewMode.TERRAIN], [0, 1],
        "Convective rest state: pressure at t=1 should match t=0 (stationary).",
    ),
    "off_equilibrium": (
        off_equilibrium, [ViewMode.PRESSURE, ViewMode.TERRAIN], [0, 1],
        "Uniform mass over a slope is not a rest state: pressure visibly drifts.",
    ),
}
