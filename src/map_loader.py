"""
map_loader.py

Source-agnostic map I/O for the Firecracker simulation.

A "map" is a fully-baked initial state: terrain, the initial
temperature/pressure/oxygen fields, their relaxation targets, and the per-type
fuel/fuel-temperature stacks. This module is the stable seam between *producers*
of that state (the synthetic generator in gen_maps.py, hand-crafted test
scenarios, or a future real-data importer) and the *consumer* (the environment).

It deliberately has no dependency on the physics engine (Simulation) or on the
generation code: it only reads, writes, validates, and selects map files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

MAP_SUFFIX = ".npz"

# Bumped whenever the on-disk field semantics change. Maps stamped with a different
# version — including legacy maps that predate the field, which read back as 0 — are
# rejected by validate_against_config with a regenerate hint.
#   v1 = SI units: temperature [K], mass [kg/m^2], terrain [m], oxygen [kg/m^3], fuel [kg/m^2].
#   v2 = + per-map synoptic ambient wind vector (ambient_wind_x/y) [m/s].
#   v3 = mass is now the shallow boundary-layer areal mass (level-lid rest state), not the
#        full hydrostatic column -- so terrain strongly channels the wind.
UNITS_VERSION = 3

# Maps live at <repo_root>/maps by default (alongside cfg/), same convention as
# config.py. A relative maps dir is anchored here so the app finds its maps
# regardless of the current working directory; absolute paths are left untouched.
_REPO_ROOT = Path(__file__).resolve().parent.parent


def _resolve_dir(maps_dir: str | Path) -> Path:
    maps_dir = Path(maps_dir)
    return maps_dir if maps_dir.is_absolute() else _REPO_ROOT / maps_dir


@dataclass
class MapData:
    """A fully-baked initial state plus the metadata needed to validate it.

    The relaxation targets (`temp_eq`, `oxygen_eq`) are optional: when omitted
    they default to copies of the initial fields, so a hand-crafted map only
    needs to supply terrain, temperature, pressure, oxygen and fuel.
    """

    terrain:           np.ndarray  # (H, W)
    air_temperatures:  np.ndarray  # (H, W)
    mass:              np.ndarray  # (H, W) shallow boundary-layer areal mass [kg/m^2]
    oxygen:            np.ndarray  # (H, W)
    fuel:              np.ndarray  # (N, H, W)
    fuel_temperatures: np.ndarray  # (N, H, W)
    fuel_type_names:   list[str]
    grid_size:         int
    temp_eq:           np.ndarray | None = None  # (H, W) temperature relaxation target
    oxygen_eq:         np.ndarray | None = None  # (H, W) oxygen replenishment target
    seed:              int | None = None
    source:            str = "unknown"
    units_version:     int = UNITS_VERSION       # field semantics version (see UNITS_VERSION)
    # Per-map synoptic (prevailing) wind [m/s]: the spatially-uniform background the
    # momentum drag relaxes toward. Spatially constant, so stored as two scalars.
    ambient_wind_x:    float = 0.0
    ambient_wind_y:    float = 0.0

    def __post_init__(self) -> None:
        if self.temp_eq is None:
            self.temp_eq = self.air_temperatures.copy()
        if self.oxygen_eq is None:
            self.oxygen_eq = self.oxygen.copy()

    @property
    def num_fuel_types(self) -> int:
        return len(self.fuel_type_names)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_map(path: str | Path, m: MapData) -> Path:
    """Write a map to a compressed .npz file, creating parent dirs as needed."""
    path = Path(path)
    if not path.is_absolute():
        path = _REPO_ROOT / path
    if path.suffix != MAP_SUFFIX:
        path = path.with_suffix(MAP_SUFFIX)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        terrain=m.terrain,
        air_temperatures=m.air_temperatures,
        mass=m.mass,
        oxygen=m.oxygen,
        fuel=m.fuel,
        fuel_temperatures=m.fuel_temperatures,
        temp_eq=m.temp_eq,
        oxygen_eq=m.oxygen_eq,
        fuel_type_names=np.array(m.fuel_type_names, dtype="U"),
        grid_size=np.int64(m.grid_size),
        seed=np.int64(-1 if m.seed is None else m.seed),
        source=np.str_(m.source),
        units_version=np.int64(m.units_version),
        ambient_wind_x=np.float64(m.ambient_wind_x),
        ambient_wind_y=np.float64(m.ambient_wind_y),
    )
    return path


def load_map(path: str | Path) -> MapData:
    """Load a map from a .npz file produced by save_map (or a compatible producer)."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        seed = int(data["seed"]) if "seed" in data else -1
        # Absent on legacy (pre-SI) maps; read back as version 0 so validation rejects them.
        units_version = int(data["units_version"]) if "units_version" in data else 0
        return MapData(
            terrain=data["terrain"].astype(np.float32),
            air_temperatures=data["air_temperatures"].astype(np.float32),
            mass=data["mass"].astype(np.float32),
            oxygen=data["oxygen"].astype(np.float32),
            fuel=data["fuel"].astype(np.float32),
            fuel_temperatures=data["fuel_temperatures"].astype(np.float32),
            temp_eq=data["temp_eq"].astype(np.float32) if "temp_eq" in data else None,
            oxygen_eq=data["oxygen_eq"].astype(np.float32) if "oxygen_eq" in data else None,
            fuel_type_names=[str(n) for n in data["fuel_type_names"]],
            grid_size=int(data["grid_size"]),
            seed=None if seed < 0 else seed,
            source=str(data["source"]) if "source" in data else "unknown",
            units_version=units_version,
            ambient_wind_x=float(data["ambient_wind_x"]) if "ambient_wind_x" in data else 0.0,
            ambient_wind_y=float(data["ambient_wind_y"]) if "ambient_wind_y" in data else 0.0,
        )


# ---------------------------------------------------------------------------
# Discovery / selection
# ---------------------------------------------------------------------------

def list_maps(maps_dir: str | Path) -> list[Path]:
    """All map files in maps_dir, sorted by name (empty list if the dir is missing)."""
    maps_dir = _resolve_dir(maps_dir)
    if not maps_dir.is_dir():
        return []
    return sorted(maps_dir.glob(f"*{MAP_SUFFIX}"))


def resolve_map(
    maps_dir: str | Path,
    name: str | None = None,
    rng: np.random.Generator | None = None,
) -> Path:
    """Resolve which map file to load.

    With a name, return that specific map (error if it does not exist). Without a
    name, return a random map from maps_dir. Raises FileNotFoundError with a hint
    toward gen_maps.py when no usable map is found.
    """
    maps_dir = _resolve_dir(maps_dir)
    if name:
        candidate = maps_dir / name
        if candidate.suffix != MAP_SUFFIX:
            candidate = candidate.with_suffix(MAP_SUFFIX)
        if not candidate.is_file():
            available = [p.name for p in list_maps(maps_dir)]
            raise FileNotFoundError(
                f"Map '{name}' not found in '{maps_dir}'. "
                f"Available: {available or '(none)'}."
            )
        return candidate

    maps = list_maps(maps_dir)
    if not maps:
        raise FileNotFoundError(
            f"No maps found in '{maps_dir}'. Generate some first, e.g.\n"
            f"    python src/gen_maps.py --count 8 --out {maps_dir}"
        )
    rng = rng if rng is not None else np.random.default_rng()
    return maps[int(rng.integers(0, len(maps)))]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_against_config(
    m: MapData, grid_size: int, fuel_type_names: list[str]
) -> None:
    """Raise ValueError if a map is incompatible with the running configuration.

    The fuel axis is positional (indexed against the per-type physics arrays), so
    the fuel-type names must match in both content and order.
    """
    if m.units_version != UNITS_VERSION:
        legacy = " (legacy nondimensional map)" if m.units_version == 0 else ""
        raise ValueError(
            f"Map units_version {m.units_version}{legacy} does not match the expected "
            f"{UNITS_VERSION} (SI units). Regenerate the maps, e.g.\n"
            f"    python src/gen_maps.py --count 8 --out maps"
        )

    expected_2d = (grid_size, grid_size)
    for field_name in ("terrain", "air_temperatures", "mass", "oxygen", "temp_eq", "oxygen_eq"):
        arr = getattr(m, field_name)
        if arr.shape != expected_2d:
            raise ValueError(
                f"Map field '{field_name}' has shape {arr.shape}, expected {expected_2d} "
                f"(grid_size={grid_size})."
            )

    n = len(fuel_type_names)
    expected_3d = (n, grid_size, grid_size)
    for field_name in ("fuel", "fuel_temperatures"):
        arr = getattr(m, field_name)
        if arr.shape != expected_3d:
            raise ValueError(
                f"Map field '{field_name}' has shape {arr.shape}, expected {expected_3d}."
            )

    if list(m.fuel_type_names) != list(fuel_type_names):
        raise ValueError(
            f"Map fuel types {m.fuel_type_names} do not match configured fuel types "
            f"{fuel_type_names} (names and order must match)."
        )
