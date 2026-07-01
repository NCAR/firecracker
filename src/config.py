"""
config.py

Loads TOML configuration for the Firecracker simulation.
The default config lives at <repo_root>/cfg/default.toml.
Pass an explicit path to load_config() to use an alternative file.
"""

import tomllib
from pathlib import Path

_DEFAULT_PATH = Path(__file__).parent.parent / "cfg" / "default.toml"


def load_config(path: str | Path | None = None) -> dict:
    p = Path(path) if path is not None else _DEFAULT_PATH
    with open(p, "rb") as f:
        return tomllib.load(f)


# Default boundary-sponge / observation-padding width [cells]; mirrors Simulation's default.
_DEFAULT_SPONGE_WIDTH = 8


def boundary_pad(cfg: dict | None) -> int:
    """Cells of padding ringing the observed grid (= the open-boundary sponge width).

    The simulation runs on a grid padded by this many cells on every side, so the entire
    sponge taper sits in the ring and the observed interior is sponge-free. The observation,
    rendering and UI use only the inner grid; physics uses inner + 2*pad. Set sponge_width = 0
    to recover the un-padded grid (sponge on the observed edges, the pre-padding behaviour).
    """
    return int((cfg or {}).get("momentum", {}).get("sponge_width", _DEFAULT_SPONGE_WIDTH))


def physics_grid_size(cfg: dict | None, observed_grid_size: int) -> int:
    """Physics grid side [cells] = observed grid + 2*boundary_pad (the padded simulation grid)."""
    return observed_grid_size + 2 * boundary_pad(cfg)
