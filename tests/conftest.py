"""
Pytest fixtures for the Firecracker physics tests.

The reusable, pytest-free building blocks (make_config, build_map, the analytic
scenarios, and the invariant probes) live in scenarios.py so they can also be
imported by tools/visualize.py without pulling in pytest. This module only adds
the make_env fixture (which needs pytest + tmp_path) and re-exports the helpers
so tests can keep importing them from `conftest`.
"""

import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Re-export the pytest-free helpers (so `from conftest import ...` still works).
from scenarios import (  # noqa: E402,F401
    DEFAULT_GRID,
    build_map,
    make_config,
    to_numpy,
    total_air_energy,
    total_mass,
    total_oxygen,
)
from map_loader import save_map           # noqa: E402
from firecracker_env import FirecrackerEnv  # noqa: E402


@pytest.fixture
def make_env(tmp_path):
    """Factory: bake a MapData to a temp dir and return a reset, headless env."""
    def _make(config: dict, map_data, seed: int = 0) -> FirecrackerEnv:
        save_map(tmp_path / "fixture.npz", map_data)
        env = FirecrackerEnv(config=config, maps_dir=str(tmp_path), map_name="fixture")
        env.reset(seed=seed)
        return env
    return _make
