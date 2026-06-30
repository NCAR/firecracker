"""
Tests for the array-agnostic field builders in gen_maps.MapGenerator.

The derived-field math (temperature, pressure, mass, oxygen, vegetation thinning) runs
on either a NumPy array (the CPU path and the test API) or a torch tensor (the batched
GPU build path). These pin down both halves of that contract:
  * NumPy in -> NumPy out, float32, unchanged from the original CPU behaviour, and
  * torch in -> torch out, same values (so the GPU build reuses the same physics).
"""

import sys
from pathlib import Path

import numpy as np
import torch

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import load_config  # noqa: E402
from gen_maps import MapGenerator  # noqa: E402

GRID = 24


def _gen() -> MapGenerator:
    return MapGenerator(load_config())


def _elevation_np() -> np.ndarray:
    # A smooth ramp of plausible elevations [m], spanning sea level to near the peak.
    z = np.linspace(0.0, 450.0, GRID * GRID, dtype=np.float32).reshape(GRID, GRID)
    return z


def _relief_np() -> np.ndarray:
    # Normalised [0, 1] relief, the input the vegetation multiplier expects.
    return np.linspace(0.0, 1.0, GRID * GRID, dtype=np.float32).reshape(GRID, GRID)


def test_numpy_in_numpy_out_float32():
    gen = _gen()
    z = _elevation_np()
    air = gen.air_temperature_profile(z)
    for field in (air, gen.pressure_profile(z), gen.boundary_layer_mass(z),
                  gen.oxygen_profile(z, air),
                  gen.altitude_vegetation_multiplier(_relief_np(), 0),
                  gen.normalize_grid(z)):
        assert isinstance(field, np.ndarray)
        assert field.dtype == np.float32


def test_torch_in_torch_out_matches_numpy():
    gen = _gen()
    z_np = _elevation_np()
    z_t = torch.as_tensor(z_np, dtype=torch.float64)
    relief_np = _relief_np()
    relief_t = torch.as_tensor(relief_np, dtype=torch.float64)

    air_np = gen.air_temperature_profile(z_np)
    air_t = gen.air_temperature_profile(z_t)
    assert isinstance(air_t, torch.Tensor)

    pairs = [
        (air_np, air_t),
        (gen.pressure_profile(z_np), gen.pressure_profile(z_t)),
        (gen.boundary_layer_mass(z_np), gen.boundary_layer_mass(z_t)),
        (gen.oxygen_profile(z_np, air_np), gen.oxygen_profile(z_t, air_t)),
        (gen.altitude_vegetation_multiplier(relief_np, 0),
         gen.altitude_vegetation_multiplier(relief_t, 0)),
        (gen.normalize_grid(z_np), gen.normalize_grid(z_t)),
    ]
    for arr_np, arr_t in pairs:
        assert isinstance(arr_t, torch.Tensor)
        # float32 numpy vs float64 torch: compare with a float32-scale tolerance.
        assert np.allclose(arr_np, arr_t.numpy(), rtol=1e-5, atol=1e-4)


def test_zero_lapse_rate_uses_isothermal_pressure():
    # The gamma <= 0 branch uses exp(); exercise it on both backends.
    gen = _gen()
    gen.temperature_lapse_rate = 0.0
    z_np = _elevation_np()
    z_t = torch.as_tensor(z_np, dtype=torch.float64)
    p_np = gen.pressure_profile(z_np)
    p_t = gen.pressure_profile(z_t)
    assert np.allclose(p_np, p_t.numpy(), rtol=1e-5, atol=1e-2)


def test_batched_normalize_is_per_map():
    # A batched (B, H, W) tensor normalises each map over its own extent: every map's
    # min -> 0 and max -> 1 independently, regardless of the other maps' ranges.
    gen = _gen()
    batch = torch.stack([
        torch.linspace(0.0, 1.0, GRID * GRID, dtype=torch.float64).reshape(GRID, GRID),
        torch.linspace(-5.0, 50.0, GRID * GRID, dtype=torch.float64).reshape(GRID, GRID),
    ])
    out = gen.normalize_grid(batch)
    assert out.shape == batch.shape
    per_map_min = out.amin(dim=(-2, -1))
    per_map_max = out.amax(dim=(-2, -1))
    assert torch.allclose(per_map_min, torch.zeros(2, dtype=out.dtype), atol=1e-9)
    assert torch.allclose(per_map_max, torch.ones(2, dtype=out.dtype), atol=1e-9)


def test_batched_sample_perlin_matches_per_map():
    # sample_perlin_grid with a list of bases stacks the per-base fields, so a future
    # batched build matches looping generate_map.
    gen = _gen()
    bases = [1, 7, 19]
    batch = gen.sample_perlin_grid(GRID, bases, scale=24.0)
    assert batch.shape == (3, GRID, GRID)
    stacked = torch.stack([gen.sample_perlin_grid(GRID, b, scale=24.0) for b in bases])
    assert torch.allclose(batch, stacked)


def test_device_property_resolves():
    # The lazy device property resolves to a real torch device without needing a GPU.
    dev = MapGenerator(load_config()).device
    assert isinstance(dev, torch.device)
    assert dev.type in ("cpu", "cuda")
