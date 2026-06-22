"""
World-model observation normalization (Phase 7).

Simulation.build_observation stacks three SI fields -- the hottest fuel temperature [K], the
total fuel mass [kg/m^2], and terrain elevation [m] -- whose raw magnitudes span several orders.
Each channel is mapped to roughly [0, 1] by the documented affine window in
physics_constants.OBS_NORM, (value - offset) / scale, so the world model sees comparable scales.
These tests call build_observation directly on synthetic tensors and pin that mapping (the
per-channel reductions, the normalization, and that it is rank-agnostic across a batch axis).
"""

import torch

import physics_constants as pc
from simulation import OBS_CHANNELS, Simulation


GRID = 8
DTYPE = torch.float64


def _obs(fuel_temperatures, fuel, terrain):
    return Simulation.build_observation(fuel_temperatures, fuel, terrain)


def _const(value, *shape):
    return torch.full(shape, float(value), dtype=DTYPE)


def test_channel_order_matches_obs_channels():
    """The stacked channel axis is exactly OBS_CHANNELS, length 3."""
    obs = _obs(_const(pc.T_REF, 2, GRID, GRID), _const(0.0, 2, GRID, GRID), _const(0.0, GRID, GRID))
    assert obs.shape == (len(OBS_CHANNELS), GRID, GRID) == (3, GRID, GRID)


def test_rest_state_maps_near_zero():
    """Ambient fuel temp (~T_REF), no fuel, sea-level terrain -> every channel ~ 0."""
    obs = _obs(_const(pc.T_REF, 2, GRID, GRID), _const(0.0, 2, GRID, GRID), _const(0.0, GRID, GRID))
    torch.testing.assert_close(obs, torch.zeros_like(obs), atol=1e-9, rtol=0.0)


def test_terrain_normalized_by_elev_max():
    """Terrain at the documented ceiling maps to 1; the channel is z / ELEV_MAX_M."""
    obs = _obs(_const(pc.T_REF, 1, GRID, GRID), _const(0.0, 1, GRID, GRID), _const(pc.ELEV_MAX_M, GRID, GRID))
    torch.testing.assert_close(obs[2], torch.ones((GRID, GRID), dtype=DTYPE), atol=1e-9, rtol=0.0)


def test_fuel_is_summed_then_normalized():
    """Fuel channel is the total mass over types, normalized by FUEL_REF_KG_M2."""
    half = pc.FUEL_REF_KG_M2 / 2.0
    obs = _obs(_const(pc.T_REF, 2, GRID, GRID), _const(half, 2, GRID, GRID), _const(0.0, GRID, GRID))
    # two types each at half the reference -> total == reference -> 1.
    torch.testing.assert_close(obs[1], torch.ones((GRID, GRID), dtype=DTYPE), atol=1e-9, rtol=0.0)


def test_fuel_temperature_uses_hottest_type_and_rises_with_fire():
    """The temperature channel takes the hottest fuel type and a flame maps near (or above) 1."""
    temps = _const(pc.T_REF, 2, GRID, GRID)
    temps[0, 4, 4] = 2300.0           # one hot cell in type 0
    obs = _obs(temps, _const(1.0, 2, GRID, GRID), _const(0.0, GRID, GRID))
    expected_hot = (2300.0 - pc.T_REF) / pc.OBS_NORM["fuel_temperature"][1]
    assert abs(float(obs[0, 4, 4]) - expected_hot) < 1e-9
    assert float(obs[0, 4, 4]) > float(obs[0, 0, 0])   # flame sits above the ambient band
    assert expected_hot > 0.9                          # a 2300 K flame lands near 1


def test_batched_matches_single_world():
    """A leading batch axis normalizes identically to stacking single-world results."""
    temps = torch.rand(3, 2, GRID, GRID, dtype=DTYPE) * 2000.0 + pc.T_REF
    fuel = torch.rand(3, 2, GRID, GRID, dtype=DTYPE) * 10.0
    terrain = torch.rand(3, GRID, GRID, dtype=DTYPE) * pc.ELEV_MAX_M
    batched = _obs(temps, fuel, terrain)
    assert batched.shape == (3, len(OBS_CHANNELS), GRID, GRID)
    for b in range(3):
        single = _obs(temps[b], fuel[b], terrain[b])
        torch.testing.assert_close(batched[b], single, atol=1e-12, rtol=0.0)
