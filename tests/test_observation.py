"""
World-model observation normalization (Phase 7).

Simulation.build_observation stacks five SI fields -- the hottest fuel temperature [K], the
total fuel mass [kg/m^2], terrain elevation [m], and the two near-surface fire-wind components
[m/s] -- whose raw magnitudes span several orders. Each channel is mapped to roughly [0, 1] by
the documented affine window in
physics_constants.OBS_NORM, (value - offset) / scale, so the world model sees comparable scales.
These tests call build_observation directly on synthetic tensors and pin that mapping (the
per-channel reductions, the normalization, and that it is rank-agnostic across a batch axis).
"""

import torch

import physics_constants as pc
from simulation import OBS_CHANNELS, Simulation


GRID = 8
DTYPE = torch.float64


def _obs(fuel_temperatures, fuel, terrain, wind_x=None, wind_y=None):
    if wind_x is None:
        wind_x = torch.zeros_like(terrain)
    if wind_y is None:
        wind_y = torch.zeros_like(terrain)
    return Simulation.build_observation(fuel_temperatures, fuel, terrain, wind_x, wind_y)


def _const(value, *shape):
    return torch.full(shape, float(value), dtype=DTYPE)


def test_channel_order_matches_obs_channels():
    """The stacked channel axis is exactly OBS_CHANNELS, length 5."""
    obs = _obs(_const(pc.T_REF, 2, GRID, GRID), _const(0.0, 2, GRID, GRID), _const(0.0, GRID, GRID))
    assert obs.shape == (len(OBS_CHANNELS), GRID, GRID) == (5, GRID, GRID)


def test_rest_state_maps_near_zero():
    """Ambient fuel temp (~T_REF), no fuel, sea-level terrain -> every channel ~ 0."""
    obs = _obs(_const(pc.T_REF, 2, GRID, GRID), _const(0.0, 2, GRID, GRID), _const(0.0, GRID, GRID))
    torch.testing.assert_close(obs, torch.zeros_like(obs), atol=1e-9, rtol=0.0)


def test_terrain_normalized_by_elev_max():
    """Terrain at the documented ceiling maps to 1; the channel is z / ELEV_MAX_M."""
    obs = _obs(_const(pc.T_REF, 1, GRID, GRID), _const(0.0, 1, GRID, GRID), _const(pc.ELEV_MAX_M, GRID, GRID))
    torch.testing.assert_close(obs[2], torch.ones((GRID, GRID), dtype=DTYPE), atol=1e-9, rtol=0.0)


def test_wind_components_signed_and_normalized_by_wind_ref():
    """The wind channels pass the fire wind through, signed about 0 and scaled by WIND_REF_M_S."""
    wx = _const(pc.WIND_REF_M_S, GRID, GRID)        # +peak -> +1
    wy = _const(-pc.WIND_REF_M_S / 2.0, GRID, GRID)  # half the peak, reversed -> -0.5
    obs = _obs(_const(pc.T_REF, 1, GRID, GRID), _const(0.0, 1, GRID, GRID),
               _const(0.0, GRID, GRID), wx, wy)
    torch.testing.assert_close(obs[3], torch.ones((GRID, GRID), dtype=DTYPE), atol=1e-9, rtol=0.0)
    torch.testing.assert_close(obs[4], _const(-0.5, GRID, GRID), atol=1e-9, rtol=0.0)


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


def test_env_observation_matches_world_model(make_env):
    """The Gymnasium env's reset/step return the same world-model observation (and its space
    matches), so the agent trains on exactly what the rollout collector pools."""
    import numpy as np
    from scenarios import si_ridge

    env = make_env(*si_ridge(grid=24, ambient=(12.0, -4.0), peak_m=300.0))

    obs, _ = env.reset(seed=0)
    assert obs.shape == (len(OBS_CHANNELS), 24, 24)
    assert env.observation_space.shape == obs.shape
    assert env.observation_space.contains(obs)
    # The env observation is build_observation cropped to the observed interior, so crop expected.
    expected = env._crop(Simulation.build_observation(
        env._fuel_temperatures, env._fuel, env._terrain, env._x_wind_fire, env._y_wind_fire
    )).detach().cpu().numpy().astype("float32")
    np.testing.assert_array_equal(obs, expected)

    step_obs, *_ = env.step(0)
    step_expected = env._crop(Simulation.build_observation(
        env._fuel_temperatures, env._fuel, env._terrain, env._x_wind_fire, env._y_wind_fire
    )).detach().cpu().numpy().astype("float32")
    np.testing.assert_array_equal(step_obs, step_expected)


def test_padding_physics_grid_and_observation_crop(make_env):
    """The env runs physics on the padded grid (observed + 2*pad) and observes only the inner
    region; an action/click at an observed cell maps through the pad to the right physics cell."""
    import numpy as np
    from scenarios import make_config, build_map

    obs_grid = 16
    env = make_env(make_config(obs_grid), build_map(grid_size=obs_grid))
    pad = env._pad
    assert pad > 0                                      # the default sponge_width padding is active
    assert env._sim_size == obs_grid + 2 * pad
    assert env._terrain.shape[-1] == env._sim_size      # state lives on the padded physics grid

    obs, _ = env.reset(seed=0)
    assert obs.shape == (len(OBS_CHANNELS), obs_grid, obs_grid)   # observation is the inner region

    # A fire spawned at observed (0, 0) is centred at physics (pad, pad) -- i.e. the observed
    # top-left corner -- proving the screen->cell offset.
    ign = float(env._sim.ignition_thresholds.flatten()[0])
    env._spawn_fire_patch(0, 0)
    hot = env._fuel_temperatures[0] > ign
    assert bool(hot[pad, pad])                          # patch centre at the padded offset
    assert not bool(hot[0, 0])                          # not at the physics corner (inside the ring)
    assert bool(env._crop(hot)[0, 0])                   # and it shows at observed (0, 0)
