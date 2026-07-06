"""
World-model observation normalization (Phase 7).

Simulation.build_observation stacks SI fields -- the air temperature [K], one temperature channel
per fuel type [K], one vegetation-mass channel per fuel type [kg/m^2], terrain elevation [m], the
two near-surface fire-wind components [m/s], and the one-hot vegetation biome -- whose raw
magnitudes span several orders. Each channel is mapped to roughly [0, 1] by the documented affine window in
physics_constants.OBS_NORM, (value - offset) / scale, so the world model sees comparable scales.
These tests call build_observation directly on synthetic tensors and pin that mapping (the
per-channel reductions, the normalization, and that it is rank-agnostic across a batch axis).
"""

import torch

import physics_constants as pc
from simulation import Simulation, BIOME_NAMES, obs_channel_names


GRID = 8
DTYPE = torch.float64
N_BIOME = len(BIOME_NAMES)


IGN_DEFAULT = 573.0   # default per-type ignition threshold [K]; ambient (T_REF) stays below it


def _obs(fuel_temperatures, fuel, terrain, wind_x=None, wind_y=None,
         air_temperatures=None, biome_onehot=None, ignition_thresholds=None):
    if wind_x is None:
        wind_x = torch.zeros_like(terrain)
    if wind_y is None:
        wind_y = torch.zeros_like(terrain)
    if air_temperatures is None:
        air_temperatures = torch.full_like(terrain, float(pc.T_REF))   # rest air -> channel ~ 0
    if biome_onehot is None:
        # Neutral all-zero biome block for tests not exercising biomes (keeps them at 0).
        biome_onehot = terrain.new_zeros((*terrain.shape[:-2], N_BIOME, *terrain.shape[-2:]))
    if ignition_thresholds is None:
        n_fuel = fuel_temperatures.shape[-3]
        ignition_thresholds = torch.full((n_fuel, 1, 1), IGN_DEFAULT, dtype=fuel_temperatures.dtype)
    return Simulation.build_observation(
        air_temperatures, fuel_temperatures, fuel, terrain, wind_x, wind_y,
        biome_onehot, ignition_thresholds,
    )


def _const(value, *shape):
    return torch.full(shape, float(value), dtype=DTYPE)


def test_channel_count_is_five_plus_two_per_fuel_type_plus_biomes():
    """The channel axis is air temperature + one temperature channel per fuel type + one mass
    channel per fuel type + terrain + 2 wind + one-hot biome + ignited mask, so
    C = 5 + 2*N_fuel + N_biome (here 2 fuel types, 3 biomes -> 12)."""
    obs = _obs(_const(pc.T_REF, 2, GRID, GRID), _const(0.0, 2, GRID, GRID), _const(0.0, GRID, GRID))
    assert obs.shape == (9 + N_BIOME, GRID, GRID)
    names = obs_channel_names(["grass", "tree"])
    assert len(names) == 9 + N_BIOME
    assert names == ("air_temperature",
                     "fuel_temperature_grass", "fuel_temperature_tree",
                     "fuel_grass", "fuel_tree", "terrain", "wind_x", "wind_y",
                     "biome_woodland", "biome_grassland", "biome_shrubland", "ignited")


def test_rest_state_maps_near_zero():
    """Ambient air/fuel temp (~T_REF), no fuel, sea-level terrain, empty biome, nothing ignited
    -> every channel ~ 0."""
    obs = _obs(_const(pc.T_REF, 2, GRID, GRID), _const(0.0, 2, GRID, GRID), _const(0.0, GRID, GRID))
    torch.testing.assert_close(obs, torch.zeros_like(obs), atol=1e-9, rtol=0.0)


def test_biome_channels_are_one_hot():
    """The N_biome channels ahead of the trailing ignited mask pass the one-hot biome through
    unchanged (identity window), in OBS_CHANNELS_BIOME order (woodland, grassland, shrubland)."""
    onehot = torch.zeros(N_BIOME, GRID, GRID, dtype=DTYPE)
    onehot[0, :4, :] = 1.0    # top half woodland
    onehot[1, 4:, :] = 1.0    # bottom half grassland
    obs = _obs(_const(pc.T_REF, 1, GRID, GRID), _const(0.0, 1, GRID, GRID),
               _const(0.0, GRID, GRID), biome_onehot=onehot)
    torch.testing.assert_close(obs[-(N_BIOME + 1):-1], onehot, atol=1e-9, rtol=0.0)


def test_ignited_channel_is_burning_mask():
    """The trailing channel is 1 exactly where some fuel type is at/above its own ignition
    threshold (the canonical burning test), reduced over types; 0 elsewhere."""
    temps = _const(pc.T_REF, 2, GRID, GRID)
    temps[0, 1, 1] = IGN_DEFAULT + 50.0   # type 0 above its threshold -> ignited
    temps[1, 2, 2] = IGN_DEFAULT - 50.0   # type 1 below its threshold -> not ignited
    obs = _obs(temps, _const(1.0, 2, GRID, GRID), _const(0.0, GRID, GRID))
    expected = torch.zeros((GRID, GRID), dtype=DTYPE)
    expected[1, 1] = 1.0
    torch.testing.assert_close(obs[-1], expected, atol=1e-9, rtol=0.0)


def test_air_temperature_channel_is_leading_and_rises_with_heat():
    """Channel 0 is the air temperature, normalized by its own (T_REF, 1000) window; a hot plume
    maps near 1 while ambient air maps to ~0."""
    air = _const(pc.T_REF, GRID, GRID)
    air[4, 4] = pc.T_REF + 1000.0     # a hot cell one full window above ambient
    obs = _obs(_const(pc.T_REF, 1, GRID, GRID), _const(0.0, 1, GRID, GRID),
               _const(0.0, GRID, GRID), air_temperatures=air)
    torch.testing.assert_close(obs[0, 0, 0], torch.zeros((), dtype=DTYPE), atol=1e-9, rtol=0.0)
    expected_hot = 1000.0 / pc.OBS_NORM["air_temperature"][1]
    assert abs(float(obs[0, 4, 4]) - expected_hot) < 1e-9
    assert abs(expected_hot - 1.0) < 1e-9


def test_terrain_normalized_by_elev_max():
    """Terrain at the documented ceiling maps to 1; the channel is z / ELEV_MAX_M.
    With 1 fuel type terrain sits at index 2*N_fuel + 1 = 3 (air, temp, mass, terrain, ...)."""
    obs = _obs(_const(pc.T_REF, 1, GRID, GRID), _const(0.0, 1, GRID, GRID), _const(pc.ELEV_MAX_M, GRID, GRID))
    torch.testing.assert_close(obs[3], torch.ones((GRID, GRID), dtype=DTYPE), atol=1e-9, rtol=0.0)


def test_wind_components_signed_and_normalized_by_wind_ref():
    """The wind channels pass the fire wind through, signed about 0 and scaled by WIND_REF_M_S.
    With 1 fuel type wind_x/wind_y are the last two channels (indices 4 and 5)."""
    wx = _const(pc.WIND_REF_M_S, GRID, GRID)        # +peak -> +1
    wy = _const(-pc.WIND_REF_M_S / 2.0, GRID, GRID)  # half the peak, reversed -> -0.5
    obs = _obs(_const(pc.T_REF, 1, GRID, GRID), _const(0.0, 1, GRID, GRID),
               _const(0.0, GRID, GRID), wx, wy)
    torch.testing.assert_close(obs[4], torch.ones((GRID, GRID), dtype=DTYPE), atol=1e-9, rtol=0.0)
    torch.testing.assert_close(obs[5], _const(-0.5, GRID, GRID), atol=1e-9, rtol=0.0)


def test_fuel_channels_are_per_type_then_normalized():
    """Each fuel type gets its own mass channel (no summing), each normalized by FUEL_REF_KG_M2.
    With 2 fuel types the mass block is channels 3 and 4 (behind air + the two temperature channels)."""
    fuel = torch.zeros(2, GRID, GRID, dtype=DTYPE)
    fuel[0] = pc.FUEL_REF_KG_M2            # type 0 at the reference -> 1
    fuel[1] = pc.FUEL_REF_KG_M2 / 2.0      # type 1 at half the reference -> 0.5
    obs = _obs(_const(pc.T_REF, 2, GRID, GRID), fuel, _const(0.0, GRID, GRID))
    torch.testing.assert_close(obs[3], torch.ones((GRID, GRID), dtype=DTYPE), atol=1e-9, rtol=0.0)
    torch.testing.assert_close(obs[4], _const(0.5, GRID, GRID), atol=1e-9, rtol=0.0)
    torch.testing.assert_close(obs[5], torch.zeros((GRID, GRID), dtype=DTYPE), atol=1e-9, rtol=0.0)  # terrain


def test_fuel_temperature_is_per_type_and_rises_with_fire():
    """Each fuel type gets its own temperature channel (no amax reduction); a flame maps near
    (or above) 1 in that type's channel and leaves the other types' channels at ambient.
    With air at index 0 the fuel-temperature block is channels 1 and 2."""
    temps = _const(pc.T_REF, 2, GRID, GRID)
    temps[0, 4, 4] = 2300.0           # one hot cell in type 0 only
    obs = _obs(temps, _const(1.0, 2, GRID, GRID), _const(0.0, GRID, GRID))
    expected_hot = (2300.0 - pc.T_REF) / pc.OBS_NORM["fuel_temperature"][1]
    assert abs(float(obs[1, 4, 4]) - expected_hot) < 1e-9   # type 0's own channel carries the flame
    assert abs(float(obs[2, 4, 4])) < 1e-9                  # type 1 stays at ambient (no cross-type max)
    assert float(obs[1, 4, 4]) > float(obs[1, 0, 0])        # flame sits above the ambient band
    assert expected_hot > 0.9                               # a 2300 K flame lands near 1


def test_batched_matches_single_world():
    """A leading batch axis normalizes identically to stacking single-world results."""
    temps = torch.rand(3, 2, GRID, GRID, dtype=DTYPE) * 2000.0 + pc.T_REF
    fuel = torch.rand(3, 2, GRID, GRID, dtype=DTYPE) * 10.0
    terrain = torch.rand(3, GRID, GRID, dtype=DTYPE) * pc.ELEV_MAX_M
    batched = _obs(temps, fuel, terrain)
    assert batched.shape == (3, 9 + N_BIOME, GRID, GRID)   # 5 fixed + 2*(2 fuel types) + biomes
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
    assert obs.shape == (len(env.obs_channels), 24, 24)
    assert env.observation_space.shape == obs.shape
    assert env.observation_space.contains(obs)
    # The env observation is build_observation cropped to the observed interior, so crop expected.
    expected = env._crop(Simulation.build_observation(
        env._air_temperatures, env._fuel_temperatures, env._fuel, env._terrain,
        env._x_wind_fire, env._y_wind_fire, env._biome_onehot, env._sim.ignition_thresholds
    )).detach().cpu().numpy().astype("float32")
    np.testing.assert_array_equal(obs, expected)

    step_obs, *_ = env.step(0)
    step_expected = env._crop(Simulation.build_observation(
        env._air_temperatures, env._fuel_temperatures, env._fuel, env._terrain,
        env._x_wind_fire, env._y_wind_fire, env._biome_onehot, env._sim.ignition_thresholds
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
    assert obs.shape == (len(env.obs_channels), obs_grid, obs_grid)   # observation is the inner region

    # A fire spawned at observed (0, 0) is centred at physics (pad, pad) -- i.e. the observed
    # top-left corner -- proving the screen->cell offset.
    ign = float(env._sim.ignition_thresholds.flatten()[0])
    env._spawn_fire_patch(0, 0)
    hot = env._fuel_temperatures[0] > ign
    assert bool(hot[pad, pad])                          # patch centre at the padded offset
    assert not bool(hot[0, 0])                          # not at the physics corner (inside the ring)
    assert bool(env._crop(hot)[0, 0])                   # and it shows at observed (0, 0)
