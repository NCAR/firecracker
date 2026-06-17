"""
Batched simulation core + rollout collector.

The same Simulation ops now run on a leading batch axis B (rank-agnostic), so one engine can
step B worlds in parallel for world-model training. These tests pin down that:

  * a B=1 batched step is bit-for-bit the single-world env step (step_fields is the shared
    source of truth),
  * worlds don't cross-contaminate (a uniform world is untouched by a violent batch-mate),
  * conservation holds per world in a batch,
  * build_observation produces the B x C x N x N tensor with the documented channels, and
  * BatchedRollout fills its pool and serves correctly-shaped minibatches.
"""

import numpy as np
import torch

from conftest import to_numpy
from simulation import Simulation, SimState, OBS_CHANNELS
from map_loader import save_map
from rollout import BatchedRollout, FireDataset
from scenarios import hot_blob, mass_gradient, off_equilibrium, uniform, build_map, make_config


def _batched_state(sim: Simulation, maps) -> SimState:
    """Stack a list of MapData into one batched SimState (mirrors BatchedRollout.reset)."""
    def stack(attr: str) -> torch.Tensor:
        arr = np.stack([getattr(m, attr) for m in maps])
        return torch.as_tensor(arr, dtype=sim.dtype, device=sim.device)

    mass, terrain, air = stack("mass"), stack("terrain"), stack("air_temperatures")
    # Per-world ambient wind (B, 1, 1); prognostic wind starts at the ambient.
    u_amb_x = torch.as_tensor([m.ambient_wind_x for m in maps], dtype=sim.dtype, device=sim.device).view(-1, 1, 1)
    u_amb_y = torch.as_tensor([m.ambient_wind_y for m in maps], dtype=sim.dtype, device=sim.device).view(-1, 1, 1)
    x_wind, y_wind = torch.zeros_like(mass) + u_amb_x, torch.zeros_like(mass) + u_amb_y
    return SimState(
        mass=mass, air_temperatures=air, ground_temperature=stack("temp_eq"),
        fuel_temperatures=stack("fuel_temperatures"), fuel=stack("fuel"),
        oxygen=stack("oxygen"), terrain=terrain,
        temp_eq=stack("temp_eq"), oxygen_eq=stack("oxygen_eq"), mass_eq=mass.clone(),
        x_wind_vel=x_wind, y_wind_vel=y_wind,
        u_amb_x=u_amb_x, u_amb_y=u_amb_y, radiant_flux=torch.zeros_like(mass),
    )


def test_batched_b1_matches_single_world(make_env):
    """A single world stepped as a B=1 batch must match the single-world env exactly.

    With B=1 the batch-global substep counts equal the lone world's, so the two code paths
    are numerically identical — this proves the rank-agnostic step_fields is a faithful
    drop-in for the env's per-tick physics.
    """
    config, m = hot_blob()
    env = make_env(config, m)
    s = _batched_state(env._sim, [m])

    for _ in range(10):
        env._sim.step_fields(s)
        env.step(0)

    for batched, single in (
        (s.mass, env._mass), (s.air_temperatures, env._air_temperatures),
        (s.oxygen, env._oxygen), (s.fuel_temperatures, env._fuel_temperatures),
    ):
        np.testing.assert_allclose(to_numpy(batched)[0], to_numpy(single), rtol=1e-5, atol=1e-6)


def test_uniform_world_unaffected_by_batchmate():
    """A uniform world (zero gradients) batched with a violent one must not change.

    Down-gradient transport/advection vanish where the field is flat, so with fire and
    relaxation off the uniform world is a fixed point regardless of its batch-mates — a
    direct check that worlds don't leak into each other through the shared ops.
    """
    _, u = uniform()
    _, h = hot_blob()
    sim = Simulation(make_config())
    s = _batched_state(sim, [u, h])

    before = to_numpy(s.air_temperatures)[0].copy()
    before_mass = to_numpy(s.mass)[0].copy()
    for _ in range(15):
        sim.step_fields(s)

    np.testing.assert_allclose(to_numpy(s.air_temperatures)[0], before, atol=1e-6)
    np.testing.assert_allclose(to_numpy(s.mass)[0], before_mass, atol=1e-6)


def test_batched_conserves_mass_energy_per_world():
    """Each world in a batch independently conserves mass, energy, and oxygen.

    Uses the closed core (no edge sponge / friction) so the totals are conserved; this
    checks the batched advection is per-world independent and conservative.
    """
    maps = [hot_blob()[1], mass_gradient()[1], off_equilibrium()[1]]
    sim = Simulation(make_config(momentum={"sponge_strength": 0.0, "drag_coeff": 0.0}))
    s = _batched_state(sim, maps)

    axes = (-2, -1)   # reduce each world's H, W, leaving the batch axis
    m0 = s.mass.sum(dim=axes)
    e0 = (s.mass * s.air_temperatures).sum(dim=axes)
    o0 = s.oxygen.sum(dim=axes)
    for _ in range(25):
        sim.step_fields(s)

    assert torch.isfinite(s.air_temperatures).all()
    np.testing.assert_allclose(to_numpy(s.mass.sum(dim=axes)), to_numpy(m0), rtol=1e-4)
    np.testing.assert_allclose(to_numpy((s.mass * s.air_temperatures).sum(dim=axes)), to_numpy(e0), rtol=1e-4)
    np.testing.assert_allclose(to_numpy(s.oxygen.sum(dim=axes)), to_numpy(o0), rtol=1e-4)


def test_build_observation_shape_and_channels():
    """build_observation stacks the documented channels into B x C x N x N."""
    B, N = 4, 8
    names = ("grass", "tree")
    fuel_temps = torch.rand(B, len(names), N, N)
    fuel = torch.rand(B, len(names), N, N)
    terrain = torch.rand(B, N, N)

    obs = Simulation.build_observation(fuel_temps, fuel, terrain)

    assert obs.shape == (B, len(OBS_CHANNELS), N, N)
    np.testing.assert_allclose(to_numpy(obs[:, 0]), to_numpy(fuel_temps.amax(dim=1)))  # hottest type
    np.testing.assert_allclose(to_numpy(obs[:, 1]), to_numpy(fuel.sum(dim=1)))         # total fuel
    np.testing.assert_allclose(to_numpy(obs[:, 2]), to_numpy(terrain))                 # terrain


def test_rollout_collects_and_samples(tmp_path):
    """BatchedRollout fills its pool and serves correctly-shaped minibatches."""
    grid = 16
    config = make_config(grid_size=grid)
    # Bake a couple of small maps the collector can sample from.
    for i in range(2):
        save_map(tmp_path / f"m_{i}.npz", build_map(grid_size=grid))

    roll = BatchedRollout(
        config, maps_dir=str(tmp_path), grid_size=grid,
        num_envs=4, steps=8, stride=2, minibatch_size=10, buffer_device="cpu", seed=0,
    )
    pool = roll.collect()

    # stride=2 over 8 steps stores 4 frames, each a batch of 4 worlds -> 16 samples.
    assert roll.capacity == (8 // 2) * 4
    assert pool.shape == (roll.capacity, len(OBS_CHANNELS), grid, grid)
    assert torch.isfinite(pool).all()

    batch = roll.sample_minibatch()
    assert batch.shape == (10, len(OBS_CHANNELS), grid, grid)
    assert batch.device.type == "cpu"


def test_rollout_stride_subsamples(tmp_path):
    """Raising stride proportionally shrinks the pool (every stride-th frame kept)."""
    grid = 16
    config = make_config(grid_size=grid)
    save_map(tmp_path / "m.npz", build_map(grid_size=grid))

    common = dict(maps_dir=str(tmp_path), grid_size=grid, num_envs=2, steps=12, buffer_device="cpu")
    dense = BatchedRollout(config, stride=1, **common)
    sparse = BatchedRollout(config, stride=4, **common)

    assert dense.collect().shape[0] == 12 * 2
    assert sparse.collect().shape[0] == (12 // 4) * 2


def test_build_dataset_and_load(tmp_path):
    """build_dataset writes sharded files + meta that FireDataset memory-maps back."""
    grid = 16
    config = make_config(grid_size=grid)
    maps_dir = tmp_path / "maps"
    maps_dir.mkdir()
    save_map(maps_dir / "m.npz", build_map(grid_size=grid))

    roll = BatchedRollout(
        config, maps_dir=str(maps_dir), grid_size=grid,
        num_envs=2, steps=4, stride=2, buffer_device="cpu", seed=0,
    )
    out = tmp_path / "data"
    rounds = 3
    meta = roll.build_dataset(out, rounds=rounds)

    per_shard = (4 // 2) * 2   # (steps // stride) * num_envs = 4
    assert meta["samples_per_shard"] == per_shard
    assert meta["total_samples"] == per_shard * rounds
    assert len(sorted(out.glob("shard_*.npy"))) == rounds
    assert (out / "meta.json").is_file()

    ds = FireDataset(out)
    assert len(ds) == per_shard * rounds
    sample = ds[0]
    assert sample.shape == (len(OBS_CHANNELS), grid, grid)

    # Last item indexes into the final shard and must match that shard's raw contents.
    last_shard = np.load(sorted(out.glob("shard_*.npy"))[-1])
    np.testing.assert_allclose(to_numpy(ds[len(ds) - 1]), last_shard[-1])


def test_build_dataset_refuses_overwrite(tmp_path):
    """Existing shards are protected unless overwrite=True."""
    import pytest
    grid = 16
    maps_dir = tmp_path / "maps"
    maps_dir.mkdir()
    save_map(maps_dir / "m.npz", build_map(grid_size=grid))
    roll = BatchedRollout(
        make_config(grid_size=grid), maps_dir=str(maps_dir), grid_size=grid,
        num_envs=2, steps=2, stride=1, buffer_device="cpu", seed=0,
    )
    out = tmp_path / "data"
    roll.build_dataset(out, rounds=1)
    with pytest.raises(FileExistsError):
        roll.build_dataset(out, rounds=1)
    roll.build_dataset(out, rounds=1, overwrite=True)   # ok with the flag
