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

import physics_constants as pc
from conftest import to_numpy
from simulation import Simulation, SimState, obs_channel_names
from map_loader import save_map
from config import boundary_pad
from rollout import (
    BatchedRollout,
    FireDataset,
    bake_normalization,
    compute_channel_minmax,
    ensure_channel_stats,
    stats_affine,
)
from scenarios import (
    hot_blob, mass_gradient, off_equilibrium, uniform, build_map, make_config, pad_map, ramp_terrain,
)


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
        mass=mass, air_temperatures=air.unsqueeze(-3), ground_temperature=stack("temp_eq"),
        fuel_temperatures=stack("fuel_temperatures"), fuel=stack("fuel"),
        oxygen=stack("oxygen"), terrain=terrain,
        temp_eq=stack("temp_eq").unsqueeze(-3), oxygen_eq=stack("oxygen_eq"), mass_eq=mass.clone(),
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
        (s.mass, env._mass), (s.air_temperatures.select(-3, 0), env._air_temperatures),
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
    e0 = (s.mass * s.air_temperatures.select(-3, 0)).sum(dim=axes)
    o0 = s.oxygen.sum(dim=axes)
    for _ in range(25):
        sim.step_fields(s)

    assert torch.isfinite(s.air_temperatures).all()
    np.testing.assert_allclose(to_numpy(s.mass.sum(dim=axes)), to_numpy(m0), rtol=1e-4)
    np.testing.assert_allclose(to_numpy((s.mass * s.air_temperatures.select(-3, 0)).sum(dim=axes)), to_numpy(e0), rtol=1e-4)
    np.testing.assert_allclose(to_numpy(s.oxygen.sum(dim=axes)), to_numpy(o0), rtol=1e-4)


def test_build_observation_shape_and_channels():
    """build_observation stacks the documented channels into B x C x N x N, each normalized
    by the per-channel affine window in physics_constants.OBS_NORM (see tests/test_observation.py)."""
    B, N = 4, 8
    names = ("grass", "tree")
    air_temps = torch.rand(B, N, N) * 1000.0 + pc.T_REF
    fuel_temps = torch.rand(B, len(names), N, N) * 1000.0   # 0..1000 K, straddles the thresholds
    fuel = torch.rand(B, len(names), N, N)
    terrain = torch.rand(B, N, N)
    wind_x = torch.rand(B, N, N) * 2.0 - 1.0   # signed
    wind_y = torch.rand(B, N, N) * 2.0 - 1.0
    ign = torch.tensor([500.0, 600.0]).view(len(names), 1, 1)   # per-type ignition thresholds [K]

    obs = Simulation.build_observation(
        air_temps.unsqueeze(-3), fuel_temps, fuel, terrain, wind_x, wind_y, ign
    )

    def norm(channel, raw):
        offset, scale = pc.OBS_NORM[channel]
        return (to_numpy(raw) - offset) / scale

    # C = 6 fixed channels (air temperature + 2 slope + 2 wind + ignited) + two per fuel type
    # (temperature + mass); 2 fuel types -> 10. Biome is not an observation channel.
    assert obs.shape == (B, len(obs_channel_names(names)), N, N)
    slope_x, slope_y = Simulation._periodic_grad(terrain, pc.DEFAULT_CELL_SIZE_M)   # observed slope
    np.testing.assert_allclose(to_numpy(obs[:, 0]), norm("air_temperature", air_temps))               # air temp
    np.testing.assert_allclose(to_numpy(obs[:, 1]), norm("fuel_temperature", fuel_temps[:, 0]))       # temp type 0
    np.testing.assert_allclose(to_numpy(obs[:, 2]), norm("fuel_temperature", fuel_temps[:, 1]))       # temp type 1
    np.testing.assert_allclose(to_numpy(obs[:, 3]), norm("fuel", fuel[:, 0]))                          # mass type 0
    np.testing.assert_allclose(to_numpy(obs[:, 4]), norm("fuel", fuel[:, 1]))                          # mass type 1
    np.testing.assert_allclose(to_numpy(obs[:, 5]), norm("slope_x", slope_x), rtol=1e-6, atol=1e-6)   # terrain slope u
    np.testing.assert_allclose(to_numpy(obs[:, 6]), norm("slope_y", slope_y), rtol=1e-6, atol=1e-6)   # terrain slope v
    np.testing.assert_allclose(to_numpy(obs[:, 7]), norm("wind_x", wind_x))                            # fire wind u
    np.testing.assert_allclose(to_numpy(obs[:, 8]), norm("wind_y", wind_y))                            # fire wind v
    expected_ignited = (fuel_temps >= ign).any(dim=1).to(fuel_temps.dtype)                             # burning mask
    np.testing.assert_allclose(to_numpy(obs[:, -1]), to_numpy(expected_ignited))


def test_rollout_collects_and_samples(tmp_path):
    """BatchedRollout fills its pool and serves correctly-shaped minibatches."""
    grid = 16
    config = make_config(grid_size=grid)
    # Bake a couple of small maps the collector can sample from (ringed out to the physics size).
    pad = boundary_pad(config)
    for i in range(2):
        save_map(tmp_path / f"m_{i}.npz", pad_map(build_map(grid_size=grid), pad))

    roll = BatchedRollout(
        config, maps_dir=str(tmp_path), grid_size=grid,
        num_envs=4, steps=8, stride=2, minibatch_size=10, buffer_device="cpu", seed=0,
    )
    pool = roll.collect()

    # stride=2 over 8 steps stores 4 frames, each a batch of 4 worlds -> 16 samples.
    assert roll.capacity == (8 // 2) * 4
    assert pool.shape == (roll.capacity, roll.num_channels, grid, grid)
    assert torch.isfinite(pool).all()

    batch = roll.sample_minibatch()
    assert batch.shape == (10, roll.num_channels, grid, grid)
    assert batch.device.type == "cpu"


def test_rollout_stride_subsamples(tmp_path):
    """Raising stride proportionally shrinks the pool (every stride-th frame kept)."""
    grid = 16
    config = make_config(grid_size=grid)
    save_map(tmp_path / "m.npz", pad_map(build_map(grid_size=grid), boundary_pad(config)))

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
    save_map(maps_dir / "m.npz", pad_map(build_map(grid_size=grid), boundary_pad(make_config(grid_size=grid))))

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
    assert sample.shape == (roll.num_channels, grid, grid)

    # Last item indexes into the final shard and must match that shard's raw contents.
    last_shard = np.load(sorted(out.glob("shard_*.npy"))[-1])
    np.testing.assert_allclose(to_numpy(ds[len(ds) - 1]), last_shard[-1])


def test_firedataset_unions_worker_subdirs(tmp_path):
    """Pointing FireDataset at a parent dir unions per-worker subdir datasets into one."""
    grid = 16
    config = make_config(grid_size=grid)
    maps_dir = tmp_path / "maps"
    maps_dir.mkdir()
    save_map(maps_dir / "m.npz", pad_map(build_map(grid_size=grid), boundary_pad(make_config(grid_size=grid))))

    parent = tmp_path / "data"
    rounds = 2
    for w in range(3):                                  # three "workers", own subdir + seed
        roll = BatchedRollout(
            config, maps_dir=str(maps_dir), grid_size=grid,
            num_envs=2, steps=4, stride=2, buffer_device="cpu", seed=w,
        )
        roll.build_dataset(parent / f"w{w}", rounds=rounds)

    per_shard = (4 // 2) * 2
    ds = FireDataset(parent)                            # parent has no meta.json -> nested
    assert len(ds) == per_shard * rounds * 3
    assert ds[0].shape == (roll.num_channels, grid, grid)


def test_firedataset_rejects_incompatible_sources(tmp_path):
    """A subdir whose meta disagrees on a structural field fails fast at load."""
    import pytest
    parent = tmp_path / "data"
    for grid in (16, 24):                               # different grid_size between workers
        maps_dir = tmp_path / f"maps{grid}"
        maps_dir.mkdir()
        save_map(maps_dir / "m.npz", pad_map(build_map(grid_size=grid), boundary_pad(make_config(grid_size=grid))))
        roll = BatchedRollout(
            make_config(grid_size=grid), maps_dir=str(maps_dir), grid_size=grid,
            num_envs=2, steps=2, stride=1, buffer_device="cpu", seed=0,
        )
        roll.build_dataset(parent / f"g{grid}", rounds=1)
    with pytest.raises(ValueError):
        FireDataset(parent)


def _build_small_dataset(tmp_path, *, nested=False):
    """Build a tiny multi-shard dataset (float32 for tight stats) and return its root dir."""
    grid = 16
    config = make_config(grid_size=grid)
    maps_dir = tmp_path / "maps"
    maps_dir.mkdir()
    # Ramped terrain (across columns) so the slope channels aren't spatially constant, plus a
    # temperature ramp across rows that crosses the woodland threshold (298 K) so the map spans all
    # three biomes (cool -> woodland; warm+low -> grassland; warm+high -> shrubland), varying the
    # per-type fuel placement. With the two ramps on independent axes every channel has real
    # variance, so standardisation drives each to a genuine unit std (not the constant-channel
    # no-op guard). temp_eq (which classify_biomes reads) defaults to the air field.
    terrain = (ramp_terrain(grid) * 300.0).astype(np.float32)
    air = (290.0 + ramp_terrain(grid).T * 30.0).astype(np.float32)   # 290..320 K across rows
    save_map(maps_dir / "m.npz", pad_map(build_map(grid_size=grid, terrain=terrain, air=air),
                                          boundary_pad(config)))

    def build(out, seed):
        BatchedRollout(
            config, maps_dir=str(maps_dir), grid_size=grid, num_envs=4, steps=8, stride=2,
            buffer_device="cpu", buffer_dtype="float32", spawn_fire=True, seed=seed,
        ).build_dataset(out, rounds=3)

    root = tmp_path / "data"
    if nested:
        for w in range(3):
            build(root / f"w{w}", seed=w)
    else:
        build(root, seed=0)
    return root


def test_bake_normalization_min_max_to_unit_range(tmp_path):
    """bake_normalization rewrites the shards so FireDataset (raw read) yields the scaled range per
    channel: [0, 1] for min-max channels, [-1, 1] about 0 for the signed (wind) channels."""
    from rollout import SIGNED_CHANNELS
    root = _build_small_dataset(tmp_path, nested=True)
    meta = FireDataset(root).meta
    n_ch = int(meta["num_channels"])
    names = list(meta["channels"])
    signed = torch.tensor([nm in SIGNED_CHANNELS for nm in names])
    assert signed.any()                                      # the small dataset carries wind_x/wind_y
    assert not FireDataset(root).is_normalized               # raw dataset, not yet flagged

    # A channel whose raw values never vary (e.g. a fully-saturated ignited mask on this tiny grid)
    # has a zero span -- bake floors its scale to 1, so it maps to a constant ~0 rather than spanning
    # its range. Note which channels vary now, before baking overwrites the shards.
    raw = FireDataset(root).get_batch(np.arange(len(FireDataset(root)))).to(torch.float64)
    per_ch_raw = raw.permute(1, 0, 2, 3).reshape(n_ch, -1)
    varying = (per_ch_raw.amax(dim=1) - per_ch_raw.amin(dim=1)) >= 1e-8

    summary = bake_normalization(root, max_samples=None, workers=3, verbose=False)  # parallel pool
    assert summary["baked"] == 3 and (root / "stats.json").is_file()

    ds = FireDataset(root)                                    # served values are already normalized
    assert ds.is_normalized
    batch = ds.get_batch(np.arange(len(ds)))                 # (N, C, H, W), no read-time transform
    per_ch = batch.to(torch.float64).permute(1, 0, 2, 3).reshape(n_ch, -1)
    lo, hi = per_ch.amin(dim=1), per_ch.amax(dim=1)

    # Min-max channels: min ~0, and a varying one reaches ~1 (a constant one stays at 0).
    mm = ~signed
    torch.testing.assert_close(lo[mm], torch.zeros(mm.sum(), dtype=torch.float64), atol=1e-4, rtol=0)
    torch.testing.assert_close(hi[mm], (varying & mm)[mm].to(torch.float64), atol=1e-4, rtol=0)
    # Signed (wind) channels: symmetric about 0 -> everything within [-1, 1] and the dominant extreme
    # hits +/-1 (unlike a min-max channel, whose min would be pinned to 0).
    sgn = signed & varying
    assert torch.all(lo[sgn] >= -1 - 1e-4) and torch.all(hi[sgn] <= 1 + 1e-4)
    torch.testing.assert_close(torch.maximum(hi[sgn].abs(), lo[sgn].abs()),
                               torch.ones(sgn.sum(), dtype=torch.float64), atol=1e-4, rtol=0)
    # __getitem__ and get_batch read the same baked values.
    torch.testing.assert_close(ds[0], batch[0])


def test_bake_normalization_idempotent(tmp_path):
    """Re-baking an already-normalized dataset is a no-op (doesn't double-scale)."""
    root = _build_small_dataset(tmp_path)
    bake_normalization(root, max_samples=None, verbose=False)
    before = FireDataset(root).get_batch(np.arange(8)).clone()
    summary = bake_normalization(root, max_samples=None, verbose=False)   # second pass
    assert summary["baked"] == 0 and summary["skipped"] == 1
    torch.testing.assert_close(FireDataset(root).get_batch(np.arange(8)), before)


def test_channel_minmax_match_manual_over_union(tmp_path):
    """Min/max over a nested (multi-worker) dataset match a direct min/max of the raw union."""
    root = _build_small_dataset(tmp_path, nested=True)
    ds = FireDataset(root)
    cmin, cmax = compute_channel_minmax(ds, max_samples=None)

    allx = ds.get_batch(np.arange(len(ds))).to(torch.float64)      # raw, un-normalized
    ref_min = allx.amin(dim=(0, 2, 3))
    ref_max = allx.amax(dim=(0, 2, 3))
    np.testing.assert_allclose(cmin, ref_min.numpy(), atol=1e-6)
    np.testing.assert_allclose(cmax, ref_max.numpy(), atol=1e-6)


def test_stats_affine_signed_channels_map_to_symmetric_range():
    """Signed channels (wind) scale to [-1, 1] about 0; others keep min-max to [0, 1]."""
    stats = {
        "channels": ["air_temperature", "wind_x", "wind_y"],
        "signed_channels": ["wind_x", "wind_y"],
        "channel_min": [10.0, -1.4, -2.0],
        "channel_max": [30.0, 1.0, 2.0],
    }
    offset, scale = stats_affine(stats)
    # Unsigned channel: offset = min, scale = max - min (forward maps [min, max] -> [0, 1]).
    np.testing.assert_allclose([offset[0], scale[0]], [10.0, 20.0])
    # Signed channels: offset 0, scale = max(|min|, |max|), so 0 stays at 0 and the sign is kept.
    np.testing.assert_allclose(offset[1:], [0.0, 0.0])
    np.testing.assert_allclose(scale[1:], [1.4, 2.0])            # max(|-1.4|,|1.0|), max(|-2|,|2|)

    # Forward transform: calm wind -> 0, and each channel's extreme lands within [-1, 1].
    fwd = lambda x, i: (x - offset[i]) / scale[i]
    assert fwd(0.0, 1) == 0.0
    np.testing.assert_allclose(fwd(-1.4, 1), -1.0)               # the dominant extreme hits -1
    assert abs(fwd(1.0, 1)) <= 1.0
    # Inverse round-trips (de-normalization for inference/eval).
    np.testing.assert_allclose(fwd(0.83, 1) * scale[1] + offset[1], 0.83)


def test_stats_affine_signed_resolves_names_from_checkpoint_key():
    """A checkpoint reuses `channels` for widths, so names live under `channel_names`; use them."""
    stats = {
        "channels": [32, 64, 128],                               # encoder WIDTHS, not names
        "channel_names": ["air_temperature", "wind_x", "wind_y"],
        "signed_channels": ["wind_x", "wind_y"],
        "channel_min": [10.0, -1.4, -2.0],
        "channel_max": [30.0, 1.0, 2.0],
    }
    offset, scale = stats_affine(stats)
    np.testing.assert_allclose(offset, [10.0, 0.0, 0.0])
    np.testing.assert_allclose(scale, [20.0, 1.4, 2.0])


def test_channel_minmax_sampling_within_full(tmp_path):
    """A capped random-sample range sits inside the full-set range and records its provenance."""
    root = _build_small_dataset(tmp_path, nested=True)   # 3 workers x 3 shards x 8 = 72 samples
    ds = FireDataset(root)
    n = len(ds)

    full_min, full_max = compute_channel_minmax(ds, max_samples=None)
    samp_min, samp_max = compute_channel_minmax(ds, max_samples=n // 2, seed=0)
    # A subset can only shrink the observed range: its min is >= the full min, its max <= the full
    # max. This is exactly why min-max defaults to a full scan (a subset under-covers the extremes).
    assert np.all(samp_min >= full_min - 1e-9)
    assert np.all(samp_max <= full_max + 1e-9)

    # Budget is met by reading whole shards, so the recorded count rounds up to a shard multiple.
    stats = ensure_channel_stats(root, max_samples=n // 2, verbose=False)
    assert stats["total_samples"] == n
    assert stats["n_samples"] % ds.per_shard == 0
    assert n // 2 <= stats["n_samples"] < n


def test_unbaked_dataset_serves_raw(tmp_path):
    """An un-baked dataset is not flagged normalized and serves the raw shard values verbatim."""
    root = _build_small_dataset(tmp_path)
    ds = FireDataset(root)
    assert not ds.is_normalized
    raw = np.load(sorted(root.glob("shard_*.npy"))[0])
    np.testing.assert_allclose(to_numpy(ds[0]), raw[0])


def test_build_dataset_refuses_overwrite(tmp_path):
    """Existing shards are protected unless overwrite=True."""
    import pytest
    grid = 16
    maps_dir = tmp_path / "maps"
    maps_dir.mkdir()
    save_map(maps_dir / "m.npz", pad_map(build_map(grid_size=grid), boundary_pad(make_config(grid_size=grid))))
    roll = BatchedRollout(
        make_config(grid_size=grid), maps_dir=str(maps_dir), grid_size=grid,
        num_envs=2, steps=2, stride=1, buffer_device="cpu", seed=0,
    )
    out = tmp_path / "data"
    roll.build_dataset(out, rounds=1)
    with pytest.raises(FileExistsError):
        roll.build_dataset(out, rounds=1)
    roll.build_dataset(out, rounds=1, overwrite=True)   # ok with the flag
