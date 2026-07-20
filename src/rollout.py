"""
rollout.py

Batched rollout collector for world-model training — decoupled from Gymnasium.

One `Simulation` steps `B` worlds in parallel on the sim device (the physics ops are
rank-agnostic, so the single-world engine batches with no special-casing). Every `stride`-th
tick's observation, a `B x C x N x N` tensor (see `Simulation.build_observation` /
`obs_channel_names`), is written into a fixed pool of independent samples; the trainer then draws
random, near-I.I.D. minibatches from that pool. Consecutive frames are highly redundant, so
subsampling (default stride=16) keeps the pool decorrelated and small.

Memory note — the pool is `capacity x C x N x N` floats, where
`capacity = (steps // stride) * num_envs`:

    steps=1024, stride=16, num_envs=64, C=3, N=256, float32
        -> (1024//16)*64*3*256*256*4  ~= 3.2 GB

That does not fit on a GPU and is heavy for host RAM, so the pool defaults to living on the
CPU (`buffer_device="cpu"`) and its dtype is configurable (`buffer_dtype="float16"` halves
it). Stepping still happens on the sim device; only the per-step observation is copied to the
pool. Size `steps`/`num_envs` to your hardware via the [rollout] config section.

Quick use (online sampling):

    from config import load_config
    from rollout import BatchedRollout

    roll = BatchedRollout(load_config())
    roll.collect()                       # fill the pool by stepping B worlds `steps` times
    batch = roll.sample_minibatch()      # (minibatch_size, C, N, N) for the world model

Static dataset for pretraining (sharded to disk, then memory-mapped):

    roll.build_dataset("data/fire", rounds=8)     # 8 shards, fresh worlds each round

    from rollout import FireDataset
    from torch.utils.data import DataLoader
    ds = FireDataset("data/fire")                 # lazily memory-maps the shards
    loader = DataLoader(ds, batch_size=256, shuffle=True, num_workers=4)

Or from the command line:

    python src/rollout.py --out data/fire --rounds 8 --spawn-fire

Multi-GPU generation -- run one process per GPU, each writing into its own subdirectory of a
shared dataset dir (distinct --seed so the workers draw different worlds, not copies):

    CUDA_VISIBLE_DEVICES=0 python src/rollout.py --out data/fire/w0 --num-envs 16 --seed 0 --spawn-fire &
    CUDA_VISIBLE_DEVICES=1 python src/rollout.py --out data/fire/w1 --num-envs 16 --seed 1 --spawn-fire &
    CUDA_VISIBLE_DEVICES=2 python src/rollout.py --out data/fire/w2 --num-envs 16 --seed 2 --spawn-fire &
    CUDA_VISIBLE_DEVICES=3 python src/rollout.py --out data/fire/w3 --num-envs 16 --seed 3 --spawn-fire &
    wait

    ds = FireDataset("data/fire")                 # unions the w0..w3 subdirs into one dataset
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from config import load_config, boundary_pad
from simulation import Simulation, SimState, obs_channel_names, lift_air_levels
from map_loader import load_map, resolve_map, list_maps, validate_against_config

DEFAULT_GRID_SIZE = 256

# Filenames used by build_dataset / FireDataset for an on-disk sharded dataset.
_META_NAME = "meta.json"
_SHARD_GLOB = "shard_*.npy"
# Per-channel normalization stats (min/max) sit at the dataset root, next to meta.json for a
# flat dataset or above the per-worker subdirs for a nested one — see compute_channel_minmax.
_STATS_NAME = "stats.json"

# Channels that are signed with a physically meaningful zero: the near-surface wind components (calm
# air is 0 m/s, the sign is the direction) and the terrain-slope components (flat ground is 0, the
# sign is the uphill direction). These are scaled symmetrically to [-1, 1] — offset 0, scale
# max(|min|, |max|) — instead of the default min-max to [0, 1], so the zero lands exactly at 0 and
# the sign is preserved (see stats_affine). Every other channel stays on min-max. Extend this set to
# add more signed channels; write_channel_minmax records the per-dataset subset into stats.json's
# "signed_channels" so the exact transform travels with the data and checkpoints.
SIGNED_CHANNELS: frozenset[str] = frozenset({"wind_x", "wind_y", "slope_x", "slope_y"})


def _resolve_dtype(name: str) -> torch.dtype:
    dt = getattr(torch, name, None)
    if not isinstance(dt, torch.dtype):
        raise ValueError(f"Unknown buffer_dtype '{name}' (try 'float32' or 'float16').")
    return dt


class BatchedRollout:
    """Hold and step `B` worlds in parallel, pooling their observations for training."""

    def __init__(
        self,
        config: dict | None = None,
        *,
        num_envs: int | None = None,
        maps_dir: str | None = None,
        grid_size: int | None = None,
        steps: int | None = None,
        stride: int | None = None,
        minibatch_size: int | None = None,
        buffer_device: str | None = None,
        buffer_dtype: str | None = None,
        spawn_fire: bool | None = None,
        seed: int | None = None,
        map_repeats: int | None = None,
        num_shards: int = 1,
        shard_index: int = 0,
        schedule_seed: int = 0,
    ):
        cfg      = config or {}
        roll_cfg = cfg.get("rollout", {})
        env_cfg  = cfg.get("environment", {})
        fire_cfg = cfg.get("fire", {})
        maps_cfg = cfg.get("maps", {})

        self._sim = Simulation(config)

        # Explicit constructor args win over config, which wins over code defaults.
        self.num_envs       = int(num_envs       if num_envs       is not None else roll_cfg.get("num_envs",       64))
        self.steps          = int(steps          if steps          is not None else roll_cfg.get("steps",          1024))
        self.stride         = int(stride         if stride         is not None else roll_cfg.get("stride",         16))
        self.minibatch_size = int(minibatch_size if minibatch_size is not None else roll_cfg.get("minibatch_size", 256))
        if self.stride < 1:
            raise ValueError(f"stride must be >= 1, got {self.stride}")
        # grid_size is the OBSERVED grid (pooled/observed); physics runs on a grid padded by _pad on
        # every side (the open-boundary sponge ring), and observe() crops the interior back out.
        self.grid_size      = int(grid_size      if grid_size      is not None else env_cfg.get("grid_size",       DEFAULT_GRID_SIZE))
        self._pad           = boundary_pad(config)
        self._sim_size      = self.grid_size + 2 * self._pad
        self._maps_dir      = str(maps_dir       if maps_dir       is not None else maps_cfg.get("dir",            "maps"))
        self._spawn_fire    = bool(spawn_fire    if spawn_fire     is not None else fire_cfg.get("spawn_fire",     False))
        self._fire_radius   = int(fire_cfg.get("spawn_radius", 5))

        self._buffer_device = torch.device(
            buffer_device if buffer_device is not None else roll_cfg.get("buffer_device", "cpu")
        )
        self._buffer_dtype = _resolve_dtype(
            buffer_dtype if buffer_dtype is not None else roll_cfg.get("buffer_dtype", "float32")
        )

        self.obs_channels = obs_channel_names(self._sim.fuel_type_names)
        self.num_channels = len(self.obs_channels)
        # Host RNG for map selection / ignition; torch generator for minibatch sampling.
        self._np_rng = np.random.default_rng(seed)
        self._gen = torch.Generator(device=self._buffer_device)
        if seed is not None:
            self._gen.manual_seed(int(seed))

        # Optional exhaustive-coverage schedule. When map_repeats is set, maps are drawn
        # WITHOUT replacement from a deterministic list in which every map appears exactly
        # `map_repeats` times, instead of the default random-with-replacement draw. Independent
        # workers share `schedule_seed` (so they agree on one global permutation) and take a
        # disjoint contiguous shard of it via (shard_index, num_shards); together the shards
        # cover every map exactly `map_repeats` times. Fire ignition still uses `seed`, so each
        # worker lights its worlds differently. `_schedule` is None in replacement mode.
        # Schedule provenance for the dataset meta. In replacement mode (map_repeats None) all four
        # are None; in exhaustive mode they record the coverage (every map used map_repeats times)
        # and this worker's disjoint slice of the shared global schedule (shard_index of num_shards,
        # permuted by schedule_seed), enough to reproduce which maps this worker drew.
        self._schedule: list[Path] | None = None
        self._schedule_pos = 0
        if map_repeats is not None:
            self._map_repeats = int(map_repeats)
            self._num_shards = int(num_shards)
            self._shard_index = int(shard_index)
            self._schedule_seed = int(schedule_seed)
            self._schedule = self._build_map_schedule(
                self._map_repeats, self._num_shards, self._shard_index, self._schedule_seed
            )
        else:
            self._map_repeats = self._num_shards = self._shard_index = self._schedule_seed = None

        self._state: SimState | None = None
        self._pool: torch.Tensor | None = None   # (capacity, C, N, N) on buffer_device
        self._filled: int = 0                     # number of valid samples in the pool

        self.reset()

    # -----------------------------------------------------------------------
    # World setup
    # -----------------------------------------------------------------------

    def _build_map_schedule(
        self, repeats: int, num_shards: int, shard_index: int, schedule_seed: int
    ) -> list[Path]:
        """This worker's ordered slice of the global exhaustive-coverage schedule.

        The global schedule is every map in `maps_dir` repeated `repeats` times, then shuffled
        with `schedule_seed`. All workers build the identical permutation and each takes the
        contiguous chunk for its `shard_index`, so across `num_shards` workers every map is used
        exactly `repeats` times. The chunk length must be divisible so shards are equal-sized.
        """
        if repeats < 1:
            raise ValueError(f"map_repeats must be >= 1, got {repeats}")
        if not 0 <= shard_index < num_shards:
            raise ValueError(f"shard_index {shard_index} out of range for num_shards {num_shards}")
        maps = list_maps(self._maps_dir)
        if not maps:
            raise FileNotFoundError(
                f"No maps found in '{self._maps_dir}' for exhaustive schedule; generate some first."
            )
        full = np.repeat(np.arange(len(maps)), repeats)      # each map index, `repeats` times
        perm = np.random.default_rng(schedule_seed).permutation(full)
        if len(perm) % num_shards != 0:
            raise ValueError(
                f"schedule length {len(perm)} (= {len(maps)} maps x {repeats}) is not divisible "
                f"by num_shards {num_shards}; choose factors that divide evenly."
            )
        shard_len = len(perm) // num_shards
        sel = perm[shard_index * shard_len:(shard_index + 1) * shard_len]
        return [maps[j] for j in sel]

    def _next_scheduled_map(self) -> Path:
        """Pop the next map from this worker's schedule (used only in exhaustive mode)."""
        if self._schedule_pos >= len(self._schedule):
            raise RuntimeError(
                f"Map schedule exhausted after {self._schedule_pos} draws; this worker's shard "
                f"holds {len(self._schedule)} maps but the run asked for more "
                f"(rounds x num_envs). Reduce rounds/num_envs or raise map_repeats."
            )
        path = self._schedule[self._schedule_pos]
        self._schedule_pos += 1
        return path

    def reset(self) -> None:
        """(Re)load `B` worlds from the maps dir and build the batched simulation state."""
        fields = {k: [] for k in ("terrain", "air_temperatures", "temp_eq", "oxygen",
                                  "oxygen_eq", "mass", "mass_eq", "x_wind_vel", "y_wind_vel",
                                  "fuel", "fuel_temperatures")}
        amb_x, amb_y = [], []
        for _ in range(self.num_envs):
            if self._schedule is not None:
                # Exhaustive mode: consume the next map from this worker's schedule.
                path = self._next_scheduled_map()
            else:
                # Sample with replacement so num_envs can exceed the number of map files.
                path = resolve_map(self._maps_dir, None, self._np_rng)
            m = load_map(path)
            validate_against_config(m, self._sim_size, self._sim.fuel_type_names)
            for k in fields:
                fields[k].append(getattr(m, k))
            amb_x.append(m.ambient_wind_x)
            amb_y.append(m.ambient_wind_y)

        def stack(name: str) -> torch.Tensor:
            arr = np.stack(fields[name])   # leading axis becomes the batch axis B
            return torch.as_tensor(arr, dtype=self._sim.dtype, device=self._sim.device)

        terrain = stack("terrain")
        mass    = stack("mass")
        air     = stack("air_temperatures")
        # Per-world synoptic ambient wind, shaped (B, 1, 1) to broadcast against (B, H, W).
        def amb(values: list) -> torch.Tensor:
            return torch.as_tensor(values, dtype=self._sim.dtype, device=self._sim.device).view(-1, 1, 1)
        u_amb_x, u_amb_y = amb(amb_x), amb(amb_y)
        # Prognostic wind starts at each map's developed (spun-up) orographic field, so every
        # world begins in its terrain-shaped state rather than relaxing into it.
        x_wind = stack("x_wind_vel")
        y_wind = stack("y_wind_vel")
        # Static terrain-channeling gain per world (Phase 6, Option 2): the fire/oxygen read the
        # prognostic wind sped up through gaps by this gain (applied each step in step_fields).
        channel_gain = self._sim.compute_channel_gain(terrain, u_amb_x, u_amb_y)

        self._state = SimState(
            mass=mass,
            air_temperatures=lift_air_levels(air, self._sim.num_air_levels),   # (L, H, W) air stack
            ground_temperature=stack("temp_eq"),   # surface skin starts at the rest profile
            fuel_temperatures=stack("fuel_temperatures"),
            fuel=stack("fuel"),
            oxygen=stack("oxygen"),
            terrain=terrain,
            temp_eq=lift_air_levels(stack("temp_eq"), self._sim.num_air_levels),
            oxygen_eq=stack("oxygen_eq"),
            mass_eq=stack("mass_eq"),
            x_wind_vel=x_wind,
            y_wind_vel=y_wind,
            u_amb_x=u_amb_x,
            u_amb_y=u_amb_y,
            radiant_flux=torch.zeros_like(mass),
            channel_gain=channel_gain,
        )
        if self._spawn_fire:
            self._ignite()
        self._filled = 0

    def _ignite(self) -> None:
        """Light a random circular patch in every world (batched analogue of the env click).

        The patch grid spans the padded physics size, but the centre is drawn from the observed
        interior (offset by _pad), so ignitions land on-screen and never in the hidden sponge ring."""
        s, n = self._state, self._sim_size
        dev = self._sim.device
        rows = torch.arange(n, device=dev).view(1, -1, 1)
        cols = torch.arange(n, device=dev).view(1, 1, -1)
        centres = self._np_rng.integers(0, self.grid_size, size=(2, self.num_envs)) + self._pad
        r0 = torch.as_tensor(centres[0], device=dev).view(-1, 1, 1)
        c0 = torch.as_tensor(centres[1], device=dev).view(-1, 1, 1)
        patch = (rows - r0) ** 2 + (cols - c0) ** 2 <= self._fire_radius ** 2   # (B, N, N)

        ign_max = float(self._sim.ignition_thresholds.max())
        # Ground ignition: heat only the surface air level (level 0). The canopy level catches only
        # once the ladder carries fire up, so a spawn must not warm the elevated levels directly.
        surf_air = s.air_temperatures[:, 0]   # (B, H, W) surface level view
        s.air_temperatures[:, 0] = torch.where(patch, torch.full_like(surf_air, ign_max * 2.0), surf_air)
        for t in range(self._sim.num_fuel_types):
            ign_t = float(self._sim.ignition_thresholds[t])
            s.fuel_temperatures[:, t] = torch.where(patch, torch.full_like(s.fuel_temperatures[:, t], ign_t * 2.0), s.fuel_temperatures[:, t])
            s.fuel[:, t] = torch.where(patch, torch.ones_like(s.fuel[:, t]), s.fuel[:, t])

    # -----------------------------------------------------------------------
    # Collection / sampling
    # -----------------------------------------------------------------------

    @property
    def capacity(self) -> int:
        """Total samples the pool holds once full: one B-batch per *stored* step.

        Consecutive ticks are highly redundant, so only every `stride`-th frame is kept —
        (steps // stride) stored frames, each contributing num_envs samples.
        """
        return (self.steps // self.stride) * self.num_envs

    def observe(self) -> torch.Tensor:
        """Current batched observation, B x C x N x N on the sim device (N = observed grid size).

        The physics runs on the padded grid; the observation crops the interior, dropping the
        sponge ring so the pooled samples match the env's observation."""
        s = self._state
        # Same near-surface wind the fire reads for spread (prognostic x channeling gain),
        # falling back to the prognostic wind when no channeling layer is present.
        wind_x = s.x_wind_fire if s.x_wind_fire is not None else s.x_wind_vel
        wind_y = s.y_wind_fire if s.y_wind_fire is not None else s.y_wind_vel
        obs = Simulation.build_observation(
            s.air_temperatures, s.fuel_temperatures, s.fuel, s.terrain, wind_x, wind_y,
            self._sim.ignition_thresholds, self._sim.cell_size_m,
        )
        if self._pad == 0:
            return obs
        p, g = self._pad, self.grid_size
        return obs[..., p:p + g, p:p + g]

    def collect(self, steps: int | None = None, stride: int | None = None) -> torch.Tensor:
        """Step the `B` worlds `steps` times, pooling every `stride`-th tick's observation.

        Sampling starts at the ignition frame (t=0, before the first step) and stores every
        `stride`-th tick thereafter (t = 0, stride, 2*stride, ...), so the very first frame of each
        episode is captured. Consecutive frames change little, so subsampling (default stride=16)
        yields more decorrelated samples per byte: `steps // stride` frames are stored, each a
        B-batch of observations. Returns the filled pool (`capacity x C x N x N` on the buffer
        device), overwriting any previously collected samples.
        """
        steps  = self.steps  if steps  is None else int(steps)
        stride = self.stride if stride is None else int(stride)
        capacity = (steps // stride) * self.num_envs
        if self._pool is None or self._pool.shape[0] != capacity:
            self._pool = torch.empty(
                (capacity, self.num_channels, self.grid_size, self.grid_size),
                dtype=self._buffer_dtype, device=self._buffer_device,
            )

        write = 0
        for t in range(steps):
            # Store every stride-th frame, starting at the ignition frame (t=0). The write guard keeps
            # exactly `capacity` frames when steps isn't a multiple of stride (t=0 would otherwise add
            # one extra frame past the pool), dropping the trailing frame rather than overflowing.
            if t % stride == 0 and write < capacity:
                obs = self.observe().to(device=self._buffer_device, dtype=self._buffer_dtype)
                self._pool[write:write + self.num_envs] = obs
                write += self.num_envs
            self._sim.step_fields(self._state)

        self._filled = write
        return self._pool

    def sample_minibatch(self, size: int | None = None) -> torch.Tensor:
        """Draw `size` random samples (uniform, with replacement) from the pool."""
        if self._filled == 0:
            raise RuntimeError("Pool is empty; call collect() before sample_minibatch().")
        size = self.minibatch_size if size is None else int(size)
        idx = torch.randint(0, self._filled, (size,), generator=self._gen, device=self._buffer_device)
        return self._pool[idx]

    # -----------------------------------------------------------------------
    # Static on-disk dataset
    # -----------------------------------------------------------------------

    def build_dataset(
        self,
        out_dir: str | Path,
        rounds: int,
        *,
        steps: int | None = None,
        stride: int | None = None,
        overwrite: bool = False,
    ) -> dict:
        """Generate a large static, sharded dataset on disk for pretraining.

        Each round re-samples fresh worlds (reset) then collect()s a pool, which is written
        as one `.npy` shard. Only one round is held in memory at a time, so the dataset can
        far exceed RAM. A `meta.json` describing the shards is written alongside them; load
        the result with `FireDataset(out_dir)`.

        Returns the metadata dict.
        """
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        existing = sorted(out.glob(_SHARD_GLOB))
        if existing and not overwrite:
            raise FileExistsError(
                f"{out} already contains {len(existing)} shard(s); pass overwrite=True to replace."
            )
        for p in existing:
            p.unlink()

        # In exhaustive mode the schedule must hold exactly rounds x num_envs maps so every
        # map is used its full `repeats` times with none left over or reused. Rewind the cursor
        # past the constructor's initial reset() so round 0 starts at the top of the schedule.
        if self._schedule is not None:
            needed = rounds * self.num_envs
            if len(self._schedule) != needed:
                raise ValueError(
                    f"Exhaustive schedule has {len(self._schedule)} maps for this worker but "
                    f"rounds x num_envs = {rounds} x {self.num_envs} = {needed}; they must match "
                    f"for exact coverage. Adjust rounds, num_envs, num_shards, or map_repeats."
                )
            self._schedule_pos = 0

        # Effective per-round schedule (collect() lets these override the attributes); record
        # what was actually used so a trajectory-aware reader can reshape each shard's flat
        # samples into (frames, envs) and know the sim-tick spacing between stored frames.
        eff_steps  = self.steps  if steps  is None else int(steps)
        eff_stride = self.stride if stride is None else int(stride)

        samples_per_shard = None
        for r in range(rounds):
            self.reset()                                   # fresh worlds each round
            pool = self.collect(steps=steps, stride=stride)
            arr = pool.cpu().numpy()                        # shards live on the host
            np.save(out / f"shard_{r:04d}.npy", arr)
            samples_per_shard = arr.shape[0]
            print(f"  [{r + 1}/{rounds}] shard_{r:04d}.npy  {arr.shape}  {arr.dtype}")

        meta = {
            "rounds":            rounds,
            "samples_per_shard": samples_per_shard,
            "total_samples":     samples_per_shard * rounds,
            "channels":          list(self.obs_channels),
            "num_channels":      self.num_channels,
            "grid_size":         self.grid_size,
            # Trajectory structure: a shard is (F, B) samples in frame-major, env-minor order
            # (see collect). F = samples_per_shard // num_envs frames per world, each world an
            # independent episode; consecutive stored frames are `stride` sim-ticks apart.
            "num_envs":          self.num_envs,
            "stride":            eff_stride,
            "steps":             eff_steps,
            # Map coverage / schedule provenance: map_repeats an int means exhaustive (every map
            # used exactly this many times, no replacement), with this worker taking shard_index of
            # num_shards from the schedule_seed-permuted global schedule; all null means random-with-
            # replacement draws (no schedule). See _build_map_schedule.
            "map_repeats":       self._map_repeats,
            "num_shards":        self._num_shards,
            "shard_index":       self._shard_index,
            "schedule_seed":     self._schedule_seed,
            "dtype":             str(np.dtype(self._buffer_dtype_np)),
            "shard_glob":        _SHARD_GLOB,
        }
        (out / _META_NAME).write_text(json.dumps(meta, indent=2))
        return meta

    @property
    def _buffer_dtype_np(self) -> np.dtype:
        """The numpy dtype the pool serialises to (mirrors the torch buffer dtype)."""
        return torch.empty(0, dtype=self._buffer_dtype).numpy().dtype


class FireDataset(Dataset):
    """Memory-mapped torch Dataset over a sharded dataset built by BatchedRollout.build_dataset.

    Two on-disk layouts are accepted, transparently:

    * Flat -- `root` itself holds the shards and a `meta.json` (a single build_dataset run).
    * Nested -- `root` holds one subdirectory per source, each a flat dataset with its own
      `meta.json`. This is what several workers produce when each writes into its own subdir
      (e.g. one per GPU); pointing FireDataset at the parent unions them into one dataset.
      All sources must agree on channels / grid_size / dtype / samples_per_shard, since they
      are interleaved into one flat, uniformly-sized shard list.

    Shards are opened lazily with np.load(mmap_mode='r') on first access (and per DataLoader
    worker after fork), so the OS pages samples in on demand and the dataset can far exceed
    RAM. Each item is one C x N x N observation. Samples are stored grouped by world within a
    shard (and by source across subdirs), so shuffle in the DataLoader for I.I.D. minibatches.

    Normalization is *baked into the shards* offline (tools/normalize_dataset.py rewrites each
    sample to the per-channel min-max scale `(x - min) / (max - min)`), not applied on read: every
    shard is read hundreds of times over a training run, so paying the scale once at prep time keeps
    the hot path a plain memmap copy. A normalised dataset carries `"normalized": true` in its meta
    and the applied min/max in a root `stats.json`; `FireDataset.is_normalized` reports the flag.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        # A meta.json directly under root marks a flat dataset; otherwise treat root as a
        # parent holding one subdirectory per source (multi-worker layout).
        source_dirs = (
            [self.root]
            if (self.root / _META_NAME).is_file()
            else [p.parent for p in sorted(self.root.glob(f"*/{_META_NAME}"))]
        )
        if not source_dirs:
            raise FileNotFoundError(
                f"No dataset found at {self.root}: neither a {_META_NAME} nor any "
                f"*/{_META_NAME} subdirectory."
            )

        self.meta = json.loads((source_dirs[0] / _META_NAME).read_text())
        self.per_shard = int(self.meta["samples_per_shard"])
        # Fields every source must share for the flat, uniformly-sized shard list to be valid.
        keys = ("samples_per_shard", "num_channels", "grid_size", "dtype")
        ref = {k: self.meta.get(k) for k in keys}

        self.shard_paths: list[Path] = []
        self._all_normalized = True                 # true only if every source is flagged normalised
        for d in source_dirs:
            meta = json.loads((d / _META_NAME).read_text())
            mism = {k: meta.get(k) for k in keys if meta.get(k) != ref[k]}
            if mism:
                raise ValueError(
                    f"Dataset source {d} is incompatible with {source_dirs[0]}: "
                    f"{mism} != {{{', '.join(f'{k}: {ref[k]!r}' for k in mism)}}}."
                )
            self._all_normalized &= bool(meta.get("normalized", False))
            self.shard_paths.extend(sorted(d.glob(meta.get("shard_glob", _SHARD_GLOB))))
        if not self.shard_paths:
            raise FileNotFoundError(f"No shards found under {self.root}.")
        self._mmaps: list[np.ndarray | None] = [None] * len(self.shard_paths)

    @property
    def is_normalized(self) -> bool:
        """Whether the on-disk shards have been normalized (every source meta flags it)."""
        return self._all_normalized

    def __len__(self) -> int:
        return self.per_shard * len(self.shard_paths)

    def _shard(self, i: int) -> np.ndarray:
        if self._mmaps[i] is None:
            self._mmaps[i] = np.load(self.shard_paths[i], mmap_mode="r")
        return self._mmaps[i]

    def __getitem__(self, idx: int) -> torch.Tensor:
        if idx < 0:
            idx += len(self)
        shard, offset = divmod(idx, self.per_shard)
        # Copy just this sample out of the (read-only) memmap so the tensor owns writable
        # memory — cheap (one C x N x N slice) and safe for in-place transforms downstream.
        return torch.from_numpy(np.array(self._shard(shard)[offset]))

    def get_batch(self, indices) -> torch.Tensor:
        """Fetch many samples at once as a single (B, C, N, N) tensor, in the given order.

        Groups the requested global indices by shard and reads each shard's samples in one
        memmap fancy-index into a preallocated buffer — far less per-sample Python/allocation
        overhead than B separate __getitem__ calls plus a collate. Used by RamBatchLoader to
        serve whole batches without DataLoader worker processes (and their shared memory).
        """
        return torch.from_numpy(self._raw_batch(indices))

    def _raw_batch(self, indices) -> np.ndarray:
        """Grouped-by-shard fancy read of the requested samples as a (B, C, N, N) numpy array.

        Groups the global indices by shard so each shard is read in a single memmap fancy-index
        into a preallocated buffer. Shared by get_batch and compute_channel_minmax.
        """
        indices = np.asarray(indices, dtype=np.int64)
        shard_ids, offsets = np.divmod(indices, self.per_shard)
        ref = self._shard(int(shard_ids[0]))
        out = np.empty((indices.shape[0], *ref.shape[1:]), dtype=ref.dtype)
        for s in np.unique(shard_ids):
            mask = shard_ids == s
            out[mask] = self._shard(int(s))[offsets[mask]]
        return out


def plan_stat_shards(dataset: FireDataset, max_samples: int | None, seed: int) -> np.ndarray:
    """Which shard indices compute_channel_minmax will read for a given `max_samples` budget.

    Sampling is done at *shard* granularity: reading a few whole shards sequentially is far cheaper
    on a capacity-tuned parallel filesystem than scattering `max_samples` single-sample reads
    across all shards (random 1 MB reads over hundreds of GB). Each shard already spans many worlds
    x timesteps, so a random subset of shards covers the value distribution well. Picks the fewest
    whole shards that meet the budget; returns all shards when the budget covers the whole set.
    """
    n_shards = len(dataset.shard_paths)
    if max_samples is None or n_shards * dataset.per_shard <= max_samples:
        return np.arange(n_shards)
    k = min(n_shards, max(1, -(-max_samples // dataset.per_shard)))   # ceil(max_samples/per_shard)
    return np.sort(np.random.default_rng(seed).choice(n_shards, size=k, replace=False))


def compute_channel_minmax(
    dataset: FireDataset, *, max_samples: int | None = None, seed: int = 0, chunk: int = 256
) -> tuple[np.ndarray, np.ndarray]:
    """Find per-channel min and max for min-max scaling of `dataset` to [0, 1].

    Most channels aren't normally distributed (bounded masses, one-hot biomes, a binary ignited
    mask, heavy-tailed temperatures), so the dataset is min-max scaled — (x - min) / (max - min) —
    rather than z-scored. This maps the bulk of each channel into [0, 1] while letting genuine
    outliers (e.g. flame-front temperatures) ride above 1, which is often exactly the signal we
    care about, instead of being flattened by a std.

    Unlike a mean/std, min and max are extreme order statistics: reading only a subset of shards
    systematically *under*-estimates the true range (the global extreme may live in an unread
    shard), which would push unread outliers past [0, 1] after scaling. So the default
    (`max_samples=None`) reads *every* shard; passing a `max_samples` budget samples whole shards
    (see plan_stat_shards) and trades that exactness for speed. Values are read in float64
    `chunk`-sample slices, so no large buffer is materialised.

    A channel whose read values never vary (max == min) has zero span; the caller floors the span
    to 1 so scaling it is a safe no-op (the constant maps to 0) rather than a divide-by-zero.

    Returns `(cmin, cmax)`, each a float64 array of length num_channels.
    """
    n_ch = int(dataset.meta["num_channels"])
    if len(dataset) == 0:
        raise ValueError(f"Dataset at {dataset.root} is empty; cannot compute channel stats.")
    shard_sel = plan_stat_shards(dataset, max_samples, seed)

    cmin = np.full(n_ch, np.inf, dtype=np.float64)    # running per-channel min
    cmax = np.full(n_ch, -np.inf, dtype=np.float64)   # running per-channel max
    for si in shard_sel:
        shard = dataset._shard(int(si))               # (per_shard, C, H, W) memmap
        m = shard.shape[0]
        for start in range(0, m, chunk):              # sequential slices of one shard
            arr = np.asarray(shard[start:start + chunk], dtype=np.float64)
            cmin = np.minimum(cmin, arr.min(axis=(0, 2, 3)))
            cmax = np.maximum(cmax, arr.max(axis=(0, 2, 3)))
    return cmin, cmax


def _signed_channel_indices(stats: dict) -> list[int]:
    """Indices of the channels `stats` marks signed (mapped to [-1, 1]; see SIGNED_CHANNELS).

    The signed set is stored by name in `stats["signed_channels"]`; resolving it to positions needs
    the ordered channel names. A raw dataset stats.json keeps those under `channels`, but a checkpoint
    config reuses `channels` for the encoder WIDTHS, so the names are stashed under `channel_names`
    there — prefer that, and fall back to `channels` only when its entries are actually strings.
    Raises if signed channels are declared but no name order is available to locate them, rather than
    silently applying the wrong (min-max) transform to a signed channel.
    """
    signed = stats.get("signed_channels")
    if not signed:
        return []
    order = stats.get("channel_names")
    if order is None:
        cand = stats.get("channels")
        order = cand if (cand and isinstance(cand[0], str)) else None
    if order is None:
        raise ValueError(
            "stats declares signed_channels but carries no channel name order (channel_names / "
            "channels) to resolve them; cannot apply the symmetric [-1, 1] transform."
        )
    signed = set(signed)
    return [i for i, name in enumerate(order) if name in signed]


def stats_affine(stats: dict) -> tuple[np.ndarray, np.ndarray]:
    """The baked normalization as an affine pair `(offset, scale)`: forward is (x - offset) / scale,
    inverse is y * scale + offset.

    Min-max datasets/checkpoints carry `channel_min`/`channel_max` (offset = min, scale = max - min);
    legacy z-scored ones carry `channel_mean`/`channel_std` (offset = mean, scale = std), so old
    checkpoints still de-normalize correctly. Channels listed in `signed_channels` (the near-surface
    wind) instead map symmetrically to [-1, 1] — offset 0, scale max(|min|, |max|) — so calm wind
    lands at 0 and the sign is preserved; this applies to the min-max form only. A zero/tiny span is
    floored to 1 so a constant channel is a safe no-op (the constant maps to 0). Returns two float64
    arrays of length num_channels.
    """
    if "channel_min" in stats and "channel_max" in stats:
        cmin = np.asarray(stats["channel_min"], dtype=np.float64)
        cmax = np.asarray(stats["channel_max"], dtype=np.float64)
        offset = cmin.copy()
        scale = cmax - cmin
        for i in _signed_channel_indices(stats):          # symmetric about 0 -> [-1, 1]
            offset[i] = 0.0
            scale[i] = max(abs(cmin[i]), abs(cmax[i]))
    else:
        offset = np.asarray(stats["channel_mean"], dtype=np.float64)
        scale = np.asarray(stats["channel_std"], dtype=np.float64).copy()
    scale[scale < 1e-8] = 1.0                             # constant channel -> unit scale (no-op)
    return offset, scale


def write_channel_minmax(
    root: str | Path, cmin: np.ndarray, cmax: np.ndarray, *, channels: list[str] | None = None,
    n_samples: int | None = None, total_samples: int | None = None,
) -> dict:
    """Write per-channel min/max to `root`/stats.json (the transform baked in by normalize_dataset).

    `n_samples`/`total_samples` (when given) record how many observations the estimate drew and
    how many the dataset holds, for provenance. Returns the stats dict that was written.
    """
    stats = {
        "channel_min": [float(x) for x in cmin],
        "channel_max": [float(x) for x in cmax],
    }
    if channels is not None:
        stats["channels"] = list(channels)
        # Record which of this dataset's channels are scaled symmetrically to [-1, 1] (signed, with a
        # meaningful zero) rather than min-max to [0, 1], so stats_affine reproduces the exact bake.
        signed = [c for c in channels if c in SIGNED_CHANNELS]
        if signed:
            stats["signed_channels"] = signed
    if n_samples is not None:
        stats["n_samples"] = int(n_samples)
    if total_samples is not None:
        stats["total_samples"] = int(total_samples)
    # Atomic publish: write to a temp file then rename, so a concurrent reader (e.g. a sibling
    # trainer in a replicate sweep) sees either no file or the complete one, never a torn write.
    root = Path(root)
    tmp = root / f".{_STATS_NAME}.tmp.{os.getpid()}"
    tmp.write_text(json.dumps(stats, indent=2))
    os.replace(tmp, root / _STATS_NAME)
    return stats


def ensure_channel_stats(
    root: str | Path, *, recompute: bool = False, max_samples: int | None = None,
    seed: int = 0, chunk: int = 256, verbose: bool = True,
) -> dict:
    """Compute + write `root`/stats.json if missing (or `recompute`), else load the existing one.

    Idempotent: the first call over a freshly built dataset finds the per-channel min/max (see
    compute_channel_minmax; the default reads every shard so the range is exact) and caches them;
    later calls just read the file back. Returns the stats dict.
    """
    stats_path = Path(root) / _STATS_NAME
    if stats_path.is_file() and not recompute:
        return json.loads(stats_path.read_text())
    ds = FireDataset(root)                                 # read raw values to measure them
    n = len(ds)
    shard_sel = plan_stat_shards(ds, max_samples, seed)
    n_used = int(len(shard_sel) * ds.per_shard)
    if verbose:
        scope = (f"all {n}" if n_used == n
                 else f"{n_used} ({len(shard_sel)} random shards) of {n}")
        print(f"measuring channel min/max over {scope} samples in '{root}' ...")
    cmin, cmax = compute_channel_minmax(ds, max_samples=max_samples, seed=seed, chunk=chunk)
    stats = write_channel_minmax(
        root, cmin, cmax, channels=list(ds.meta.get("channels", [])),
        n_samples=n_used, total_samples=n,
    )
    if verbose:
        names = stats.get("channels") or [f"ch{i}" for i in range(len(cmin))]
        signed = set(stats.get("signed_channels", []))
        for name, lo, hi in zip(names, cmin, cmax):
            tag = "  -> [-1,1] signed" if name in signed else ""
            print(f"  {name:<18} min {lo:+.4g}  max {hi:+.4g}{tag}")
    return stats


def _dataset_sources(root: Path) -> list[Path]:
    """The source dirs of a dataset: `root` itself if flat, else its per-worker subdirs."""
    if (root / _META_NAME).is_file():
        return [root]
    return [p.parent for p in sorted(root.glob(f"*/{_META_NAME}"))]


def _drop_page_cache(path: str) -> None:
    """Flush `path` and drop its page-cache pages (best effort, Linux only).

    Baking rewrites hundreds of GB; without this the freshly written pages pile up as dirty cache
    faster than the filesystem flushes them, and under a memory cgroup that climbs to the limit and
    OOM-kills the job. Fsyncing then advising DONTNEED turns those pages clean and evicts them, so a
    many-worker bake stays within a bounded resident footprint. A no-op where fadvise is absent."""
    if not hasattr(os, "posix_fadvise"):
        return
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)                                     # dirty -> clean so DONTNEED can evict
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    except OSError:
        pass


def _bake_shard(job: tuple) -> int:
    """Rewrite one shard to min-max scaled values into its temp path; returns the sample count.

    `offset`/`scale` are (1, C, 1, 1) float32 (min and max-min per channel); the sample maps to
    (x - offset) / scale. The float32 working buffer is bounded to `chunk` samples; the output keeps
    the shard's stored dtype. Reads the pristine original (never mutated here) and writes a sibling
    temp file, so the source dir is only swapped in once all its shards are baked. Each shard's read
    and written pages are dropped from cache afterwards to bound memory use.
    """
    src_path, tmp_path, offset, scale, chunk, dtype_str = job
    dtype = np.dtype(dtype_str)
    raw = np.load(src_path, mmap_mode="r")                   # (m, C, H, W), read-only
    n = int(raw.shape[0])
    out = np.empty(raw.shape, dtype=dtype)
    for start in range(0, n, chunk):
        block = np.asarray(raw[start:start + chunk], dtype=np.float32)
        out[start:start + chunk] = ((block - offset) / scale).astype(dtype)
    np.save(tmp_path, out)
    del raw, out                                             # release the mmap + output buffer
    _drop_page_cache(tmp_path)                               # evict the just-written pages
    _drop_page_cache(src_path)                               # evict the read-in original pages
    return n


def bake_normalization(
    root: str | Path, *, max_samples: int | None = None, seed: int = 0, chunk: int = 256,
    workers: int = 1, keep_backup: bool = False, verbose: bool = True,
) -> dict:
    """Normalize a dataset on disk: rewrite every shard to the per-channel min-max scale (x-min)/(max-min).

    Measures (or reuses) the per-channel min/max -> `root`/stats.json, then rewrites every shard to
    scaled values in the stored dtype and flags each source meta `"normalized": true`, so
    FireDataset serves already-scaled data with no per-read cost. Individual shards are baked in
    parallel across `workers` processes (I/O-bound; scale it to the node's cpus). Idempotent:
    already-flagged sources are skipped, so a re-run (or requeued job) is a safe no-op.

    Safe/resumable: each source's shards are baked from the pristine originals into a sibling
    `<src>.norm.tmp`, and the directory is swapped in only once all its shards are written — an
    interrupted run never leaves half-scaled shards live. Returns a summary dict.
    """
    root = Path(root)
    sources = _dataset_sources(root)
    if not sources:
        raise FileNotFoundError(f"No dataset found at {root} to normalize.")

    def flagged(d: Path) -> bool:
        return bool(json.loads((d / _META_NAME).read_text()).get("normalized", False))

    pending = [d for d in sources if not flagged(d)]
    if not pending:
        if verbose:
            print(f"all {len(sources)} source(s) already normalized; nothing to do.")
        return {"baked": 0, "skipped": len(sources), "sources": len(sources), "shards": 0}
    if len(pending) < len(sources) and not (root / _STATS_NAME).is_file():
        # Some sources baked, some raw, and no cached stats -> a fresh estimate would be measured
        # from a mix of scaled and unscaled shards. Refuse rather than bake a wrong transform.
        raise RuntimeError(
            f"{root} is partially normalized ({len(sources) - len(pending)}/{len(sources)} sources) "
            f"but has no {_STATS_NAME}; re-stage the raw dataset and bake it in one pass."
        )

    # Stats are measured from the (still-raw) data; reuses stats.json when already present.
    stats = ensure_channel_stats(root, max_samples=max_samples, seed=seed, chunk=chunk,
                                 verbose=verbose)
    off64, scale64 = stats_affine(stats)                 # min and (max-min), constant-channel floored
    offset = off64.astype(np.float32).reshape(1, -1, 1, 1)
    scale = scale64.astype(np.float32).reshape(1, -1, 1, 1)

    # Fan out one task per shard (across all pending sources) into fresh per-source temp dirs.
    plans = []          # (source_dir, tmp_dir, meta)
    tasks = []          # per-shard bake jobs for the worker pool
    for d in pending:
        meta = json.loads((d / _META_NAME).read_text())
        shards = sorted(d.glob(meta.get("shard_glob", _SHARD_GLOB)))
        tmp = d.parent / (d.name + ".norm.tmp")
        if tmp.exists():                                     # clear a stale tmp from an aborted run
            shutil.rmtree(tmp)
        tmp.mkdir()
        plans.append((d, tmp, meta))
        for sp in shards:
            tasks.append((str(sp), str(tmp / sp.name), offset, scale, chunk, meta["dtype"]))

    if verbose:
        print(f"baking {len(tasks)} shard(s) across {len(pending)} source(s) "
              f"with {workers} worker(s) ...")
    if workers > 1 and len(tasks) > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as ex:
            n_samples = sum(ex.map(_bake_shard, tasks))
    else:
        n_samples = sum(_bake_shard(t) for t in tasks)

    # All shards baked -> finalise each source: flag its meta, then swap the dir in atomically.
    for d, tmp, meta in plans:
        (tmp / _META_NAME).write_text(json.dumps({**meta, "normalized": True}, indent=2))
        bak = d.parent / (d.name + ".raw.bak")
        if bak.exists():
            shutil.rmtree(bak)
        os.replace(d, bak)                                   # move raw original aside
        os.replace(tmp, d)                                   # move baked dir into place
        if not keep_backup:
            shutil.rmtree(bak)
        if verbose:
            print(f"  {d.name}: baked")
    return {"baked": len(pending), "skipped": len(sources) - len(pending),
            "sources": len(sources), "shards": len(tasks), "samples": n_samples}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a static sharded Firecracker dataset for pretraining")
    parser.add_argument("--out", required=True, help="output directory for shards + meta.json")
    parser.add_argument("--rounds", type=int, default=8, help="number of shards (each is one reset+collect)")
    parser.add_argument("--config", metavar="PATH", help="path to a TOML config file")
    parser.add_argument("--maps-dir", help="directory to sample maps from (overrides config)")
    parser.add_argument("--num-envs", type=int, help="worlds stepped in parallel (overrides config)")
    parser.add_argument("--steps", type=int, help="ticks stepped per round (overrides config)")
    parser.add_argument("--stride", type=int, help="store every stride-th frame (overrides config)")
    parser.add_argument("--buffer-dtype", help="shard dtype, e.g. float32 or float16 (overrides config)")
    parser.add_argument("--spawn-fire", action="store_true", help="ignite each world at reset")
    parser.add_argument("--seed", type=int, default=None, help="seed for map selection / ignition")
    parser.add_argument("--overwrite", action="store_true", help="replace existing shards in --out")
    parser.add_argument("--map-repeats", type=int, default=None,
                        help="exhaustive mode: use every map exactly this many times "
                             "(without replacement) instead of random sampling with replacement")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="exhaustive mode: total number of workers sharing the schedule")
    parser.add_argument("--shard-index", type=int, default=0,
                        help="exhaustive mode: this worker's index in [0, num-shards)")
    parser.add_argument("--schedule-seed", type=int, default=0,
                        help="exhaustive mode: shared seed all workers use to permute the "
                             "global schedule (must match across workers)")
    args = parser.parse_args()

    roll = BatchedRollout(
        load_config(args.config),
        maps_dir=args.maps_dir,
        num_envs=args.num_envs,
        steps=args.steps,
        stride=args.stride,
        buffer_dtype=args.buffer_dtype,
        spawn_fire=True if args.spawn_fire else None,
        seed=args.seed,
        map_repeats=args.map_repeats,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        schedule_seed=args.schedule_seed,
    )
    print(
        f"Building {args.rounds} shard(s) of {roll.capacity} samples "
        f"({roll.num_channels}x{roll.grid_size}x{roll.grid_size}) into '{args.out}/' ..."
    )
    meta = roll.build_dataset(args.out, args.rounds, overwrite=args.overwrite)
    print(f"Done. {meta['total_samples']} total samples across {meta['rounds']} shard(s).")


if __name__ == "__main__":
    main()
