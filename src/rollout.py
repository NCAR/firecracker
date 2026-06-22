"""
rollout.py

Batched rollout collector for world-model training — decoupled from Gymnasium.

One `Simulation` steps `B` worlds in parallel on the sim device (the physics ops are
rank-agnostic, so the single-world engine batches with no special-casing). Every `stride`-th
tick's observation, a `B x C x N x N` tensor (see `Simulation.build_observation` /
`OBS_CHANNELS`), is written into a fixed pool of independent samples; the trainer then draws
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
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from config import load_config
from simulation import Simulation, SimState, OBS_CHANNELS
from map_loader import load_map, resolve_map, validate_against_config

DEFAULT_GRID_SIZE = 256

# Filenames used by build_dataset / FireDataset for an on-disk sharded dataset.
_META_NAME = "meta.json"
_SHARD_GLOB = "shard_*.npy"


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
        self.grid_size      = int(grid_size      if grid_size      is not None else env_cfg.get("grid_size",       DEFAULT_GRID_SIZE))
        self._maps_dir      = str(maps_dir       if maps_dir       is not None else maps_cfg.get("dir",            "maps"))
        self._spawn_fire    = bool(spawn_fire    if spawn_fire     is not None else fire_cfg.get("spawn_fire",     False))
        self._fire_radius   = int(fire_cfg.get("spawn_radius", 5))

        self._buffer_device = torch.device(
            buffer_device if buffer_device is not None else roll_cfg.get("buffer_device", "cpu")
        )
        self._buffer_dtype = _resolve_dtype(
            buffer_dtype if buffer_dtype is not None else roll_cfg.get("buffer_dtype", "float32")
        )

        self.num_channels = len(OBS_CHANNELS)
        # Host RNG for map selection / ignition; torch generator for minibatch sampling.
        self._np_rng = np.random.default_rng(seed)
        self._gen = torch.Generator(device=self._buffer_device)
        if seed is not None:
            self._gen.manual_seed(int(seed))

        self._state: SimState | None = None
        self._pool: torch.Tensor | None = None   # (capacity, C, N, N) on buffer_device
        self._filled: int = 0                     # number of valid samples in the pool

        self.reset()

    # -----------------------------------------------------------------------
    # World setup
    # -----------------------------------------------------------------------

    def reset(self) -> None:
        """(Re)load `B` worlds from the maps dir and build the batched simulation state."""
        fields = {k: [] for k in ("terrain", "air_temperatures", "temp_eq", "oxygen",
                                  "oxygen_eq", "mass", "fuel", "fuel_temperatures")}
        amb_x, amb_y = [], []
        for _ in range(self.num_envs):
            # Sample with replacement so num_envs can exceed the number of map files.
            path = resolve_map(self._maps_dir, None, self._np_rng)
            m = load_map(path)
            validate_against_config(m, self.grid_size, self._sim.fuel_type_names)
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
        # Prognostic wind starts at the ambient (broadcast to full fields) so each world
        # begins near its steady state.
        x_wind = torch.zeros_like(mass) + u_amb_x
        y_wind = torch.zeros_like(mass) + u_amb_y
        # Static terrain-channeling gain per world (Phase 6, Option 2): the fire/oxygen read the
        # prognostic wind sped up through gaps by this gain (applied each step in step_fields).
        channel_gain = self._sim.compute_channel_gain(terrain, u_amb_x, u_amb_y)

        self._state = SimState(
            mass=mass,
            air_temperatures=air,
            ground_temperature=stack("temp_eq"),   # surface skin starts at the rest profile
            fuel_temperatures=stack("fuel_temperatures"),
            fuel=stack("fuel"),
            oxygen=stack("oxygen"),
            terrain=terrain,
            temp_eq=stack("temp_eq"),
            oxygen_eq=stack("oxygen_eq"),
            mass_eq=mass.clone(),
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
        """Light a random circular patch in every world (batched analogue of the env click)."""
        s, n = self._state, self.grid_size
        dev = self._sim.device
        rows = torch.arange(n, device=dev).view(1, -1, 1)
        cols = torch.arange(n, device=dev).view(1, 1, -1)
        r0 = torch.as_tensor(self._np_rng.integers(0, n, size=self.num_envs), device=dev).view(-1, 1, 1)
        c0 = torch.as_tensor(self._np_rng.integers(0, n, size=self.num_envs), device=dev).view(-1, 1, 1)
        patch = (rows - r0) ** 2 + (cols - c0) ** 2 <= self._fire_radius ** 2   # (B, N, N)

        ign_max = float(self._sim.ignition_thresholds.max())
        s.air_temperatures = torch.where(patch, torch.full_like(s.air_temperatures, ign_max * 2.0), s.air_temperatures)
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
        """Current batched observation, B x C x N x N on the sim device."""
        s = self._state
        return Simulation.build_observation(s.fuel_temperatures, s.fuel, s.terrain)

    def collect(self, steps: int | None = None, stride: int | None = None) -> torch.Tensor:
        """Step the `B` worlds `steps` times, pooling every `stride`-th tick's observation.

        Consecutive frames change little, so subsampling (default stride=16) yields more
        decorrelated samples per byte: `steps // stride` frames are stored, each a B-batch of
        observations. Returns the filled pool (`capacity x C x N x N` on the buffer device),
        overwriting any previously collected samples.
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
            self._sim.step_fields(self._state)
            if (t + 1) % stride == 0:   # store every stride-th frame
                obs = self.observe().to(device=self._buffer_device, dtype=self._buffer_dtype)
                self._pool[write:write + self.num_envs] = obs
                write += self.num_envs

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
            "channels":          list(OBS_CHANNELS),
            "num_channels":      self.num_channels,
            "grid_size":         self.grid_size,
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

    Shards are opened lazily with np.load(mmap_mode='r') on first access (and per DataLoader
    worker after fork), so the OS pages samples in on demand and the dataset can far exceed
    RAM. Each item is one C x N x N observation. Samples are stored grouped by world within a
    shard, so shuffle in the DataLoader for I.I.D. minibatches.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.meta = json.loads((self.root / _META_NAME).read_text())
        self.shard_paths = sorted(self.root.glob(self.meta.get("shard_glob", _SHARD_GLOB)))
        if not self.shard_paths:
            raise FileNotFoundError(f"No shards found in {self.root}.")
        self.per_shard = int(self.meta["samples_per_shard"])
        self._mmaps: list[np.ndarray | None] = [None] * len(self.shard_paths)

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
    )
    print(
        f"Building {args.rounds} shard(s) of {roll.capacity} samples "
        f"({roll.num_channels}x{roll.grid_size}x{roll.grid_size}) into '{args.out}/' ..."
    )
    meta = roll.build_dataset(args.out, args.rounds, overwrite=args.overwrite)
    print(f"Done. {meta['total_samples']} total samples across {meta['rounds']} shard(s).")


if __name__ == "__main__":
    main()
