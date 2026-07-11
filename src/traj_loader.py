"""
traj_loader.py

Trajectory-aware views over a FireDataset's flat shards, for next-latent-state
prediction / world-model training.

A shard built by BatchedRollout.collect holds (F, B) observations in frame-major,
env-minor order: for each of F stored frames, B envs are written contiguously. Each
env is an independent world, so env e of shard s is one episode of F frames spaced
`stride` sim-ticks apart (see rollout.collect and the trajectory fields in meta.json).
This module recovers those episodes and enumerates fixed-length frame windows over
them by mapping each window to the global flat indices FireDataset already serves --
no data is moved or reshaped on disk.

Quick use (single-step (x_t, x_{t+1}) pairs):

    from rollout import FireDataset
    from traj_loader import TrajectoryDataset, TrajectoryBatchLoader, split_episodes

    ds   = FireDataset("data/fire")
    traj = TrajectoryDataset(ds, window=2)          # num_envs/stride read from meta
    train_w, val_w = split_episodes(traj, val_frac=0.05, seed=0)
    loader = TrajectoryBatchLoader(traj, train_w, batch_size=64, shuffle=True, seed=0)
    for batch in loader:                            # (batch, window, C, H, W)
        x_t, x_next = batch[:, 0], batch[:, 1]

Older datasets whose meta.json predates the num_envs/stride fields need those passed
explicitly: TrajectoryDataset(ds, num_envs=4, stride=64).
"""

from __future__ import annotations

import queue
import threading
from typing import Iterator

import numpy as np
import torch


class TrajectoryDataset:
    """Enumerate fixed-length frame windows over the episodes in a FireDataset.

    An episode is one (shard, env): env e of shard s, all F stored frames of that world.
    A window is `window` frames spaced `step` frames apart; windows start densely at
    frames 0 .. F-1-span within each episode (span = (window-1) * step). This class holds
    no pixels -- window_indices maps a window to the global flat FireDataset indices, which
    get_batch reads in one grouped-by-shard pass.

    Args:
        dataset:  a FireDataset (uses .meta, .per_shard, .shard_paths, .get_batch).
        num_envs: B, the env axis within a shard. Defaults to meta["num_envs"]; pass
                  explicitly for datasets whose meta predates that field.
        stride:   sim-ticks between stored frames (kept for reference / dt). meta fallback.
        window:   frames per window (2 = (x_t, x_{t+1}) pairs; >2 for multi-step rollout).
        step:     frame gap between successive window elements (1 = adjacent stored frames).
    """

    def __init__(self, dataset, *, num_envs=None, stride=None, window=2, step=1):
        self.ds = dataset
        meta = dataset.meta
        if num_envs is None and "num_envs" not in meta:
            raise ValueError(
                "dataset meta has no 'num_envs' (built before the trajectory fields were "
                "added); pass num_envs=... explicitly."
            )
        self.B = int(num_envs if num_envs is not None else meta["num_envs"])
        self.stride = int(stride if stride is not None else meta.get("stride", 1))
        self.per_shard = int(dataset.per_shard)
        if self.B < 1 or self.per_shard % self.B != 0:
            raise ValueError(
                f"per_shard {self.per_shard} not divisible by num_envs {self.B}; "
                f"wrong num_envs for this dataset?"
            )
        self.F = self.per_shard // self.B                  # frames per episode
        self.n_shards = len(dataset.shard_paths)
        self.window = int(window)
        self.step = int(step)
        if self.window < 1 or self.step < 1:
            raise ValueError("window and step must be >= 1")
        self.span = (self.window - 1) * self.step          # frames a window spans
        if self.span >= self.F:
            raise ValueError(
                f"window span {self.span} >= episode length {self.F}; shorten window/step."
            )
        # One episode per (shard, env). Windows start densely at frames 0 .. starts_per_ep-1.
        self.episodes = [(s, e) for s in range(self.n_shards) for e in range(self.B)]
        self.starts_per_ep = self.F - self.span

    def __len__(self) -> int:
        return len(self.episodes) * self.starts_per_ep

    def _global(self, shard: int, env: int, frame: int) -> int:
        # Flat FireDataset index of (shard, env, frame): shard block, then frame-major/env-minor.
        return shard * self.per_shard + frame * self.B + env

    def window_indices(self, win_id: int) -> list[int]:
        """The `window` global flat indices making up window `win_id`, in time order."""
        ep, k = divmod(int(win_id), self.starts_per_ep)
        shard, env = self.episodes[ep]
        return [self._global(shard, env, k + j * self.step) for j in range(self.window)]

    def episode_windows(self, episode_ids) -> np.ndarray:
        """All window ids belonging to the given episode ids (used to build train/val splits)."""
        spe = self.starts_per_ep
        if len(episode_ids) == 0:
            return np.empty(0, dtype=np.int64)
        return np.concatenate(
            [np.arange(int(e) * spe, (int(e) + 1) * spe, dtype=np.int64) for e in episode_ids]
        )


def split_episodes(traj: TrajectoryDataset, val_frac: float, seed: int):
    """Train/val window ids split at EPISODE granularity.

    A whole world lands entirely in train or entirely in val, so x_t and x_{t+1} never
    straddle the split -- otherwise the next-state target would leak across it. Returns
    (train_window_ids, val_window_ids).
    """
    n_ep = len(traj.episodes)
    perm = np.random.default_rng(seed).permutation(n_ep)
    n_val = max(1, round(n_ep * val_frac))
    val_ep, train_ep = perm[:n_val], perm[n_val:]
    return traj.episode_windows(train_ep), traj.episode_windows(val_ep)


class TrajectoryBatchLoader:
    """Thread-prefetched loader yielding (batch, window, C, H, W) tensors over frame windows.

    Mirrors ram_loader.RamBatchLoader: one background thread stages the next batch via a
    single vectorized FireDataset.get_batch (grouped by shard), overlapping GPU compute and
    spawning no worker processes (so it is safe under the heavy process packing that crashes
    DataLoader's shared-memory workers). All windows in a batch are flattened to one index
    list, read once, then reshaped back to (batch, window, ...).

    Args:
        traj:        a TrajectoryDataset.
        window_ids:  window ids this loader draws from (e.g. a train or val split).
        batch_size:  windows per batch.
        shuffle:     reshuffle the window ids each epoch (seeded, reproducible).
        seed:        seed for the shuffle RNG.
        drop_last:   drop a trailing partial batch (train) or keep it (val).
        pin_memory:  pin each batch for faster host->device copies (CUDA only).
        prefetch:    max batches the background thread may stage ahead.
    """

    def __init__(self, traj, window_ids, batch_size, *, shuffle, seed,
                 drop_last=True, pin_memory=False, prefetch=3):
        self.traj = traj
        self.window_ids = np.asarray(window_ids, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.pin_memory = bool(pin_memory)
        self.prefetch = max(1, int(prefetch))
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        n = len(self.window_ids)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def _chunks(self):
        order = self._rng.permutation(self.window_ids) if self.shuffle else self.window_ids
        n = len(order)
        stop = (n // self.batch_size) * self.batch_size if self.drop_last else n
        for i in range(0, stop, self.batch_size):
            yield order[i : i + self.batch_size]

    def _make_batch(self, ids) -> torch.Tensor:
        # (batch, window) global indices -> one flat read -> (batch, window, C, H, W).
        idx = np.stack([self.traj.window_indices(int(w)) for w in ids])
        frames = self.traj.ds.get_batch(idx.reshape(-1))       # (batch*window, C, H, W)
        batch = frames.view(len(ids), self.traj.window, *frames.shape[1:])
        return batch.pin_memory() if self.pin_memory else batch

    def __iter__(self) -> Iterator[torch.Tensor]:
        q: queue.Queue = queue.Queue(maxsize=self.prefetch)
        sentinel = object()

        def producer() -> None:
            try:
                for chunk in self._chunks():
                    q.put(self._make_batch(chunk))
            except Exception as exc:  # surface a fetch error to the consumer instead of hanging
                q.put(exc)
            else:
                q.put(sentinel)

        thread = threading.Thread(target=producer, daemon=True)
        thread.start()
        while True:
            item = q.get()
            if item is sentinel:
                break
            if isinstance(item, Exception):
                raise item
            yield item
        thread.join()
