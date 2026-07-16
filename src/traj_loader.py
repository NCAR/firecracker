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

from collections import deque
from concurrent.futures import ThreadPoolExecutor
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

    def __init__(self, dataset, *, num_envs=None, stride=None, window=2, step=1, partial=False):
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
        self.partial = bool(partial)
        if self.window < 1 or self.step < 1:
            raise ValueError("window and step must be >= 1")
        self.span = (self.window - 1) * self.step          # frames a full window spans
        # A full window needs `span` future frames. A PARTIAL window only needs one target frame
        # (t0 + one step); the rest are padded and flagged by window_mask, so tail frames can still be
        # a window start -- this is what gives the encoder coverage of the late-episode states. When
        # partial is off, behavior (starts, indices, mask) is identical to the fixed-window original.
        min_span = self.step if self.partial else self.span
        if min_span >= self.F:
            raise ValueError(
                f"window {'min-span' if self.partial else 'span'} {min_span} >= episode length "
                f"{self.F}; shorten window/step."
            )
        # One episode per (shard, env). Windows start densely at frames 0 .. starts_per_ep-1.
        self.episodes = [(s, e) for s in range(self.n_shards) for e in range(self.B)]
        self.starts_per_ep = self.F - min_span

    def __len__(self) -> int:
        return len(self.episodes) * self.starts_per_ep

    def _global(self, shard: int, env: int, frame: int) -> int:
        # Flat FireDataset index of (shard, env, frame): shard block, then frame-major/env-minor.
        return shard * self.per_shard + frame * self.B + env

    def window_indices(self, win_id: int) -> list[int]:
        """The `window` global flat indices making up window `win_id`, in time order. Frames past the
        episode end are clamped to the last stored frame (only reachable for partial windows; the
        companion window_mask flags those padded slots). For full windows the clamp never fires, so the
        indices are identical to the fixed-window original."""
        ep, k = divmod(int(win_id), self.starts_per_ep)
        shard, env = self.episodes[ep]
        last = self.F - 1
        return [self._global(shard, env, min(k + j * self.step, last)) for j in range(self.window)]

    def window_mask(self, win_id: int) -> np.ndarray:
        """Boolean (window,) mask: which window elements are real future frames vs padded past the
        episode end. All-True for full (non-partial) windows."""
        _, k = divmod(int(win_id), self.starts_per_ep)
        last = self.F - 1
        return np.array([k + j * self.step <= last for j in range(self.window)], dtype=bool)

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
    """Thread-pool-prefetched loader yielding (batch, window, C, H, W) tensors over frame windows.

    A pool of `num_workers` background threads stages batches concurrently, each via a single
    vectorized FireDataset.get_batch (grouped by shard), overlapping GPU compute and spawning no
    worker PROCESSES (so it is safe under the heavy process packing that crashes DataLoader's
    shared-memory workers). All windows in a batch are flattened to one index list, read once,
    then reshaped back to (batch, window, ...).

    Assembling one batch is the scattered gather of `batch*window` frames plus an optional
    pin_memory -- both release the GIL for the bulk copy, so several threads genuinely overlap.
    With a single thread the fast batch-256 GPU step outran the loader (GPU-starved); the pool
    keeps `prefetch` batches in flight so the next is ready when the GPU asks. Batches are yielded
    in their shuffled order (FIFO on the futures), so results stay reproducible for a given seed.

    Args:
        traj:        a TrajectoryDataset.
        window_ids:  window ids this loader draws from (e.g. a train or val split).
        batch_size:  windows per batch.
        shuffle:     reshuffle the window ids each epoch (seeded, reproducible).
        seed:        seed for the shuffle RNG.
        drop_last:   drop a trailing partial batch (train) or keep it (val).
        pin_memory:  pin each batch for faster host->device copies (CUDA only).
        prefetch:    max batches staged ahead (depth of the in-flight pipeline).
        num_workers: threads assembling batches concurrently.
    """

    def __init__(self, traj, window_ids, batch_size, *, shuffle, seed,
                 drop_last=True, pin_memory=False, prefetch=6, num_workers=4):
        self.traj = traj
        self.window_ids = np.asarray(window_ids, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.pin_memory = bool(pin_memory)
        self.num_workers = max(1, int(num_workers))
        # Keep at least as many batches in flight as workers, or the pool can never fill.
        self.prefetch = max(self.num_workers, int(prefetch))
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

    def _make_batch(self, ids):
        # (batch, window) global indices -> one flat read -> (batch, window, C, H, W).
        idx = np.stack([self.traj.window_indices(int(w)) for w in ids])
        frames = self.traj.ds.get_batch(idx.reshape(-1))       # (batch*window, C, H, W)
        batch = frames.view(len(ids), self.traj.window, *frames.shape[1:])
        batch = batch.pin_memory() if self.pin_memory else batch
        # Partial (variable-horizon) datasets also emit a (batch, window) validity mask so the trainer
        # can zero the loss on padded steps. Fixed-window datasets keep the original bare-tensor yield.
        if getattr(self.traj, "partial", False):
            masks = np.stack([self.traj.window_mask(int(w)) for w in ids])
            mask = torch.from_numpy(masks)
            return (batch, mask.pin_memory()) if self.pin_memory else (batch, mask)
        return batch

    def __iter__(self) -> Iterator[torch.Tensor]:
        chunks = self._chunks()
        with ThreadPoolExecutor(max_workers=self.num_workers) as ex:
            futures: deque = deque()
            # Prime the pipeline with up to `prefetch` batches, then keep it topped up: pop the
            # oldest future (preserving order), submit one more, yield. .result() re-raises any
            # assembly error in the consumer. shutdown on __exit__ drains in-flight batches.
            for _ in range(self.prefetch):
                try:
                    futures.append(ex.submit(self._make_batch, next(chunks)))
                except StopIteration:
                    break
            while futures:
                batch = futures.popleft().result()
                try:
                    futures.append(ex.submit(self._make_batch, next(chunks)))
                except StopIteration:
                    pass
                yield batch
