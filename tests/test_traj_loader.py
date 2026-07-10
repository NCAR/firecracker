"""
Trajectory-aware view over a built FireDataset (src/traj_loader.py).

A shard is (F, B) observations in frame-major, env-minor order (rollout.collect), so env e
of shard s is one world's F-frame episode. These tests build a tiny on-disk dataset and pin
down that TrajectoryDataset recovers that structure exactly:

  * build_dataset now records num_envs / stride / steps in meta, and F = per_shard // num_envs,
  * window_indices maps a window to the right global FireDataset indices -- verified against the
    raw shard reshaped to (F, B, C, H, W), the ground truth for the layout,
  * TrajectoryBatchLoader yields (batch, window, C, H, W) whose frames are the same world's
    consecutive stored frames, and
  * split_episodes partitions at the episode level (no world's frames leak across train/val).
"""

import numpy as np
import torch

from conftest import to_numpy, make_config, build_map, pad_map
from config import boundary_pad
from map_loader import save_map
from rollout import BatchedRollout, FireDataset
from traj_loader import TrajectoryDataset, TrajectoryBatchLoader, split_episodes


def _build(tmp_path, *, num_envs, steps, stride, rounds, grid=16):
    """Build a tiny sharded dataset and return (root, FireDataset, raw shard arrays)."""
    config = make_config(grid_size=grid)
    maps_dir = tmp_path / "maps"
    maps_dir.mkdir()
    # A few distinct maps so parallel envs are genuinely different worlds.
    for i in range(4):
        save_map(maps_dir / f"m_{i}.npz", pad_map(build_map(grid_size=grid), boundary_pad(config)))
    out = tmp_path / "data"
    BatchedRollout(
        config, maps_dir=str(maps_dir), grid_size=grid,
        num_envs=num_envs, steps=steps, stride=stride,
        buffer_device="cpu", spawn_fire=True, seed=0,
    ).build_dataset(out, rounds=rounds)
    ds = FireDataset(out)
    raw = [np.load(p) for p in ds.shard_paths]          # (per_shard, C, H, W) each
    return out, ds, raw


def test_meta_records_trajectory_fields(tmp_path):
    """build_dataset writes num_envs / stride / steps so a reader can reshape unambiguously."""
    _, ds, _ = _build(tmp_path, num_envs=2, steps=6, stride=2, rounds=2)
    assert ds.meta["num_envs"] == 2
    assert ds.meta["stride"] == 2
    assert ds.meta["steps"] == 6
    # F = steps // stride = 3 frames per world; per_shard = F * B = 6.
    assert ds.meta["samples_per_shard"] == (6 // 2) * 2


def test_trajectory_shape_matches_meta(tmp_path):
    _, ds, _ = _build(tmp_path, num_envs=2, steps=6, stride=2, rounds=2)
    traj = TrajectoryDataset(ds, window=2)               # num_envs/stride from meta
    assert traj.B == 2
    assert traj.F == 3                                   # steps // stride
    assert traj.n_shards == 2
    assert traj.stride == 2
    assert traj.starts_per_ep == 3 - 1                   # F - span, span = (2-1)*1
    assert len(traj.episodes) == 2 * 2                   # n_shards * num_envs
    assert len(traj) == len(traj.episodes) * traj.starts_per_ep


def test_window_indices_match_raw_shard_layout(tmp_path):
    """window_indices must address the same pixels as the raw shard reshaped to (F, B, C, H, W).

    That reshape is the ground truth for the frame-major/env-minor write order, so matching it
    proves the flat-index arithmetic (shard*per_shard + frame*B + env) is correct.
    """
    _, ds, raw = _build(tmp_path, num_envs=3, steps=8, stride=2, rounds=2)
    traj = TrajectoryDataset(ds, window=4, step=1)       # a full-length episode window (F=4)
    F, B = traj.F, traj.B
    assert (F, B) == (4, 3)

    # Ground-truth episodes: reshape each shard to (F, B, C, H, W); env e is a trajectory.
    truth = [r.reshape(F, B, *r.shape[1:]) for r in raw]

    for ep_id, (shard, env) in enumerate(traj.episodes):
        win_id = ep_id * traj.starts_per_ep              # the sole full-length window (f0 = 0)
        idx = traj.window_indices(win_id)
        # Each global index round-trips through FireDataset to that world's frame in time order.
        got = np.stack([to_numpy(ds[i]) for i in idx])   # (window, C, H, W)
        want = truth[shard][:, env]                       # (F, C, H, W) = env's whole episode
        np.testing.assert_array_equal(got, want)


def test_loader_yields_consecutive_frames_of_one_world(tmp_path):
    """TrajectoryBatchLoader batches (x_t, x_{t+1}) that are one world's adjacent stored frames."""
    _, ds, raw = _build(tmp_path, num_envs=3, steps=8, stride=2, rounds=2)
    traj = TrajectoryDataset(ds, window=2, step=1)
    F, B = traj.F, traj.B
    truth = [r.reshape(F, B, *r.shape[1:]) for r in raw]

    # No shuffle + drop_last=False -> deterministic full coverage in window-id order.
    all_ids = np.arange(len(traj))
    loader = TrajectoryBatchLoader(traj, all_ids, batch_size=5, shuffle=False, seed=0,
                                   drop_last=False)
    seen = 0
    for batch in loader:
        assert batch.shape[1:] == (2, ds.meta["num_channels"], 16, 16)
        for b in range(batch.shape[0]):
            ep_id, k = divmod(int(all_ids[seen + b]), traj.starts_per_ep)
            shard, env = traj.episodes[ep_id]
            np.testing.assert_array_equal(to_numpy(batch[b, 0]), truth[shard][k, env])
            np.testing.assert_array_equal(to_numpy(batch[b, 1]), truth[shard][k + 1, env])
        seen += batch.shape[0]
    assert seen == len(traj)                              # every window served exactly once


def test_split_is_episode_disjoint(tmp_path):
    """split_episodes puts a whole world entirely in train or val (no cross-split frame leak)."""
    _, ds, _ = _build(tmp_path, num_envs=4, steps=8, stride=2, rounds=3)
    traj = TrajectoryDataset(ds, window=2)
    train_w, val_w = split_episodes(traj, val_frac=0.25, seed=0)

    assert len(train_w) + len(val_w) == len(traj)        # a partition
    assert set(train_w).isdisjoint(val_w)

    # Map every window back to its owning episode; the two sides share no episode.
    def episodes_of(win_ids):
        return {int(w) // traj.starts_per_ep for w in win_ids}
    assert episodes_of(train_w).isdisjoint(episodes_of(val_w))
    assert len(episodes_of(train_w)) + len(episodes_of(val_w)) == len(traj.episodes)


def test_missing_num_envs_requires_explicit_arg(tmp_path):
    """A dataset whose meta predates the num_envs field must be given it explicitly."""
    _, ds, _ = _build(tmp_path, num_envs=2, steps=6, stride=2, rounds=1)
    ds.meta.pop("num_envs")                               # simulate a pre-upgrade dataset
    try:
        TrajectoryDataset(ds, window=2)
    except ValueError as exc:
        assert "num_envs" in str(exc)
    else:
        raise AssertionError("expected ValueError when num_envs is absent from meta")
    # Explicit override still works.
    traj = TrajectoryDataset(ds, num_envs=2, stride=2, window=2)
    assert traj.B == 2 and traj.F == 3
