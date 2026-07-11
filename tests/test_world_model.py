"""
Latent world model: dynamics head + trainer (src/dynamics.py, src/train_world_model.py).

Builds a tiny trajectory dataset and pins down that:

  * LatentTransition predicts on the unit sphere (residual, re-normalized), and the cosine
    prediction loss drives down when the head is overfit to fixed latent targets (frozen AE),
  * --freeze-ae updates only the dynamics head, leaving the autoencoder weights untouched,
  * build_autoencoder with --init-from restores the exact autoencoder weights from a checkpoint,
  * a full main() smoke run over the tiny dataset writes metrics.csv + checkpoints.
"""

import argparse
import sys

import numpy as np
import torch

from conftest import make_config, build_map, pad_map
from config import boundary_pad, load_config
from map_loader import save_map
from rollout import BatchedRollout, FireDataset
from strided_autoencoder import ConvAutoencoder
from dynamics import LatentTransition
from traj_loader import TrajectoryDataset, TrajectoryBatchLoader, split_episodes
import train_world_model as twm


GRID = 16
CHANNELS = (8, 16)          # 2 pooling stages -> 16 / 2**2 = 4, fits GRID=16
LATENT = 32


def _build_dataset(tmp_path):
    """Build a tiny normalized-shape dataset and return (root, FireDataset)."""
    config = make_config(grid_size=GRID)
    maps_dir = tmp_path / "maps"
    maps_dir.mkdir()
    for i in range(4):
        save_map(maps_dir / f"m_{i}.npz", pad_map(build_map(grid_size=GRID), boundary_pad(config)))
    out = tmp_path / "data"
    BatchedRollout(
        config, maps_dir=str(maps_dir), grid_size=GRID,
        num_envs=2, steps=8, stride=2, buffer_device="cpu", spawn_fire=True, seed=0,
    ).build_dataset(out, rounds=2)
    return out, FireDataset(out)


def _ae(in_channels):
    return ConvAutoencoder(in_channels=in_channels, grid_size=GRID,
                           channels=CHANNELS, latent_dim=LATENT)


def test_latent_transition_predicts_on_unit_sphere():
    head = LatentTransition(LATENT, depth=2)
    z = torch.nn.functional.normalize(torch.randn(5, LATENT), dim=-1)
    z_hat = head(z)
    assert z_hat.shape == (5, LATENT)
    norms = z_hat.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_head_overfits_fixed_targets(tmp_path):
    """With the AE frozen, the head must drive the cosine prediction loss down on fixed targets."""
    _, ds = _build_dataset(tmp_path)
    model = _ae(ds.meta["num_channels"])
    for p in model.parameters():
        p.requires_grad_(False)
    head = LatentTransition(LATENT, depth=2)
    traj = TrajectoryDataset(ds, window=2)
    train_w, _ = split_episodes(traj, val_frac=0.25, seed=0)
    loader = TrajectoryBatchLoader(traj, train_w, batch_size=4, shuffle=True, seed=0, drop_last=True)
    opt = torch.optim.Adam(head.parameters(), lr=1e-2)
    device = torch.device("cpu")

    first = twm.run_epoch(model, head, loader, device, opt, None, torch.nn.MSELoss(),
                          1.0, 1.0, True, 1.0, 1, "train", 0)[1]     # pred component
    for epoch in range(2, 61):
        last = twm.run_epoch(model, head, loader, device, opt, None, torch.nn.MSELoss(),
                             1.0, 1.0, True, 1.0, epoch, "train", 0)[1]
    assert last < first * 0.5, f"pred loss did not fall: {first:.4f} -> {last:.4f}"
    assert last < 0.3


def test_freeze_ae_updates_only_head(tmp_path):
    """--freeze-ae must leave every autoencoder weight unchanged and move the head's weights."""
    _, ds = _build_dataset(tmp_path)
    model = _ae(ds.meta["num_channels"])
    for p in model.parameters():
        p.requires_grad_(False)
    head = LatentTransition(LATENT, depth=2)
    ae_before = [p.detach().clone() for p in model.parameters()]
    head_before = [p.detach().clone() for p in head.parameters()]

    traj = TrajectoryDataset(ds, window=2)
    train_w, _ = split_episodes(traj, val_frac=0.25, seed=0)
    loader = TrajectoryBatchLoader(traj, train_w, batch_size=4, shuffle=True, seed=0, drop_last=True)
    opt = torch.optim.Adam(twm._trained_params(model, head, freeze_ae=True), lr=1e-2)
    _, pred, rec, _, _ = twm.run_epoch(model, head, loader, torch.device("cpu"), opt, None,
                                       torch.nn.MSELoss(), 1.0, 1.0, True, 1.0, 1, "train", 0)

    assert rec == 0.0                                   # reconstruction dropped when the AE is frozen
    for a, b in zip(model.parameters(), ae_before):
        assert torch.equal(a, b)                        # AE weights frozen
    assert any(not torch.equal(a, b) for a, b in zip(head.parameters(), head_before))  # head moved


def test_init_from_restores_ae_weights(tmp_path):
    """build_autoencoder(--init-from) rebuilds the arch from the checkpoint and loads its weights."""
    data_root, ds = _build_dataset(tmp_path)
    in_ch = ds.meta["num_channels"]
    ae = _ae(in_ch)
    ckpt_path = tmp_path / "ae.pt"
    model_config = {
        "arch": "pooled", "in_channels": in_ch, "grid_size": GRID,
        "channels": list(CHANNELS), "latent_dim": LATENT,
        "normalize_latent": True, "bottleneck_channels": None, "weight_dtype": "float32",
    }
    torch.save({"model_state": ae.state_dict(), "model_config": model_config, "epoch": 1}, ckpt_path)

    args = argparse.Namespace(init_from=str(ckpt_path), data=str(data_root))
    loaded, mc, _stats = twm.build_autoencoder(args, {}, torch.device("cpu"), torch.float32)
    assert mc["channels"] == list(CHANNELS) and mc["latent_dim"] == LATENT
    for a, b in zip(loaded.parameters(), ae.parameters()):
        assert torch.equal(a, b)


def test_main_smoke_freeze(tmp_path, monkeypatch):
    """End-to-end main() run: pretrained-init + freeze-ae writes metrics.csv and checkpoints."""
    data_root, ds = _build_dataset(tmp_path)
    in_ch = ds.meta["num_channels"]
    ae = _ae(in_ch)
    ckpt_path = tmp_path / "ae.pt"
    model_config = {
        "arch": "pooled", "in_channels": in_ch, "grid_size": GRID,
        "channels": list(CHANNELS), "latent_dim": LATENT,
        "normalize_latent": True, "bottleneck_channels": None, "weight_dtype": "float32",
    }
    torch.save({"model_state": ae.state_dict(), "model_config": model_config, "epoch": 1}, ckpt_path)

    out = tmp_path / "wm"
    argv = [
        "train_world_model.py",
        "--data", str(data_root),
        "--init-from", str(ckpt_path),
        "--freeze-ae",
        "--window", "2",
        "--val-frac", "0.25",
        "--epochs", "2",
        "--batch-size", "4",
        "--out", str(out),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    twm.main()

    assert (out / "metrics.csv").is_file()
    assert (out / "best.pt").is_file()
    assert (out / "last.pt").is_file()
    lines = (out / "metrics.csv").read_text().strip().splitlines()
    assert len(lines) == 3                              # header + 2 epochs
    ckpt = torch.load(out / "best.pt", map_location="cpu")
    assert ckpt["freeze_ae"] is True
    assert "dynamics_state" in ckpt and "model_state" in ckpt
