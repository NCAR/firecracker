"""
Tests for the decode-supervised latent-rollout world model
(src/train_world_model_rollout.py, src/wm_losses.py, and the traj_loader partial-window mode).
"""

import sys

import numpy as np
import pytest
import torch

from conftest import make_config, build_map, pad_map
from config import boundary_pad
from map_loader import save_map
from rollout import BatchedRollout, FireDataset
from traj_loader import TrajectoryDataset, TrajectoryBatchLoader
import wm_losses as wml
import train_world_model_rollout as twmr

GRID = 16


def _build_dataset(tmp_path):
    config = make_config(grid_size=GRID)
    maps_dir = tmp_path / "maps"; maps_dir.mkdir()
    for i in range(4):
        save_map(maps_dir / f"m_{i}.npz", pad_map(build_map(grid_size=GRID), boundary_pad(config)))
    out = tmp_path / "data"
    BatchedRollout(config, maps_dir=str(maps_dir), grid_size=GRID, num_envs=2, steps=8, stride=2,
                   buffer_device="cpu", spawn_fire=True, seed=0).build_dataset(out, rounds=2)
    return out, FireDataset(out)


# --- traj_loader partial mode: backward compat + mask correctness ------------

def test_partial_windows_cover_tail_and_mask(tmp_path):
    _, ds = _build_dataset(tmp_path)
    full = TrajectoryDataset(ds, num_envs=2, window=3, step=1, partial=False)
    part = TrajectoryDataset(ds, num_envs=2, window=3, step=1, partial=True)
    F = full.F
    # partial allows more starts per episode (down to t0 + one target) than full windows.
    assert full.starts_per_ep == F - 2          # span = (3-1)*1
    assert part.starts_per_ep == F - 1          # min_span = step = 1
    assert part.starts_per_ep > full.starts_per_ep
    # a tail-starting partial window is padded: its mask has a False, and indices clamp (no OOB).
    last = part.starts_per_ep - 1               # start = F-2 within episode 0 -> only 2 real frames
    m = part.window_mask(last)
    assert m.tolist() == [True, True, False]
    idx = part.window_indices(last)
    assert all(0 <= i < len(ds) for i in idx)   # clamped, in range
    # full windows are always fully valid.
    assert part.window_mask(0).all()


def test_partial_off_is_backward_compatible(tmp_path):
    """With partial off, windows + the loader yield exactly the original behavior (bare tensor)."""
    _, ds = _build_dataset(tmp_path)
    full = TrajectoryDataset(ds, num_envs=2, window=3, step=1, partial=False)
    assert not getattr(full, "partial")
    for w in range(len(full)):
        assert full.window_mask(w).all()        # every element valid, no padding
    loader = TrajectoryBatchLoader(full, np.arange(len(full)), 4, shuffle=False, seed=0, drop_last=False)
    batch = next(iter(loader))
    assert isinstance(batch, torch.Tensor) and batch.shape[1] == 3   # NOT a (batch, mask) tuple


def test_partial_loader_yields_mask(tmp_path):
    _, ds = _build_dataset(tmp_path)
    part = TrajectoryDataset(ds, num_envs=2, window=3, step=1, partial=True)
    loader = TrajectoryBatchLoader(part, np.arange(len(part)), 4, shuffle=False, seed=0, drop_last=False)
    batch = next(iter(loader))
    assert isinstance(batch, tuple) and len(batch) == 2
    x, mask = batch
    assert x.shape[0] == mask.shape[0] and mask.shape[1] == 3 and mask.dtype == torch.bool


# --- loss: masking + fire weighting ----------------------------------------

def test_masked_weighted_bce_ignores_padded_steps():
    B, S, C, H, W = 2, 3, 4, 8, 8
    logits = torch.zeros(B, S, C, H, W, requires_grad=True)
    target = torch.rand(B, S, C, H, W)
    cw = torch.ones(C)
    full = torch.ones(B, S)
    part = full.clone(); part[:, 2] = 0                 # mask out the last step
    l_full = wml.masked_weighted_bce(logits, target, full, cw)
    l_part = wml.masked_weighted_bce(logits, target, part, cw)
    assert torch.isfinite(l_full) and torch.isfinite(l_part)
    # a fully-masked step must not contribute: grad on that step's logits is zero.
    l_part.backward()
    assert logits.grad[:, 2].abs().sum() == 0
    assert logits.grad[:, 0].abs().sum() > 0


def test_channel_gains_rescale_compressed_channel():
    names = ["air_temperature", "terrain"]
    frames = torch.zeros(50, 2, 4, 4)
    frames[:, 0] = 0.02                                  # a compressed "temperature" channel
    frames[:, 1] = torch.rand(50, 4, 4)                  # a well-spread channel
    gains = wml.compute_channel_gains(frames, names, pct=99.0, max_gain=25.0)
    assert gains[0] > 5.0                                # temperature gets a large gain
    assert abs(float(gains[1]) - 1.0) < 1e-6             # non-temperature untouched
    out = wml.apply_gains(frames, gains)
    assert out.max() <= 1.0 and out[:, 0].mean() > frames[:, 0].mean()


# --- change-supervision loss + persistence baseline ------------------------

def test_masked_change_bce_zero_when_perfect():
    """Decoding exactly to the target gives ~0 change loss regardless of how much changed."""
    B, S, C, H, W = 2, 3, 4, 8, 8
    target = (torch.rand(B, S, C, H, W) > 0.5).float()
    prev = (torch.rand(B, S, C, H, W) > 0.5).float()
    logits = torch.where(target > 0.5, 30.0, -30.0)      # sigmoid -> target
    mask = torch.ones(B, S)
    loss = wml.masked_change_bce(logits, target, prev, mask, torch.ones(C))
    assert float(loss) < 1e-3


def test_masked_change_bce_penalizes_persistence():
    """The anti-identity property: a persistence prediction (decode == prev) is maximally wrong on
    the changed cells, so its change loss must be large -- much larger than the perfect prediction."""
    B, S, C, H, W = 2, 3, 4, 8, 8
    target = (torch.rand(B, S, C, H, W) > 0.5).float()
    prev = 1.0 - target                                  # every cell changed (max change), prev != target
    mask = torch.ones(B, S)
    cw = torch.ones(C)
    persist_logits = torch.where(prev > 0.5, 30.0, -30.0)    # decode == prev (identity/persistence)
    perfect_logits = torch.where(target > 0.5, 30.0, -30.0)  # decode == target
    l_persist = wml.masked_change_bce(persist_logits, target, prev, mask, cw)
    l_perfect = wml.masked_change_bce(perfect_logits, target, prev, mask, cw)
    assert float(l_persist) > 10.0                       # BCE ~30 on every (all-changed) cell
    assert float(l_persist) > 100 * float(l_perfect + 1e-6)


def test_masked_change_bce_normalizes_by_changed_cells():
    """The mean is over changed cells only, so changing HOW MANY cells differ (at fixed per-cell
    error) must not scale the loss -- unlike a full-frame average that would shrink it."""
    B, S, C, H, W = 1, 1, 1, 8, 8
    logits = torch.zeros(B, S, C, H, W)                  # sigmoid = 0.5 everywhere -> BCE = ln2 on target=1
    mask = torch.ones(B, S)
    cw = torch.ones(C)
    few = torch.zeros(B, S, C, H, W); few[..., 0, :2] = 1.0      # 2 changed cells (target=1, prev=0)
    many = torch.zeros(B, S, C, H, W); many[..., :4, :] = 1.0    # 32 changed cells
    prev = torch.zeros(B, S, C, H, W)
    l_few = wml.masked_change_bce(logits, few, prev, mask, cw)
    l_many = wml.masked_change_bce(logits, many, prev, mask, cw)
    assert abs(float(l_few) - float(l_many)) < 1e-5      # both ~ln2, count-invariant


def test_masked_change_bce_ignores_padded_steps():
    B, S, C, H, W = 2, 3, 4, 8, 8
    target = (torch.rand(B, S, C, H, W) > 0.5).float()
    prev = 1.0 - target
    logits = torch.zeros(B, S, C, H, W, requires_grad=True)
    mask = torch.ones(B, S); mask[:, 2] = 0              # mask the last transition step
    loss = wml.masked_change_bce(logits, target, prev, mask, torch.ones(C))
    loss.backward()
    assert logits.grad[:, 2].abs().sum() == 0
    assert logits.grad[:, 0].abs().sum() > 0


def test_persistence_mae_matches_hand_computation():
    B, S, C, H, W = 1, 2, 2, 4, 4
    target = torch.zeros(B, S, C, H, W)
    prev = torch.zeros(B, S, C, H, W)
    target[0, 0, 0] = 1.0                                # channel 0: whole 4x4 differs by 1 on step 0
    mask = torch.ones(B, S)
    mae = wml.persistence_mae(target, prev, mask)
    # channel 0: 16 cells of |1| on step 0, 0 on step 1, averaged over 2*16 valid cells = 0.5
    assert torch.allclose(mae, torch.tensor([0.5, 0.0]), atol=1e-6)


def test_persistence_mae_respects_mask():
    B, S, C, H, W = 1, 2, 1, 4, 4
    target = torch.zeros(B, S, C, H, W); target[0, 1] = 1.0      # only the (to-be-masked) step differs
    prev = torch.zeros(B, S, C, H, W)
    mask = torch.ones(B, S); mask[:, 1] = 0
    mae = wml.persistence_mae(target, prev, mask)
    assert float(mae[0]) == 0.0                          # the only difference is on the masked step


# --- dynamics head norm option ---------------------------------------------

def test_dynamics_norm_option_builds_expected_layers():
    from dynamics import LatentTransition
    import torch.nn as nn
    bn = LatentTransition(32, depth=2, norm="batchnorm")
    ln = LatentTransition(32, depth=2, norm="layernorm")
    no = LatentTransition(32, depth=2, norm="none")
    assert bn.norm == "batchnorm" and ln.norm == "layernorm" and no.norm == "none"
    assert any(isinstance(m, nn.BatchNorm1d) for m in bn.net) and not any(isinstance(m, nn.LayerNorm) for m in bn.net)
    assert any(isinstance(m, nn.LayerNorm) for m in ln.net) and not any(isinstance(m, nn.BatchNorm1d) for m in ln.net)
    assert not any(isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)) for m in no.net)
    # default stays batchnorm so old checkpoints / the SPR trainer rebuild identically.
    assert LatentTransition(32, depth=2).norm == "batchnorm"
    with pytest.raises(ValueError):
        LatentTransition(32, depth=2, norm="groupnorm")


def test_layernorm_head_is_batch_independent():
    """LayerNorm normalizes per-sample, so a row's output must not depend on the rest of the batch --
    the property BatchNorm lacks and the reason it's preferred for the recurrently-applied head."""
    from dynamics import LatentTransition
    ln = LatentTransition(16, depth=2, norm="layernorm", normalize_output=False).eval()
    z = torch.randn(8, 16)
    out_full = ln(z)
    out_solo = ln(z[:1])                                 # same first row, alone
    assert torch.allclose(out_full[:1], out_solo, atol=1e-5)


# --- trainer smoke + gradient path -----------------------------------------

def test_rollout_trainer_smoke(tmp_path, monkeypatch):
    data_root, _ = _build_dataset(tmp_path)
    out = tmp_path / "wm_roll"
    argv = ["train_world_model_rollout.py", "--data", str(data_root),
            "--channels", "8,16", "--latent-dim", "32",
            "--window", "3", "--step", "1", "--val-frac", "0.25",
            "--epochs", "2", "--batch-size", "4", "--num-envs", "2", "--out", str(out)]
    monkeypatch.setattr(sys, "argv", argv)
    twmr.main()
    assert (out / "metrics.csv").is_file() and (out / "best.pt").is_file() and (out / "last.pt").is_file()
    lines = (out / "metrics.csv").read_text().strip().splitlines()
    assert len(lines) == 3                               # header + 2 epochs
    ck = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
    assert ck["regime"] == "decode_rollout"
    assert ck["dynamics_config"]["residual"] is True and ck["dynamics_config"]["normalize_output"] is False
    assert ck["dynamics_config"]["norm"] == "layernorm"          # new rollout runs default to LayerNorm
    assert ck["model_config"]["normalize_latent"] is False and ck["model_config"]["latent_bn"] is True
    assert "channel_gains" in ck["model_config"]
    assert "optimizer_state" in ck                       # resumable


def test_rollout_trainer_two_term_loss(tmp_path, monkeypatch):
    """The --w-change>0 decomposition runs end-to-end, writes the new anchor/change + static/dynamic
    columns, and records the two-term recipe in loss_config."""
    data_root, _ = _build_dataset(tmp_path)
    out = tmp_path / "wm_roll_2term"
    argv = ["train_world_model_rollout.py", "--data", str(data_root),
            "--channels", "8,16", "--latent-dim", "32",
            "--window", "3", "--step", "1", "--val-frac", "0.25",
            "--w-change", "1.0", "--w-anchor", "0.2", "--change-eps", "1e-3",
            "--epochs", "2", "--batch-size", "4", "--num-envs", "2", "--out", str(out)]
    monkeypatch.setattr(sys, "argv", argv)
    twmr.main()
    header = (out / "metrics.csv").read_text().splitlines()[0].split(",")
    for col in ("train_anchor", "train_change", "val_anchor", "val_change",
                "val_static_mae", "val_dynamic_mae", "persist_static_mae", "persist_dynamic_mae"):
        assert col in header
    ck = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
    lc = ck["loss_config"]
    assert lc["w_change"] == 1.0 and lc["w_anchor"] == 0.2 and lc["change_eps"] == 1e-3
    assert lc["anchor_front"] == 0.0            # w_change>0 -> front emphasis moves to the change term


def test_rollout_gradient_reaches_encoder_decoder_dynamics(tmp_path):
    """The single reconstruction loss must reach the encoder, the decoder, AND the dynamics head."""
    _, ds = _build_dataset(tmp_path)
    from strided_autoencoder import ConvAutoencoder
    from dynamics import LatentTransition
    model = ConvAutoencoder(in_channels=ds.meta["num_channels"], grid_size=GRID, channels=(8, 16),
                            latent_dim=32, normalize_latent=False, latent_bn=True).float()
    dyn = LatentTransition(32, depth=2, residual=True, normalize_output=False).float()
    part = TrajectoryDataset(ds, num_envs=2, window=3, step=1, partial=True)
    loader = TrajectoryBatchLoader(part, np.arange(len(part)), 4, shuffle=False, seed=0, drop_last=True)
    x, mask = next(iter(loader))
    logits = twmr.rollout_logits(model, dyn, x[:, 0].float(), x.shape[1], freeze_encoder=False)
    cw = torch.ones(ds.meta["num_channels"])
    loss = wml.masked_weighted_bce(logits, x.float(), mask, cw)
    loss.backward()
    g = lambda m: sum(p.grad.abs().sum() for p in m.parameters() if p.grad is not None)
    assert g(model.encoder) > 0 and g(model.decoder) > 0 and g(dyn) > 0
