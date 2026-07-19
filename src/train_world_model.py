"""
train_world_model.py

Train a latent world model on Firecracker rollout trajectories: an autoencoder
(strided_autoencoder.ConvAutoencoder) plus a latent transition head (dynamics.LatentTransition)
that predicts the next latent from the current one.

Data are drawn as short frame windows over episodes (traj_loader.TrajectoryDataset), each window
a run of consecutive stored frames from one world. For a window (x_t, x_{t+1}) the loss splits into
two DISJOINT gradient paths through a single shared online latent z_t (BYOL/SPR-style asymmetry):

  z_t          = encode(x_t)                       # ONLINE encoder, unit-norm (normalize_latent=True)
  z_hat        = transition(z_t)                   # predicted next latent, re-normalized to the sphere
  z_{t+1}^ema  = ema_encode(x_{t+1})               # EMA (momentum) encoder, stop-grad, unit-norm

  reconstruction (anchor)  = loss_fn(decode(z_t), x_t)              # decoder -> encoder; NOT the head
  prediction (self-pred)   = 1 - cos(z_hat, stopgrad(z_{t+1}^ema)) # head -> encoder; NOT the decoder

The second frame x_{t+1} is NEVER decoded: its only role is as the prediction target, encoded by an
exponential-moving-average copy of the encoder (updated after each step, no gradient). Gradients
therefore flow along exactly two paths that meet only at the encoder: the reconstruction path
(decoder->encoder) and the self-predictive path (dynamics head->encoder). The EMA target -- rather
than the online encoder's own stop-grad output -- decouples the target from the fast-moving online
weights and stabilizes training, guarding against representational collapse (together with the L2
normalization and the decoder anchor; no variance/covariance VICReg term is needed).

Two modes, selected by --init-from:
  * from scratch     -- build a fresh autoencoder from config/CLI and train encoder+decoder+head
                        end to end (reconstruction + prediction).
  * pretrained init  -- load an autoencoder checkpoint (from train_autoencoder.py) and continue.
                        Add --freeze-ae to hold the autoencoder fixed and train only the dynamics
                        head (reconstruction is then constant and dropped).

Example:

    # from a pretrained autoencoder, train just the dynamics head
    python src/train_world_model.py --data data/fire --init-from runs/ae/best.pt \
        --freeze-ae --epochs 30 --out runs/wm

    # end to end from scratch
    python src/train_world_model.py --data data/fire --epochs 50 --out runs/wm
"""

from __future__ import annotations

import argparse
import copy
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import wandb
except ImportError:                          # --wandb then errors out with an install hint
    wandb = None

from config import load_config
from rollout import FireDataset
from ram_loader import prewarm_shards
from traj_loader import TrajectoryDataset, TrajectoryBatchLoader, split_episodes
from dynamics import LatentTransition, RewardHead
# Reuse the autoencoder trainer's device/dtype/stats helpers and its architecture registry so the
# two entrypoints stay in lockstep (same arch names, same weight-dtype handling) with no copy.
from train_autoencoder import (
    ARCHITECTURES,
    WEIGHT_DTYPES,
    _resolve_device,
    _resolve_weight_dtype,
    _load_channel_stats,
    per_block_grad_norms,
)


def cosine_prediction_loss(z_hat: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
    """1 - cosine similarity between the prediction and the (stop-grad) target latent.

    Both are expected to be unit-norm; the stop-gradient on the target is applied by the caller.
    """
    return (1.0 - F.cosine_similarity(z_hat, z_target, dim=-1)).mean()


@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    """In-place EMA update of the target encoder: theta_ema <- decay*theta_ema + (1-decay)*theta_online.

    Parameters are momentum-blended; buffers (BatchNorm running stats, num_batches_tracked) are copied
    straight across, keeping the target's normalization synced to the online encoder without a lerp on
    the integer counter. Runs under no_grad -- the EMA weights never receive gradients.
    """
    for pe, pm in zip(ema_model.parameters(), model.parameters()):
        pe.mul_(decay).add_(pm.detach(), alpha=1.0 - decay)
    for be, bm in zip(ema_model.buffers(), model.buffers()):
        be.copy_(bm)


def build_autoencoder(args, config, device, weight_dtype):
    """Construct the autoencoder, either fresh from config/CLI or loaded from an --init-from ckpt.

    Returns (model, model_config, channel_stats). When loading, the architecture is rebuilt from
    the checkpoint's own model_config so it always matches the saved weights.
    """
    if args.init_from is not None:
        ckpt = torch.load(args.init_from, map_location=device)
        mc = dict(ckpt["model_config"])
        arch = mc["arch"]
        kwargs = {
            "in_channels": mc["in_channels"],
            "grid_size": mc["grid_size"],
            "channels": tuple(mc["channels"]),
            "latent_dim": mc["latent_dim"],
            "normalize_latent": mc["normalize_latent"],
        }
        if mc.get("bottleneck_channels") is not None:
            kwargs["bottleneck_channels"] = mc["bottleneck_channels"]
        if mc.get("blocks_per_stage") is not None:
            kwargs["blocks_per_stage"] = tuple(mc["blocks_per_stage"])
        if mc.get("latent_bn"):
            kwargs["latent_bn"] = True
        model = ARCHITECTURES[arch](**kwargs).to(device=device, dtype=weight_dtype)
        model.load_state_dict(ckpt["model_state"])
        # The checkpoint may be bf16 (that's what ships in models/); building at weight_dtype before
        # load_state_dict upcasts the weights to that dtype (bf16->f32 is lossless). Record the dtype
        # the model actually runs in, not the source checkpoint's, so a later rebuild matches.
        mc["weight_dtype"] = str(weight_dtype).removeprefix("torch.")
        print(f"init: loaded autoencoder from '{args.init_from}' (arch={arch}, "
              f"src dtype {ckpt['model_config'].get('weight_dtype', '?')} -> "
              f"{mc['weight_dtype']})")
        # Prefer the stats the AE was trained under (carried in its config); fall back to the data.
        # Min-max datasets carry channel_min/channel_max; legacy z-scored ones channel_mean/std.
        stats = None
        if "channel_min" in mc and "channel_max" in mc:
            stats = {"channel_min": mc["channel_min"], "channel_max": mc["channel_max"]}
            # Carry the signed-channel marking + names so the symmetric [-1, 1] wind transform is
            # reproduced (stats_affine locates signed channels by name via channel_names).
            for k in ("signed_channels", "channel_names"):
                if k in mc:
                    stats[k] = mc[k]
        elif "channel_mean" in mc and "channel_std" in mc:
            stats = {"channel_mean": mc["channel_mean"], "channel_std": mc["channel_std"]}
        return model, mc, stats

    # From scratch: resolve the architecture from the [autoencoder] config, CLI overriding.
    ae_cfg = config.get("autoencoder", {})
    arch = args.arch or ae_cfg.get("arch", "pooled")
    arch_cfg = ae_cfg.get(arch, {})
    in_channels = int(args._in_channels)
    grid_size = int(args._grid_size)
    kwargs = {"in_channels": in_channels, "grid_size": grid_size}
    channels = args.channels if args.channels is not None else arch_cfg.get("channels")
    if channels is not None:
        kwargs["channels"] = tuple(channels)
    latent_dim = args.latent_dim if args.latent_dim is not None else arch_cfg.get("latent_dim")
    if latent_dim is not None:
        kwargs["latent_dim"] = latent_dim
    if arch_cfg.get("normalize_latent") is not None:
        kwargs["normalize_latent"] = arch_cfg["normalize_latent"]
    if arch_cfg.get("bottleneck_channels") is not None:
        kwargs["bottleneck_channels"] = arch_cfg["bottleneck_channels"]
    # blocks_per_stage is tied to the config's `channels`; skip it if --channels overrode the width
    # list to a different length (stale, would mismatch the stage count).
    cfg_bps = arch_cfg.get("blocks_per_stage")
    if cfg_bps is not None and (channels is None or len(cfg_bps) == len(tuple(channels))):
        kwargs["blocks_per_stage"] = tuple(cfg_bps)
    if args.latent_bn or arch_cfg.get("latent_bn"):
        kwargs["latent_bn"] = True
    model = ARCHITECTURES[arch](**kwargs).to(device=device, dtype=weight_dtype)
    model_config = {
        "arch": arch,
        "in_channels": in_channels,
        "grid_size": grid_size,
        "channels": list(model.channels),
        "blocks_per_stage": list(model.blocks_per_stage),
        "latent_dim": model.latent_dim,
        "normalize_latent": model.normalize_latent,
        "bottleneck_channels": model.bottleneck_channels,
        "latent_bn": model.latent_bn,
        "weight_dtype": str(weight_dtype).removeprefix("torch."),
    }
    return model, model_config, _load_channel_stats(args.data)


def covariance_loss(z):
    """VICReg off-diagonal covariance penalty: push cross-dimension covariances of the latent to
    zero. BatchNorm(affine=False) already fixes per-dim variance, so a low participation ratio is
    dim *correlation* (variance piled on a few axes); decorrelating raises the latent's effective rank."""
    z = z - z.mean(0, keepdim=True)
    n, d = z.shape
    cov = (z.T @ z) / (n - 1)
    off_sq = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
    return off_sq / d


def _newly_ignited(x_prev, x_cur, ignited_channel, thr=0.5):
    """Count cells that ignite between two frames: ignited in x_cur but not in x_prev. Returns (B,)
    float counts (the normalized 'ignited' mask is ~binary, so a 0.5 threshold reads it cleanly)."""
    prev = x_prev[:, ignited_channel] > thr
    cur = x_cur[:, ignited_channel] > thr
    return (cur & ~prev).flatten(1).sum(dim=1).float()


def run_epoch(
    model, transition, ema_model, loader, device, optimizer, scaler,
    rec_loss_fn, pred_weight, rec_weight, freeze_ae, ema_decay, max_grad_norm, epoch, tag, log_interval,
    out_activation="none", cov_weight=0.0, rollout_steps=1,
    reward_head=None, ignited_channel=None, reward_weight=0.0, pred_discount=1.0,
):
    """One pass over `loader`. Trains when `optimizer` is given, else evaluates (no grad).

    The next-frame target latent comes from `ema_model` (the momentum encoder, stop-grad); when
    training and not frozen, `ema_model` is EMA-updated toward `model` after each optimizer step.
    For a frozen AE `ema_model` is None and the (fixed) online encoder supplies the target directly.

    Three disjoint losses meet only at the encoder: (1) self-prediction unrolls `transition` K steps
    from the root latent z0 against stop-grad EMA targets; (2) reconstruction decodes ONE random
    window frame per sample via its own independent encode (never through the heads); (3) the reward
    head predicts the one-step newly-ignited count from each rolled-out latent (grad -> dynamics +
    encoder, never the decoder). Reconstruction/reward are skipped when their component is absent.

    Returns (mean_total, mean_pred, mean_rec, mean_mae, grad_norm_mean, grad_norm_max, mean_reward).
    `mean_mae` is a loss-independent reconstruction MAE yardstick over the reconstructed frame
    (sigmoid applied for the bce head so arms are comparable); it and mean_rec are 0 for a frozen AE.
    """
    training = optimizer is not None
    transition.train(training)
    if reward_head is not None:
        reward_head.train(training)
    model.train(training and not freeze_ae)                # a frozen AE stays in eval (BN uses running stats)
    use_amp = scaler is not None and device.type == "cuda"
    param_dtype = next(model.parameters()).dtype

    tot = tot_pred = tot_rec = tot_mae = tot_reward = 0.0
    seen = 0
    gnorm_sum = 0.0
    gnorm_max = 0.0
    gnorm_steps = 0
    per_step_sum = None      # running n-weighted sum of the per-step cosine loss (len = K); the drift curve
    grad_ctx = torch.enable_grad() if training else torch.no_grad()
    with grad_ctx:
        for step, batch in enumerate(loader):
            # (B, window, C, H, W). Frame k (0..window-1) is t + k*step; we unroll K=rollout_steps
            # of them and reconstruct every frame we touch.
            x = batch.to(device=device, dtype=param_dtype, non_blocking=True)
            K = min(rollout_steps, x.shape[1] - 1)                  # rollout length, capped by the window
            if training:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                # --- PREDICTION + REWARD: unroll the dynamics head K times from the root latent z0. The
                # head consumes its OWN previous output (a true rollout, so error compounds and persistence
                # is penalized at longer horizons); each step is scored against the stop-grad EMA target of
                # the real future frame, and the reward head predicts that step's newly-ignited count from
                # the rolled-out latent. Gradient reaches the encoder ONLY through z0; future frames are
                # encoded under no_grad and never decoded here, and reward targets are counts from the data
                # (no grad). A frozen AE needs no encoder graph, so z0 is encoded under no_grad too.
                enc_ctx = torch.no_grad() if freeze_ae else torch.enable_grad()
                with enc_ctx:
                    z0 = model.encode(x[:, 0])
                target_encoder = model if ema_model is None else ema_model
                z_hat = z0
                pred = 0.0
                step_losses = []                                   # per-step cosine, for the drift breakdown
                reward = torch.zeros((), device=device)
                # Horizon discount: weight training step k by gamma^(k-1) so the well-conditioned near-step
                # gradient isn't swamped by far-step BPTT noise (flat-averaged long unrolls fail to optimize;
                # see the K=8 plateau). gamma=1 -> flat mean (unchanged). Applied to TRAINING only: validation
                # keeps the flat mean so val_pred stays a fixed, comparable metric across every K arm.
                gamma = pred_discount if training else 1.0
                wsum = 0.0
                for k in range(1, K + 1):
                    z_hat = transition(z_hat)                       # unit-norm prediction, fed back in
                    with torch.no_grad():
                        z_tgt = target_encoder.encode(x[:, k])     # EMA, stop-grad; x[:,k] not decoded here
                    step_loss = cosine_prediction_loss(z_hat, z_tgt)
                    w = gamma ** (k - 1)
                    pred = pred + w * step_loss
                    wsum += w
                    step_losses.append(step_loss)                  # RAW per-step (undiscounted) for the drift curve
                    if reward_head is not None:
                        r_hat = reward_head(z_hat)                  # reward of transition (k-1 -> k)
                        with torch.no_grad():
                            r_tgt = torch.log1p(_newly_ignited(x[:, k - 1], x[:, k], ignited_channel))
                        reward = reward + F.smooth_l1_loss(r_hat, r_tgt)
                pred = pred / wsum
                if reward_head is not None:
                    reward = reward / K

                # --- RECONSTRUCTION anchor on ONE RANDOM window frame per sample, via its OWN encode->decode.
                # A fresh independent encode (not z0, not the EMA target) means the recon graph runs
                # decoder->encoder only and NEVER touches `transition` -- the two losses stay on disjoint
                # paths. Because the frame index is uniform over 0..K across steps, every position (incl.
                # the last-K*step episode tail that is otherwise only ever a rollout target) is grounded in
                # expectation, at a flat one decode per step.
                if freeze_ae:
                    rec = torch.zeros((), device=device)
                    mae = torch.zeros((), device=device)
                else:
                    b = x.shape[0]
                    j = torch.randint(0, K + 1, (b,), device=x.device)
                    x_rec = x[torch.arange(b, device=x.device), j]  # (B,C,H,W): a random frame per sample
                    z_rec = model.encode(x_rec)                     # independent encode; grad -> encoder only
                    dec = model.decode(z_rec)
                    rec = rec_loss_fn(dec, x_rec)
                    # Loss-independent reconstruction MAE (sigmoid for the logits/bce head), a yardstick
                    # comparable across arms regardless of the rec loss.
                    with torch.no_grad():
                        rt = dec.sigmoid() if out_activation == "sigmoid" else dec
                        mae = F.l1_loss(rt, x_rec)

                loss = pred_weight * pred + rec_weight * rec
                if reward_head is not None and reward_weight > 0:
                    loss = loss + reward_weight * reward
                if training and not freeze_ae and cov_weight > 0:
                    loss = loss + cov_weight * covariance_loss(z0)

            if training:
                gnorm_val = 0.0
                if use_amp:
                    scaler.scale(loss).backward()
                    if max_grad_norm > 0:
                        scaler.unscale_(optimizer)
                        gnorm_val = float(torch.nn.utils.clip_grad_norm_(_trained_params(model, transition, freeze_ae, reward_head), max_grad_norm))
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if max_grad_norm > 0:
                        gnorm_val = float(torch.nn.utils.clip_grad_norm_(_trained_params(model, transition, freeze_ae, reward_head), max_grad_norm))
                    optimizer.step()
                # After the online weights move, pull the EMA target encoder toward them.
                if ema_model is not None:
                    update_ema(ema_model, model, ema_decay)
                if max_grad_norm > 0 and np.isfinite(gnorm_val):
                    gnorm_sum += gnorm_val
                    gnorm_max = max(gnorm_max, gnorm_val)
                    gnorm_steps += 1

            n = x.shape[0]
            tot += loss.item() * n
            tot_pred += pred.item() * n
            tot_rec += rec.item() * n
            tot_mae += mae.item() * n
            tot_reward += reward.item() * n
            if step_losses:
                if per_step_sum is None:
                    per_step_sum = [0.0] * len(step_losses)
                for i, sl in enumerate(step_losses):
                    per_step_sum[i] += sl.item() * n
            seen += n
            if training and log_interval and step % log_interval == 0:
                print(f"  epoch {epoch:3d} [{tag}] step {step:5d}/{len(loader)}  "
                      f"loss {loss.item():.6f}  pred {pred.item():.6f}  rec {rec.item():.6f}")

    denom = max(seen, 1)
    per_step = [s / denom for s in per_step_sum] if per_step_sum is not None else []
    return (tot / denom, tot_pred / denom, tot_rec / denom, tot_mae / denom,
            gnorm_sum / max(gnorm_steps, 1), gnorm_max, tot_reward / denom, per_step)


def _trained_params(model, transition, freeze_ae, reward_head=None):
    """The parameters the optimizer updates: the dynamics head always, the AE unless frozen, and the
    reward head when present."""
    params = list(transition.parameters())
    if not freeze_ae:
        params = list(model.parameters()) + params
    if reward_head is not None:
        params += list(reward_head.parameters())
    return params


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a latent world model (autoencoder + dynamics head) on Firecracker trajectories."
    )
    # Data / trajectory windowing
    parser.add_argument("--data", default="data/fire", help="dataset dir (shards + meta.json)")
    parser.add_argument("--config", metavar="PATH", help="path to a TOML config file")
    parser.add_argument("--num-envs", type=int, default=None,
                        help="envs-per-shard override for datasets whose meta predates the field")
    parser.add_argument("--window", type=int, default=2,
                        help="frames per trajectory window (v1 trains on the first (t, t+1) pair)")
    parser.add_argument("--step", type=int, default=1, help="frame gap within a window")
    parser.add_argument("--rollout-steps", type=int, default=1,
                        help="dynamics-head unroll length K (needs window >= K+1); the head is fed its "
                             "own output for K steps and every one of the K+1 frames is reconstructed")
    parser.add_argument("--val-rollout-steps", type=int, default=None,
                        help="unroll length used for VALIDATION only (default: same as --rollout-steps). "
                             "Decouple it (e.g. a fixed 8) so runs trained at different --rollout-steps K are "
                             "compared on the SAME long-horizon rollout; needs window >= this + 1. The per-step "
                             "cosine (val_pred_step_NN) is logged so the drift curve is visible.")
    parser.add_argument("--horizon-discount", type=float, default=1.0,
                        help="discount gamma for the multi-step prediction loss: training step k is weighted "
                             "gamma^(k-1) (loss normalized by the weight sum). 1.0 = flat mean (default). "
                             "<1 down-weights far rollout steps so the well-conditioned near-step gradient "
                             "dominates -- fixes the flat-averaged long-unroll (large-K) optimization plateau. "
                             "TRAINING only; validation stays a flat mean so val_pred is comparable across K.")
    parser.add_argument("--val-frac", type=float, default=0.05, help="fraction of episodes held out")
    parser.add_argument("--prewarm", action="store_true", help="stream shards into page cache first")
    # Model
    parser.add_argument("--init-from", metavar="CKPT", default=None,
                        help="autoencoder checkpoint to initialize from; omit to train from scratch")
    parser.add_argument("--freeze-ae", action="store_true",
                        help="hold the (pretrained) autoencoder fixed and train only the dynamics head")
    parser.add_argument("--arch", default=None, help="autoencoder arch (from-scratch only)")
    parser.add_argument("--channels", type=lambda s: [int(p) for p in s.split(",") if p.strip()],
                        default=None, help="encoder channel widths, e.g. 32,64,128 (from-scratch only)")
    parser.add_argument("--latent-dim", type=int, default=None, help="latent width (from-scratch only)")
    parser.add_argument("--latent-bn", action="store_true",
                        help="BatchNorm(affine=False) on the latent before L2-norm (anti-collapse)")
    parser.add_argument("--dyn-depth", type=int, default=2, help="dynamics head hidden layers")
    parser.add_argument("--dyn-hidden", type=int, default=None,
                        help="dynamics head hidden width (default 1024)")
    parser.add_argument("--dyn-norm", choices=("batchnorm", "layernorm", "none"), default="batchnorm",
                        help="dynamics-head hidden-layer normalization. BatchNorm (default, historical) "
                             "blends running stats across rollout depths when the head is unrolled "
                             "(train/eval mismatch at K>1); LayerNorm is per-sample and depth-invariant, "
                             "so it is the right choice for multi-step rollouts and for A/Bs that vary K.")
    parser.add_argument("--reward-weight", type=float, default=0.0,
                        help="weight on the one-step reward loss (0 = no reward head); the reward is the "
                             "newly-ignited-cell count, predicted from each rolled-out latent")
    parser.add_argument("--reward-depth", type=int, default=2, help="reward head hidden layers")
    parser.add_argument("--reward-hidden", type=int, default=None,
                        help="reward head hidden width (default 256)")
    parser.add_argument("--ema-decay", type=float, default=0.996,
                        help="momentum for the EMA target encoder that supplies the prediction target "
                             "(theta_ema <- decay*theta_ema + (1-decay)*theta_online each step). "
                             "Ignored with --freeze-ae (the fixed encoder is its own target).")
    # Loss weights
    parser.add_argument("--pred-weight", type=float, default=2.0, help="weight on the prediction loss")
    parser.add_argument("--rec-weight", type=float, default=0.2, help="weight on the reconstruction loss")
    parser.add_argument("--cov-weight", type=float, default=0.0,
                        help="weight on the VICReg covariance/decorrelation penalty (0 = off)")
    parser.add_argument("--rec-loss", choices=("mse", "huber", "bce"), default="mse",
                        help="reconstruction loss: mse | huber | bce. 'bce' uses BCEWithLogitsLoss "
                             "(decoder emits logits; sigmoid is applied only for reconstructions), so "
                             "pair it with --out-activation sigmoid on the min-max [0,1] data.")
    parser.add_argument("--huber-delta", type=float, default=0.1,
                        help="Huber delta when --rec-loss huber. Inputs are min-max scaled to [0,1], "
                             "so a small delta (~0.1) is where robustness engages (default: 0.1).")
    parser.add_argument("--out-activation", choices=("none", "sigmoid"), default="none",
                        help="decoder output activation used when forming reconstructions: 'none' "
                             "(raw linear head) or 'sigmoid' (pair with --rec-loss bce). Recorded in "
                             "the checkpoint so eval/viewers reconstruct the right values.")
    # Optimization
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--adam-eps", type=float, default=1e-3)
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="global grad-norm clip (0 = off)")
    parser.add_argument("--amp", action="store_true", help="mixed precision (float32 master weights)")
    parser.add_argument("--weight-dtype", choices=tuple(WEIGHT_DTYPES), default="float32")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=0)
    parser.add_argument("--out", default="runs/world_model", help="output dir for checkpoints + metrics")
    parser.add_argument("--resume", default=None,
                        help="path to a world-model checkpoint (last.pt) to resume from: restores the "
                             "autoencoder, dynamics head, reward head, EMA target encoder, optimizer, "
                             "GradScaler, best-pred, and continues at the next epoch. Architecture must "
                             "match (validated). metrics.csv is appended to rather than truncated. "
                             "Mutually exclusive with --init-from")
    parser.add_argument("--save-interval", type=int, default=0,
                        help="also write epoch_NNN.pt every N epochs (0 = only best.pt + last.pt)")
    # Weights & Biases (matches the autoencoder trainer's flags so tools/sweep.py drives both the
    # same way: it injects --wandb-group / --wandb-name per replicate when --wandb is forwarded).
    parser.add_argument("--wandb", action="store_true", help="log the run to Weights & Biases")
    parser.add_argument("--wandb-project", default="firecracker-world-model", help="W&B project")
    parser.add_argument("--wandb-entity", default=None, help="W&B entity (team/user)")
    parser.add_argument("--wandb-group", default=None, help="W&B group shared by replicates")
    parser.add_argument("--wandb-name", default=None, help="W&B run name (default: seed<seed>)")
    parser.add_argument("--wandb-mode", default="online", choices=("online", "offline", "disabled"))
    args = parser.parse_args()
    if args.resume is not None and args.init_from is not None:
        raise SystemExit("--resume and --init-from are mutually exclusive (resume restores the weights).")

    torch.manual_seed(args.seed)
    config = load_config(args.config)
    device = _resolve_device(args.device)
    if args.amp and WEIGHT_DTYPES[args.weight_dtype] is not torch.float32:
        raise SystemExit("--amp keeps float32 master weights; drop --weight-dtype or --amp.")
    weight_dtype = _resolve_weight_dtype(args.weight_dtype, device)
    print(f"device: {device}  weight-dtype: {str(weight_dtype).removeprefix('torch.')}")

    # 1. Dataset + trajectory windows ----------------------------------------
    dataset = FireDataset(args.data)
    if args.prewarm:
        prewarm_shards(dataset)
    if not dataset.is_normalized:
        print(f"WARNING: dataset at '{args.data}' is not marked normalized; bake it first with "
              f"python tools/normalize_dataset.py --data {args.data}")
    args._in_channels = int(dataset.meta["num_channels"])
    args._grid_size = int(dataset.meta["grid_size"])
    # Validation may roll out a different (longer) horizon than training so runs trained at different
    # K compare on one common long-horizon metric; the window must cover whichever is larger.
    val_rollout_steps = args.val_rollout_steps if args.val_rollout_steps is not None else args.rollout_steps
    max_rollout = max(args.rollout_steps, val_rollout_steps)
    if args.window < max_rollout + 1:
        raise SystemExit(f"--window ({args.window}) must be >= max(rollout_steps, val_rollout_steps) + 1 "
                         f"({max_rollout + 1}); train K={args.rollout_steps}, val K={val_rollout_steps}")
    traj = TrajectoryDataset(dataset, num_envs=args.num_envs, window=args.window, step=args.step)
    train_w, val_w = split_episodes(traj, val_frac=args.val_frac, seed=args.seed)
    pin = device.type == "cuda"
    train_loader = TrajectoryBatchLoader(traj, train_w, args.batch_size, shuffle=True,
                                         seed=args.seed, drop_last=True, pin_memory=pin)
    val_loader = TrajectoryBatchLoader(traj, val_w, args.batch_size, shuffle=False,
                                       seed=args.seed, drop_last=False, pin_memory=pin)
    print(f"dataset: {len(dataset)} frames -> {len(traj)} windows "
          f"({len(traj.episodes)} episodes of {traj.F} frames, B={traj.B}); "
          f"{len(train_w)} train / {len(val_w)} val windows")

    # 2. Model: autoencoder + dynamics head ----------------------------------
    model, model_config, channel_stats = build_autoencoder(args, config, device, weight_dtype)
    if args.freeze_ae:
        if args.init_from is None:
            raise SystemExit("--freeze-ae requires --init-from (nothing to freeze when training from scratch).")
        for p in model.parameters():
            p.requires_grad_(False)
    # EMA target encoder: a momentum copy of the whole autoencoder (only its encode() is used; the
    # decoder half rides along, EMA-updated but unused). None for a frozen AE -- a fixed encoder is
    # already a stable target, so it serves as its own target with no EMA bookkeeping.
    ema_model = None
    if not args.freeze_ae:
        ema_model = copy.deepcopy(model)
        for p in ema_model.parameters():
            p.requires_grad_(False)
        ema_model.eval()
    transition = LatentTransition(
        model.latent_dim, hidden_dim=args.dyn_hidden, depth=args.dyn_depth, norm=args.dyn_norm,
    ).to(device=device, dtype=weight_dtype)
    # Record the RESOLVED hidden width (transition.hidden_dim), not the raw --dyn-hidden which is
    # None when defaulted, so a rebuild reproduces the exact head even if the default later changes.
    dyn_config = {"latent_dim": model.latent_dim, "hidden_dim": transition.hidden_dim,
                  "depth": args.dyn_depth, "norm": transition.norm,
                  # Sim-ticks a single dynamics step represents: the dataset's stored-frame spacing
                  # (traj.stride) times the trajectory frame gap (--step). Recorded so the rollout
                  # viewer can advance the physics exactly one training step per model step without
                  # being told the stride on the command line.
                  "ticks_per_step": traj.stride * args.step}
    n_ae = sum(p.numel() for p in model.parameters())
    n_dyn = sum(p.numel() for p in transition.parameters())

    # Optional reward head. The reward is the newly-ignited-cell count, read off the dataset's binary
    # 'ignited' channel on the fly (no new dataset). Resolve that channel by name from the meta.
    reward_head = None
    reward_config = None
    ignited_channel = None
    if args.reward_weight > 0:
        names = list(dataset.meta.get("channels", []))
        if "ignited" in names:
            ignited_channel = names.index("ignited")
        else:
            ignited_channel = args._in_channels - 1
            print(f"WARNING: no 'ignited' channel name in meta; assuming last channel "
                  f"({ignited_channel}) for the reward target")
        reward_head = RewardHead(
            model.latent_dim, hidden_dim=args.reward_hidden, depth=args.reward_depth,
        ).to(device=device, dtype=weight_dtype)
        reward_config = {"latent_dim": model.latent_dim, "hidden_dim": reward_head.hidden_dim,
                         "depth": args.reward_depth, "ignited_channel": ignited_channel}
    n_rew = sum(p.numel() for p in reward_head.parameters()) if reward_head is not None else 0
    print(f"model: autoencoder params={n_ae:,} ({'frozen' if args.freeze_ae else 'trainable'}) + "
          f"dynamics params={n_dyn:,}"
          + (f" + reward params={n_rew:,} (ignited ch {ignited_channel})" if reward_head else "")
          + f"  latent_dim={model.latent_dim}")

    if args.rec_loss == "huber":
        rec_loss_fn = nn.HuberLoss(delta=args.huber_delta)
    elif args.rec_loss == "bce":
        # Decoder emits logits; BCEWithLogits fuses the sigmoid in (stable). Reconstructions/val_mae
        # apply sigmoid explicitly (see run_epoch / out_activation).
        rec_loss_fn = nn.BCEWithLogitsLoss()
    else:
        rec_loss_fn = nn.MSELoss()
    # Recorded so eval/viewers know whether decode() outputs logits (apply sigmoid) or raw values.
    model_config["out_activation"] = args.out_activation
    optimizer = torch.optim.AdamW(_trained_params(model, transition, args.freeze_ae, reward_head),
                                  lr=args.lr, weight_decay=args.weight_decay, eps=args.adam_eps)
    scaler = torch.cuda.amp.GradScaler() if (args.amp and device.type == "cuda") else None

    # Resume: restore the FULL training state (all weights + EMA target + optimizer + scaler) and
    # continue at the next epoch. The architecture is rebuilt from CLI/config above, so guard against a
    # silent mismatch that would let load_state_dict succeed on a differently-shaped model and corrupt
    # the run. `best_pred` and `start_epoch` seed the loop below (inf/1, or restored here).
    start_epoch = 1
    best_pred = float("inf")
    resume_wandb_id = None
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device)
        ck_cfg = ckpt.get("model_config", {})
        mismatch = {
            k: (ck_cfg.get(k), model_config.get(k))
            for k in ("arch", "in_channels", "grid_size", "channels", "latent_dim")
            if ck_cfg.get(k) != model_config.get(k)
        }
        if mismatch:
            raise SystemExit(
                f"--resume checkpoint architecture does not match this run: {mismatch}. "
                "Pass the same --arch/--channels/--latent-dim/--latent-bn the checkpoint was trained with."
            )
        model.load_state_dict(ckpt["model_state"])
        transition.load_state_dict(ckpt["dynamics_state"])
        if (reward_head is not None) != ("reward_state" in ckpt):
            raise SystemExit(
                "--resume reward-head mismatch: the checkpoint "
                f"{'has' if 'reward_state' in ckpt else 'has no'} reward head but this run "
                f"{'has one' if reward_head is not None else 'has none'} "
                "(set --reward-weight to match the checkpoint).")
        if reward_head is not None:
            reward_head.load_state_dict(ckpt["reward_state"])
        # The EMA target encoder carries its own momentum history; rebuilding it from the online
        # weights would throw that away, so restore it explicitly (None for a frozen AE).
        if ema_model is not None and ckpt.get("ema_state") is not None:
            ema_model.load_state_dict(ckpt["ema_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if scaler is not None and ckpt.get("scaler_state") is not None:
            scaler.load_state_dict(ckpt["scaler_state"])
        # Older checkpoints predate best_pred; fall back to that epoch's val_loss so best.pt is not
        # overwritten by a worse model on the first resumed epoch.
        best_pred = ckpt.get("best_pred", ckpt.get("val_loss", float("inf")))
        start_epoch = int(ckpt["epoch"]) + 1
        # Reattach to the original W&B run (if any) so resumed epochs extend the same history.
        resume_wandb_id = ckpt.get("wandb_run_id")
        print(f"resume: loaded '{args.resume}' (epoch {ckpt['epoch']}, best_pred {best_pred:.6f}); "
              f"continuing at epoch {start_epoch}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Optional Weights & Biases run. `run` is None when --wandb is off, turning every log call
    # below into a no-op. Replicates share --wandb-group (injected per-run by tools/sweep.py) so
    # W&B averages them, exactly as for the autoencoder sweeps.
    run = None
    if args.wandb:
        if wandb is None:
            raise SystemExit("--wandb was passed but wandb is not installed (pip install wandb)")
        run = wandb.init(
            project=args.wandb_project, entity=args.wandb_entity, group=args.wandb_group,
            name=args.wandb_name or f"seed{args.seed}", mode=args.wandb_mode,
            id=resume_wandb_id, resume="allow" if resume_wandb_id is not None else None,
            config={
                **model_config, **dyn_config,
                "seed": args.seed, "epochs": args.epochs, "batch_size": args.batch_size,
                "lr": args.lr, "weight_decay": args.weight_decay, "adam_eps": args.adam_eps,
                "max_grad_norm": args.max_grad_norm, "amp": scaler is not None,
                "window": args.window, "step": args.step, "rollout_steps": args.rollout_steps,
                "val_rollout_steps": val_rollout_steps,
                "horizon_discount": args.horizon_discount,
                "val_frac": args.val_frac,
                "pred_weight": args.pred_weight, "rec_weight": args.rec_weight,
                "rec_loss": args.rec_loss, "out_activation": args.out_activation,
                "huber_delta": args.huber_delta, "freeze_ae": args.freeze_ae,
                "ema_decay": None if args.freeze_ae else args.ema_decay,
                "init_from": args.init_from, "dyn_params": n_dyn, "ae_params": n_ae,
                "reward_weight": args.reward_weight, "reward_params": n_rew,
                "n_train_windows": len(train_w), "n_val_windows": len(val_w),
            },
        )

    def save_checkpoint(path, epoch, val_loss, best_pred):
        ckpt = {
            "epoch": epoch,
            "val_loss": val_loss,          # total loss at the saved epoch (reference)
            "best_pred": best_pred,        # the val_pred that selected this checkpoint
            "model_state": model.state_dict(),
            "dynamics_state": transition.state_dict(),
            "model_config": model_config,
            "dynamics_config": dyn_config,
            "freeze_ae": args.freeze_ae,
            "wandb_run_id": run.id if run is not None else None,
            # Full optimizer/scaler/EMA state so --resume continues in place (not just a warm restart).
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict() if scaler is not None else None,
            "ema_state": ema_model.state_dict() if ema_model is not None else None,
        }
        if reward_head is not None:
            ckpt["reward_state"] = reward_head.state_dict()
            ckpt["reward_config"] = reward_config
        if channel_stats is not None:
            ckpt["model_config"] = {**model_config, **channel_stats}
        torch.save(ckpt, path)

    # On resume, append to the existing log (and skip the header) so earlier epochs are preserved.
    metrics_path = out_dir / "metrics.csv"
    resuming_log = args.resume is not None and metrics_path.exists()
    metrics_file = metrics_path.open("a" if resuming_log else "w", newline="")
    metrics_writer = csv.writer(metrics_file)
    # Per-step validation cosine columns (val_pred_step_01..NN over the val rollout horizon) expose the
    # drift curve so runs trained at different K are compared step-by-step, not just on the mean val_pred.
    val_step_cols = [f"val_pred_step_{i:02d}" for i in range(1, val_rollout_steps + 1)]
    if not resuming_log:
        metrics_writer.writerow(["epoch", "train_loss", "train_pred", "train_rec", "train_reward", "train_mae",
                                 "val_loss", "val_pred", "val_rec", "val_reward", "val_mae",
                                 "grad_norm_mean", "grad_norm_max", "seconds"] + val_step_cols)

    # 3. Train ---------------------------------------------------------------
    # best.pt is selected on val_pred (the cosine next-latent loss): it's the world model's actual
    # objective and, unlike val_loss (pred + a Huber/BCE rec of different scales), is comparable
    # across the linear-Huber and sigmoid-BCE arms of the A/B. (best_pred seeded above: inf, or
    # restored from --resume.)
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_pred, tr_rec, tr_mae, gmean, gmax, tr_rew, _ = run_epoch(
            model, transition, ema_model, train_loader, device, optimizer, scaler,
            rec_loss_fn, args.pred_weight, args.rec_weight, args.freeze_ae, args.ema_decay,
            args.max_grad_norm, epoch, "train", args.log_interval, args.out_activation,
            cov_weight=args.cov_weight, rollout_steps=args.rollout_steps,
            reward_head=reward_head, ignited_channel=ignited_channel, reward_weight=args.reward_weight,
            pred_discount=args.horizon_discount,
        )
        va_loss, va_pred, va_rec, va_mae, _, _, va_rew, va_per_step = run_epoch(
            model, transition, ema_model, val_loader, device, None, None,
            rec_loss_fn, args.pred_weight, args.rec_weight, args.freeze_ae, args.ema_decay,
            0.0, epoch, "val", 0, args.out_activation, rollout_steps=val_rollout_steps,
            reward_head=reward_head, ignited_channel=ignited_channel, reward_weight=args.reward_weight,
        )
        dt = time.time() - t0
        rew_str = f" reward {tr_rew:.4f}/{va_rew:.4f}" if reward_head is not None else ""
        print(f"epoch {epoch:3d}/{args.epochs}  train {tr_loss:.6f} (pred {tr_pred:.6f} rec {tr_rec:.6f} mae {tr_mae:.6f})  "
              f"val {va_loss:.6f} (pred {va_pred:.6f} rec {va_rec:.6f} mae {va_mae:.6f})  "
              f"grad(mean {gmean:.3f} max {gmax:.3f}){rew_str}  ({dt:.1f}s)")
        metrics_writer.writerow([epoch, f"{tr_loss:.6f}", f"{tr_pred:.6f}", f"{tr_rec:.6f}", f"{tr_rew:.6f}",
                                 f"{tr_mae:.6f}", f"{va_loss:.6f}", f"{va_pred:.6f}", f"{va_rec:.6f}",
                                 f"{va_rew:.6f}", f"{va_mae:.6f}", f"{gmean:.6f}", f"{gmax:.6f}", f"{dt:.2f}"]
                                + [f"{v:.6f}" for v in va_per_step])
        metrics_file.flush()

        if va_pred < best_pred:
            best_pred = va_pred
            save_checkpoint(out_dir / "best.pt", epoch, va_loss, best_pred)
            print(f"  new best val_pred {va_pred:.6f} -> {out_dir / 'best.pt'}")
        # Overwrite last.pt every epoch so a walltime kill mid-run is fully resumable (--resume last.pt).
        save_checkpoint(out_dir / "last.pt", epoch, va_loss, best_pred)
        if args.save_interval and epoch % args.save_interval == 0:
            save_checkpoint(out_dir / f"epoch_{epoch:03d}.pt", epoch, va_loss, best_pred)

        if run is not None:
            run.log({
                "epoch": epoch,
                "train_loss": tr_loss, "train_pred": tr_pred, "train_rec": tr_rec, "train_mae": tr_mae,
                "val_loss": va_loss, "val_pred": va_pred, "val_rec": va_rec, "val_mae": va_mae,
                "train_reward": tr_rew, "val_reward": va_rew, "best_pred": best_pred,
                "grad_norm_mean": gmean, "grad_norm_max": gmax,
                "lr": optimizer.param_groups[0]["lr"], "epoch_seconds": dt,
                **{col: v for col, v in zip(val_step_cols, va_per_step)},
            }, step=epoch)

    metrics_file.close()
    save_checkpoint(out_dir / "last.pt", args.epochs, best_pred, best_pred)
    print(f"done. best val_pred {best_pred:.6f}; checkpoints in '{out_dir}/'.")
    if run is not None:
        run.summary["best_pred"] = best_pred
        run.finish()


if __name__ == "__main__":
    main()
