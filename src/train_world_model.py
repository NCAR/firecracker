"""
train_world_model.py

Train a latent world model on Firecracker rollout trajectories: an autoencoder
(strided_autoencoder.ConvAutoencoder) plus a latent transition head (dynamics.LatentTransition)
that predicts the next latent from the current one.

Data are drawn as short frame windows over episodes (traj_loader.TrajectoryDataset), each window
a run of consecutive stored frames from one world. For a window (x_t, x_{t+1}):

  z_t      = encode(x_t)                      # unit-norm latent (normalize_latent=True)
  z_{t+1}  = encode(x_{t+1})                  # unit-norm latent
  z_hat    = transition(z_t)                  # predicted next latent, re-normalized to the sphere

  prediction loss  = 1 - cos(z_hat, stopgrad(z_{t+1}))     # SimSiam/BYOL-style, cosine on the sphere
  reconstruction   = loss_fn(decode(z_t), x_t) + loss_fn(decode(z_{t+1}), x_{t+1})

The L2 normalization on both encode() and transition() outputs, together with the decoder anchor,
prevents representational collapse -- so no variance/covariance (VICReg) term is needed. The
stop-gradient on the target latent is what makes the predictive objective non-trivial.

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
from dynamics import LatentTransition
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
    model = ARCHITECTURES[arch](**kwargs).to(device=device, dtype=weight_dtype)
    model_config = {
        "arch": arch,
        "in_channels": in_channels,
        "grid_size": grid_size,
        "channels": list(model.channels),
        "latent_dim": model.latent_dim,
        "normalize_latent": model.normalize_latent,
        "bottleneck_channels": model.bottleneck_channels,
        "weight_dtype": str(weight_dtype).removeprefix("torch."),
    }
    return model, model_config, _load_channel_stats(args.data)


def run_epoch(
    model, transition, loader, device, optimizer, scaler,
    rec_loss_fn, pred_weight, rec_weight, freeze_ae, max_grad_norm, epoch, tag, log_interval,
    out_activation="none",
):
    """One pass over `loader`. Trains when `optimizer` is given, else evaluates (no grad).

    Returns (mean_total, mean_pred, mean_rec, mean_mae, grad_norm_mean, grad_norm_max). `mean_mae`
    is a loss-independent reconstruction MAE yardstick (sigmoid applied when out_activation=='sigmoid'
    so the bce/sigmoid and linear arms are directly comparable); it is 0 for a frozen AE.
    """
    training = optimizer is not None
    transition.train(training)
    model.train(training and not freeze_ae)                # a frozen AE stays in eval (BN uses running stats)
    use_amp = scaler is not None and device.type == "cuda"
    param_dtype = next(model.parameters()).dtype

    tot = tot_pred = tot_rec = tot_mae = 0.0
    seen = 0
    gnorm_sum = 0.0
    gnorm_max = 0.0
    gnorm_steps = 0
    grad_ctx = torch.enable_grad() if training else torch.no_grad()
    with grad_ctx:
        for step, batch in enumerate(loader):
            # (B, window, C, H, W); v1 uses the first two frames as the (t, t+1) pair.
            x = batch.to(device=device, dtype=param_dtype, non_blocking=True)
            x_t, x_next = x[:, 0], x[:, 1]
            if training:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                # A frozen AE needs no encoder graph; encode under no_grad to save memory.
                enc_ctx = torch.no_grad() if freeze_ae else torch.enable_grad()
                with enc_ctx:
                    z_t = model.encode(x_t)
                    z_next = model.encode(x_next)
                z_hat = transition(z_t)
                pred = cosine_prediction_loss(z_hat, z_next.detach())
                if freeze_ae:
                    rec = torch.zeros((), device=device)
                    mae = torch.zeros((), device=device)
                else:
                    dec_t, dec_next = model.decode(z_t), model.decode(z_next)
                    rec = 0.5 * (rec_loss_fn(dec_t, x_t) + rec_loss_fn(dec_next, x_next))
                    # Loss-independent reconstruction MAE (sigmoid for the logits/bce head), so the
                    # arms are comparable regardless of which rec loss trained them.
                    with torch.no_grad():
                        rt = dec_t.sigmoid() if out_activation == "sigmoid" else dec_t
                        rn = dec_next.sigmoid() if out_activation == "sigmoid" else dec_next
                        mae = 0.5 * (F.l1_loss(rt, x_t) + F.l1_loss(rn, x_next))
                loss = pred_weight * pred + rec_weight * rec

            if training:
                gnorm_val = 0.0
                if use_amp:
                    scaler.scale(loss).backward()
                    if max_grad_norm > 0:
                        scaler.unscale_(optimizer)
                        gnorm_val = float(torch.nn.utils.clip_grad_norm_(_trained_params(model, transition, freeze_ae), max_grad_norm))
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if max_grad_norm > 0:
                        gnorm_val = float(torch.nn.utils.clip_grad_norm_(_trained_params(model, transition, freeze_ae), max_grad_norm))
                    optimizer.step()
                if max_grad_norm > 0 and np.isfinite(gnorm_val):
                    gnorm_sum += gnorm_val
                    gnorm_max = max(gnorm_max, gnorm_val)
                    gnorm_steps += 1

            n = x.shape[0]
            tot += loss.item() * n
            tot_pred += pred.item() * n
            tot_rec += float(rec) * n
            tot_mae += float(mae) * n
            seen += n
            if training and log_interval and step % log_interval == 0:
                print(f"  epoch {epoch:3d} [{tag}] step {step:5d}/{len(loader)}  "
                      f"loss {loss.item():.6f}  pred {pred.item():.6f}  rec {float(rec):.6f}")

    denom = max(seen, 1)
    return (tot / denom, tot_pred / denom, tot_rec / denom, tot_mae / denom,
            gnorm_sum / max(gnorm_steps, 1), gnorm_max)


def _trained_params(model, transition, freeze_ae):
    """The parameters the optimizer updates (dynamics head only when the AE is frozen)."""
    if freeze_ae:
        return list(transition.parameters())
    return list(model.parameters()) + list(transition.parameters())


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
    parser.add_argument("--dyn-depth", type=int, default=2, help="dynamics head hidden layers")
    parser.add_argument("--dyn-hidden", type=int, default=None,
                        help="dynamics head hidden width (default 1024)")
    # Loss weights
    parser.add_argument("--pred-weight", type=float, default=1.0, help="weight on the prediction loss")
    parser.add_argument("--rec-weight", type=float, default=1.0, help="weight on the reconstruction loss")
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
    # Weights & Biases (matches the autoencoder trainer's flags so tools/sweep.py drives both the
    # same way: it injects --wandb-group / --wandb-name per replicate when --wandb is forwarded).
    parser.add_argument("--wandb", action="store_true", help="log the run to Weights & Biases")
    parser.add_argument("--wandb-project", default="firecracker-world-model", help="W&B project")
    parser.add_argument("--wandb-entity", default=None, help="W&B entity (team/user)")
    parser.add_argument("--wandb-group", default=None, help="W&B group shared by replicates")
    parser.add_argument("--wandb-name", default=None, help="W&B run name (default: seed<seed>)")
    parser.add_argument("--wandb-mode", default="online", choices=("online", "offline", "disabled"))
    args = parser.parse_args()

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
    transition = LatentTransition(
        model.latent_dim, hidden_dim=args.dyn_hidden, depth=args.dyn_depth,
    ).to(device=device, dtype=weight_dtype)
    # Record the RESOLVED hidden width (transition.hidden_dim), not the raw --dyn-hidden which is
    # None when defaulted, so a rebuild reproduces the exact head even if the default later changes.
    dyn_config = {"latent_dim": model.latent_dim, "hidden_dim": transition.hidden_dim,
                  "depth": args.dyn_depth}
    n_ae = sum(p.numel() for p in model.parameters())
    n_dyn = sum(p.numel() for p in transition.parameters())
    print(f"model: autoencoder params={n_ae:,} ({'frozen' if args.freeze_ae else 'trainable'}) + "
          f"dynamics params={n_dyn:,}  latent_dim={model.latent_dim}")

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
    optimizer = torch.optim.AdamW(_trained_params(model, transition, args.freeze_ae),
                                  lr=args.lr, weight_decay=args.weight_decay, eps=args.adam_eps)
    scaler = torch.cuda.amp.GradScaler() if (args.amp and device.type == "cuda") else None

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
            config={
                **model_config, **dyn_config,
                "seed": args.seed, "epochs": args.epochs, "batch_size": args.batch_size,
                "lr": args.lr, "weight_decay": args.weight_decay, "adam_eps": args.adam_eps,
                "max_grad_norm": args.max_grad_norm, "amp": scaler is not None,
                "window": args.window, "step": args.step, "val_frac": args.val_frac,
                "pred_weight": args.pred_weight, "rec_weight": args.rec_weight,
                "rec_loss": args.rec_loss, "out_activation": args.out_activation,
                "huber_delta": args.huber_delta, "freeze_ae": args.freeze_ae,
                "init_from": args.init_from, "dyn_params": n_dyn, "ae_params": n_ae,
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
        }
        if channel_stats is not None:
            ckpt["model_config"] = {**model_config, **channel_stats}
        torch.save(ckpt, path)

    metrics_path = out_dir / "metrics.csv"
    metrics_file = metrics_path.open("w", newline="")
    metrics_writer = csv.writer(metrics_file)
    metrics_writer.writerow(["epoch", "train_loss", "train_pred", "train_rec", "train_mae",
                             "val_loss", "val_pred", "val_rec", "val_mae",
                             "grad_norm_mean", "grad_norm_max", "seconds"])

    # 3. Train ---------------------------------------------------------------
    # best.pt is selected on val_pred (the cosine next-latent loss): it's the world model's actual
    # objective and, unlike val_loss (pred + a Huber/BCE rec of different scales), is comparable
    # across the linear-Huber and sigmoid-BCE arms of the A/B.
    best_pred = float("inf")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_pred, tr_rec, tr_mae, gmean, gmax = run_epoch(
            model, transition, train_loader, device, optimizer, scaler,
            rec_loss_fn, args.pred_weight, args.rec_weight, args.freeze_ae,
            args.max_grad_norm, epoch, "train", args.log_interval, args.out_activation,
        )
        va_loss, va_pred, va_rec, va_mae, _, _ = run_epoch(
            model, transition, val_loader, device, None, None,
            rec_loss_fn, args.pred_weight, args.rec_weight, args.freeze_ae,
            0.0, epoch, "val", 0, args.out_activation,
        )
        dt = time.time() - t0
        print(f"epoch {epoch:3d}/{args.epochs}  train {tr_loss:.6f} (pred {tr_pred:.6f} rec {tr_rec:.6f} mae {tr_mae:.6f})  "
              f"val {va_loss:.6f} (pred {va_pred:.6f} rec {va_rec:.6f} mae {va_mae:.6f})  "
              f"grad(mean {gmean:.3f} max {gmax:.3f})  ({dt:.1f}s)")
        metrics_writer.writerow([epoch, f"{tr_loss:.6f}", f"{tr_pred:.6f}", f"{tr_rec:.6f}", f"{tr_mae:.6f}",
                                 f"{va_loss:.6f}", f"{va_pred:.6f}", f"{va_rec:.6f}", f"{va_mae:.6f}",
                                 f"{gmean:.6f}", f"{gmax:.6f}", f"{dt:.2f}"])
        metrics_file.flush()

        if va_pred < best_pred:
            best_pred = va_pred
            save_checkpoint(out_dir / "best.pt", epoch, va_loss, best_pred)
            print(f"  new best val_pred {va_pred:.6f} -> {out_dir / 'best.pt'}")

        if run is not None:
            run.log({
                "epoch": epoch,
                "train_loss": tr_loss, "train_pred": tr_pred, "train_rec": tr_rec, "train_mae": tr_mae,
                "val_loss": va_loss, "val_pred": va_pred, "val_rec": va_rec, "val_mae": va_mae,
                "best_pred": best_pred,
                "grad_norm_mean": gmean, "grad_norm_max": gmax,
                "lr": optimizer.param_groups[0]["lr"], "epoch_seconds": dt,
            }, step=epoch)

    metrics_file.close()
    save_checkpoint(out_dir / "last.pt", args.epochs, best_pred, best_pred)
    print(f"done. best val_pred {best_pred:.6f}; checkpoints in '{out_dir}/'.")
    if run is not None:
        run.summary["best_pred"] = best_pred
        run.finish()


if __name__ == "__main__":
    main()
