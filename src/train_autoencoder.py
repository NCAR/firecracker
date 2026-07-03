"""
train_autoencoder.py

Train the convolutional autoencoder (autoencoder.ConvAutoencoder) to reconstruct Firecracker
world-model observations collected from batched fire rollouts.

Pipeline
--------
1. (once) Build a static, sharded, memory-mapped dataset of observations by stepping batched
   fire worlds — see rollout.BatchedRollout.build_dataset / rollout.FireDataset. Pass --build
   to generate it; thereafter the script trains directly off the shards on disk.
2. Train the autoencoder under per-pixel MSE between the input observation and its
   reconstruction, holding out a validation split to track generalisation.

Dataset sizing (defaults)
-------------------------
One "rollout" is a single env's 4096-step trajectory sampled every 64 steps, i.e. 64
observations. Stepping num_envs=64 worlds in parallel for --rounds=16 yields
16 * 64 * 64 = 65,536 observations (1024 rollouts) written as 16 shards. Each observation is a
C x N x N tensor (C = 4 + one channel per fuel type, N = 256), already normalised by
build_observation; the exact channel count travels in the dataset meta.

At N=256, C=5 the full set is ~86 GB in float32 / ~43 GB in float16 on disk, so --buffer-dtype
defaults to float16 for the dataset; the model still trains in float32.

Quick use
---------
    # Build the 65k-sample dataset (igniting each world), then train:
    python src/train_autoencoder.py --build --spawn-fire --data data/fire --epochs 50

    # Train on an already-built dataset:
    python src/train_autoencoder.py --data data/fire --epochs 50 --batch-size 64

Checkpoints (model weights + the config needed to rebuild the architecture) are written to
--out; the lowest-val-loss one is also copied to <out>/best.pt.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

try:
    import wandb
except ImportError:  # optional dependency; only needed when --wandb is passed
    wandb = None

from autoencoder import ConvAutoencoder
from config import load_config
from ram_loader import RamBatchLoader
from rollout import BatchedRollout, FireDataset
from strided_autoencoder import StridedConvAutoencoder
from strided_vae import StridedConvVAE
from vae import ConvVAE

# Selectable autoencoder architectures. All share the same constructor signature. The plain
# autoencoders use a (reconstruction, latent) forward contract; the VAE variants (arch names in
# VAE_ARCHS) instead return (reconstruction, latent, mu, logvar) and add a KL term to the loss
# (weighted by --beta). --arch picks one by key.
ARCHITECTURES = {
    "shuffle": ConvAutoencoder,         # pixel-unshuffle/shuffle resampling
    "strided": StridedConvAutoencoder,  # 2x2 stride-2 conv / transposed-conv resampling
    "shuffle-vae": ConvVAE,             # variational, pixel-unshuffle/shuffle resampling
    "strided-vae": StridedConvVAE,      # variational, 2x2 stride-2 conv resampling
}

# Architectures with a probabilistic bottleneck: their forward returns (x_hat, z, mu, logvar)
# and training adds beta * KL(N(mu, sigma^2) || N(0, I)) to the reconstruction loss.
VAE_ARCHS = frozenset({"shuffle-vae", "strided-vae"})


def _resolve_device(name: str | None) -> torch.device:
    """Pick the training device: an explicit value wins, else CUDA if present (mirrors
    simulation._resolve_device so the trainer follows the same convention)."""
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_channel_stats(data_dir: str) -> dict | None:
    """The per-channel mean/std baked into this dataset (its `stats.json`), or None if absent.

    Recorded in the checkpoint so an inference/eval script can standardise raw inputs the same way
    the training shards were standardised (and de-normalize reconstructions)."""
    path = Path(data_dir) / "stats.json"
    if not path.is_file():
        return None
    stats = json.loads(path.read_text())
    if "channel_mean" not in stats or "channel_std" not in stats:
        return None
    return stats


def _parse_channels(spec: str) -> tuple[int, ...]:
    """Parse an encoder-width spec like '32,64,128' into a tuple of ints."""
    try:
        widths = tuple(int(p) for p in spec.split(",") if p.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid --channels {spec!r}: {exc}") from None
    if not widths:
        raise argparse.ArgumentTypeError("--channels must list at least one width")
    return widths


def build_dataset_if_needed(args: argparse.Namespace, config: dict) -> None:
    """Generate the sharded observation dataset under args.data when asked or when missing.

    The rollout sizing (steps / stride / num_envs / rounds) is taken from the CLI, which
    defaults to the 65,536-sample layout documented in the module docstring.
    """
    data_dir = Path(args.data)
    # A dataset is "present" either flat (meta.json in data_dir) or nested (one subdir per
    # source, each with its own meta.json) -- the multi-GPU generation layout FireDataset
    # unions. Recognise both so pointing --data at a nested parent doesn't trigger a rebuild.
    present = (data_dir / "meta.json").exists() or any(data_dir.glob("*/meta.json"))
    if present and not args.build:
        return  # dataset already present and no rebuild requested

    roll = BatchedRollout(
        config,
        num_envs=args.num_envs,
        steps=args.steps,
        stride=args.stride,
        buffer_dtype=args.buffer_dtype,
        spawn_fire=True if args.spawn_fire else None,
        seed=args.seed,
    )
    per_rollout = roll.steps // roll.stride            # observations from one env trajectory
    total = roll.capacity * args.rounds                 # capacity already folds in num_envs
    print(
        f"Building {args.rounds} shard(s) into '{data_dir}/' : "
        f"{roll.num_envs} envs x {args.rounds} rounds = {roll.num_envs * args.rounds} rollouts, "
        f"{per_rollout} obs/rollout -> {total} observations "
        f"({roll.num_channels}x{roll.grid_size}x{roll.grid_size}, {args.buffer_dtype}) ..."
    )
    info = roll.build_dataset(data_dir, args.rounds, overwrite=args.build)
    print(f"  done: {info['total_samples']} samples across {info['rounds']} shard(s).")


def run_epoch(
    model: ConvAutoencoder,
    loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.cuda.amp.GradScaler | None,
    log_interval: int,
    epoch: int,
    tag: str,
    max_grad_norm: float = 0.0,
    extra_loss_fn: nn.Module | None = None,
    is_vae: bool = False,
    beta: float = 0.0,
) -> tuple[float, float, float, float, float]:
    """Run one pass over `loader`. Trains when `optimizer` is given, else evaluates.

    When training with `max_grad_norm > 0`, gradients are clipped to that global L2 norm
    (after unscaling under AMP). The clipper returns the *pre-clip* total norm, which we
    track to expose gradient spikes.

    `extra_loss_fn`, if given, is evaluated (no grad) alongside the primary loss so a common
    yardstick (e.g. reconstruction MSE) can be reported regardless of the training `loss_fn`.

    When `is_vae`, the model's forward returns `(x_hat, z, mu, logvar)` and the optimised loss is
    the reconstruction loss plus `beta * model.kl_divergence(mu, logvar)`; the plain KL term is
    tracked and returned so it can be reported separately.

    Returns `(mean_loss, grad_norm_mean, grad_norm_max, extra_mean, kl_mean)`; grad-norm stats
    are the pre-clip totals over the epoch (0.0 for an eval pass), `extra_mean` is NaN when
    `extra_loss_fn` is None, and `kl_mean` is NaN when `is_vae` is False.
    """
    training = optimizer is not None
    model.train(training)
    use_amp = scaler is not None and device.type == "cuda"

    total_loss = 0.0
    total_seen = 0
    extra_sum = 0.0
    kl_sum = 0.0
    gnorm_sum = 0.0
    gnorm_max = 0.0
    gnorm_steps = 0
    grad_ctx = torch.enable_grad() if training else torch.no_grad()
    with grad_ctx:
        for step, batch in enumerate(loader):
            # Shards may be stored in float16; the model trains in float32.
            x = batch.to(device=device, dtype=torch.float32, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                out = model(x)
                x_hat = out[0]
                loss = loss_fn(x_hat, x)
                if is_vae:
                    kl = model.kl_divergence(out[2], out[3])  # out == (x_hat, z, mu, logvar)
                    loss = loss + beta * kl
                if extra_loss_fn is not None:
                    extra = extra_loss_fn(x_hat, x)

            if training:
                if use_amp:
                    scaler.scale(loss).backward()
                    # Unscale before clipping so the norm/threshold are in real units.
                    if max_grad_norm > 0:
                        scaler.unscale_(optimizer)
                        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                        gnorm_val = float(gnorm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if max_grad_norm > 0:
                        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                        gnorm_val = float(gnorm)
                    optimizer.step()
                # Skip non-finite norms: under AMP a scaled-gradient overflow yields an inf/nan
                # norm and the scaler skips that step, so it isn't a real (taken) update.
                if max_grad_norm > 0 and math.isfinite(gnorm_val):
                    gnorm_sum += gnorm_val
                    gnorm_max = max(gnorm_max, gnorm_val)
                    gnorm_steps += 1

            n = x.shape[0]
            total_loss += loss.item() * n
            if extra_loss_fn is not None:
                extra_sum += extra.item() * n
            if is_vae:
                kl_sum += kl.item() * n
            total_seen += n
            if training and log_interval and step % log_interval == 0:
                print(
                    f"  epoch {epoch:3d} [{tag}] step {step:5d}/{len(loader)}  "
                    f"loss {loss.item():.6f}"
                )

    mean_loss = total_loss / max(total_seen, 1)
    gnorm_mean = gnorm_sum / max(gnorm_steps, 1)
    extra_mean = extra_sum / max(total_seen, 1) if extra_loss_fn is not None else float("nan")
    kl_mean = kl_sum / max(total_seen, 1) if is_vae else float("nan")
    return mean_loss, gnorm_mean, gnorm_max, extra_mean, kl_mean


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the ConvAutoencoder on Firecracker rollout observations."
    )
    # Data / dataset construction
    parser.add_argument("--data", default="data/fire", help="dataset dir (shards + meta.json)")
    parser.add_argument("--config", metavar="PATH", help="path to a TOML config file")
    parser.add_argument("--build", action="store_true",
                        help="(re)build the dataset before training (overwrites existing shards)")
    parser.add_argument("--rounds", type=int, default=16,
                        help="shards to generate: rounds * num_envs * (steps//stride) samples")
    parser.add_argument("--num-envs", type=int, default=64, help="worlds stepped in parallel")
    parser.add_argument("--steps", type=int, default=4096, help="ticks per rollout")
    parser.add_argument("--stride", type=int, default=64, help="store every stride-th frame")
    parser.add_argument("--buffer-dtype", default="float16",
                        help="shard storage dtype, float16 (default) or float32")
    parser.add_argument("--spawn-fire", action="store_true", help="ignite each world at reset")
    # Optimisation
    parser.add_argument("--epochs", type=int, default=50, help="training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="minibatch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="AdamW learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="AdamW weight decay")
    parser.add_argument("--adam-eps", type=float, default=1e-3,
                        help="AdamW epsilon; a larger floor on the update denominator guards "
                             "against oversized steps when a param's second moment collapses "
                             "(sparse/intermittent gradients)")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                        help="clip gradients to this global L2 norm each step (0 disables)")
    parser.add_argument("--loss", choices=("mse", "huber"), default="mse",
                        help="reconstruction loss: 'mse' (default) or 'huber' (robust, bounds "
                             "the gradient of large residuals)")
    parser.add_argument("--huber-delta", type=float, default=1.0,
                        help="Huber transition point (only for --loss huber); residuals below it "
                             "are quadratic, above it linear. With per-channel standardisation the "
                             "inputs are ~unit-variance (roughly [-3,3]), so 0.5-1.0 is where "
                             "robustness starts to engage (default: 1.0)")
    parser.add_argument("--val-frac", type=float, default=0.05,
                        help="fraction of samples held out for validation")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader worker processes (only used by --loader dataloader)")
    parser.add_argument("--loader", choices=("ram", "dataloader"), default="ram",
                        help="'ram' (default): thread-prefetched in-RAM batch loader, no worker "
                             "processes (safe under heavy packing, no shared-memory crash). "
                             "'dataloader': stock torch DataLoader with --num-workers workers")
    parser.add_argument("--amp", action="store_true",
                        help="use mixed-precision autocast + GradScaler (CUDA only)")
    # Model
    parser.add_argument("--arch", choices=tuple(ARCHITECTURES), default=None,
                        help="autoencoder architecture: 'shuffle'/'strided' (pixel-shuffle vs "
                             "2x2 stride-2 conv resampling) or their variational '-vae' variants "
                             "(add a KL term, weighted by --beta); default from [autoencoder].arch")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="KL weight for the '-vae' architectures (beta-VAE): the loss is "
                             "reconstruction + beta * KL(posterior || N(0, I)). Ignored by the "
                             "non-variational archs (default: 1.0)")
    parser.add_argument("--channels", type=_parse_channels, default=None,
                        help="encoder channel widths per downsample stage; the last is the conv "
                             "channel count before flattening, e.g. 16,32,64,128,256,512; default "
                             "from [autoencoder.<arch>].channels")
    parser.add_argument("--latent-dim", type=int, default=None,
                        help="width of the dense latent vector the flattened feature map "
                             "projects to; default from [autoencoder.<arch>].latent_dim")
    # Bookkeeping
    parser.add_argument("--out", default="checkpoints", help="dir to write checkpoints into")
    parser.add_argument("--save-interval", type=int, default=5,
                        help="write a checkpoint every N epochs (best is always saved)")
    parser.add_argument("--log-interval", type=int, default=100,
                        help="print the train loss every N steps (0 to silence)")
    parser.add_argument("--device", help="torch device override, e.g. cuda or cpu")
    parser.add_argument("--seed", type=int, default=0, help="seed for rollout + train/val split")
    # Weights & Biases (optional). Replicate runs share --wandb-group so W&B can average them.
    parser.add_argument("--wandb", action="store_true",
                        help="log config + per-epoch metrics to Weights & Biases")
    parser.add_argument("--wandb-project", default="firecracker-autoencoder",
                        help="W&B project name (default: firecracker-autoencoder)")
    parser.add_argument("--wandb-entity", default=None,
                        help="W&B entity (team/user); default from your W&B login")
    parser.add_argument("--wandb-group", default=None,
                        help="W&B group; shared across replicate runs so they aggregate")
    parser.add_argument("--wandb-name", default=None,
                        help="W&B run name (default: derived from the seed)")
    parser.add_argument("--wandb-mode", default=None,
                        choices=("online", "offline", "disabled"),
                        help="W&B mode; default online, or honour $WANDB_MODE if set")
    args = parser.parse_args()

    config = load_config(args.config)
    device = _resolve_device(args.device)
    torch.manual_seed(args.seed)
    print(f"device: {device}")

    # Model architecture: CLI flags override the [autoencoder] config, which in turn overrides
    # the model constructor defaults. `channels`/`latent_dim` are read from the per-arch subtable.
    ae_cfg = config.get("autoencoder", {})
    arch = args.arch or ae_cfg.get("arch", "shuffle")
    arch_cfg = ae_cfg.get(arch, {})
    channels = args.channels if args.channels is not None else arch_cfg.get("channels")
    channels = tuple(channels) if channels is not None else None
    latent_dim = args.latent_dim if args.latent_dim is not None else arch_cfg.get("latent_dim")

    # 1. Dataset --------------------------------------------------------------
    build_dataset_if_needed(args, config)
    dataset = FireDataset(args.data)
    # Normalization is baked into the shards offline (tools/normalize_dataset.py) so the training
    # hot path stays a plain memmap copy. Warn if the data hasn't been standardized; the applied
    # per-channel mean/std travel in <data>/stats.json (recorded in the checkpoint for inference).
    channel_stats = _load_channel_stats(args.data)
    if dataset.is_normalized:
        print(f"input: per-channel zero-mean/unit-std baked into shards ('{args.data}')")
    else:
        print(f"WARNING: dataset at '{args.data}' is not marked normalized; train on standardized "
              f"data by baking it first: python tools/normalize_dataset.py --data {args.data}")
    grid_size = int(dataset.meta["grid_size"])
    in_channels = int(dataset.meta["num_channels"])
    n_total = len(dataset)
    n_val = max(1, int(round(n_total * args.val_frac)))
    n_train = n_total - n_val
    print(
        f"dataset: {n_total} samples ({in_channels}x{grid_size}x{grid_size}) "
        f"-> {n_train} train / {n_val} val  [loader={args.loader}]"
    )

    pin = device.type == "cuda"
    if args.loader == "ram":
        # Seeded index split (mirrors random_split's role) + thread-prefetched batch loaders.
        perm = np.random.default_rng(args.seed).permutation(n_total)
        val_idx, train_idx = perm[:n_val], perm[n_val:]
        train_loader = RamBatchLoader(
            dataset, train_idx, args.batch_size, shuffle=True, seed=args.seed,
            drop_last=True, pin_memory=pin,
        )
        val_loader = RamBatchLoader(
            dataset, val_idx, args.batch_size, shuffle=False, seed=args.seed,
            drop_last=False, pin_memory=pin,
        )
    else:
        gen = torch.Generator().manual_seed(args.seed)
        train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=gen)
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=pin, drop_last=True,
            persistent_workers=args.num_workers > 0,
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=pin,
            persistent_workers=args.num_workers > 0,
        )

    # 2. Model / optimiser ----------------------------------------------------
    # Pass only the settings that were resolved; None lets the constructor default apply.
    model_kwargs = {"in_channels": in_channels, "grid_size": grid_size}
    if channels is not None:
        model_kwargs["channels"] = channels
    if latent_dim is not None:
        model_kwargs["latent_dim"] = latent_dim
    model = ARCHITECTURES[arch](**model_kwargs).to(device)
    is_vae = arch in VAE_ARCHS
    n_params = sum(p.numel() for p in model.parameters())
    conv_shape = f"{model.conv_channels}x{model.conv_spatial}x{model.conv_spatial}"
    vae_note = f" beta={args.beta}" if is_vae else ""
    print(f"model: arch={arch} channels={tuple(model.channels)} conv_latent={conv_shape} "
          f"latent_dim={model.latent_dim} params={n_params:,}{vae_note}")

    loss_fn = nn.HuberLoss(delta=args.huber_delta) if args.loss == "huber" else nn.MSELoss()
    # A loss-independent yardstick: always report validation reconstruction MSE so runs trained
    # with different --loss (mse vs huber) are directly comparable. Redundant with val_loss when
    # --loss mse, but cheap (val is a small held-out split, evaluated without gradients).
    val_mse_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay, eps=args.adam_eps)
    scaler = torch.cuda.amp.GradScaler() if (args.amp and device.type == "cuda") else None

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    # The config an inference/eval script needs to rebuild the exact architecture.
    model_config = {
        "arch": arch,
        "in_channels": in_channels,
        "grid_size": grid_size,
        "channels": list(model.channels),
        "latent_dim": model.latent_dim,
    }
    # The KL weight the VAE was trained under (informational; not needed to rebuild the model).
    if is_vae:
        model_config["beta"] = args.beta
    # Carry the input standardisation so inference/eval can reproduce the exact channel scaling
    # the model was trained under (the transform baked into the shards; absent if none was found).
    if channel_stats is not None:
        model_config["channel_mean"] = channel_stats["channel_mean"]
        model_config["channel_std"] = channel_stats["channel_std"]

    # Optional Weights & Biases run. `run` is None when --wandb is not passed, which turns
    # every log call below into a no-op. Replicates share --wandb-group so W&B averages them.
    run = None
    if args.wandb:
        if wandb is None:
            raise SystemExit("--wandb was passed but wandb is not installed (pip install wandb)")
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            name=args.wandb_name or f"seed{args.seed}",
            mode=args.wandb_mode,
            config={
                **model_config,
                "seed": args.seed,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "adam_eps": args.adam_eps,
                "max_grad_norm": args.max_grad_norm,
                "loss": args.loss,
                "huber_delta": args.huber_delta if args.loss == "huber" else None,
                "beta": args.beta if is_vae else None,
                "amp": scaler is not None,
                "val_frac": args.val_frac,
                "n_train": n_train,
                "n_val": n_val,
                "params": n_params,
            },
        )

    def save_checkpoint(path: Path, epoch: int, val_loss: float) -> None:
        torch.save(
            {
                "epoch": epoch,
                "val_loss": val_loss,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "model_config": model_config,
            },
            path,
        )

    # A machine-readable per-epoch log so replicate runs (e.g. tools/sweep.py) can be
    # aggregated/averaged after the fact. Flushed each epoch to survive interruptions.
    metrics_path = out_dir / "metrics.csv"
    metrics_file = metrics_path.open("w", newline="")
    metrics_writer = csv.writer(metrics_file)
    metrics_writer.writerow(
        ["epoch", "train_loss", "val_loss", "val_mse", "train_kl", "val_kl",
         "grad_norm_mean", "grad_norm_max", "seconds"]
    )

    # 3. Train ----------------------------------------------------------------
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, gnorm_mean, gnorm_max, _, train_kl = run_epoch(
            model, train_loader, device, loss_fn, optimizer, scaler,
            args.log_interval, epoch, "train", max_grad_norm=args.max_grad_norm,
            is_vae=is_vae, beta=args.beta,
        )
        val_loss, _, _, val_mse, val_kl = run_epoch(
            model, val_loader, device, loss_fn, None, None, 0, epoch, "val",
            extra_loss_fn=val_mse_fn, is_vae=is_vae, beta=args.beta,
        )
        dt = time.time() - t0
        kl_note = f"  train_kl {train_kl:.4f}  val_kl {val_kl:.4f}" if is_vae else ""
        print(
            f"epoch {epoch:3d}/{args.epochs}  train {train_loss:.6f}  val {val_loss:.6f}  "
            f"val_mse {val_mse:.6f}{kl_note}  "
            f"grad_norm(mean {gnorm_mean:.3f}, max {gnorm_max:.3f})  ({dt:.1f}s)"
        )
        metrics_writer.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{val_mse:.6f}",
                                 f"{train_kl:.6f}", f"{val_kl:.6f}",
                                 f"{gnorm_mean:.6f}", f"{gnorm_max:.6f}", f"{dt:.2f}"])
        metrics_file.flush()

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(out_dir / "best.pt", epoch, val_loss)
            print(f"  new best val {val_loss:.6f} -> {out_dir / 'best.pt'}")
        if args.save_interval and epoch % args.save_interval == 0:
            save_checkpoint(out_dir / f"epoch_{epoch:03d}.pt", epoch, val_loss)

        if run is not None:
            run.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_mse": val_mse,
                "train_kl": train_kl,
                "val_kl": val_kl,
                "best_val": best_val,
                "grad_norm_mean": gnorm_mean,
                "grad_norm_max": gnorm_max,
                "lr": optimizer.param_groups[0]["lr"],
                "epoch_seconds": dt,
            }, step=epoch)

    metrics_file.close()
    save_checkpoint(out_dir / "last.pt", args.epochs, best_val)
    print(f"done. best val loss {best_val:.6f}; checkpoints in '{out_dir}/'.")
    if run is not None:
        run.summary["best_val"] = best_val
        run.finish()


if __name__ == "__main__":
    main()
