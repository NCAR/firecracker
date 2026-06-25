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
C x N x N tensor (C = len(OBS_CHANNELS) = 5, N = 256), already normalised by build_observation.

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
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from autoencoder import ConvAutoencoder
from config import load_config
from rollout import BatchedRollout, FireDataset


def _resolve_device(name: str | None) -> torch.device:
    """Pick the training device: an explicit value wins, else CUDA if present (mirrors
    simulation._resolve_device so the trainer follows the same convention)."""
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    meta = data_dir / "meta.json"
    if meta.exists() and not args.build:
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
) -> float:
    """Run one pass over `loader`. Trains when `optimizer` is given, else evaluates.

    Returns the sample-weighted mean reconstruction loss over the epoch.
    """
    training = optimizer is not None
    model.train(training)
    use_amp = scaler is not None and device.type == "cuda"

    total_loss = 0.0
    total_seen = 0
    grad_ctx = torch.enable_grad() if training else torch.no_grad()
    with grad_ctx:
        for step, batch in enumerate(loader):
            # Shards may be stored in float16; the model trains in float32.
            x = batch.to(device=device, dtype=torch.float32, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                x_hat, _ = model(x)
                loss = loss_fn(x_hat, x)

            if training:
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            n = x.shape[0]
            total_loss += loss.item() * n
            total_seen += n
            if training and log_interval and step % log_interval == 0:
                print(
                    f"  epoch {epoch:3d} [{tag}] step {step:5d}/{len(loader)}  "
                    f"loss {loss.item():.6f}"
                )

    return total_loss / max(total_seen, 1)


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
    parser.add_argument("--lr", type=float, default=1e-3, help="AdamW learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="AdamW weight decay")
    parser.add_argument("--val-frac", type=float, default=0.05,
                        help="fraction of samples held out for validation")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker processes")
    parser.add_argument("--amp", action="store_true",
                        help="use mixed-precision autocast + GradScaler (CUDA only)")
    # Model
    parser.add_argument("--latent-dim", type=int, default=256, help="bottleneck latent size")
    parser.add_argument("--channels", type=_parse_channels, default=(32, 64, 128),
                        help="encoder channel widths per downsample stage, e.g. 32,64,128")
    # Bookkeeping
    parser.add_argument("--out", default="checkpoints", help="dir to write checkpoints into")
    parser.add_argument("--save-interval", type=int, default=5,
                        help="write a checkpoint every N epochs (best is always saved)")
    parser.add_argument("--log-interval", type=int, default=100,
                        help="print the train loss every N steps (0 to silence)")
    parser.add_argument("--device", help="torch device override, e.g. cuda or cpu")
    parser.add_argument("--seed", type=int, default=0, help="seed for rollout + train/val split")
    args = parser.parse_args()

    config = load_config(args.config)
    device = _resolve_device(args.device)
    torch.manual_seed(args.seed)
    print(f"device: {device}")

    # 1. Dataset --------------------------------------------------------------
    build_dataset_if_needed(args, config)
    dataset = FireDataset(args.data)
    grid_size = int(dataset.meta["grid_size"])
    in_channels = int(dataset.meta["num_channels"])
    n_total = len(dataset)
    n_val = max(1, int(round(n_total * args.val_frac)))
    n_train = n_total - n_val
    gen = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=gen)
    print(
        f"dataset: {n_total} samples ({in_channels}x{grid_size}x{grid_size}) "
        f"-> {n_train} train / {n_val} val"
    )

    pin = device.type == "cuda"
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
    model = ConvAutoencoder(
        in_channels=in_channels,
        grid_size=grid_size,
        channels=args.channels,
        latent_dim=args.latent_dim,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: channels={args.channels} latent_dim={args.latent_dim} params={n_params:,}")

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler() if (args.amp and device.type == "cuda") else None

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    # The config an inference/eval script needs to rebuild the exact architecture.
    model_config = {
        "in_channels": in_channels,
        "grid_size": grid_size,
        "channels": list(args.channels),
        "latent_dim": args.latent_dim,
    }

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

    # 3. Train ----------------------------------------------------------------
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = run_epoch(model, train_loader, device, loss_fn, optimizer, scaler,
                               args.log_interval, epoch, "train")
        val_loss = run_epoch(model, val_loader, device, loss_fn, None, None, 0, epoch, "val")
        dt = time.time() - t0
        print(
            f"epoch {epoch:3d}/{args.epochs}  train {train_loss:.6f}  "
            f"val {val_loss:.6f}  ({dt:.1f}s)"
        )

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(out_dir / "best.pt", epoch, val_loss)
            print(f"  new best val {val_loss:.6f} -> {out_dir / 'best.pt'}")
        if args.save_interval and epoch % args.save_interval == 0:
            save_checkpoint(out_dir / f"epoch_{epoch:03d}.pt", epoch, val_loss)

    save_checkpoint(out_dir / "last.pt", args.epochs, best_val)
    print(f"done. best val loss {best_val:.6f}; checkpoints in '{out_dir}/'.")


if __name__ == "__main__":
    main()
