"""
train_autoencoder.py

Train a convolutional autoencoder (strided_autoencoder.ConvAutoencoder, the "pooled" arch) to
reconstruct Firecracker world-model observations collected from batched fire rollouts.

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
C x N x N tensor (C = 5 + two channels per fuel type + three one-hot biome channels, N = 256),
already normalised by build_observation; the exact channel count travels in the dataset meta.

At N=256, C=10 (one fuel type) the full set is ~172 GB in float32 / ~86 GB in float16 on disk, so
--buffer-dtype defaults to float16 for the dataset.

Model precision
---------------
--weight-dtype selects the dtype the model's parameters (and forward/backward compute) run in:
bfloat16 (default) halves weight + activation memory and, on Ampere/Hopper (A100/H100), runs on
the bf16 tensor cores for a speedup, while bf16's f32-range exponent avoids the underflow that
makes pure float16 training diverge. float16 is available (fastest on V100, whose tensor cores are
fp16-only) but is numerically fragile with AdamW; float32 keeps the old full-precision behavior.
On CPU (no GPU) half precision is slow / partially unsupported, so the trainer auto-falls back to
float32 there. The chosen dtype is recorded in the checkpoint so eval/inference rebuilds it the
same way (and likewise up-casts to float32 on CPU, which is lossless from bf16).

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

from config import load_config
from losses import MSSSIML1Loss
from ram_loader import RamBatchLoader, prewarm_shards
from rollout import BatchedRollout, FireDataset
from strided_autoencoder import ConvAutoencoder

# Selectable autoencoder architectures, keyed by --arch. Only the parameter-free 2x2 average-pool
# encoder ("pooled") remains; the learned strided-conv downsampler was evaluated and dropped (no
# quality gain over avg-pool, higher inter-seed variance, +3.5M params).
ARCHITECTURES = {
    "pooled": ConvAutoencoder,  # 2x2 average-pool downsample (parameter-free)
}

# Names accepted by --weight-dtype, mapped to the torch dtype the model's params/compute run in.
# Mirrored (by string) into the checkpoint's model_config so eval/inference rebuilds the same dtype.
WEIGHT_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _resolve_weight_dtype(name: str, device: torch.device) -> torch.dtype:
    """Map a --weight-dtype name to a torch dtype, falling back to float32 on CPU.

    Half precision (float16/bfloat16) is slow and partially unimplemented on CPU, so a half
    request there is downgraded to float32 (with a note) rather than run slowly or error out.
    """
    dtype = WEIGHT_DTYPES[name]
    if dtype is not torch.float32 and device.type == "cpu":
        print(f"note: --weight-dtype {name} requested but device is CPU; "
              f"training in float32 (half precision is slow/unsupported on CPU).")
        return torch.float32
    return dtype


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


def _block_of(param_name: str) -> str:
    """Group a parameter under the block it belongs to, for per-block grad-norm logging.

    `encoder.blocks.3.conv1.weight` -> `encoder.blocks.3`; `to_latent.weight` -> `to_latent`.
    The blocks come out in depth order (encoder stages, bottleneck, decoder stages), so the
    reported norms read as a gradient-vs-depth profile."""
    parts = param_name.split(".")
    if len(parts) >= 3 and parts[1] == "blocks":
        return ".".join(parts[:3])
    return parts[0]


def per_block_grad_norms(model: nn.Module) -> dict[str, float]:
    """Total L2 grad norm of each block from the current (post-backward) `.grad` tensors.

    Call after gradients are in real units (post-unscale under AMP). Returns block -> norm; a
    block whose gradients are all None (unused) is omitted."""
    sq: dict[str, float] = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        sq[_block_of(name)] = sq.get(_block_of(name), 0.0) + float(p.grad.detach().pow(2).sum())
    return {b: math.sqrt(v) for b, v in sq.items()}


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
    collect_layer_grads: bool = False,
) -> tuple[float, float, float, float, dict[str, float]]:
    """Run one pass over `loader`. Trains when `optimizer` is given, else evaluates.

    When training with `max_grad_norm > 0`, gradients are clipped to that global L2 norm
    (after unscaling under AMP). The clipper returns the *pre-clip* total norm, which we
    track to expose gradient spikes.

    `extra_loss_fn`, if given, is evaluated (no grad) alongside the primary loss so a common
    yardstick (e.g. reconstruction MSE) can be reported regardless of the training `loss_fn`.

    With `collect_layer_grads`, the per-block grad L2 norms (see `per_block_grad_norms`) are
    accumulated over the epoch and returned averaged -- a gradient-vs-depth profile for spotting
    vanishing gradients. Non-finite (AMP-overflow) steps are excluded, matching the global norm.

    Returns `(mean_loss, grad_norm_mean, grad_norm_max, extra_mean, layer_grad_means)`; grad-norm
    stats are the pre-clip totals over the epoch (0.0 for an eval pass), `extra_mean` is NaN when
    `extra_loss_fn` is None, and `layer_grad_means` is empty unless `collect_layer_grads`.
    """
    training = optimizer is not None
    model.train(training)
    use_amp = scaler is not None and device.type == "cuda"
    # Feed inputs in the model's own parameter dtype (float32 / float16 / bfloat16). Under AMP the
    # params are float32 and autocast handles the per-op down-cast, so f32 inputs are correct there.
    param_dtype = next(model.parameters()).dtype

    total_loss = 0.0
    total_seen = 0
    extra_sum = 0.0
    gnorm_sum = 0.0
    gnorm_max = 0.0
    gnorm_steps = 0
    layer_gnorm_sum: dict[str, float] = {}
    layer_gnorm_steps = 0
    collect_layers = training and collect_layer_grads
    grad_ctx = torch.enable_grad() if training else torch.no_grad()
    with grad_ctx:
        for step, batch in enumerate(loader):
            # Shards may be stored in float16; move them to the model's param dtype (see above).
            x = batch.to(device=device, dtype=param_dtype, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                out = model(x)
                x_hat = out[0]
                loss = loss_fn(x_hat, x)
                if extra_loss_fn is not None:
                    extra = extra_loss_fn(x_hat, x)

            if training:
                step_block: dict[str, float] = {}
                # Grads must be in real units before we read (per-block) or clip them, so unscale
                # under AMP whenever either is wanted. Per-block norms are read pre-clip so they
                # match the pre-clip global norm (clipping only rescales the whole vector uniformly).
                need_real_grads = max_grad_norm > 0 or collect_layers
                if use_amp:
                    scaler.scale(loss).backward()
                    if need_real_grads:
                        scaler.unscale_(optimizer)
                    if collect_layers:
                        step_block = per_block_grad_norms(model)
                    if max_grad_norm > 0:
                        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                        gnorm_val = float(gnorm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if collect_layers:
                        step_block = per_block_grad_norms(model)
                    if max_grad_norm > 0:
                        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                        gnorm_val = float(gnorm)
                    optimizer.step()
                # Skip non-finite norms: under AMP a scaled-gradient overflow yields an inf/nan
                # norm and the scaler skips that step, so it isn't a real (taken) update.
                grads_finite = (
                    math.isfinite(gnorm_val) if max_grad_norm > 0
                    else all(math.isfinite(v) for v in step_block.values())
                )
                if max_grad_norm > 0 and grads_finite:
                    gnorm_sum += gnorm_val
                    gnorm_max = max(gnorm_max, gnorm_val)
                    gnorm_steps += 1
                if collect_layers and grads_finite:
                    for b, v in step_block.items():
                        layer_gnorm_sum[b] = layer_gnorm_sum.get(b, 0.0) + v
                    layer_gnorm_steps += 1

            n = x.shape[0]
            total_loss += loss.item() * n
            if extra_loss_fn is not None:
                extra_sum += extra.item() * n
            total_seen += n
            if training and log_interval and step % log_interval == 0:
                print(
                    f"  epoch {epoch:3d} [{tag}] step {step:5d}/{len(loader)}  "
                    f"loss {loss.item():.6f}"
                )

    mean_loss = total_loss / max(total_seen, 1)
    gnorm_mean = gnorm_sum / max(gnorm_steps, 1)
    extra_mean = extra_sum / max(total_seen, 1) if extra_loss_fn is not None else float("nan")
    layer_grad_means = {b: s / layer_gnorm_steps for b, s in layer_gnorm_sum.items()} \
        if layer_gnorm_steps else {}
    return mean_loss, gnorm_mean, gnorm_max, extra_mean, layer_grad_means


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a convolutional autoencoder on Firecracker rollout observations."
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
    parser.add_argument("--log-layer-grads", action="store_true",
                        help="log per-block (per-stage) grad L2 norms each epoch to "
                             "layer_grad_norms.csv (and W&B as grad_norm/<block>) -- a "
                             "gradient-vs-depth profile for spotting vanishing gradients")
    parser.add_argument("--loss", choices=("mse", "huber", "ms-ssim+l1"), default="mse",
                        help="reconstruction loss: 'mse' (default), 'huber' (robust, bounds the "
                             "gradient of large residuals), or 'ms-ssim+l1' (multi-scale structural "
                             "similarity + L1; rewards preserving local contrast/structure, so it "
                             "penalises blur that mse tolerates). val_mse is always logged so the "
                             "three are comparable")
    parser.add_argument("--huber-delta", type=float, default=1.0,
                        help="Huber transition point (only for --loss huber); residuals below it "
                             "are quadratic, above it linear. With per-channel standardisation the "
                             "inputs are ~unit-variance (roughly [-3,3]), so 0.5-1.0 is where "
                             "robustness starts to engage (default: 1.0)")
    parser.add_argument("--ms-ssim-alpha", type=float, default=0.84,
                        help="(only for --loss ms-ssim+l1) weight on the MS-SSIM term vs L1: "
                             "alpha*(1-MS-SSIM) + (1-alpha)*L1 (default: 0.84, per Zhao et al.)")
    parser.add_argument("--ms-ssim-data-range", type=float, default=6.0,
                        help="(only for --loss ms-ssim+l1) SSIM dynamic range L for its C1/C2 "
                             "constants; the inputs are per-channel standardised, so ~6.0 (a +/-3 "
                             "sigma span) suits them (default: 6.0)")
    parser.add_argument("--val-frac", type=float, default=0.05,
                        help="fraction of samples held out for validation")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader worker processes (only used by --loader dataloader)")
    parser.add_argument("--loader", choices=("ram", "dataloader"), default="ram",
                        help="'ram' (default): thread-prefetched in-RAM batch loader, no worker "
                             "processes (safe under heavy packing, no shared-memory crash). "
                             "'dataloader': stock torch DataLoader with --num-workers workers")
    parser.add_argument("--prewarm", action=argparse.BooleanOptionalAction, default=True,
                        help="stream all shards sequentially in a background thread at startup to "
                             "warm the OS page cache, overlapping the first epoch (default: on). "
                             "Avoids the ~30-min epoch-1 tax from a shuffled first read faulting "
                             "pages in random order off the parallel filesystem. --no-prewarm "
                             "disables it")
    parser.add_argument("--amp", action="store_true",
                        help="use mixed-precision autocast + GradScaler (CUDA only). Keeps float32 "
                             "master weights; mutually exclusive with a half --weight-dtype")
    parser.add_argument("--weight-dtype", choices=tuple(WEIGHT_DTYPES), default="bfloat16",
                        help="dtype for the model's parameters + forward/backward compute: "
                             "'bfloat16' (default; half memory, stable, tensor-core accelerated on "
                             "A100/H100), 'float16' (fastest on V100 but numerically fragile with "
                             "AdamW), or 'float32' (full precision). Auto-downgraded to float32 on "
                             "CPU. Recorded in the checkpoint for eval/inference")
    # Model
    parser.add_argument("--arch", choices=tuple(ARCHITECTURES), default=None,
                        help="autoencoder architecture: 'pooled' (2x2 average-pool encoder, "
                             "resize-conv decoder); default from [autoencoder].arch")
    parser.add_argument("--channels", type=_parse_channels, default=None,
                        help="encoder channel widths per downsample stage; the last is the conv "
                             "channel count before flattening, e.g. 16,32,64,128,256,512; default "
                             "from [autoencoder.<arch>].channels")
    parser.add_argument("--latent-dim", type=int, default=None,
                        help="width of the dense latent vector the flattened feature map "
                             "projects to; default from [autoencoder.<arch>].latent_dim")
    parser.add_argument("--normalize-latent", action=argparse.BooleanOptionalAction, default=None,
                        help="L2-normalize the latent to unit magnitude (project onto the unit "
                             "hypersphere). --no-normalize-latent disables it; default from "
                             "[autoencoder.<arch>].normalize_latent (on if unset)")
    # Bookkeeping
    parser.add_argument("--out", default="checkpoints", help="dir to write checkpoints into")
    parser.add_argument("--save-interval", type=int, default=5,
                        help="write a checkpoint every N epochs (best is always saved)")
    parser.add_argument("--resume", default=None,
                        help="path to a checkpoint to resume from: restores model, optimizer, "
                             "GradScaler, best-val, and continues at the next epoch. The "
                             "architecture/channels/latent must match the checkpoint (they are "
                             "validated). metrics.csv is appended to rather than truncated")
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
    # AMP (float32 master weights + autocast) and a half --weight-dtype are two different low-
    # precision strategies; combining them (a GradScaler over already-half params) is incoherent.
    if args.amp and WEIGHT_DTYPES[args.weight_dtype] is not torch.float32:
        raise SystemExit("--amp keeps float32 master weights and cannot be combined with "
                         f"--weight-dtype {args.weight_dtype}; pass one or the other.")
    weight_dtype = _resolve_weight_dtype(args.weight_dtype, device)
    print(f"device: {device}  weight-dtype: {str(weight_dtype).removeprefix('torch.')}")

    # Model architecture: CLI flags override the [autoencoder] config, which in turn overrides
    # the model constructor defaults. `channels`/`latent_dim` are read from the per-arch subtable.
    ae_cfg = config.get("autoencoder", {})
    arch = args.arch or ae_cfg.get("arch", "pooled")
    arch_cfg = ae_cfg.get(arch, {})
    channels = args.channels if args.channels is not None else arch_cfg.get("channels")
    channels = tuple(channels) if channels is not None else None
    latent_dim = args.latent_dim if args.latent_dim is not None else arch_cfg.get("latent_dim")
    normalize_latent = (
        args.normalize_latent if args.normalize_latent is not None
        else arch_cfg.get("normalize_latent")
    )

    # 1. Dataset --------------------------------------------------------------
    build_dataset_if_needed(args, config)
    dataset = FireDataset(args.data)
    # Decouple the first read of the data from training: stream all shards sequentially into the
    # (node-shared) page cache on a background thread so epoch 1 overlaps the load instead of
    # driving scattered random reads off GLADE. Fire-and-forget -- the thread is a daemon.
    if args.prewarm:
        print(f"prewarm: streaming {len(dataset.shard_paths)} shards into page cache (background)")
        prewarm_shards(dataset)
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
    if normalize_latent is not None:
        model_kwargs["normalize_latent"] = normalize_latent
    model = ARCHITECTURES[arch](**model_kwargs).to(device=device, dtype=weight_dtype)
    n_params = sum(p.numel() for p in model.parameters())
    conv_shape = f"{model.conv_channels}x{model.conv_spatial}x{model.conv_spatial}"
    print(f"model: arch={arch} channels={tuple(model.channels)} conv_latent={conv_shape} "
          f"latent_dim={model.latent_dim} normalize_latent={model.normalize_latent} "
          f"params={n_params:,}")

    if args.loss == "huber":
        loss_fn: nn.Module = nn.HuberLoss(delta=args.huber_delta)
    elif args.loss == "ms-ssim+l1":
        loss_fn = MSSSIML1Loss(data_range=args.ms_ssim_data_range, alpha=args.ms_ssim_alpha)
    else:
        loss_fn = nn.MSELoss()
    # A loss-independent yardstick: always report validation reconstruction MSE so runs trained
    # with different --loss (mse / huber / ms-ssim+l1) are directly comparable. Redundant with
    # val_loss when --loss mse, but cheap (val is a small held-out split, evaluated without grads).
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
        "normalize_latent": model.normalize_latent,
        # The dtype the weights are saved in; eval rebuilds the model in it (or up-casts to float32
        # on CPU, which is lossless from bfloat16). AMP trains float32 weights, so record float32.
        "weight_dtype": str(weight_dtype).removeprefix("torch."),
    }
    # Carry the input standardisation so inference/eval can reproduce the exact channel scaling
    # the model was trained under (the transform baked into the shards; absent if none was found).
    if channel_stats is not None:
        model_config["channel_mean"] = channel_stats["channel_mean"]
        model_config["channel_std"] = channel_stats["channel_std"]

    # Resume: reload model/optimizer/scaler state and continue at the next epoch. The architecture
    # is rebuilt from CLI/config above (not the checkpoint), so guard against a silent mismatch that
    # would make load_state_dict succeed on a differently-shaped model or corrupt the run.
    start_epoch = 1
    best_val = float("inf")
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
                "Pass the same --arch/--channels/--latent-dim the checkpoint was trained with."
            )
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if scaler is not None and ckpt.get("scaler_state") is not None:
            scaler.load_state_dict(ckpt["scaler_state"])
        # Older checkpoints predate best_val; fall back to that epoch's val_loss so best.pt is not
        # overwritten by a worse model on the first resumed epoch.
        best_val = ckpt.get("best_val", ckpt.get("val_loss", float("inf")))
        start_epoch = int(ckpt["epoch"]) + 1
        # Reattach to the original W&B run (if any) so the resumed epochs extend the same history
        # rather than opening a duplicate run.
        resume_wandb_id = ckpt.get("wandb_run_id")
        print(f"resume: loaded '{args.resume}' (epoch {ckpt['epoch']}, best_val {best_val:.6f}); "
              f"continuing at epoch {start_epoch}")

    # Optional Weights & Biases run. `run` is None when --wandb is not passed, which turns
    # every log call below into a no-op. Replicates share --wandb-group so W&B averages them.
    run = None
    if args.wandb:
        if wandb is None:
            raise SystemExit("--wandb was passed but wandb is not installed (pip install wandb)")
        # On resume, reattach to the checkpoint's run id (resume="allow" continues it, or starts a
        # fresh run under that id if the server has no history for it). A first run passes id=None,
        # letting W&B mint one; that id is then saved into every checkpoint below.
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            name=args.wandb_name or f"seed{args.seed}",
            mode=args.wandb_mode,
            id=resume_wandb_id,
            resume="allow" if resume_wandb_id is not None else None,
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
                "ms_ssim_alpha": args.ms_ssim_alpha if args.loss == "ms-ssim+l1" else None,
                "ms_ssim_data_range": args.ms_ssim_data_range if args.loss == "ms-ssim+l1" else None,
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
                "best_val": best_val,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                # Present only under AMP; restored on --resume so loss scaling continues seamlessly.
                "scaler_state": scaler.state_dict() if scaler is not None else None,
                # The W&B run id (if logging), so --resume reattaches to the same run.
                "wandb_run_id": run.id if run is not None else None,
                "model_config": model_config,
            },
            path,
        )

    # A machine-readable per-epoch log so replicate runs (e.g. tools/sweep.py) can be
    # aggregated/averaged after the fact. Flushed each epoch to survive interruptions. On resume,
    # append to the existing log (and skip the header) so earlier epochs are preserved.
    metrics_path = out_dir / "metrics.csv"
    resuming_log = args.resume is not None and metrics_path.exists()
    metrics_file = metrics_path.open("a" if resuming_log else "w", newline="")
    metrics_writer = csv.writer(metrics_file)
    if not resuming_log:
        metrics_writer.writerow(
            ["epoch", "train_loss", "val_loss", "val_mse",
             "grad_norm_mean", "grad_norm_max", "seconds"]
        )

    # Optional per-block grad-norm log (long format: one row per block per epoch), written only
    # when --log-layer-grads is set. Kept separate so the main metrics.csv schema is unchanged.
    layer_grad_file = None
    layer_grad_writer = None
    if args.log_layer_grads:
        layer_grad_path = out_dir / "layer_grad_norms.csv"
        resuming_lg = args.resume is not None and layer_grad_path.exists()
        layer_grad_file = layer_grad_path.open("a" if resuming_lg else "w", newline="")
        layer_grad_writer = csv.writer(layer_grad_file)
        if not resuming_lg:
            layer_grad_writer.writerow(["epoch", "block", "grad_norm"])

    # 3. Train ----------------------------------------------------------------
    # `best_val` and `start_epoch` were seeded above (inf / 1, or restored from --resume).
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_loss, gnorm_mean, gnorm_max, _, layer_gnorms = run_epoch(
            model, train_loader, device, loss_fn, optimizer, scaler,
            args.log_interval, epoch, "train", max_grad_norm=args.max_grad_norm,
            collect_layer_grads=args.log_layer_grads,
        )
        val_loss, _, _, val_mse, _ = run_epoch(
            model, val_loader, device, loss_fn, None, None, 0, epoch, "val",
            extra_loss_fn=val_mse_fn,
        )
        dt = time.time() - t0
        print(
            f"epoch {epoch:3d}/{args.epochs}  train {train_loss:.6f}  val {val_loss:.6f}  "
            f"val_mse {val_mse:.6f}  "
            f"grad_norm(mean {gnorm_mean:.3f}, max {gnorm_max:.3f})  ({dt:.1f}s)"
        )
        metrics_writer.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{val_mse:.6f}",
                                 f"{gnorm_mean:.6f}", f"{gnorm_max:.6f}", f"{dt:.2f}"])
        metrics_file.flush()

        if layer_grad_writer is not None:
            for block, gn in layer_gnorms.items():
                layer_grad_writer.writerow([epoch, block, f"{gn:.6e}"])
            layer_grad_file.flush()

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(out_dir / "best.pt", epoch, val_loss)
            print(f"  new best val {val_loss:.6f} -> {out_dir / 'best.pt'}")
        if args.save_interval and epoch % args.save_interval == 0:
            save_checkpoint(out_dir / f"epoch_{epoch:03d}.pt", epoch, val_loss)

        if run is not None:
            log_data = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_mse": val_mse,
                "best_val": best_val,
                "grad_norm_mean": gnorm_mean,
                "grad_norm_max": gnorm_max,
                "lr": optimizer.param_groups[0]["lr"],
                "epoch_seconds": dt,
            }
            log_data.update({f"grad_norm/{block}": gn for block, gn in layer_gnorms.items()})
            run.log(log_data, step=epoch)

    metrics_file.close()
    if layer_grad_file is not None:
        layer_grad_file.close()
    save_checkpoint(out_dir / "last.pt", args.epochs, best_val)
    print(f"done. best val loss {best_val:.6f}; checkpoints in '{out_dir}/'.")
    if run is not None:
        run.summary["best_val"] = best_val
        run.finish()


if __name__ == "__main__":
    main()
