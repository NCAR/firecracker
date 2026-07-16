"""
train_world_model_rollout.py

Decode-supervised latent-rollout world model (the "rollout" regime; distinct from the SPR/BYOL scheme
in train_world_model.py, which is left untouched). Encode frame 0 ONCE, roll an additive-residual
dynamics head forward, DECODE every rolled step, and supervise each decoded step against the real
future frame with a single fire-weighted reconstruction loss -- gradients flow decoder -> BPTT through
the dynamics -> the one t0 encoder call. No SPR cosine loss, no EMA target, no reward head, no latent
L2-norm (collapse is prevented by reconstruction itself + latent BatchNorm).

Why this regime: the SPR model's dynamics collapsed (predicted fire DECAYS over the rollout while the
real fire grows) because a latent-space cosine loss is minimized by a near-static latent. Supervising
the decoded rollout against the real, growing future forces the dynamics to actually propagate fire.

Key pieces:
  * uniform start sampling with pad-and-mask (traj_loader partial=True) so the encoder also sees late,
    fully-developed states; the loss is masked past each sequence's horizon (batch/BN width stay fixed).
  * per-channel + spatial-fire-front weighting so the sparse fire front and dynamic channels aren't
    drowned by the static structure (wm_losses.masked_weighted_bce).
  * on-the-fly temperature renormalization (wm_losses.compute_channel_gains) -- the min-max bake
    compressed temperatures into ~[0,0.03]; rescale so a high percentile maps to ~1. Recorded in the
    checkpoint so eval/inference reproduce it.

Backward compatibility: this is a NEW entrypoint. It reuses train_autoencoder's device/dtype/stats
helpers and the ConvAutoencoder/LatentTransition flags that already exist; it writes checkpoints in the
same dict shape as train_world_model.py plus a `loss_config`/`channel_gains` block.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

try:
    import wandb
except ImportError:
    wandb = None

from config import load_config
from rollout import FireDataset
from ram_loader import prewarm_shards
from traj_loader import TrajectoryDataset, TrajectoryBatchLoader, split_episodes
from dynamics import LatentTransition
from train_autoencoder import (
    ARCHITECTURES, WEIGHT_DTYPES, _resolve_device, _resolve_weight_dtype, _load_channel_stats,
)
import wm_losses as wml


# --- model construction -----------------------------------------------------

def build_ae(args, config, device, weight_dtype):
    """Build the autoencoder for the rollout regime: NO latent L2-norm by default (normalize_latent
    off), latent BatchNorm ON (anti-collapse + well-scaled latent). From scratch or from --init-from.
    Returns (model, model_config, channel_stats)."""
    normalize_latent = bool(args.normalize_latent)
    latent_bn = not args.no_latent_bn
    if args.init_from is not None:
        # WARM-START: reuse a pretrained AE's conv weights to bootstrap, but build the model with THIS
        # run's latent scheme (normalize_latent / latent_bn), not the AE's -- the goal is transfer of
        # spatial features, not the AE's exact latent geometry. The pretrained AE may have been trained
        # on differently-normalized (e.g. z-scored) data, so its inputs here are OOD; the channel stats
        # therefore come from the NEW dataset, never the AE. Load is non-strict because a latent-BN this
        # run adds (or the AE's absence of one) leaves a few buffers to initialize fresh.
        ckpt = torch.load(args.init_from, map_location=device)
        amc = ckpt["model_config"]
        kwargs = {"in_channels": amc["in_channels"], "grid_size": amc["grid_size"],
                  "channels": tuple(amc["channels"]), "latent_dim": amc["latent_dim"],
                  "normalize_latent": normalize_latent}
        if amc.get("bottleneck_channels") is not None:
            kwargs["bottleneck_channels"] = amc["bottleneck_channels"]
        if latent_bn:
            kwargs["latent_bn"] = True
        model = ARCHITECTURES[amc["arch"]](**kwargs).to(device=device, dtype=weight_dtype)
        missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
        fresh = sorted({m.split(".")[0] for m in missing})
        print(f"init(warm-start): AE '{args.init_from}' arch={amc['arch']}; loaded non-strict "
              f"(fresh-init: {fresh or 'none'}; unexpected: {len(unexpected)})")
        _shrink_and_perturb(model, ckpt["model_state"], shrink=args.warmstart_shrink,
                            noise=args.warmstart_noise)
        mcfg = {"arch": amc["arch"], "in_channels": amc["in_channels"], "grid_size": amc["grid_size"],
                "channels": list(model.channels), "latent_dim": model.latent_dim,
                "normalize_latent": model.normalize_latent, "bottleneck_channels": model.bottleneck_channels,
                "latent_bn": model.latent_bn, "weight_dtype": str(weight_dtype).removeprefix("torch."),
                "out_activation": "sigmoid", "warm_started_from": str(args.init_from)}
        return model, mcfg, _load_channel_stats(args.data)          # NEW data's stats, not the AE's

    ae_cfg = config.get("autoencoder", {})
    arch = args.arch or ae_cfg.get("arch", "pooled")
    arch_cfg = ae_cfg.get(arch, {})
    in_channels, grid_size = int(args._in_channels), int(args._grid_size)
    kwargs = {"in_channels": in_channels, "grid_size": grid_size, "normalize_latent": normalize_latent}
    channels = args.channels if args.channels is not None else arch_cfg.get("channels")
    if channels is not None:
        kwargs["channels"] = tuple(channels)
    latent_dim = args.latent_dim if args.latent_dim is not None else arch_cfg.get("latent_dim")
    if latent_dim is not None:
        kwargs["latent_dim"] = latent_dim
    if arch_cfg.get("bottleneck_channels") is not None:
        kwargs["bottleneck_channels"] = arch_cfg["bottleneck_channels"]
    if latent_bn:
        kwargs["latent_bn"] = True
    model = ARCHITECTURES[arch](**kwargs).to(device=device, dtype=weight_dtype)
    model_config = {
        "arch": arch, "in_channels": in_channels, "grid_size": grid_size,
        "channels": list(model.channels), "latent_dim": model.latent_dim,
        "normalize_latent": model.normalize_latent, "bottleneck_channels": model.bottleneck_channels,
        "latent_bn": model.latent_bn, "weight_dtype": str(weight_dtype).removeprefix("torch."),
        "out_activation": "sigmoid",
    }
    return model, model_config, _load_channel_stats(args.data)


@torch.no_grad()
def _shrink_and_perturb(model, loaded_state, *, shrink=1.0, noise=0.0):
    """Ash & Adams (2020) shrink-and-perturb on the warm-started weights to counter loss of plasticity:
    theta <- shrink*theta + noise*eps, eps ~ N(0, std(theta_layer)). Only params present in the loaded
    AE state are touched (freshly-initialized ones are left alone). A no-op at shrink=1, noise=0. Small
    shrink / noise nudges the pretrained weights off their converged basin so gradient descent can still
    reshape them for the new (OOD) data, while keeping most of the transferred spatial structure."""
    if shrink == 1.0 and noise == 0.0:
        return
    for name, p in model.named_parameters():
        if name not in loaded_state:
            continue
        s = float(p.detach().float().std()) or 1.0
        p.mul_(shrink)
        if noise > 0.0:
            p.add_(torch.randn_like(p) * (noise * s))
    print(f"warm-start shrink-and-perturb: shrink={shrink} noise={noise} (plasticity aid)")


def encoder_param_names(prefixes=("encoder.", "enc_project.", "to_latent.")):
    return prefixes


def trained_params(model, dynamics, *, freeze_ae, freeze_encoder):
    """Parameters the optimizer updates, honoring the freeze tier."""
    params = list(dynamics.parameters())
    if freeze_ae:
        for p in model.parameters():
            p.requires_grad_(False)
        return params
    enc_pref = encoder_param_names()
    for n, p in model.named_parameters():
        if freeze_encoder and n.startswith(enc_pref):
            p.requires_grad_(False)
        else:
            params.append(p)
    return params


# --- rollout + loss ---------------------------------------------------------

def rollout_logits(model, dynamics, x0, steps, *, freeze_encoder, return_latents=False):
    """Encode x0 once, roll `steps-1` dynamics steps, decode EVERY step. Returns (B, steps, C, H, W)
    decoder logits. Gradient reaches the encoder only through z0; through the dynamics via BPTT.
    With `return_latents`, also returns the per-step latents (B, steps, latent_dim) so the caller can
    track how the rolled latent's magnitude drifts across the rollout (unbounded here -- no output norm)."""
    # no_grad only when the encoder is frozen; otherwise inherit the caller's grad mode (nullcontext),
    # so a val pass under torch.no_grad() stays grad-free instead of being force-enabled here.
    enc_ctx = torch.no_grad() if freeze_encoder else nullcontext()
    with enc_ctx:
        z = model.encode(x0)
    zs = [z]
    outs = [model.decode(z)]
    for _ in range(1, steps):
        z = dynamics(z)
        zs.append(z)
        outs.append(model.decode(z))
    logits = torch.stack(outs, dim=1)
    if return_latents:
        return logits, torch.stack(zs, dim=1)       # (B, steps, latent_dim)
    return logits


@dataclass
class EpochStats:
    """One epoch's aggregates. Split out (instead of an ever-growing tuple) so anchor/change and the
    static/dynamic MAE splits have named homes."""
    loss: float                 # optimized objective: anchor (w_change==0) or w_anchor*anchor + w_change*change
    anchor: float               # L_anchor: full-frame channel-weighted recon (static fidelity)
    change: float               # L_change: change-cell BCE (0.0 when w_change==0)
    mae: np.ndarray             # (C,) per-channel model MAE over valid steps
    persist: np.ndarray         # (C,) per-channel MAE of the copy-previous-frame baseline
    fire_mae: float             # mean model MAE over the dynamic (fire) channels
    grad_mean: float
    grad_max: float
    lnorm: np.ndarray           # (S,) per-rollout-step latent L2 norm
    skipped: int


def run_epoch(model, dynamics, loader, device, optimizer, scaler, *, gains, channel_w, ignited_idx,
              anchor_front, front_dilate, w_anchor, w_change, change_eps, max_grad_norm,
              freeze_ae, freeze_encoder, use_amp, training, fire_idx):
    model.train(training and not freeze_ae)
    dynamics.train(training)
    tot = anchor_tot = change_tot = n_seen = 0.0
    gsum = gmax = 0.0
    gclip = skipped = 0                               # gclip: optimizer steps with a finite grad norm;
                                                      # skipped: fp16-overflow steps the scaler discards
    mae_sum = np.zeros(len(channel_w))
    persist_sum = np.zeros(len(channel_w))            # copy-previous-frame baseline MAE (the bar to beat)
    lnorm_sum = None                                  # per-rollout-step latent L2 norm, lazily sized to S
    for batch in loader:
        x, mask = batch                                   # partial loader -> (x, mask)
        x = x.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        if gains is not None:
            x = wml.apply_gains(x, gains)                 # temperature renorm (targets + inputs)
        b, s = x.shape[:2]
        with torch.set_grad_enabled(training), torch.autocast(device_type=device.type, enabled=use_amp):
            logits, zs = rollout_logits(model, dynamics, x[:, 0], s, freeze_encoder=freeze_encoder,
                                        return_latents=True)
            # L_anchor: full-frame reconstruction that keeps the static majority (terrain/fuel) faithful.
            anchor = wml.masked_weighted_bce(logits, x, mask, channel_w, ignited_idx=ignited_idx,
                                             front_weight=anchor_front, front_dilate=front_dilate)
            if w_change > 0:
                # L_change: BCE on cells that change between consecutive real frames (steps 1..K), so a
                # persistence prediction can't win. prev = x[:, :-1], target = x[:, 1:].
                change = wml.masked_change_bce(logits[:, 1:], x[:, 1:], x[:, :-1], mask[:, 1:], channel_w,
                                               eps=change_eps, ignited_idx=ignited_idx)
                loss = w_anchor * anchor + w_change * change
            else:
                change = torch.zeros((), device=logits.device)      # single-term (backward-compatible)
                loss = anchor
        if training:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                if max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    gn = float(nn.utils.clip_grad_norm_(_grad_params(model, dynamics), max_grad_norm))
                    # A non-finite norm means an fp16 grad overflowed; scaler.step() will skip this
                    # update and back off the loss scale. Don't fold it into the stats -- it poisons
                    # the mean/max with inf/nan -- just count it.
                    if math.isfinite(gn):
                        gsum += gn; gmax = max(gmax, gn); gclip += 1
                    else:
                        skipped += 1
                scaler.step(optimizer); scaler.update()
            else:
                loss.backward()
                if max_grad_norm > 0:
                    gn = float(nn.utils.clip_grad_norm_(_grad_params(model, dynamics), max_grad_norm))
                    if math.isfinite(gn):
                        gsum += gn; gmax = max(gmax, gn); gclip += 1
                    else:
                        skipped += 1
                optimizer.step()
        with torch.no_grad():
            mae = wml.per_channel_recon_stats(logits.float().sigmoid(), x.float(), mask)
            # Persistence baseline: MAE of copying the previous real frame over the transition steps.
            # Data-only, so it's the fixed bar the model must beat -- model dyn MAE >= this == collapse.
            persist = wml.persistence_mae(x[:, 1:].float(), x[:, :-1].float(), mask[:, 1:])
            # Per-step latent magnitude (mean L2 norm over the batch), NOT masked: drift accumulates on
            # every rolled latent regardless of whether its decoded step is a padded/masked target.
            step_norm = zs.detach().float().norm(dim=-1).mean(dim=0)      # (S,)
        if lnorm_sum is None:
            lnorm_sum = np.zeros(step_norm.shape[0])
        lnorm_sum += step_norm.cpu().numpy() * b
        mae_sum += mae.cpu().numpy() * b
        persist_sum += persist.cpu().numpy() * b
        tot += float(loss.item()) * b
        anchor_tot += float(anchor.item()) * b
        change_tot += float(change.item()) * b
        n_seen += b
    inv = 1.0 / max(n_seen, 1)
    mae_mean = mae_sum * inv
    fire_mae = float(mae_mean[fire_idx].mean()) if len(fire_idx) else 0.0
    lnorm_mean = (lnorm_sum * inv) if lnorm_sum is not None else np.zeros(1)
    return EpochStats(loss=tot * inv, anchor=anchor_tot * inv, change=change_tot * inv,
                      mae=mae_mean, persist=persist_sum * inv, fire_mae=fire_mae,
                      grad_mean=gsum / max(gclip, 1), grad_max=gmax, lnorm=lnorm_mean, skipped=skipped)


def _grad_params(model, dynamics):
    return [p for p in (*model.parameters(), *dynamics.parameters()) if p.requires_grad]


# --- main -------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Decode-supervised latent-rollout world model.")
    p.add_argument("--data", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--num-envs", type=int, default=None)
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--window", type=int, default=9, help="decoded steps per sequence (t0..t_{K}); K=window-1")
    p.add_argument("--step", type=int, default=1, help="stored-frame gap per rollout step")
    p.add_argument("--prewarm", action="store_true")
    # model / regime
    p.add_argument("--init-from", default=None, help="warm-start encoder+decoder from a pretrained AE")
    p.add_argument("--warmstart-shrink", type=float, default=1.0,
                   help="shrink-and-perturb: scale warm-started weights (<1 restores plasticity; 1=off)")
    p.add_argument("--warmstart-noise", type=float, default=0.0,
                   help="shrink-and-perturb: add gaussian noise = this * per-layer weight std (0=off)")
    p.add_argument("--freeze-ae", action="store_true", help="freeze encoder+decoder, train only dynamics")
    p.add_argument("--freeze-encoder", action="store_true", help="freeze encoder, train decoder+dynamics")
    p.add_argument("--arch", default=None)
    p.add_argument("--channels", type=lambda s: [int(x) for x in s.split(",") if x.strip()], default=None)
    p.add_argument("--latent-dim", type=int, default=None)
    p.add_argument("--normalize-latent", action="store_true", help="re-enable latent L2-norm (default OFF)")
    p.add_argument("--no-latent-bn", action="store_true", help="disable latent BatchNorm (default ON)")
    p.add_argument("--dyn-depth", type=int, default=2)
    p.add_argument("--dyn-hidden", type=int, default=None)
    p.add_argument("--dyn-norm", default="layernorm", choices=("batchnorm", "layernorm", "none"),
                   help="dynamics-head hidden-layer norm; layernorm avoids BatchNorm's rollout-depth "
                        "stat mixing when the head is applied K times (default: layernorm)")
    # loss
    p.add_argument("--w-temp", type=float, default=4.0, help="per-channel weight on temperature channels")
    p.add_argument("--w-wind", type=float, default=3.0, help="per-channel weight on wind channels")
    p.add_argument("--w-ignited", type=float, default=6.0, help="per-channel weight on the ignited mask")
    p.add_argument("--front-weight", type=float, default=4.0, help="extra weight on burning-front cells")
    p.add_argument("--front-dilate", type=int, default=2, help="dilation radius of the fire-front mask")
    # two-term (anchor + change) decomposition -- OFF by default (w-change 0 == the original loss)
    p.add_argument("--w-change", type=float, default=0.0,
                   help="weight on the change-cell BCE term; >0 enables the anti-identity decomposition "
                        "(loss = w_anchor*anchor + w_change*change). 0 = original single-term loss.")
    p.add_argument("--w-anchor", type=float, default=0.2,
                   help="weight on the full-frame reconstruction anchor (only used when --w-change>0)")
    p.add_argument("--change-eps", type=float, default=1e-3,
                   help="min per-cell |target-prev| (on gain-normalized [0,1] channels) counted as change")
    p.add_argument("--anchor-front", type=float, default=None,
                   help="front up-weight inside the anchor term; default: front-weight when w-change==0 "
                        "(backward compat), else 0 (front emphasis lives in the change term)")
    p.add_argument("--temp-pct", type=float, default=99.0, help="percentile mapped to 1 when renorming temps")
    p.add_argument("--temp-max-gain", type=float, default=25.0)
    p.add_argument("--no-temp-renorm", action="store_true", help="disable Option-B temperature rescaling")
    # optim
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--adam-eps", type=float, default=1e-8)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--weight-dtype", default="float32", choices=list(WEIGHT_DTYPES))
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--loader-workers", type=int, default=4)
    # bookkeeping
    p.add_argument("--out", default="runs/wm_rollout")
    p.add_argument("--resume", default=None)
    p.add_argument("--save-interval", type=int, default=0)
    p.add_argument("--log-interval", type=int, default=0)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", default="firecracker-world-model")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-group", default=None)
    p.add_argument("--wandb-name", default=None)
    p.add_argument("--wandb-mode", default="online", choices=("online", "offline", "disabled"))
    args = p.parse_args()
    if args.resume is not None and args.init_from is not None:
        raise SystemExit("--resume and --init-from are mutually exclusive.")
    if args.freeze_ae and args.freeze_encoder:
        raise SystemExit("--freeze-ae already freezes the encoder; drop --freeze-encoder.")
    if (args.freeze_ae or args.freeze_encoder) and args.init_from is None and args.resume is None:
        raise SystemExit("--freeze-* requires --init-from (nothing to freeze when training from scratch).")

    torch.manual_seed(args.seed)
    config = load_config(args.config)
    device = _resolve_device(args.device)
    weight_dtype = _resolve_weight_dtype(args.weight_dtype, device)
    if args.amp and WEIGHT_DTYPES[args.weight_dtype] is not torch.float32:
        raise SystemExit("--amp expects --weight-dtype float32 (see the bf16-AMP-stall note).")

    # data ------------------------------------------------------------------
    dataset = FireDataset(args.data)
    if not dataset.is_normalized:
        print(f"WARNING: dataset '{args.data}' is not marked normalized; bake it first "
              f"(python tools/normalize_dataset.py --data {args.data}).")
    args._in_channels = int(dataset.meta["num_channels"])
    args._grid_size = int(dataset.meta["grid_size"])
    names = list(dataset.meta.get("channels", [f"ch{i}" for i in range(args._in_channels)]))
    if args.window < 2:
        raise SystemExit("--window must be >= 2 (t0 + at least one target).")
    traj = TrajectoryDataset(dataset, num_envs=args.num_envs, window=args.window, step=args.step,
                             partial=True)
    train_w, val_w = split_episodes(traj, val_frac=args.val_frac, seed=args.seed)
    if args.prewarm:
        prewarm_shards(dataset)
    pin = device.type == "cuda"
    train_loader = TrajectoryBatchLoader(traj, train_w, args.batch_size, shuffle=True, seed=args.seed,
                                         drop_last=True, pin_memory=pin, num_workers=args.loader_workers)
    val_loader = TrajectoryBatchLoader(traj, val_w, args.batch_size, shuffle=False, seed=args.seed,
                                       drop_last=False, pin_memory=pin, num_workers=args.loader_workers)
    print(f"dataset: {len(dataset)} frames -> {len(traj)} windows (partial); "
          f"{len(train_w)} train / {len(val_w)} val; window={args.window} step={args.step}")

    # model -----------------------------------------------------------------
    model, model_config, channel_stats = build_ae(args, config, device, weight_dtype)
    if channel_stats is not None:
        # Merge ONLY the numeric normalization stats. A blind {**mc, **stats} would clobber
        # model_config["channels"] (encoder widths) with the stats' "channels" (per-channel NAMES) --
        # the exact pollution that broke a prior artifact. Keep the widths; stash names separately.
        for k in ("channel_min", "channel_max", "channel_mean", "channel_std"):
            if k in channel_stats:
                model_config[k] = channel_stats[k]
        if "channels" in channel_stats:
            model_config["channel_names"] = channel_stats["channels"]
    # The dynamics head re-normalizes its output onto the unit sphere IFF the encoder does (normalize
    # _latent): the rolled latents z_1..z_K must live on the same manifold the decoder was trained on,
    # or decode() sees off-distribution inputs. So a from-scratch (normalize_latent=False) run gets a
    # pure additive residual, while a warm-start from an L2-normalized AE keeps every rolled step on the
    # sphere its decoder expects. Residual (skip-through-time) is always on.
    dyn_norm = bool(model.normalize_latent)
    dynamics = LatentTransition(model.latent_dim, hidden_dim=args.dyn_hidden, depth=args.dyn_depth,
                                residual=True, normalize_output=dyn_norm, norm=args.dyn_norm
                                ).to(device=device, dtype=weight_dtype)
    dyn_config = {"latent_dim": model.latent_dim, "hidden_dim": dynamics.hidden_dim,
                  "depth": args.dyn_depth, "residual": True, "normalize_output": dyn_norm,
                  "norm": dynamics.norm}

    # loss weights + temperature renorm gains -------------------------------
    channel_w = wml.build_channel_weights(names, temp=args.w_temp, wind=args.w_wind, ignited=args.w_ignited)
    ignited_idx = wml.ignited_index(names)
    roles = wml.channel_roles(names)
    fire_idx = sorted(roles["temp"] + roles["wind"] + roles["ignited"])
    static_idx = sorted(roles["static"])
    # anchor_front: when the change term is off, the anchor keeps the original front weighting (exact
    # backward compat); when it's on, the front emphasis lives in the change mask so the anchor drops
    # it (default 0) unless explicitly overridden.
    if args.anchor_front is not None:
        anchor_front = args.anchor_front
    else:
        anchor_front = 0.0 if args.w_change > 0 else args.front_weight
    gains = None
    if not args.no_temp_renorm:
        sample = _sample_frames(train_loader, n_batches=4)
        gains = wml.compute_channel_gains(sample, names, pct=args.temp_pct, max_gain=args.temp_max_gain)
        applied = {names[i]: round(float(g), 2) for i, g in enumerate(gains) if abs(float(g) - 1.0) > 1e-6}
        print(f"temp renorm gains (Option B): {applied}")
        gains = gains.to(device)
    loss_config = {"w_temp": args.w_temp, "w_wind": args.w_wind, "w_ignited": args.w_ignited,
                   "front_weight": args.front_weight, "front_dilate": args.front_dilate,
                   "temp_pct": args.temp_pct, "no_temp_renorm": args.no_temp_renorm,
                   "w_anchor": args.w_anchor, "w_change": args.w_change, "change_eps": args.change_eps,
                   "anchor_front": anchor_front}
    model_config["channel_gains"] = None if gains is None else gains.cpu().tolist()
    model_config["fire_channel_weights"] = channel_w.tolist()

    optimizer = torch.optim.AdamW(trained_params(model, dynamics, freeze_ae=args.freeze_ae,
                                                 freeze_encoder=args.freeze_encoder),
                                  lr=args.lr, weight_decay=args.weight_decay, eps=args.adam_eps)
    scaler = torch.cuda.amp.GradScaler() if (args.amp and device.type == "cuda") else None

    n_ae = sum(p.numel() for p in model.parameters())
    n_dyn = sum(p.numel() for p in dynamics.parameters())
    print(f"model: ae={n_ae:,} (freeze_ae={args.freeze_ae} freeze_enc={args.freeze_encoder}) + "
          f"dyn={n_dyn:,}  latent_dim={model.latent_dim} normalize_latent={model.normalize_latent} "
          f"latent_bn={model.latent_bn}")

    # resume ----------------------------------------------------------------
    start_epoch, best_val, resume_wandb_id = 1, float("inf"), None
    if args.resume is not None:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model_state"]); dynamics.load_state_dict(ck["dynamics_state"])
        optimizer.load_state_dict(ck["optimizer_state"])
        if scaler is not None and ck.get("scaler_state") is not None:
            scaler.load_state_dict(ck["scaler_state"])
        best_val = ck.get("best_val", ck.get("val_loss", float("inf")))
        start_epoch = int(ck["epoch"]) + 1
        resume_wandb_id = ck.get("wandb_run_id")
        print(f"resume: '{args.resume}' epoch {ck['epoch']} -> continuing at {start_epoch}")

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    run = None
    if args.wandb:
        if wandb is None:
            raise SystemExit("--wandb but wandb not installed.")
        run = wandb.init(project=args.wandb_project, entity=args.wandb_entity, group=args.wandb_group,
                         name=args.wandb_name or f"seed{args.seed}", mode=args.wandb_mode,
                         id=resume_wandb_id, resume="allow" if resume_wandb_id else None,
                         config={**model_config, **dyn_config, **loss_config,
                                 "seed": args.seed, "epochs": args.epochs, "batch_size": args.batch_size,
                                 "lr": args.lr, "window": args.window, "step": args.step})

    def save(path, epoch, val_loss):
        ck = {"epoch": epoch, "val_loss": val_loss, "best_val": best_val, "regime": "decode_rollout",
              "model_state": model.state_dict(), "model_config": model_config,
              "dynamics_state": dynamics.state_dict(), "dynamics_config": dyn_config,
              "loss_config": loss_config, "freeze_ae": args.freeze_ae, "freeze_encoder": args.freeze_encoder,
              "wandb_run_id": run.id if run is not None else None,
              "optimizer_state": optimizer.state_dict(),
              "scaler_state": scaler.state_dict() if scaler is not None else None}
        torch.save(ck, path)

    metrics_path = out_dir / "metrics.csv"
    resuming_log = args.resume is not None and metrics_path.exists()
    mf = metrics_path.open("a" if resuming_log else "w", newline="")
    mw = csv.writer(mf)
    if not resuming_log:
        mw.writerow(["epoch", "train_loss", "val_loss", "val_mae", "val_fire_mae",
                     "train_anchor", "train_change", "val_anchor", "val_change",
                     "val_static_mae", "val_dynamic_mae", "persist_static_mae", "persist_dynamic_mae",
                     "grad_norm_mean", "grad_norm_max", "skipped_steps", "seconds"])

    def _split(vec):
        """(static_mean, dynamic_mean) of a (C,) per-channel vector."""
        stat = float(vec[static_idx].mean()) if static_idx else 0.0
        dyn = float(vec[fire_idx].mean()) if fire_idx else 0.0
        return stat, dyn

    common = dict(gains=gains, channel_w=channel_w.to(device), ignited_idx=ignited_idx,
                  anchor_front=anchor_front, front_dilate=args.front_dilate,
                  w_anchor=args.w_anchor, w_change=args.w_change, change_eps=args.change_eps,
                  max_grad_norm=args.max_grad_norm, freeze_ae=args.freeze_ae,
                  freeze_encoder=args.freeze_encoder, use_amp=bool(scaler is not None), fire_idx=fire_idx)
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, dynamics, train_loader, device, optimizer, scaler, training=True, **common)
        va = run_epoch(model, dynamics, val_loader, device, None, None, training=False, **common)
        va_lnorm = va.lnorm
        va_static_mae, va_dynamic_mae = _split(va.mae)
        persist_static_mae, persist_dynamic_mae = _split(va.persist)
        beats_persistence = va_dynamic_mae < persist_dynamic_mae     # False == still learning identity
        # Latent drift: ratio of the last rolled step's mean latent norm to z0's. ~1 = stable magnitude;
        # a growing ratio is the unbounded-latent drift the no-output-norm dynamics risks past the horizon.
        drift = float(va_lnorm[-1] / va_lnorm[0]) if va_lnorm[0] > 0 and len(va_lnorm) > 1 else 1.0
        dt = time.time() - t0
        print(f"epoch {epoch:3d}/{args.epochs}  train {tr.loss:.5f}  val {va.loss:.5f}  "
              f"(anc {va.anchor:.4f} chg {va.change:.4f})  "
              f"mae stat {va_static_mae:.4f}/dyn {va_dynamic_mae:.4f} "
              f"(persist dyn {persist_dynamic_mae:.4f} {'BEATS' if beats_persistence else 'IDENTITY'})  "
              f"grad(mean {tr.grad_mean:.2f} max {tr.grad_max:.2f} skip {tr.skipped})  "
              f"latent(z0 {va_lnorm[0]:.2f} zK {va_lnorm[-1]:.2f} drift {drift:.2f})  ({dt:.0f}s)")
        mw.writerow([epoch, f"{tr.loss:.6f}", f"{va.loss:.6f}", f"{va.mae.mean():.6f}", f"{va.fire_mae:.6f}",
                     f"{tr.anchor:.6f}", f"{tr.change:.6f}", f"{va.anchor:.6f}", f"{va.change:.6f}",
                     f"{va_static_mae:.6f}", f"{va_dynamic_mae:.6f}",
                     f"{persist_static_mae:.6f}", f"{persist_dynamic_mae:.6f}",
                     f"{tr.grad_mean:.6f}", f"{tr.grad_max:.6f}", tr.skipped, f"{dt:.2f}"]); mf.flush()
        if va.loss < best_val:
            best_val = va.loss
            save(out_dir / "best.pt", epoch, va.loss)
            print(f"  new best val {va.loss:.5f} -> {out_dir/'best.pt'}")
        save(out_dir / "last.pt", epoch, va.loss)
        if args.save_interval and epoch % args.save_interval == 0:
            save(out_dir / f"epoch_{epoch:03d}.pt", epoch, va.loss)
        if run is not None:
            log = {"epoch": epoch, "train_loss": tr.loss, "val_loss": va.loss, "val_mae": float(va.mae.mean()),
                   "val_fire_mae": va.fire_mae, "best_val": best_val,
                   "train/anchor": tr.anchor, "train/change": tr.change,
                   "val/anchor": va.anchor, "val/change": va.change,
                   "val/mae_static": va_static_mae, "val/mae_dynamic": va_dynamic_mae,
                   "baseline/persist_static": persist_static_mae, "baseline/persist_dynamic": persist_dynamic_mae,
                   "val/beats_persistence": float(beats_persistence),
                   "grad_norm_mean": tr.grad_mean, "grad_norm_max": tr.grad_max, "skipped_steps": tr.skipped,
                   "epoch_seconds": dt, "val/latent_norm_z0": float(va_lnorm[0]),
                   "val/latent_norm_zK": float(va_lnorm[-1]), "val/latent_drift_ratio": drift}
            # Per-step latent norm as separate series so the drift-vs-rollout-step curve is visible in W&B.
            for k, v in enumerate(va_lnorm):
                log[f"val/latent_norm_step_{k:02d}"] = float(v)
            run.log(log, step=epoch)
    mf.close()
    print(f"done. best val {best_val:.5f}; checkpoints in '{out_dir}/'.")
    if run is not None:
        run.summary["best_val"] = best_val; run.finish()


def _sample_frames(loader, n_batches=4):
    """Grab a few batches' frames (B*S, C, H, W) to estimate percentile gains."""
    got = []
    for i, batch in enumerate(loader):
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        got.append(x.reshape(-1, *x.shape[2:]))
        if i + 1 >= n_batches:
            break
    return torch.cat(got, dim=0)


if __name__ == "__main__":
    main()
