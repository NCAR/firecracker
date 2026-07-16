"""
wm_losses.py

Losses and normalization helpers for the decode-supervised latent-rollout world model
(src/train_world_model_rollout.py).

The world model encodes frame 0 once, rolls the (additive-residual) dynamics head forward, and decodes
EVERY rolled step, supervising each decoded step against the real future frame with a single
reconstruction loss (BPTT through the rollout). Two problems the plain per-pixel loss has on this data
are addressed here:

  * static, spatially-large channels (terrain / biome / fuel layout) dominate a per-pixel-averaged
    loss, so the sparse fire front and the near-baseline dynamic channels wash out. -> per-channel
    weights (emphasize temperatures / wind / ignited) and a spatial weight on the fire front.
  * min-max baking compressed the temperature channels into a sliver of [0,1] (baked air_temp p99
    ~0.03). -> `compute_channel_gains` rescales those channels so a high percentile maps to ~1, on the
    fly, no re-bake. The gains are recorded in the checkpoint so eval/inference reproduce them.

The decoder emits logits (sigmoid + BCE), matching the trained decoder head.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


# Channel-role resolution ----------------------------------------------------

def channel_roles(names: list[str]) -> dict[str, list[int]]:
    """Group channel indices by role from their names (temperatures / wind / ignited / static)."""
    roles: dict[str, list[int]] = {"temp": [], "wind": [], "ignited": [], "static": []}
    for i, n in enumerate(names):
        nl = n.lower()
        if "temperature" in nl:
            roles["temp"].append(i)
        elif nl.startswith("wind"):
            roles["wind"].append(i)
        elif nl == "ignited":
            roles["ignited"].append(i)
        else:
            roles["static"].append(i)
    return roles


def ignited_index(names: list[str]) -> int | None:
    for i, n in enumerate(names):
        if n.lower() == "ignited":
            return i
    return None


def build_channel_weights(names: list[str], *, temp: float, wind: float, ignited: float,
                          default: float = 1.0) -> torch.Tensor:
    """Per-channel loss weight (C,): emphasize the dynamic fire channels, everything else = `default`."""
    roles = channel_roles(names)
    w = torch.full((len(names),), float(default))
    for i in roles["temp"]:
        w[i] = temp
    for i in roles["wind"]:
        w[i] = wind
    for i in roles["ignited"]:
        w[i] = ignited
    return w


# Temperature (and any compressed-channel) renormalization -- Option B ---------

def compute_channel_gains(sample_frames: torch.Tensor, names: list[str], *, pct: float = 99.0,
                          max_gain: float = 25.0, roles=("temp",)) -> torch.Tensor:
    """Per-channel multiplicative gain (C,) that rescales compressed channels so their `pct` percentile
    maps to ~1.0, then clamps back into [0,1]. Only channels in `roles` get a gain != 1; the rest are 1.

    `sample_frames`: (N, C, H, W) baked-normalized frames (a modest sample is plenty). The gains are a
    pure affine on already-baked data, so no dataset re-bake is needed; record them in the checkpoint.
    """
    role_map = channel_roles(names)
    targets = set()
    for r in roles:
        targets.update(role_map.get(r, []))
    gains = torch.ones(len(names))
    flat = sample_frames.flatten(2)                        # (N, C, H*W)
    for c in sorted(targets):
        # np.percentile (not torch.quantile, which errors above ~16M elements) so a realistic sample of
        # a full-res channel -- tens of millions of values -- works.
        vals = flat[:, c].reshape(-1).float().cpu().numpy()
        if vals.size:
            gains[c] = min(1.0 / max(float(np.percentile(vals, pct)), 1e-4), float(max_gain))
    return gains


def apply_gains(x: torch.Tensor, gains: torch.Tensor) -> torch.Tensor:
    """Apply per-channel gains and clamp to [0,1]. `x`: (..., C, H, W); `gains`: (C,)."""
    shape = [1] * x.dim()
    shape[-3] = gains.numel()
    return (x * gains.view(shape).to(x)).clamp_(0.0, 1.0)


# Masked, weighted, multi-step reconstruction loss ---------------------------

def masked_weighted_bce(logits: torch.Tensor, target: torch.Tensor, step_mask: torch.Tensor,
                        channel_w: torch.Tensor, *, ignited_idx: int | None = None,
                        front_weight: float = 0.0, front_dilate: int = 1) -> torch.Tensor:
    """Per-pixel BCE-with-logits, weighted per channel and (optionally) up-weighted on the fire front,
    masked to the valid rollout steps, reduced to a weighted mean.

    Shapes: logits, target = (B, S, C, H, W); step_mask = (B, S) bool/float; channel_w = (C,).
    S = window = decoded steps (t0 .. tK). `front_weight` up-weights cells in (a dilation of) the
    target burning region to beat the sparse-fire imbalance.
    """
    B, S, C, H, W = logits.shape
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")   # (B,S,C,H,W)

    w = channel_w.to(bce).view(1, 1, C, 1, 1)
    weight = w * step_mask.to(bce).view(B, S, 1, 1, 1)
    if front_weight > 0 and ignited_idx is not None:
        ign = target[:, :, ignited_idx:ignited_idx + 1]                          # (B,S,1,H,W)
        if front_dilate > 0:
            k = 2 * front_dilate + 1
            ign = F.max_pool2d(ign.reshape(B * S, 1, H, W), k, 1, front_dilate).reshape(B, S, 1, H, W)
        weight = weight * (1.0 + front_weight * (ign > 0.5).to(bce))             # (B,S,C,H,W)

    num = (bce * weight).sum()
    den = weight.expand_as(bce).sum().clamp_min(1.0)
    return num / den


def masked_change_bce(logits: torch.Tensor, target: torch.Tensor, prev: torch.Tensor,
                      step_mask: torch.Tensor, channel_w: torch.Tensor, *, eps: float = 1e-3,
                      ignited_idx: int | None = None, front_weight: float = 0.0,
                      front_dilate: int = 1) -> torch.Tensor:
    """Per-pixel BCE-with-logits restricted to cells that CHANGE between consecutive real frames,
    normalized over that changed-cell mass alone.

    Supervises the *transition* (target vs the previous real frame `prev`) rather than the absolute
    state. The per-cell weight is nonzero only where |target - prev| > eps, and the denominator sums
    only those cells, so the loss magnitude is constant per step regardless of how little moved. A
    persistence prediction (decode == prev) is maximally wrong on every changed cell, so identity is no
    longer a loss minimum. This is the `L_change` term; `masked_weighted_bce` is the `L_anchor` term
    that preserves the static majority (see docs/wm_two_term_loss_plan.md).

    Shapes: logits, target, prev = (B, S, C, H, W); step_mask = (B, S); channel_w = (C,). Here S is the
    number of *transition* steps supervised (rollout steps 1..K; step 0 has no previous real frame, so
    callers pass the [:, 1:] slice for logits/target and [:, :-1] for prev).
    """
    B, S, C, H, W = logits.shape
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")   # (B,S,C,H,W)
    changed = ((target - prev).abs() > eps).to(bce)                              # (B,S,C,H,W)

    w = channel_w.to(bce).view(1, 1, C, 1, 1)
    weight = w * step_mask.to(bce).view(B, S, 1, 1, 1) * changed
    if front_weight > 0 and ignited_idx is not None:
        ign = target[:, :, ignited_idx:ignited_idx + 1]                          # (B,S,1,H,W)
        if front_dilate > 0:
            k = 2 * front_dilate + 1
            ign = F.max_pool2d(ign.reshape(B * S, 1, H, W), k, 1, front_dilate).reshape(B, S, 1, H, W)
        weight = weight * (1.0 + front_weight * (ign > 0.5).to(bce))             # (B,S,C,H,W)

    num = (bce * weight).sum()
    den = weight.sum().clamp_min(1.0)
    return num / den


@torch.no_grad()
def persistence_mae(target: torch.Tensor, prev: torch.Tensor, step_mask: torch.Tensor):
    """Diagnostic (not a loss): per-channel MAE of the copy-previous-frame baseline, |target - prev|,
    reduced over the valid transition steps exactly like `per_channel_recon_stats`. Data-only (no
    model): the fixed bar the model's dynamic-channel MAE must beat -- if the model's MAE is >= this,
    it has only learned persistence. Shapes: target, prev = (B, S, C, H, W); step_mask = (B, S).
    Returns a (C,) tensor."""
    B, S, C, H, W = target.shape
    m = step_mask.to(target).view(B, S, 1, 1, 1)
    ae = (target - prev).abs() * m
    denom = m.expand_as(ae).sum(dim=(0, 1, 3, 4)).clamp_min(1.0)
    return ae.sum(dim=(0, 1, 3, 4)) / denom


@torch.no_grad()
def per_channel_recon_stats(pred: torch.Tensor, target: torch.Tensor, step_mask: torch.Tensor):
    """Diagnostic (not a loss): per-channel MAE over the valid steps. pred is in [0,1] (post-sigmoid).
    Shapes as above. Returns a (C,) tensor of mean-absolute-error per channel."""
    B, S, C, H, W = pred.shape
    m = step_mask.to(pred).view(B, S, 1, 1, 1)
    ae = (pred - target).abs() * m
    denom = m.expand_as(ae).sum(dim=(0, 1, 3, 4)).clamp_min(1.0)
    return ae.sum(dim=(0, 1, 3, 4)) / denom
