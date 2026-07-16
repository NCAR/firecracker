"""
rollout_check.py

Headless open-loop rollout fidelity check for a decode-rollout world-model checkpoint
(src/train_world_model_rollout.py). No display, no pygame -- prints a per-step table so we can
tell whether the learned dynamics actually ADVANCE the fire front or just freeze (identity) /
explode.

For a batch of held-out windows it encodes frame 0 once, rolls the dynamics head K steps, decodes
every step (exactly rollout_logits), then on the `ignited` channel reports per step:
  * real_frac  -- fraction of cells burning in the REAL frame at that step
  * pred_frac  -- fraction the model predicts (should track real_frac, not stay at step-0)
  * persist    -- the copy-frame-0 baseline (constant real_frac[0]); if pred hugs this -> frozen
  * IoU        -- spatial overlap of predicted vs real burning cells
  * dyn_MAE    -- mean abs error over the dynamic (temp/wind/ignited) channels

Usage:
  python tools/rollout_check.py --ckpt runs/.../best.pt --data $SCRATCH/firecracker/stage/fire/w0 \
      --num-envs 8 --window 9 --n-windows 64
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from rollout import FireDataset
from traj_loader import TrajectoryDataset, TrajectoryBatchLoader, split_episodes
from dynamics import LatentTransition
from strided_autoencoder import ConvAutoencoder
import wm_losses as wml


def build_from_ckpt(ckpt, device):
    mc = ckpt["model_config"]
    ae = ConvAutoencoder(
        in_channels=mc["in_channels"], grid_size=mc["grid_size"], channels=tuple(mc["channels"]),
        latent_dim=mc["latent_dim"], normalize_latent=mc.get("normalize_latent", False),
        bottleneck_channels=mc.get("bottleneck_channels"), latent_bn=mc.get("latent_bn", False),
    ).to(device).eval()
    ae.load_state_dict(ckpt["model_state"])
    dc = ckpt["dynamics_config"]
    dyn = LatentTransition(
        dc["latent_dim"], hidden_dim=dc["hidden_dim"], depth=dc["depth"], residual=dc["residual"],
        normalize_output=dc["normalize_output"], norm=dc["norm"],
    ).to(device).eval()
    dyn.load_state_dict(ckpt["dynamics_state"])
    return ae, dyn, mc


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--num-envs", type=int, default=8)
    ap.add_argument("--window", type=int, default=9)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-windows", type=int, default=64)
    ap.add_argument("--thr", type=float, default=0.5, help="ignited burning threshold")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    torch.set_num_threads(min(8, torch.get_num_threads()))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    ae, dyn, mc = build_from_ckpt(ckpt, device)

    names = list(FireDataset(args.data).meta.get("channels"))
    roles = wml.channel_roles(names)
    ign = wml.ignited_index(names)
    dyn_idx = sorted(roles["temp"] + roles["wind"] + roles["ignited"])
    gains = torch.tensor(mc["channel_gains"], dtype=torch.float32, device=device) \
        if mc.get("channel_gains") is not None else None

    ds = FireDataset(args.data)
    traj = TrajectoryDataset(ds, num_envs=args.num_envs, window=args.window, step=args.step, partial=True)
    _, val_w = split_episodes(traj, val_frac=args.val_frac, seed=args.seed)
    picks = val_w[: args.n_windows]
    loader = TrajectoryBatchLoader(traj, picks, len(picks), shuffle=False, seed=args.seed,
                                   drop_last=False, num_workers=2)
    x, mask = next(iter(loader))
    x = x.to(device).float()
    if gains is not None:
        x = wml.apply_gains(x, gains)

    from train_world_model_rollout import rollout_logits
    logits = rollout_logits(ae, dyn, x[:, 0], x.shape[1], freeze_encoder=True)
    pred = torch.sigmoid(logits)                                   # (N, S, C, H, W) in [0,1]

    S = x.shape[1]
    real_b = (x[:, :, ign] > args.thr)                            # (N, S, H, W)
    pred_b = (pred[:, :, ign] > args.thr)
    base_b = real_b[:, :1].expand(-1, S, -1, -1)                  # persistence: hold frame 0
    dyn_mae = (pred[:, :, dyn_idx] - x[:, :, dyn_idx]).abs().mean(dim=(0, 2, 3, 4))

    def frac(b): return b.float().mean(dim=(0, 2, 3))             # (S,)
    def iou(a, c):
        inter = (a & c).float().sum(dim=(0, 2, 3))
        union = (a | c).float().sum(dim=(0, 2, 3)).clamp_min(1.0)
        return inter / union

    rf, pf, bf = frac(real_b), frac(pred_b), frac(base_b)
    iou_pred, iou_persist = iou(pred_b, real_b), iou(base_b, real_b)

    print(f"\nckpt: {args.ckpt}")
    print(f"windows={len(picks)}  steps={S}  ignited_ch={ign}  thr={args.thr}  device={device}")
    print(f"{'step':>4} {'real_frac':>10} {'pred_frac':>10} {'persist':>9} "
          f"{'IoU_pred':>9} {'IoU_persist':>11} {'dyn_MAE':>8}")
    for k in range(S):
        print(f"{k:>4} {rf[k]:>10.5f} {pf[k]:>10.5f} {bf[k]:>9.5f} "
              f"{iou_pred[k]:>9.3f} {iou_persist[k]:>11.3f} {dyn_mae[k]:>8.5f}")

    # Verdict on the LAST step: does the model track the real front better than freezing frame 0?
    print(f"\nreal fire growth over rollout: {rf[0]:.5f} -> {rf[-1]:.5f} "
          f"({(rf[-1]-rf[0])/max(rf[0],1e-6)*100:+.1f}%)")
    print(f"pred fire growth over rollout: {pf[0]:.5f} -> {pf[-1]:.5f} "
          f"({(pf[-1]-pf[0])/max(pf[0],1e-6)*100:+.1f}%)")
    print(f"final-step IoU  pred vs real = {iou_pred[-1]:.3f}   persist vs real = {iou_persist[-1]:.3f}   "
          f"-> model {'BEATS' if iou_pred[-1] > iou_persist[-1] else 'DOES NOT BEAT'} persistence spatially")


if __name__ == "__main__":
    main()
