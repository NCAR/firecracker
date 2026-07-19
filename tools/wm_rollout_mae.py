"""
wm_rollout_mae.py

Decoded 16-step open-loop rollout fidelity for a frozen-AE cosine-pred world model
(src/train_world_model.py; out_activation='none', so decode() emits values directly, NOT logits).

For N random trajectories (episodes) it anchors at the episode BEGINNING, picks a random start offset
s in [start_min, start_max], encodes the real frame at s ONCE, then rolls the dynamics head `steps`
times feeding its own output; every step it decodes the latent and scores the decoded frame against
the REAL frame at s+k. Reports per-channel MAE (fuel_temperature_* and ignited called out), averaged
over all steps and trajectories, plus a persistence baseline (freeze the start frame) and a per-step
drift table. MAE is in the model's normalized [0,1] space; temperature channels are also de-normalized
to physical units via channel_min/channel_max.

Usage:
  python tools/wm_rollout_mae.py --model models/wm_dynamics_deep11222_l256_14ch_k4_best_bf16.pt \
      --data /glade/campaign/cisl/aiml/mjessey/firecracker/data/fire --n-traj 1024 --steps 16
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from rollout import FireDataset
from traj_loader import TrajectoryDataset, TrajectoryBatchLoader
from dynamics import LatentTransition
from strided_autoencoder import ConvAutoencoder


def build_from_ckpt(ckpt, device):
    mc = ckpt["model_config"]
    ae = ConvAutoencoder(
        in_channels=mc["in_channels"], grid_size=mc["grid_size"], channels=tuple(mc["channels"]),
        latent_dim=mc["latent_dim"], normalize_latent=mc.get("normalize_latent", True),
        bottleneck_channels=mc.get("bottleneck_channels"), latent_bn=mc.get("latent_bn", False),
        blocks_per_stage=mc.get("blocks_per_stage"),
    ).to(device).eval()
    # bf16-on-disk weights -> float32 for accurate eval; non-float buffers kept as stored.
    ae.load_state_dict({k: (v.float() if torch.is_floating_point(v) else v)
                        for k, v in ckpt["model_state"].items()})
    dc = ckpt["dynamics_config"]
    dyn = LatentTransition(
        dc["latent_dim"], hidden_dim=dc["hidden_dim"], depth=dc["depth"],
        residual=dc.get("residual", True), normalize_output=dc.get("normalize_output", True),
        norm=dc.get("norm", "batchnorm"),
    ).to(device).eval()
    dyn.load_state_dict({k: v.float() for k, v in ckpt["dynamics_state"].items()})
    return ae, dyn, mc


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n-traj", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--start-min", type=int, default=1)
    ap.add_argument("--start-max", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--num-envs", type=int, default=None, help="episode env axis (default: meta)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    torch.set_num_threads(min(8, torch.get_num_threads()))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    ae, dyn, mc = build_from_ckpt(ckpt, device)
    names = list(mc["channel_names"])
    C = len(names)
    ft_idx = [i for i, n in enumerate(names) if n.startswith("fuel_temperature")]
    ign_idx = names.index("ignited")
    cmin = torch.tensor(mc["channel_min"], dtype=torch.float32)
    cmax = torch.tensor(mc["channel_max"], dtype=torch.float32)
    span = (cmax - cmin).clamp_min(1e-8)                       # physical range per channel

    ds = FireDataset(args.data)
    W = args.start_max + args.steps + 1                        # frames 0..(start_max+steps) needed
    traj = TrajectoryDataset(ds, num_envs=args.num_envs, window=W, step=1, partial=False)
    spe = traj.starts_per_ep
    n_ep = len(traj.episodes)
    if traj.F < W:
        raise SystemExit(f"episode length F={traj.F} < needed window {W}; reduce steps/start-max")
    n_use = min(args.n_traj, n_ep)
    ep_ids = np.random.default_rng(args.seed).permutation(n_ep)[:n_use]
    picks = [int(e) * spe for e in ep_ids]                     # k=0 -> anchored at episode start
    loader = TrajectoryBatchLoader(traj, picks, args.batch, shuffle=False, seed=args.seed,
                                   drop_last=False, num_workers=4,
                                   pin_memory=(device.type == "cuda"))
    srng = np.random.default_rng(args.seed + 1)

    S = args.steps
    err_sum = torch.zeros(S, C)          # summed over samples of per-sample mean-abs-err (per channel)
    persist_sum = torch.zeros(S, C)      # persistence baseline (freeze start frame)
    n_seen = 0
    print(f"model={args.model}\ndata={args.data}\nepisodes_available={n_ep} using={n_use} "
          f"steps={S} start=[{args.start_min},{args.start_max}] device={device}", flush=True)

    for x in loader:                                           # x: (N, W, C, H, W) on CPU
        N = x.shape[0]
        s = srng.integers(args.start_min, args.start_max + 1, size=N)
        ar = torch.arange(N)
        x0 = x[ar, torch.from_numpy(s)].to(device).float()     # start frame per sample
        z = ae.encode(x0)
        for k in range(1, S + 1):
            z = dyn(z)
            xhat = ae.decode(z)                                # (N,C,H,W) normalized values
            tgt = x[ar, torch.from_numpy(s + k)].to(device).float()
            err = (xhat - tgt).abs().mean(dim=(2, 3)).cpu()    # (N,C) per-sample per-channel MAE
            per = (x0 - tgt).abs().mean(dim=(2, 3)).cpu()      # persistence (freeze x0)
            err_sum[k - 1] += err.sum(0)
            persist_sum[k - 1] += per.sum(0)
        n_seen += N
        print(f"  processed {n_seen}/{n_use} trajectories", flush=True)

    mae = err_sum / n_seen                                     # (S,C) per-step per-channel MAE
    per_mae = persist_sum / n_seen

    def agg(m, idx):                                           # mean over steps and the given channels
        return m[:, idx].mean().item() if isinstance(idx, list) else m[:, idx].mean().item()

    ft_mae = agg(mae, ft_idx)
    ign_mae = agg(mae, ign_idx)
    ft_persist = agg(per_mae, ft_idx)
    ign_persist = agg(per_mae, ign_idx)
    # de-normalized (physical) fuel-temperature MAE: normalized err * (max-min), averaged over the 4 ch
    ft_phys = (mae[:, ft_idx] * span[ft_idx]).mean().item()

    print("\n================ RESULTS (mean over {} steps x {} trajectories) ================".format(S, n_seen))
    print(f"  fuel_temp  MAE (normalized [0,1]) = {ft_mae:.6f}   [persistence {ft_persist:.6f}]")
    print(f"  ignited    MAE (normalized [0,1]) = {ign_mae:.6f}   [persistence {ign_persist:.6f}]")
    print(f"  fuel_temp  MAE (physical units)   = {ft_phys:.4f}  (de-normalized via channel_min/max)")
    print(f"  -> model {'BEATS' if ft_mae < ft_persist else 'does NOT beat'} persistence on fuel_temp; "
          f"{'BEATS' if ign_mae < ign_persist else 'does NOT beat'} persistence on ignited")

    print("\n  per fuel_temperature_* channel (normalized MAE, mean over steps):")
    for i in ft_idx:
        print(f"    {names[i]:<28} {mae[:, i].mean().item():.6f}")

    print("\n  per-step drift (normalized MAE):")
    print(f"  {'step':>4} {'fuel_temp':>10} {'ignited':>10} {'persist_ft':>11} {'persist_ign':>12}")
    for k in range(S):
        print(f"  {k+1:>4} {mae[k, ft_idx].mean().item():>10.6f} {mae[k, ign_idx].item():>10.6f} "
              f"{per_mae[k, ft_idx].mean().item():>11.6f} {per_mae[k, ign_idx].item():>12.6f}")


if __name__ == "__main__":
    main()
