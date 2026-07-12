"""Evaluate a checkpoint over the ENTIRE dataset, reporting Huber (delta=1.0) and MSE.

Loads the model exactly as train_autoencoder saved it (arch + config from the checkpoint,
params cast to model_config['weight_dtype']), streams every sample of the FireDataset
sequentially in batches, and reports the sample-weighted mean of both losses.

Usage:
    python tools/eval_full.py --data <dir> --ckpt models/strided_huber_best.pt [--batch-size 64]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from strided_autoencoder import ConvAutoencoder
from rollout import FireDataset

ARCHITECTURES = {"pooled": ConvAutoencoder}
DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def resolve_arch(mc: dict, model_state: dict) -> str:
    """Resolve the architecture key, treating a legacy 'strided'-labelled avg-pool checkpoint
    (no per-stage 4x4 downsample convs) as 'pooled'. See model_viewer.resolve_arch."""
    arch = mc.get("arch", "pooled")
    if arch == "strided" and not any(".downsample." in k for k in model_state):
        return "pooled"
    return arch


def build_model(mc: dict, model_state: dict, device: torch.device, dtype: torch.dtype) -> nn.Module:
    m = ARCHITECTURES[resolve_arch(mc, model_state)](
        in_channels=mc["in_channels"], grid_size=mc["grid_size"],
        channels=tuple(mc["channels"]), latent_dim=mc["latent_dim"],
        # Absent in pre-normalization checkpoints; they were trained without it.
        normalize_latent=mc.get("normalize_latent", False),
        # Absent in pre-bottleneck checkpoints; None reproduces the un-bottlenecked model.
        bottleneck_channels=mc.get("bottleneck_channels"),
    )
    return m.to(device=device, dtype=dtype)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--huber-delta", type=float, default=1.0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    mc = ckpt["model_config"]
    # Rebuild in the dtype the weights are stored in (matches how a bf16-trained model runs);
    # half precision is unsupported on CPU, so up-cast to float32 there (lossless from bf16).
    want = DTYPES[mc.get("weight_dtype", "float32")]
    dtype = torch.float32 if (want is not torch.float32 and device.type == "cpu") else want
    model = build_model(mc, ckpt["model_state"], device, dtype)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    ds = FireDataset(args.data)
    if not ds.is_normalized:
        raise SystemExit(
            f"dataset at {args.data} is NOT baked-normalized; the model was trained on per-channel "
            f"min-max scaled inputs, so evaluating raw shards is invalid. Point --data at the "
            f"normalized (staged) copy.")
    n = len(ds)
    print(f"ckpt={args.ckpt}", flush=True)
    print(f"  weight_dtype(saved)={mc.get('weight_dtype')}  eval_dtype={str(dtype).removeprefix('torch.')}"
          f"  device={device}", flush=True)
    print(f"  dataset={args.data}  samples={n}  batch_size={args.batch_size}", flush=True)

    huber_fn = nn.HuberLoss(delta=args.huber_delta)
    mse_fn = nn.MSELoss()

    huber_sum = 0.0
    mse_sum = 0.0
    seen = 0
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, n, args.batch_size):
            idx = np.arange(start, min(start + args.batch_size, n))
            x = ds.get_batch(idx).to(device=device, dtype=dtype, non_blocking=True)
            x_hat = model(x)[0]
            b = x.shape[0]
            # Compute the reported metrics in float32 so the accumulation is faithful even when
            # the forward ran in bf16 (the reconstruction is still a bf16-precision estimate).
            xf, xhf = x.float(), x_hat.float()
            huber_sum += huber_fn(xhf, xf).item() * b
            mse_sum += mse_fn(xhf, xf).item() * b
            seen += b
            if (start // args.batch_size) % 100 == 0:
                print(f"  ..{seen}/{n}  ({time.time()-t0:.0f}s)", flush=True)

    print(f"RESULT ckpt={args.ckpt} dtype={str(dtype).removeprefix('torch.')} "
          f"n={seen} huber_delta={args.huber_delta} "
          f"huber={huber_sum/seen:.6f} mse={mse_sum/seen:.6f} "
          f"elapsed={time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
