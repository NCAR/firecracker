# Trained autoencoder checkpoints

Best replicate from each 16-replicate grouped-parallel sweep
(50 epochs, seeds 0–15). Each file is a `torch.save` dict with keys
`epoch`, `val_loss`, `model_state`, and `model_config`. The Adam
`optimizer_state` has been stripped (weights are bit-exact; only training
resumption is unavailable), which cuts each file from ~99 MiB to ~33 MiB in
float32; the `*_bf16.pt` files halve that again to ~21 MiB.

| File                             | Grouping / loss  | In ch | Dtype    | Source run                                         | Best epoch | Val loss |
|----------------------------------|------------------|------:|----------|----------------------------------------------------|-----------:|---------:|
| `strided_best.pt`                | strided / mse    |     8 | float32  | `sweep_strided_20260703_011310/run_02`             |         49 | 0.249810 |
| `strided_huber_best_bf16.pt`     | strided / huber  |     8 | bfloat16 | `sweep_strided_huber_20260705_*/`                  |         49 | 0.095380 |
| `strided_huber_16ch_best_bf16.pt`| strided / huber  |    16 | bfloat16 | `sweep_strided_huber_lr1e3_16ch_20260706_112844/run_09` | 48 | 0.091211 |
| `strided_huber_16ch_arch2_best_bf16.pt` | strided / huber (arch2) | 16 | bfloat16 | `sweep_strided_arch2_16ch_20260706_161842/run_03` | 50 | 0.084998 |
| `strided_huber_16ch_arch3_best_bf16.pt` | strided / huber (arch3) | 16 | bfloat16 | `lr1e3-arch3-16ch_20260707_163330/run_07` | 50 | 0.085238 |
| `residual_double_conv_16ch_best_bf16.pt` | strided / huber (residual double-conv) | 16 | bfloat16 | `residual-double-conv-16ch_20260707_232304/run_15` | 49 | 0.077465 |

The `arch2` model is the retuned strided stack — tapered widths
`[64, 64, 128, 128, 256, 256]` with a 512-d latent — vs. the earlier 16ch
model's `[16, 32, 64, 128, 256, 512]` / 256-d latent; both are recorded in each
checkpoint's `model_config`. It is the best (lowest-val-loss) replicate of the
`strided-16ch-arch2` (lr 1e-3, batch 64) sweep.

The `arch3` model is the reworked strided stack — widths `[32, 64, 128, 256,
256, 256]` / 512-d latent using **parameter-free avg-pool downsample /
nearest-neighbor upsample** resampling (vs. arch2's learned strided/transposed
convs). It is the best (lowest-val-loss) replicate of the `lr1e3-arch3-16ch`
(lr 1e-3, batch 64, huber) sweep. Despite a marginally higher best-val than
arch2 (0.0852 vs 0.0850), the arch3 sweep trained ~2.2× faster and its 16
replicates agreed ~10× more tightly (across-seed val CoV 0.40% vs 5.4%).

The `residual-double-conv` model keeps arch3's widths `[32, 64, 128, 256, 256,
256]` / 512-d latent and parameter-free avg-pool / nearest-neighbor resampling,
but replaces each single conv with a residual double-conv block (two 3×3 convs
plus a skip connection per stage). It is the best (lowest-val-loss) replicate of
the `residual-double-conv-16ch` (lr 1e-3, batch 64, huber) sweep, reaching
val-loss 0.0775 — the lowest of any 16ch model to date.

Selected as the lowest-val-loss replicate in each sweep. Per-channel input
normalization stats live with the dataset (`<data>/stats.json`) and the
architecture is recorded in each checkpoint's `model_config` (which for these
models also carries the `channel_mean`/`channel_std` inline). `*_bf16.pt` files
store `model_state` in bfloat16 (`model_config.weight_dtype = "bfloat16"`), ~half
the float32 size.

Load with:

```python
import torch
ckpt = torch.load("models/strided_best.pt", map_location="cpu")
model = build_model(ckpt["model_config"])   # see src/train_autoencoder.py
model.load_state_dict(ckpt["model_state"])
```
