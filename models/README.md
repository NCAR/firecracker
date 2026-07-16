# Trained autoencoder checkpoints

Best replicate from each 16-replicate grouped-parallel sweep
(50 epochs, seeds 0–15). Each file is a `torch.save` dict with keys
`epoch`, `val_loss`, `model_state`, and `model_config`. The Adam
`optimizer_state` has been stripped (weights are bit-exact; only training
resumption is unavailable), which cuts the float32 size roughly to model weights
alone; the `*_bf16.pt` files store those at ~2 bytes/param (so file size scales
with the model — e.g. ~21 MiB for the ~10.8M-param residual-double-conv, ~49 MiB
for the 24.5M-param conv-bottleneck).

| File                             | Grouping / loss  | In ch | Dtype    | Source run                                         | Best epoch | Val loss |
|----------------------------------|------------------|------:|----------|----------------------------------------------------|-----------:|---------:|
| `strided_best.pt`                | strided / mse    |     8 | float32  | `sweep_strided_20260703_011310/run_02`             |         49 | 0.249810 |
| `strided_huber_best_bf16.pt`     | strided / huber  |     8 | bfloat16 | `sweep_strided_huber_20260705_*/`                  |         49 | 0.095380 |
| `strided_huber_16ch_best_bf16.pt`| strided / huber  |    16 | bfloat16 | `sweep_strided_huber_lr1e3_16ch_20260706_112844/run_09` | 48 | 0.091211 |
| `strided_huber_16ch_arch2_best_bf16.pt` | strided / huber (arch2) | 16 | bfloat16 | `sweep_strided_arch2_16ch_20260706_161842/run_03` | 50 | 0.084998 |
| `strided_huber_16ch_arch3_best_bf16.pt` | strided / huber (arch3) | 16 | bfloat16 | `lr1e3-arch3-16ch_20260707_163330/run_07` | 50 | 0.085238 |
| `residual_double_conv_16ch_best_bf16.pt` | strided / huber (residual double-conv) | 16 | bfloat16 | `residual-double-conv-16ch_20260707_232304/run_15` | 49 | 0.077465 |
| `conv_bottleneck_256_16ch_best_bf16.pt` | pooled / huber (1×1 channel bottleneck) | 16 | bfloat16 | `exp-conv-bottleneck_20260709_204409/run_07` | 49 | 0.073867 |

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

The `conv-bottleneck` model uses the 5-stage `[32, 64, 128, 256, 512]` avg-pool
stack (an 8×8×512 encoder map, not downsampled to 4×4) with a 512-d latent, and
inserts a 1×1-conv channel bottleneck around the latent: the encoder map is
compressed 512→256 channels (1×1 conv + BatchNorm) before flattening and expanded
256→512 after, preserving the 8×8 spatial grid while shrinking the flattened width
32768→16384 (and the two dense latent projections with it) — 24.5M params vs the
41M an un-bottlenecked 8×8×512 stack would need. It is the best (lowest-val-loss)
replicate of the `exp-conv-bottleneck` (lr 1e-3, batch 64, huber) sweep, reaching
val-loss 0.0739 — the lowest of any 16ch model to date.

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

## World model checkpoints

World models (`src/train_world_model.py`), not autoencoders: a dynamics head that predicts the next
latent, on top of a `conv_bottleneck_256` autoencoder (frozen in the first, trained end-to-end in the
second). The second also carries a **reward head**.

| File                                            | Base AE                          | In ch | Dtype    | Source run                                          | Best epoch | Val pred (1−cos) |
|-------------------------------------------------|----------------------------------|------:|----------|-----------------------------------------------------|-----------:|-----------------:|
| `world_model_convbn256_16ch_best_bf16.pt`       | conv-bottleneck-256 (frozen)     |    16 | bfloat16 | `wm-frozen-convbn256-lr1e3_20260711_000856/run_14`  |         50 | 0.017485 |
| `world_model_rollout_reward_16ch_best_bf16.pt`  | conv-bottleneck-256 (end-to-end) |    16 | bfloat16 | `wm-bnbce-rollout-reward_20260713_182122/run_00`    |         40 | 0.028538 |
| `world_model_rollout_ln_16ch_best_bf16.pt`      | conv-bottleneck-256 (end-to-end) |    16 | bfloat16 | `wm-rollout-ln-b128_20260715_014354/run_03`         |         13 | 0.360788† |

† `world_model_rollout_ln`'s value is **not** a cosine-pred loss but the decode-supervised
fire-weighted masked BCE over the full 9-step rollout (see its paragraph below); the two world-model
loss columns are not comparable to each other or to the autoencoders.

The val "pred" is the cosine next-latent prediction loss `1 − cos(ẑ_{t+1}, sg(z_{t+1}))`, so it is
**not** comparable to the autoencoders' reconstruction (huber/mse) losses above. The two world models'
pred losses are also not directly comparable to each other: the frozen model's `0.017485` is a
**single-step** prediction, while the end-to-end model's `0.028538` is the **mean over a 5-step latent
rollout** (`--rollout-steps 5`, the head fed its own output), a strictly harder target.

`world_model_convbn256_16ch_best_bf16.pt` is the lowest-val-loss replicate (`run_14`) of the
`wm-frozen-convbn256-lr1e3` 16-replicate sweep (50 epochs, **frozen** encoder/decoder). Because the AE
was frozen, its `model_state` is the `conv_bottleneck_256` autoencoder unchanged; only `dynamics_state`
was trained.

`world_model_rollout_reward_16ch_best_bf16.pt` is the best replicate (`run_00`) of the
`wm-bnbce-rollout-reward` 8-replicate sweep, ranked by a **rebalanced** val_loss
(`2·pred + 0.2·rec + 1·reward`, min 0.17734 at epoch 41). Here the whole network is trained
**end-to-end** — encoder, decoder, dynamics, and a reward head — with a BYOL/SPR EMA target, a
5-step latent rollout, a random-frame reconstruction anchor (`sigmoid`/BCE decoder, so
`model_config.out_activation = "sigmoid"` and `decode()` emits **logits** — apply a sigmoid for
reconstructions), and a **reward head** that predicts the per-step newly-ignited-cell count as
`log1p(count)` (apply `expm1` to recover the count; `reward_config.ignited_channel = 15`). The saved
checkpoint is epoch **40**, the lowest-`val_pred` (`best_pred = 0.028538`) epoch, not epoch 41: the run
was walltime-killed at epoch 41 before that epoch's weights were checkpointed, and epoch 40 is both the
prediction-optimal epoch and a statistical tie on the rebalanced loss. Unlike the older models this one
was trained on a **min-max [0,1]** dataset, so `model_config` carries `channel_min`/`channel_max`
(verified against the dataset `stats.json`) inline plus the 16 `channel_names`.

`world_model_rollout_ln_16ch_best_bf16.pt` is the best replicate (`run_03`, lowest val_loss and
lowest val_fire_mae) of the `wm-rollout-ln-b128` 4-replicate sweep, a different **decode-supervised
latent-rollout** regime (`src/train_world_model_rollout.py`): encode t0 **once**, roll the dynamics
head forward K=8 additive-residual steps, **decode every step**, and supervise each decoded step
against the real future frame with a single fire-weighted masked BCE (BPTT through the rollout). There
is **no cosine-pred loss and no reward head** here. The saved checkpoint is epoch **13** (val_loss
0.360788, val_fire_mae 0.0258). Two things distinguish it from the reward model: the dynamics head uses
**LayerNorm** (`dynamics_config.norm = "layernorm"`) rather than BatchNorm — per-sample, depth-invariant
normalization that avoids BN's running-stat mixing across rollout depths (measured latent drift
z_K/z_0 ≈ 1.04) — and it is **not** `normalize_output` (`False`), so the rolled latents are kept near
the encoder's `latent_bn`-centered z0 distribution only by the decode supervision, valid within the K=8
horizon. Like the reward model it is min-max [0,1] data (`model_config` carries `channel_min`/
`channel_max` + 16 `channel_names` inline, verified against the dataset `stats.json`), a `sigmoid`/BCE
decoder that emits **logits** (`out_activation = "sigmoid"`; apply a sigmoid for reconstructions), and
the fire-weighting recipe in `loss_config` (`{w_temp 4, w_wind 3, w_ignited 6, front_weight 4,
front_dilate 2, temp_pct 99}`). Its dynamics head is wider/deeper than the reward model's
(`hidden_dim 1024, depth 4`).

Beyond the autoencoder keys, all three files carry `dynamics_state`, `dynamics_config`
(`{latent_dim, hidden_dim, depth, ...}`, with `hidden_dim` resolved to its actual width), and
`freeze_ae`. `world_model_rollout_reward` additionally carries `reward_state` + `reward_config`
(`{latent_dim, hidden_dim, depth, ignited_channel}`); `world_model_rollout_ln` has **no** reward head,
and its `dynamics_config` also records `residual`, `normalize_output`, and `norm`. Rebuild with:

```python
import torch
from strided_autoencoder import ConvAutoencoder
from dynamics import LatentTransition, RewardHead

ckpt = torch.load("models/world_model_rollout_reward_16ch_best_bf16.pt", map_location="cpu")
mc, dc = ckpt["model_config"], ckpt["dynamics_config"]
ae = ConvAutoencoder(
    in_channels=mc["in_channels"], grid_size=mc["grid_size"], channels=tuple(mc["channels"]),
    latent_dim=mc["latent_dim"], normalize_latent=mc["normalize_latent"],
    bottleneck_channels=mc.get("bottleneck_channels"), latent_bn=bool(mc.get("latent_bn")),
)
ae.load_state_dict(ckpt["model_state"]); ae.eval()
# Pass the optional dynamics_config fields with backward-compatible defaults so both the
# BatchNorm reward model and the LayerNorm decode-rollout model rebuild + strict-load correctly.
dyn = LatentTransition(
    dc["latent_dim"], hidden_dim=dc["hidden_dim"], depth=dc["depth"],
    residual=dc.get("residual", True), normalize_output=dc.get("normalize_output", True),
    norm=dc.get("norm", "batchnorm"),
)
dyn.load_state_dict(ckpt["dynamics_state"]); dyn.eval()

# Reward head: world_model_rollout_reward only (world_model_rollout_ln has none — guard on the key).
if "reward_config" in ckpt:
    rc = ckpt["reward_config"]
    rew = RewardHead(rc["latent_dim"], hidden_dim=rc["hidden_dim"], depth=rc["depth"])
    rew.load_state_dict(ckpt["reward_state"]); rew.eval()

# z_t = ae.encode(x);  z_hat = dyn(z_t);  logits = ae.decode(z_hat)
# x_hat = logits.sigmoid() if mc.get("out_activation") == "sigmoid" else logits
# reward_count = rew(z_hat).expm1()
```

Or view it rolling forward against the live physics:
`python tools/rollout_viewer.py --model models/world_model_convbn256_16ch_best_bf16.pt`.
