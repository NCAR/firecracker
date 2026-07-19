# Trained autoencoder checkpoints

Best replicate from each grouped-parallel sweep (typically 16 replicates /
50 epochs, seeds 0–15; per-model counts noted below). Each file is a `torch.save` dict with keys
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
| `pooled_huber_deep11222_l256_14ch_best_bf16.pt` | pooled / huber (deep 11222, latent 256) | 14 | bfloat16 | `pooled-huber-deep11222-l256-14ch_20260717_161036/run_07` | 20 | 0.003027‡ |

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

The `pooled-deep11222` model keeps the conv-bottleneck's 5-stage `[32, 64, 128,
256, 512]` avg-pool stack and 1×1 channel bottleneck (512→256→512, 8×8 grid
preserved), but deepens the three high-width stages to **two residual blocks
each** (`blocks_per_stage = [1, 1, 2, 2, 2]`, decoder mirrored) and drops the
latent to **256-d** — reallocating params from the dense latent projection into
conv depth at a near-constant ~23.85M total. It is the best (lowest-val-loss)
replicate (`run_07`) of the `pooled-huber-deep11222-l256-14ch` (**8**-replicate,
**20**-epoch, lr default, huber δ=0.1) sweep. ‡ Its val-loss `0.003027` is **not
comparable** to the 16ch models above: this is the new **14-channel**, **min-max
[0, 1]** dataset (`model_config` carries `channel_min`/`channel_max`,
`signed_channels`, and the 14 `channel_names` inline, verified against the dataset
`stats.json`), whereas the 16ch models are z-scored (`channel_mean`/`channel_std`)
— different targets, different loss scale. This is the current AE baseline for the
separately-trained dynamics head.

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
latent, on top of a frozen or end-to-end autoencoder. Earlier ones sit on the `conv_bottleneck_256` AE
(frozen in the first, end-to-end in the next two, the second of which also carries a **reward head**);
the newest sits on the frozen `pooled-deep11222` 14ch AE.

| File                                            | Base AE                          | In ch | Dtype    | Source run                                          | Best epoch | Val pred (1−cos) |
|-------------------------------------------------|----------------------------------|------:|----------|-----------------------------------------------------|-----------:|-----------------:|
| `world_model_convbn256_16ch_best_bf16.pt`       | conv-bottleneck-256 (frozen)     |    16 | bfloat16 | `wm-frozen-convbn256-lr1e3_20260711_000856/run_14`  |         50 | 0.017485 |
| `world_model_rollout_reward_16ch_best_bf16.pt`  | conv-bottleneck-256 (end-to-end) |    16 | bfloat16 | `wm-bnbce-rollout-reward_20260713_182122/run_00`    |         40 | 0.028538 |
| `world_model_rollout_ln_16ch_best_bf16.pt`      | conv-bottleneck-256 (end-to-end) |    16 | bfloat16 | `wm-rollout-ln-b128_20260715_014354/run_03`         |         13 | 0.360788† |
| `wm_dynamics_deep11222_l256_14ch_k4_best_bf16.pt` | pooled-deep11222 (frozen)      |    14 | bfloat16 | `wm-frozen-deep11222-k4_20260718_171731/run_01`     |         19 | 0.000525‡ |
| `wm_dynamics_deep11222_l256_14ch_k8g05_best_bf16.pt` | pooled-deep11222 (frozen)    |    14 | bfloat16 | `wm-frozen-deep11222-k8-g05_20260718_231007/run_03` |         20 | 0.000524‡ |
| `wm_ft_deep11222_l256_14ch_k4_fire1_best_bf16.pt` | pooled-deep11222 (**end-to-end**) |  14 | bfloat16 | `wm-ft-huber-k4-fire1_20260719_013718/run_03`       |         16 | 0.002717§ |

‡ `wm_dynamics_deep11222`'s value is the **mean cosine-pred loss over a fixed 8-step rollout**, not a
single step, so it is not directly comparable to `world_model_convbn256`'s single-step `0.017485`
(different horizon *and* different AE/dataset). † `world_model_rollout_ln`'s value is **not** a cosine-pred loss but the decode-supervised
fire-weighted masked BCE over the full 9-step rollout (see its paragraph below); the two world-model
loss columns are not comparable to each other or to the autoencoders. § `wm_ft_deep11222`'s value is
the decode-supervised fire-weighted masked **Huber** over the K=4 rollout (a regression decoder,
`out_activation="none"` — **not** logits), so it is not comparable to any column above.

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

`wm_dynamics_deep11222_l256_14ch_k4_best_bf16.pt` is the best replicate (`run_01`) of the **K=4** arm
of a rollout-length A/B (`wm-frozen-deep11222-k{1,4,8}`, 8-replicate, 20-epoch, frozen
`pooled-deep11222` 14ch AE), where the dynamics head is unrolled K steps in training but **every** arm
is validated on a fixed **8-step** rollout so the `val_pred` is one comparable long-horizon metric.
K=4 won at `val_pred 0.000525` — lower at every rollout step than K=1 (`0.00068`); the flat-averaged
K=8 arm failed to optimize (plateaued ~0.0093 as the deep-BPTT far-step gradient swamped near-step
learning) and only trains once the far steps are down-weighted by a horizon discount (`--horizon-discount`;
γ=0.5 broke the plateau — see the separate `k8-g05` run). The head is a **LayerNorm** residual MLP
(`dynamics_config = {latent_dim 256, hidden_dim 1024, depth 3, norm "layernorm", residual, normalize_output}`)
predicting a unit-sphere next-latent; the frozen AE is the `pooled-deep11222` 14ch min-max model above
(so `model_state` is that AE unchanged, only `dynamics_state` was trained), with `channel_min`/
`channel_max`, `signed_channels`, and the 14 `channel_names` carried inline. Trained `float32` + AMP
(so the on-disk master weights were float32 before the bf16 down-cast); no reward head.

`wm_dynamics_deep11222_l256_14ch_k8g05_best_bf16.pt` is the best replicate (`run_03`, epoch 20) of the
**K=8 + horizon-discount γ=0.5** run (`wm-frozen-deep11222-k8-g05`, `--horizon-discount 0.5`, otherwise
identical to the K=4 arm). Background: the flat-averaged K=8 arm and a milder γ=0.8 both *plateaued* at
val_pred ≈ 0.0093–0.010 — the long-BPTT far-step gradient swamps near-step learning — whereas γ=0.5
(88% of the loss weight on steps 1–3) broke the plateau and converged all the way down to
`val_pred 0.000524`. That's **17.7× below** the flat/γ=0.8 plateau and a **statistical dead heat with
K=4** (0.000524 vs 0.000525; identical `val_pred_step_01..08` drift curves). So the discount *fully
rescues* long-horizon training, but K=8-discounted does **not** beat the simpler K=4 sweet spot — they
converge to the same solution. Config, geometry, and provenance are identical to the k4 checkpoint
(frozen AE, LayerNorm depth-3/1024 unit-sphere head), differing only in `horizon_discount=0.5` and
`train_rollout_steps=8`.

`wm_ft_deep11222_l256_14ch_k4_fire1_best_bf16.pt` is the first **stage-3** checkpoint: the AE and the
K=4 head are fine-tuned **end-to-end** (`freeze_ae=False`, `freeze_encoder=False`) in the
decode-supervised latent-rollout regime (`src/train_world_model_rollout.py`), warm-started from the
frozen-K=4 world model above. Unlike `world_model_rollout_ln`, the decoder is a **regression** head
(`out_activation="none"` — `decode()` emits values in ~[0,1] directly, **no sigmoid**) trained with a
fire-weighted masked **Huber** loss, matching the Huber-trained pooled AE. It is the best replicate
(`run_03`, epoch 16, val_loss 0.002717) of the **fire1** arm of a fire-weighting grid — moderate
weights `loss_config = {w_temp 4, w_wind 3, w_ignited 6, front_weight 4, front_dilate 2}`; a 4×-heavier
**fire3** arm scored slightly *worse*. **Caveat — this model does not beat persistence.** Every
replicate of both arms stayed at **IDENTITY** for all 20 epochs: best dynamic-channel MAE ≈ **0.0115**
vs the copy-previous-frame baseline **0.0038** (~3× worse). It reconstructs and its static channels are
sharp, but fire-weighting alone did not dislodge the identity collapse on these low-change trajectories.
It is saved for **rollout-viewer inspection** (`tools/rollout_viewer.py --model
models/wm_ft_deep11222_l256_14ch_k4_fire1_best_bf16.pt`) to see what the fine-tune actually learned, not
as a capability win. Head geometry and inline normalization (`channel_min`/`channel_max`,
`signed_channels`, 14 `channel_names`) match the k4 checkpoint; `model_config` also records `latent_bn`
and `warm_started_from`.

Beyond the autoencoder keys, all six files carry `dynamics_state`, `dynamics_config`
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
