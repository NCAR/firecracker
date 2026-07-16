# Plan: decode-supervised latent-rollout world model

Status: DRAFT (not yet implemented). Target: a working fire-front forecast for the Friday demo,
fast to train, plus a reward signal the fire-blocker agent can read off the decoded rollout.

## 0. Why

**CORRECTION (2026-07-15).** An earlier diagnostic *double-normalized* the already-baked `[0,1]` data
and wrongly concluded the encoder discards fire. Fed correctly, the autoencoder at `t+0` reconstructs
burning at ~43–89% recall (fire-size dependent), temperatures at corr ~0.5–0.8, static channels 0.9+,
and only **wind** is genuinely broken (corr ~0). The reward-head numbers were from the same buggy probe
and are moot (head is dropped anyway).

The real failure is the **dynamics rollout**: rolling the head forward, the predicted fire *decays*
(burning-cell recall 43%→13% over 5 steps) while the real fire *grows*. This is the cosine
self-prediction "easy-target" collapse — a near-static latent minimizes step-to-step cosine, so the
head never learns that fire spreads.

Implication for the fix: force the **dynamics** to propagate fire. Decoding every rolled step and
supervising against the real (growing) future frame does exactly that — a static/decaying rollout is
heavily penalized. Temperature renorm still helps (baked `air_temp` p99 = 0.029 — 99% of values under
0.03); **wind** is a separate, still-open AE-reconstruction gap to investigate.

## 1. The new regime (one loss, BPTT through the rollout)

Encode ONCE, roll the dynamics head forward, **decode every step**, and supervise each decoded step
against the real future frame. Single objective; gradients flow decoder → BPTT through dynamics →
the single `t_0` encoder call.

```
z_0   = encode(x_0)                       # ONE encoder call per sequence
z_k   = z_{k-1} + f(z_{k-1})              # additive residual dynamics (NO output L2-norm)
x̂_k   = decode(z_k)                       # decode every step k = 0..K
L     = Σ_k  mask_k · W ⊙ recon(x̂_k, x_k) # fire-weighted, masked per-step reconstruction
```

Dropped vs. the old scheme: SPR cosine loss, EMA target encoder, reward head, latent L2 norm,
separate random-frame anchor. Kept: decoder, dynamics head (residual), `latent_bn` (anti-collapse).

### Design decisions (rationale, from the design discussion)

| Decision | Why |
|---|---|
| Single reconstruction loss, BPTT | Directly optimizes the decoded multi-step forecast (the demo metric); no latent-cosine escape hatch that let the encoder discard fire. |
| Additive residual dynamics, **no L2 norm** | Collapse is prevented by reconstruction itself (constant latent → identical decodes → huge loss) plus `latent_bn` per-dim variance. Dropping the output norm gives a clean identity highway for BPTT and lets magnitude carry intensity. **Watch long-horizon drift** (unbounded latent) — mitigate by training K ≥ inference horizon. |
| Uniform start sampling + **pad-and-mask** | Encoder must encode arbitrary states at inference, but window-starts structurally exclude the last `span` frames (most-developed fire). Uniform starts + fixed-K rollout with the loss **masked** past each sequence's horizon covers the tail while keeping batch size and BN width fixed (mask the loss, never the BN forward). |
| **Fire-weighted per-channel loss** | A per-pixel-averaged loss is dominated by static structure; the sparse fire front + compressed temperatures wash out. This is the single most important fix — the architecture change alone will still produce blurry, fire-poor predictions without it. |
| Include the `t+0` term (`decode(z_0)` vs `x_0`) | Anchors the encoder's representation of the present; one extra cheap decode; stabilizes the rollout. |
| No reward head | The agent reads reward from the decoded forecast (count predicted newly-ignited cells) — more grounded than a scalar head that only learned a blurry proxy. |

### Cost per training step (vs. old)
Old: ~7 encodes + 1 decode + K transitions (the K EMA encodes dominate; encode is the ~2.5× expensive
half). New: **1 encode + (K+1) decodes + K transitions** — the expensive encoder runs once; the extra
work is cheap decodes. Pad-and-mask waste lands only on the cheap decode/dynamics steps. Net ~30% less
AE compute + no EMA encoder in memory.

## 2. File layout — new code isolated, old regime untouched

New files (no impact on the existing trainer):
- `src/train_world_model_rollout.py` — the new trainer + CLI. Standalone entrypoint; does not import
  from `train_world_model.py` except shared helpers already factored into `train_autoencoder.py`.
- `src/wm_losses.py` — `weighted_recon_loss(pred, target, *, channel_weights, mask, fire_channel,
  front_weight, ...)`; builds the per-channel + spatial-fire weighting and applies the step mask.
- `tools/eval_wm.py` — per-channel reconstruction breakdown (corr + MAE per channel, burning-cell
  recall) and reward-skill-vs-mean-baseline, generalized from the diagnostic scripts. Works on both
  regimes.
- `tests/test_world_model_rollout.py` — smoke, mask correctness, gradient-path, drift check.
- `sweep_wm_rollout.pbs` — launcher (UNTRACKED, per repo convention; not committed).

Touched core modules — **additive, default-off, backward compatible only** (see §3):
- `src/traj_loader.py` — opt-in partial-window enumeration returning a validity mask.
- `src/model_viewer.py` / `tools/rollout_viewer.py` — honor `normalize_output`/`residual` and
  `normalize_latent=False` when rebuilding (defaults reproduce old behavior).

Unchanged (reused via flags they already expose):
- `src/strided_autoencoder.py` — `normalize_latent=False, latent_bn=True` are existing kwargs.
- `src/dynamics.py` — `LatentTransition(residual=True, normalize_output=False)` are existing kwargs.
  RewardHead simply not constructed.

## 3. Core changes & backward-compatibility contract

Every touched core module must keep old checkpoints and the old trainer bit-identical:

1. **`traj_loader.py`** — add an opt-in enumeration mode (e.g. `partial_windows=False` on
   `TrajectoryDataset`, or a sibling `RolloutWindowLoader`) that also emits window starts in the tail
   and returns a per-sample `valid_len` / boolean `mask` alongside the `(B, K+1, C, H, W)` batch.
   Default OFF reproduces today's exact windows and today's return signature. New trainer opts in.
   Regression test: with the flag off, enumerated windows + batches are identical to current output.

2. **`dynamics.py`** — no code change; but the new trainer must record `residual` and
   `normalize_output` into `dynamics_config`. Rebuilders default these to `True` when the key is
   absent, so pre-existing checkpoints (which omit them) rebuild exactly as before.

3. **`strided_autoencoder.py`** — no change; `model_config` already records `normalize_latent` and
   `latent_bn`. New model sets `normalize_latent=False`.

4. **Viewer (`model_viewer.load_model` + `rollout_viewer.load_world_model`)** — read
   `normalize_output`/`residual` from `dynamics_config` (default `True`) and pass them to
   `LatentTransition`; `normalize_latent=False` already flows through. Additive; old models unaffected.
   (We already added `latent_bn` here.)

5. **Checkpoint format** — same dict keys as today plus `dynamics_config.{residual,normalize_output}`;
   `reward_state` simply absent. `--resume` (already ported) works unchanged.

## 4. Loss design (`wm_losses.py`)

Decoder stays sigmoid; all channels are min-max [0,1]. Per-step, per-channel loss with two weightings:

- **Per-channel weight vector `W_c`** — upweight the dynamic/fire channels (ignited, air + fuel
  temperatures, wind_x/y) and compensate the temperature channels whose min-max range is compressed
  into ~[0,0.1] (so their raw error is tiny). Start from a hand-set fire-emphasis vector; expose as a
  CLI/config knob. Optionally derive from inverse per-channel variance.
- **Spatial fire-front weight** — weight cells in/near the burning region higher (dilate the `ignited`
  mask) to beat the sparsity imbalance that makes "predict not-burning" a local minimum.
- **Channel loss** — MSE/Huber for the continuous channels; weighted BCE (`pos_weight`) or focal loss
  for the binary `ignited` channel. Recommended default: Huber for continuous + `pos_weight` BCE for
  ignited, summed with `W_c`.
- **Temperature rescaling** — either bake a per-channel gain into `W_c`, or (better, if time) renorm
  temps by a realistic dynamic max instead of the rare global extreme. No new dataset needed for the
  weight-only route.
- **Masking** — multiply each step's loss by `mask_k` (from §3.1); normalize by the number of valid
  (step, cell, channel) entries so the loss scale is horizon-independent.

## 5. Training recipe (fast, for Friday)

Sensible starting point (tune after the first short run):
- **`--rollout-steps K` = 8, `--step 1`, window = 9** — confirmed inference horizon is 8 steps, each
  step = 1 stored frame = 64 ticks, so 8 steps = 512 physics steps ahead. Train K = inference horizon,
  so the model is never rolled *beyond* what it was trained on → the unbounded-latent drift risk (no
  L2 norm) is bounded away. Optional 2→8 curriculum if BPTT is twitchy at K=8.
- `--epochs` 20–30 (old curves flattened by ~25; don't run 50).
- `--batch-size` 128, `--amp --weight-dtype float32`, grad clip 1.0 (keep — cheap insurance for the
  shared dynamics-weight gradient accumulation).
- `latent_bn=True`, `normalize_latent=False`, dynamics `residual=True, normalize_output=False`.
- Data pipeline: RAM loader / prewarm to avoid the I/O stalls that ballooned the 16 h run.
- 4-replicate (not 8/16) to save time; pick best by the **eval below**, not raw val loss.

## 6. Eval (`tools/eval_wm.py`) — so failure can't hide again

- **Per-channel reconstruction**: corr + MAE per channel, and burning-cell **recall/precision**, at
  `t+0` and at each rollout step k (does fire persist/spread across the roll?).
- **Reward skill**: newly-ignited count read from the decoded rollout vs. truth, reported as
  R²/skill against the **predict-the-mean baseline** (never raw SmoothL1).
- **Long-horizon drift**: latent norm and per-channel corr as k grows past training K.
- Wire into the metrics/W&B log per epoch (at least a few channels) so the collapse is visible live.

## 7. Testing (`tests/test_world_model_rollout.py`)

- Smoke: from-scratch tiny run produces metrics + checkpoint with the new `dynamics_config` keys.
- **Mask correctness**: tail-starting sequences contribute zero loss/grad past their horizon; BN sees
  full batch width every step.
- **Gradient path**: reconstruction grad reaches decoder, dynamics (all K steps), and the single
  encoder; no EMA/reward params exist.
- **Backward-compat**: `traj_loader` with the new flag OFF yields identical windows to current; an old
  checkpoint still loads in the viewer (dynamics defaults `normalize_output=True`).
- Drift guard: latent norm stays finite over a longer-than-K rollout.

## 8. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Blurry, fire-poor predictions (mean-hedging) — the failure repeats faster | The fire-weighted loss (§4) is mandatory, not optional; verify via the per-channel eval before trusting the run. |
| BPTT instability at larger K | Grad clip (kept), modest K, optional 2→K curriculum. Per-step supervision already shortens credit assignment. |
| Long-horizon latent drift (no L2 bound) | Train K ≥ inference horizon; `latent_bn`; add a light latent-magnitude penalty or soft output BN if drift appears. |
| BN in a recurrently-applied dynamics head mixes rollout-depth stats | Watch; consider a smaller/BN-free dynamics head if the rollout behaves oddly. |
| Padding waste from tail starts | Lands on cheap decode/dynamics only; or length-bucket batches for zero waste if BN gets fussy. |

## 9. Milestones (3-day)

- **Day 1**: `wm_losses.py` + `train_world_model_rollout.py` + `traj_loader` opt-in mask; smoke test
  green; kick off a short 20-epoch/4-replicate run.
- **Day 2**: `tools/eval_wm.py`; read the per-channel eval; iterate on `W_c`/front weight/K; retrain if
  needed; confirm the rollout viewer renders a spreading fire (viewer compat change).
- **Day 3**: fire-blocker agent reads reward from the decoded rollout; assemble the demo.

## 10. Decisions
- **RESOLVED — horizon = 8** (K=8, step=1, window=9; each step = 64 ticks → 512 steps ahead). Train
  K = inference horizon, so no extrapolation beyond training and the no-L2 drift risk is bounded.
- **RESOLVED — keep sigmoid + BCE decoder** for now (may switch continuous channels to linear+Huber
  later; the loss module is built so that swap is a config change, not a rewrite).
- **PENDING — temperature normalization.** Option A = loss-weight only (no data work; signal stays in
  ~[0,0.1]). Option B = renormalize temps by a realistic (percentile) max, on-the-fly, no re-bake —
  RECOMMENDED, optionally with a modest loss-weight on top. Decide before the first run; the trainer
  will read the per-channel normalization constants from config either way.
