# Plan: two-term (anchor + change) rollout loss & static/dynamic logging

## Motivation

The decode-supervised rollout world model collapses to **identity** (`z_{t+1} ≈ z_t`,
observed latent drift `zK/z0 ≈ 1.04`, `val_fire_mae` plateaus early). Cause: the current
single-term loss (`wm_losses.masked_weighted_bce`) averages per-pixel BCE over the whole
frame, which is dominated by static/near-static cells. Predicting persistence (copy the
current frame) is therefore a strong local minimum — it's *correct* on the static majority
and only wrong on the thin, sparse fire front, whose gradient is diluted ~100×.

Fix (already agreed): decompose the objective into two terms with **separate denominators**
so persistence can't win, *without* losing static fidelity:

```
L = w_anchor · L_anchor  +  w_change · L_change      (w_change ≫ w_anchor)
```

- **L_anchor** — full-frame, channel-weighted recon, normalized over *all* cells. Keeps
  static channels faithful and keeps the dynamics head from corrupting the terrain/fuel
  subspace of the latent as it rolls. Low weight, spatially large → cheap to satisfy, can't
  re-create the identity minimum.
- **L_change** — BCE restricted to cells that actually change between consecutive *real*
  frames, normalized over the **changed-cell mass only**. Persistence scores the *worst*
  possible on this term (it predicts zero change where change is maximal), so identity is no
  longer a minimum. Constant magnitude per step regardless of how little moved.

Plus diagnostics so we can *prove* collapse instead of inferring it:
- **persistence baseline** (MAE of "copy previous real frame") logged next to the model's MAE;
- **per-channel MAE split static vs dynamic**, so we can watch static fidelity as `w_change` rises.

## Non-goals (this change)

- No architecture change (no dynamics-head re-init, no non-residual head, no frame stride).
  Those are separate levers held in reserve if this doesn't move drift.
- Keep default behavior backward compatible: `w_change = 0` reproduces today's loss exactly.

---

## Design

### 1. `src/wm_losses.py` — new change term + persistence/split helpers

**a. `masked_change_bce(logits, target, prev, step_mask, channel_w, *, eps, ignited_idx=None,
front_weight=0.0, front_dilate=1)`**

- Shapes: `logits, target, prev = (B, S', C, H, W)` where `S' = S-1` (steps `1..K`; step 0 has
  no previous real frame). `prev[:, s] = x_{t+s}` is the real frame one step earlier.
- `change = (target - prev).abs()`  →  `change_mask = (change > eps)` per (cell, channel).
- `bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")` (BCE on soft
  targets is what the existing loss already does for continuous channels — keep it uniform).
- `weight = channel_w · step_mask · change_mask` (optionally still `· (1 + front_weight·…)`,
  but default front off here — the change mask *is* the sharp front).
- `den = weight.sum().clamp_min(1.0)` → normalized over changed cells only.
- Return `(bce · weight).sum() / den`.
- Guard: if `den` is (near) 1.0 because a batch had ~no change, the clamp keeps it finite;
  the term just contributes ~0 gradient that batch. Fine.

**b. `@torch.no_grad() persistence_mae(target, prev, step_mask)`** — per-channel MAE of the
copy-previous-frame baseline: `|target - prev|` reduced exactly like `per_channel_recon_stats`
(over B, S', H, W). Returns `(C,)`. Data-only (no model), so it's the fixed bar the model must beat.

**c. Small role-split helper** (or do it in the trainer): given the `(C,)` MAE vector and
`channel_roles(names)`, return `(static_mae, dynamic_mae)` where dynamic = `temp+wind+ignited`,
static = `roles["static"]`. Keep it in the trainer next to `fire_idx` since roles are already
computed there (line 380–381).

`masked_weighted_bce` stays as-is and becomes **L_anchor** (called with `front_weight=0` so the
front emphasis lives only in the change term; see arg notes).

### 2. `src/train_world_model_rollout.py` — wire the two terms + metrics

**Loss (run_epoch, ~line 203–207):**
```python
logits, zs = rollout_logits(...)
anchor = wml.masked_weighted_bce(logits, x, mask, channel_w,
                                 ignited_idx=ignited_idx, front_weight=anchor_front, front_dilate=front_dilate)
if w_change > 0:
    change = wml.masked_change_bce(logits[:, 1:], x[:, 1:], x[:, :-1], mask[:, 1:], channel_w,
                                   eps=change_eps, ignited_idx=ignited_idx)
    loss = w_anchor * anchor + w_change * change
else:
    loss = anchor            # backward-compatible path
```
Track `anchor` and `change` scalars separately for logging (accumulate like `tot`).

**Return shape (run_epoch):** the 7-tuple is already unwieldy; switch to a small dataclass
`EpochStats` (loss, anchor, change, mae_vec `(C,)`, fire_mae, persist_vec `(C,)`, gmean, gmax,
lnorm, skipped). Update both call sites (line 455–458) and everything downstream. This is
cleaner than a 12-tuple and localizes future additions.

**Per-channel stats (line 232–233):** also compute `persist = wml.persistence_mae(x[:,1:], x[:,:-1], mask[:,1:])`
under `no_grad`, accumulate like `mae_sum`. In the caller, split both `mae_vec` and `persist_vec`
into static/dynamic via roles.

**Args (argparse, near line 284–288):**
- `--w-anchor` (float, default `0.2`) — weight on L_anchor.
- `--w-change` (float, default `0.0`) — weight on L_change. **Default 0 = today's behavior.**
  Turn it on (e.g. `1.0`) to enable the decomposition.
- `--change-eps` (float, default `1e-3`) — min per-cell delta (post-gain, in [0,1]) to count as changed.
- `--anchor-front` (float, default `0.0`) — front up-weight *inside* the anchor term. Default 0
  so, once `w_change>0`, front emphasis isn't double-counted; set >0 to keep some front weight
  in the anchor if wanted. (Existing `--front-weight`/`--front-dilate` remain but, when
  `w_change>0`, we route front emphasis through the change mask, not the anchor.)

**loss_config (line 389–391):** add `w_anchor`, `w_change`, `change_eps`, `anchor_front` so the
recipe is recorded in the checkpoint and W&B config. **Bump/record** so old checkpoints (no
keys) still load with `.get(..., default)`.

**best.pt selection:** unchanged for now (`va_loss = w_anchor·anchor + w_change·change`). Note in
code that with `w_change>0` the objective changes, so `val_loss` isn't comparable across the
w_change=0 → w_change>0 boundary. (We rank runs by val_loss only *within* a fixed recipe — see
`compare-runs-by-val-loss` memory.)

### 3. Logging (CSV + print + W&B)

**metrics.csv header (line 446–447):** add columns
`train_anchor, train_change, val_anchor, val_change, val_static_mae, val_dynamic_mae,
persist_static_mae, persist_dynamic_mae` (before `skipped_steps, seconds`). Append matching values
in the row write (line 467–468). Note: appending columns breaks header alignment for a *resumed*
old CSV — guard by only writing the new header when not resuming (already the case, line 445), and
document that resuming a pre-change run's CSV isn't supported (new run dir instead).

**print line (line 463–466):** add a compact `chg {val_change:.4f}` and
`mae stat {val_static_mae:.4f}/dyn {val_dynamic_mae:.4f} (persist {persist_dynamic_mae:.4f})`
so collapse is visible in the console: **model dyn MAE ≥ persist dyn MAE ⇒ collapsed.**

**W&B log (line 477–485):** add
`train/anchor, train/change, val/anchor, val/change, val/mae_static, val/mae_dynamic,
baseline/persist_static, baseline/persist_dynamic`, and a derived
`val/beats_persistence = float(val_dynamic_mae < persist_dynamic_mae)` for an at-a-glance panel.

### 4. Tests (`tests/test_world_model_rollout.py`)

- `masked_change_bce`: (i) zero when `logits` decode exactly to `target`; (ii) a **persistence
  prediction** (`logits = logit(prev)`, via a clamped inverse-sigmoid) yields a **large** loss —
  the anti-identity property, the key test; (iii) denominator normalizes — scaling the number of
  changed cells doesn't scale the mean; (iv) shape/masking: masked steps contribute nothing.
- `persistence_mae`: matches a hand-computed `|target-prev|` mean on a tiny tensor; respects mask.
- `run_epoch` smoke test with `--w-change>0`: returns finite loss, `EpochStats` fields populated,
  static/dynamic split lengths correct.
- Backward-compat: `w_change=0` ⇒ `loss == masked_weighted_bce(...)` bit-for-bit (guards the default path).

### 5. Checkpoint / deployment compatibility

- `dynamics_config` / `model_config` unchanged → existing exported `.pt` and the rollout viewer
  keep loading. Only `loss_config` gains keys (read with `.get`), so **no viewer change needed**.
- `models/README.md`: no change required (loss recipe isn't part of the load path). Optionally note
  the new recipe when we export a non-collapsed checkpoint.

---

## Validation (before trusting a sweep)

1. **Unit tests** green (esp. the persistence anti-identity test).
2. **Single-GPU short run** (2–3 epochs, `--w-change 1.0`): confirm
   - `val/change` decreases,
   - `val/mae_dynamic` drops **below** `baseline/persist_dynamic` within a few epochs
     (`beats_persistence → 1`) — the actual success signal,
   - `val/mae_static` does **not** blow up as `w_change` bites (guardrail on `w_anchor`),
   - latent `drift` moves off ~1.04.
3. **Rollout viewer** spot-check on the short-run checkpoint: fire front should *advance*
   step-to-step, not freeze.
4. Only then launch the replicate sweep (reuse existing `sweep_wm.pbs`, add `--w-change`,
   `--w-anchor` to the arg set; keep 4 replicates / 1-per-GPU as before).

## Rollout / sequencing

1. `wm_losses.py`: `masked_change_bce` + `persistence_mae` + tests → **land + test first** (pure,
   no trainer coupling).
2. `train_world_model_rollout.py`: `EpochStats`, wire two terms, args, splits, logging.
3. CSV/print/W&B columns.
4. Short single-GPU validation run → viewer check.
5. Sweep.

## Risks & mitigations

- **Static fidelity regression** (the terrain-forgetting concern): mitigated by the anchor term +
  `val/mae_static` guardrail. If static MAE climbs, raise `w_anchor` or lower `w_change`.
- **`w_change` too high → training instability / fp16 overflow**: `skipped_steps` already logged;
  watch it. Change term is normalized (constant scale), so scaler behavior should be stable.
- **eps mis-set**: too large → misses real change (under-trains front); too small → static
  numerical noise counts as change (re-dilutes). `1e-3` on gain-normalized [0,1] channels is a
  starting point; tune from the fraction-of-cells-changed (could log it).
- **Objective discontinuity at `w_change` 0→>0**: `val_loss` not comparable across that boundary;
  documented, and `beats_persistence`/`mae_dynamic` are the recipe-invariant signals to compare on.
