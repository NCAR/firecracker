"""
rollout_viewer.py

Rollout visualiser for a latent world model (src/train_world_model.py): it rolls the dynamics head
forward in latent space and decodes each step, so you watch the neural world evolve and see where it
diverges from the physics. Two modes:

  ONLINE (default) -- roll against the LIVE physics simulation.
      python tools/rollout_viewer.py --model models/world_model_convbn256_16ch_best_bf16.pt
    The model was trained on jumps of `--ticks-per-step` sim-ticks; that stride is read from the
    checkpoint by default (dyn_config["ticks_per_step"] = dataset stride x trajectory --step), so one
    dynamics step means that many physics ticks. Rather than burst-simulate them (which would freeze
    the UI), physics runs at `--ticks-per-second` (default 10) and the dynamics head advances once
    every `--ticks-per-step` ticks -- the model updates every ticks_per_step/ticks_per_second seconds
    while the physics panel animates smoothly. A re-anchors the latent to the current frame.

  OFFLINE (--data DIR) -- roll against RECORDED dataset frames, no simulation at all.
      python tools/rollout_viewer.py --model <ckpt> --data data/fire --num-envs 32
    A dataset episode already holds the ground-truth frames at exactly the model's stride, so this
    encodes frame 0, rolls the head over the episode, and shows decoded-rollout vs. recorded-physics
    with instant frame scrubbing. Far faster than online (one encode/decode per frame, no physics),
    and the fairer comparison. LEFT/RIGHT scrub frames, N loads a new episode, SPACE toggles autoplay.

Both reuse model_viewer.ComparisonViewer for the panels (de-standardisation, channel layout, surface
builders, TAB error map). Shared controls: TAB reconstruction<->error, number/s/b switch view, ESC
quit. Trained open-loop error compounds (single-step training, ~8-frame horizon), so expect drift.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pygame
import torch

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import load_config
from firecracker_env import FirecrackerEnv, ViewMode, TARGET_FPS
from dynamics import LatentTransition
from rollout import FireDataset
from rendering import build_color_surface, build_legend_panel, build_biome_legend_panel, heat_colormap
from model_viewer import (
    ComparisonViewer, RECONSTRUCTABLE, load_model, resolve_model_path, _ModelPanel,
    PAD, BG, TITLE_H,
)


# ---------------------------------------------------------------------------
# Shared: load a world-model checkpoint (autoencoder + dynamics head)
# ---------------------------------------------------------------------------

def load_world_model(path: Path, device: torch.device):
    """Load a world-model checkpoint -> (autoencoder, ae_config, dynamics_head, trained_ticks_per_step).

    Reuses model_viewer.load_model for the autoencoder half (it reads model_state/model_config),
    then rebuilds the LatentTransition head from dynamics_config and loads dynamics_state.
    trained_ticks_per_step is the sim-ticks a single dynamics step spans (dyn_config["ticks_per_step"],
    written by the trainers), or None for older checkpoints that predate the field.
    """
    ae_model, ae_cfg = load_model(path, device)                       # autoencoder + its config
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    if "dynamics_state" not in ckpt or "dynamics_config" not in ckpt:
        raise SystemExit(
            f"'{path}' is not a world-model checkpoint (missing dynamics_state/dynamics_config); "
            f"it looks like a plain autoencoder -- use src/model_viewer.py for that."
        )
    dc = ckpt["dynamics_config"]
    # Match the autoencoder's load dtype (f32 on CPU where half is slow; the trained dtype on GPU).
    dyn_dtype = next(ae_model.parameters()).dtype
    # Rebuild with the checkpoint's own head geometry. Defaults reproduce pre-existing checkpoints that
    # omit these keys: residual/normalize_output default True (the SPR regime) and norm defaults to
    # batchnorm (the historical layer). The decode-rollout regime records normalize_output=False and
    # norm=layernorm, both of which MUST be honored -- otherwise the head re-normalizes rolled latents the
    # decoder never saw normalized, and a layernorm state_dict won't even load into a batchnorm module.
    dynamics = LatentTransition(
        dc["latent_dim"], hidden_dim=dc.get("hidden_dim"), depth=dc.get("depth", 2),
        residual=dc.get("residual", True), normalize_output=dc.get("normalize_output", True),
        norm=dc.get("norm", "batchnorm"),
    ).to(device=device, dtype=dyn_dtype)
    dynamics.load_state_dict(ckpt["dynamics_state"])
    dynamics.eval()
    trained_ticks_per_step = dc.get("ticks_per_step")
    return ae_model, ae_cfg, dynamics, trained_ticks_per_step


def _validate_shapes(ae_cfg: dict, env: FirecrackerEnv, path: Path) -> None:
    if ae_cfg["grid_size"] != env.grid_size:
        raise SystemExit(f"model '{path}' trained on grid_size={ae_cfg['grid_size']} but this "
                         f"config uses {env.grid_size}; pass a matching --config.")
    if ae_cfg["in_channels"] != len(env.obs_channels):
        raise SystemExit(f"model '{path}' expects {ae_cfg['in_channels']} channels but this config's "
                         f"fuel types give {len(env.obs_channels)}.")


# ---------------------------------------------------------------------------
# ONLINE: roll against the live simulation
# ---------------------------------------------------------------------------

def _slerp(a: torch.Tensor, b: torch.Tensor, t: float) -> torch.Tensor:
    """Spherical linear interpolation between two unit-norm latents a, b (each (1, D)) at fraction
    t in [0, 1]. The dynamics head emits unit-norm latents (normalize_latent), so interpolating on the
    hypersphere -- rather than a straight lerp that dips inside it -- keeps every in-between latent on
    the manifold the decoder was trained on. Falls back to a renormalized lerp when a, b are nearly
    (anti)parallel and the sine denominator underflows."""
    if t <= 0.0:
        return a
    if t >= 1.0:
        return b
    dot = (a * b).sum(-1, keepdim=True).clamp(-1.0, 1.0)
    omega = torch.arccos(dot)
    sin_omega = torch.sin(omega)
    if float(sin_omega.abs().min()) < 1e-6:                    # ~(anti)parallel: lerp + renormalize
        out = a + t * (b - a)
        return out / out.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return (torch.sin((1.0 - t) * omega) / sin_omega) * a + (torch.sin(t * omega) / sin_omega) * b


class RolloutViewer(ComparisonViewer):
    """ComparisonViewer whose single model panel shows an open-loop latent rollout against the live
    physics. Only the panel's *source* differs from the base class: instead of encode(current obs) ->
    decode, it decodes a latent advanced by the dynamics head. Rather than holding one latent and
    jumping it once per --ticks-per-step, it keeps the current latent z0 and PREFETCHES the next one
    z1 = dynamics(z0); each physics tick it decodes a slerp(z0, z1, frac) so the prediction morphs
    smoothly across the stride instead of stepping. With interpolation off it holds z0 and the decode
    is cached (re-run only when the latent changes), so a slow host isn't decoding every physics tick."""

    def __init__(self, env: FirecrackerEnv, ae_model, ae_cfg: dict, dynamics: torch.nn.Module,
                 interpolate: bool = True) -> None:
        super().__init__(env, [(ae_model, ae_cfg, "Neural rollout")])
        self.dynamics = dynamics
        self.interpolate = interpolate
        self._z0: torch.Tensor | None = None   # current latent (integer dynamics step)
        self._z1: torch.Tensor | None = None   # prefetched next latent = dynamics(z0)
        self._frac = 0.0                        # interpolation position z0 -> z1 within the stride
        self._steps = 0
        self._z_version = 0          # bumped whenever the DISPLAYED latent changes (anchor/advance/frac)
        self._recon_version = -1     # version the cached decode was made at

    def _next(self, z: torch.Tensor) -> torch.Tensor:
        md = next(self.dynamics.parameters()).dtype
        with torch.no_grad():
            return self.dynamics(z.to(md))

    def anchor_to_current(self) -> None:
        """(Re)set the rollout to the encoding of the current physics observation and prefetch the
        next latent so the stride can be interpolated from the very first tick."""
        p = self.panels[0]
        gt = self.env._observation()
        x = torch.from_numpy(gt).to(self.device)
        x_in = (x - p.offset) / p.scale if p.offset is not None else x
        md = next(p.model.parameters()).dtype
        with torch.no_grad():
            self._z0 = p.model.encode(x_in.unsqueeze(0).to(md))
        self._z1 = self._next(self._z0)
        self._steps = 0
        self._frac = 0.0
        self._z_version += 1
        self._gt_obs = None          # force the next render to recompute, even if the env isn't dirty

    def advance_rollout(self) -> None:
        """Complete a stride: the prefetched next latent becomes current, and the one after it is
        prefetched. Resets the interpolation fraction (anchoring first if not yet anchored)."""
        if self._z0 is None:
            self.anchor_to_current()
            return
        self._z0 = self._z1
        self._z1 = self._next(self._z0)
        self._steps += 1
        self._frac = 0.0
        self._z_version += 1
        self._gt_obs = None          # force the next render to recompute, even if the env isn't dirty

    def set_frac(self, frac: float) -> None:
        """Set the interpolation position within the current stride (0 = z0, 1 = z1). No-op (and no
        decode) when interpolation is off, preserving the cached one-decode-per-stride behavior."""
        if not self.interpolate:
            return
        frac = min(max(frac, 0.0), 1.0)
        if frac != self._frac:
            self._frac = frac
            self._z_version += 1
            self._gt_obs = None

    def _display_latent(self) -> torch.Tensor:
        """The latent actually decoded: the slerp between z0 and its prefetched successor when
        interpolating, else just z0 (the stepwise-held latent)."""
        if not self.interpolate or self._z1 is None or self._frac <= 0.0:
            return self._z0
        return _slerp(self._z0.float(), self._z1.float(), self._frac)

    def _compute_reconstruction(self) -> None:
        """Override: the model panel decodes the (possibly interpolated) rolled latent, cached across
        ticks by version. The ground-truth obs is refreshed every call (cheap) so the TAB error map
        tracks the live physics as it advances toward the model's prediction."""
        p = self.panels[0]
        self._gt_obs = self.env._observation()                        # live physics, for the error map
        if self._z0 is None:
            self.anchor_to_current()
        if self._recon_version == self._z_version:
            return                                                    # latent unchanged -> reuse decode
        md = next(p.model.parameters()).dtype
        with torch.no_grad():
            x_hat = p.model.decode(self._display_latent().to(md))[0].float()
        if p.out_sigmoid:                                        # logits -> [0,1] reconstruction
            x_hat = torch.sigmoid(x_hat)
        recon = x_hat * p.scale + p.offset if p.offset is not None else x_hat
        recon = recon.detach().cpu().numpy().astype(np.float32)
        p.recon_obs = recon
        p.recon_phys = recon * self.scales[:, None, None] + self.offsets[:, None, None]
        self._recon_version = self._z_version

    def _model_title(self, i: int, mode) -> str:
        pos = self._steps + self._frac if self.interpolate else float(self._steps)
        base = f"Neural rollout  (t+{pos:.2f})" if self.interpolate else f"Neural rollout  (t+{self._steps})"
        return base if mode in RECONSTRUCTABLE else f"{base} (n/a)"


def run_online(args, config, env, ae_model, ae_cfg, dynamics) -> None:
    viewer = RolloutViewer(env, ae_model, ae_cfg, dynamics, interpolate=args.interpolate)
    env.reset()
    viewer.anchor_to_current()

    tick_interval = 1.0 / max(args.ticks_per_second, 1e-6)
    mode = "smoothly interpolated (slerp) every tick" if args.interpolate else "stepwise (cached)"
    print(f"online: physics at {args.ticks_per_second} ticks/s; dynamics step every "
          f"{args.ticks_per_step} ticks (~{args.ticks_per_step / args.ticks_per_second:.0f} s per model step); "
          f"prediction {mode}")

    phys_ticks = 0
    last_tick = time.monotonic()
    a_prev = False
    while env._running:
        now = time.monotonic()
        pressed = pygame.key.get_pressed()          # render() pumps the queue; edge-detect A (anchor)
        a_down = pressed[pygame.K_a]
        if a_down and not a_prev:
            viewer.anchor_to_current()
            phys_ticks = 0
        a_prev = a_down

        stepped = False
        if env._step_once:                          # '.' single physics tick
            env._step_once = False
            env.step(None)
            phys_ticks += 1
            stepped = True
        elif not env._paused and now - last_tick >= tick_interval:
            env.step(None)                          # one tick, throttled to ticks_per_second
            phys_ticks += 1
            last_tick = now
            stepped = True

        if stepped:
            if phys_ticks >= args.ticks_per_step:
                viewer.advance_rollout()            # a full stride elapsed -> the prefetched latent becomes current
                phys_ticks = 0
            viewer.set_frac(phys_ticks / max(args.ticks_per_step, 1))  # morph z0 -> z1 across the stride

        if env._reset_requested:
            env.reset()
            viewer.anchor_to_current()
            phys_ticks = 0

        viewer.render()
    viewer.close()


# ---------------------------------------------------------------------------
# OFFLINE: roll against recorded dataset frames (no simulation)
# ---------------------------------------------------------------------------

class OfflineRolloutViewer(ComparisonViewer):
    """Compare an open-loop rollout against RECORDED dataset frames. A dataset episode is one world's
    F frames at the training stride; this encodes frame 0, rolls the head over the episode, decodes
    each, and shows recorded (left) vs. rollout (right). No simulation -- frames are precomputed, so
    scrubbing is instant. Reuses the base class's surface builders / de-standardisation / error map;
    the left "physics" slot is repainted from the recorded frame instead of the live env."""

    def __init__(self, env, ae_model, ae_cfg, dynamics, dataset: FireDataset, num_envs: int,
                 interval: float) -> None:
        super().__init__(env, [(ae_model, ae_cfg, "Neural rollout")])   # -> 2 panel slots (recorded | rollout)
        self.dynamics = dynamics
        self.ds = dataset
        self.B = int(num_envs)
        if self.ds.per_shard % self.B:
            raise SystemExit(f"per_shard {self.ds.per_shard} not divisible by num_envs {self.B}.")
        self.F = self.ds.per_shard // self.B
        self.n_episodes = len(self.ds.shard_paths) * self.B
        self.interval = interval
        # A scratch panel to render the recorded frame through the base class's _recon_surface.
        self._rec_panel = _ModelPanel(ae_model, ae_cfg, "recorded", self.device)
        self._rng = np.random.default_rng()
        self._k = 0
        self._auto = True
        self._left_prev = self._right_prev = self._n_prev = False
        # Per-episode sequences (F, C, H, W): recorded + rollout, in obs and physical units.
        self._gt_obs_seq = self._gt_phys_seq = None
        self._pred_obs_seq = self._pred_phys_seq = None
        self.load_episode(int(self._rng.integers(self.n_episodes)))

    def load_episode(self, ep: int) -> None:
        """Load episode `ep` (= shard*B + env) and precompute the recorded frames and the rollout."""
        shard, env_i = divmod(int(ep) % self.n_episodes, self.B)
        per = self.ds.per_shard
        idx = [shard * per + f * self.B + env_i for f in range(self.F)]   # frame-major/env-minor layout
        frames = self.ds.get_batch(idx).to(self.device).float()          # (F, C, H, W)
        p = self.panels[0]
        md = next(p.model.parameters()).dtype
        offset, scale = p.offset, p.scale

        # Work in affine-OBS space (build_observation's ~[0,1] windows) throughout, normalizing with
        # the CHECKPOINT's stats -- exactly like the online path scales env._observation(). Shards are
        # already affine-OBS unless the dataset was baked-normalized (then un-scale, assuming its
        # stats match the checkpoint). This lets a raw, un-normalized tiny dataset work directly.
        affine = frames * scale + offset if (self.ds.is_normalized and offset is not None) else frames
        model_in = (affine - offset) / scale if offset is not None else affine   # normalize w/ ckpt stats

        def unnorm(t):   # model output (normalized) -> affine-OBS
            return t * scale + offset if offset is not None else t

        def dec(zk):     # decode -> [0,1] recon (sigmoid on a logits/bce-trained head), then affine-OBS
            o = p.model.decode(zk.to(md))[0].float()
            return unnorm(o.sigmoid() if p.out_sigmoid else o)

        with torch.no_grad():
            gt_obs = affine                                              # (F, C, H, W) affine-OBS
            z = p.model.encode(model_in[0:1].to(md))                     # anchor on frame 0
            zs = [z]
            for _ in range(self.F - 1):
                z = self.dynamics(z.to(next(self.dynamics.parameters()).dtype))
                zs.append(z)
            preds = [dec(zk) for zk in zs]
            pred_obs = torch.stack(preds)                                # (F, C, H, W) affine-OBS

        self._gt_obs_seq = gt_obs.cpu().numpy().astype(np.float32)
        self._pred_obs_seq = pred_obs.cpu().numpy().astype(np.float32)
        sc, off = self.scales[:, None, None], self.offsets[:, None, None]
        self._gt_phys_seq = self._gt_obs_seq * sc + off
        self._pred_phys_seq = self._pred_obs_seq * sc + off
        self._ep = int(ep) % self.n_episodes
        self._k = 0
        print(f"episode {self._ep}/{self.n_episodes - 1} (shard {shard}, env {env_i}); {self.F} frames")

    def next_frame(self) -> None:
        self._k = (self._k + 1) % self.F

    def render(self) -> None:
        env = self.env
        env._running, env._current_mode, _click = env._handle_events()   # mode keys / pause / quit
        pressed = pygame.key.get_pressed()

        def edge(key, prev_attr):
            down = pressed[key]
            fired = down and not getattr(self, prev_attr)
            setattr(self, prev_attr, down)
            return fired

        if edge(pygame.K_TAB, "_tab_prev"):
            self._show_error = not self._show_error
        if edge(pygame.K_m, "_m_prev"):
            self._show_burning_pred = not self._show_burning_pred
        if edge(pygame.K_LEFT, "_left_prev"):
            self._k = max(0, self._k - 1); self._auto = False
        if edge(pygame.K_RIGHT, "_right_prev"):
            self._k = min(self.F - 1, self._k + 1); self._auto = False
        if edge(pygame.K_n, "_n_prev"):
            self.load_episode(int(self._rng.integers(self.n_episodes)))
        if env._step_once:
            env._step_once = False
            self.next_frame(); self._auto = False

        mode, k = env._current_mode, self._k
        # Point the base-class helpers at frame k: gt for the error map, and each panel's cached recon.
        self._gt_obs = self._gt_obs_seq[k]
        self._rec_panel.recon_obs = self._gt_obs_seq[k]
        self._rec_panel.recon_phys = self._gt_phys_seq[k]
        self.panels[0].recon_obs = self._pred_obs_seq[k]
        self.panels[0].recon_phys = self._pred_phys_seq[k]

        self._screen.fill(BG)
        xs = [PAD + i * (self.sim_px + PAD) for i in range(self.n_panels)]   # 2 slots
        rec = self._recon_surface(mode, self._rec_panel) or self._placeholder_surface()
        self._blit_panel(rec, xs[0], f"Recorded  (frame {k}/{self.F - 1})")

        if self._show_error:
            err = self._error_field(mode, self.panels[0])
            surf = build_color_surface(err, self.scale, 0.0, float(err.max()) or 1.0, heat_colormap)
            self._blit_panel(surf, xs[1], self._error_title(0, err))
        else:
            surf = self._recon_surface(mode, self.panels[0]) or self._placeholder_surface()
            self._blit_panel(surf, xs[1], f"Rollout  (t+{k})")

        if mode == ViewMode.BIOME:
            panel = build_biome_legend_panel(self.sim_px, self._legend_font, self._legend_title_font)
        else:
            panel = build_legend_panel(env._legend_specs(), self.sim_px, self._legend_font,
                                       self._legend_title_font)
        self._screen.blit(panel, (xs[-1] + self.sim_px + PAD, TITLE_H))
        pygame.display.flip()
        self._clock.tick(TARGET_FPS)


def run_offline(args, config, env, ae_model, ae_cfg, dynamics) -> None:
    dataset = FireDataset(args.data)
    num_envs = args.num_envs if args.num_envs is not None else dataset.meta.get("num_envs")
    if num_envs is None:
        raise SystemExit("dataset meta has no 'num_envs' (built before that field); pass --num-envs "
                         "(the campaign data/fire was built with --num-envs 32).")
    if dataset.meta.get("num_channels") != ae_cfg["in_channels"] or dataset.meta.get("grid_size") != ae_cfg["grid_size"]:
        raise SystemExit("dataset shape does not match the model "
                         f"({dataset.meta.get('num_channels')}x{dataset.meta.get('grid_size')} vs "
                         f"{ae_cfg['in_channels']}x{ae_cfg['grid_size']}).")
    # The viewer normalizes frames with the CHECKPOINT's stats, so a raw dataset is preferred. A
    # baked-normalized dataset is only correct if it was normalized with those same (training) stats.
    if dataset.is_normalized:
        print(f"note: dataset '{args.data}' is normalized; assuming its stats match the checkpoint. "
              f"A dataset baked with its OWN stats will render wrong -- prefer a raw dataset.")
    else:
        print(f"dataset '{args.data}' is raw; normalizing with the checkpoint's channel stats.")
    viewer = OfflineRolloutViewer(env, ae_model, ae_cfg, dynamics, dataset, num_envs, args.interval)
    print("offline: LEFT/RIGHT scrub frames, N new episode, SPACE autoplay, TAB error, ESC quit")

    last = time.monotonic()
    while env._running:
        now = time.monotonic()
        if viewer._auto and not env._paused and now - last >= args.interval:
            viewer.next_frame()
            last = now
        viewer.render()
    viewer.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rollout of a latent world model vs. live physics (online) or recorded frames (offline)."
    )
    parser.add_argument("--model", required=True, metavar="PATH",
                        help="world-model checkpoint (runs/<wm_run>/run_NN/best.pt or models/*.pt)")
    parser.add_argument("--config", metavar="PATH", help="path to a TOML config file")
    parser.add_argument("--map", dest="map_name", default=None, help="specific map to load (online)")
    parser.add_argument("--maps-dir", default=None, help="directory to sample the map from (online)")
    # Offline mode
    parser.add_argument("--data", default=None, metavar="DIR",
                        help="dataset dir -> OFFLINE mode: roll against recorded frames (no simulation)")
    parser.add_argument("--num-envs", type=int, default=None,
                        help="envs-per-shard for datasets whose meta predates the field (data/fire: 32)")
    # Online cadence
    parser.add_argument("--ticks-per-step", type=int, default=None,
                        help="physics ticks per dynamics step; defaults to the training stride recorded "
                             "in the checkpoint (falling back to 256 for checkpoints that predate it)")
    parser.add_argument("--ticks-per-second", type=float, default=10.0,
                        help="online physics tick rate; keeps the UI responsive (default 10)")
    parser.add_argument("--interpolate", action="store_true",
                        help="online: smoothly slerp-interpolate the prediction to the prefetched next "
                             "latent every tick instead of holding each predicted latent and jumping "
                             "once per stride (the default stepwise behavior, which decodes once per "
                             "stride and is cheaper on CPU)")
    parser.add_argument("--interval", type=float, default=0.3,
                        help="seconds between frames in offline autoplay (default 0.3)")
    args = parser.parse_args()

    config = load_config(args.config)
    env = FirecrackerEnv(config=config, render_mode="rgb_array",
                         map_name=args.map_name, maps_dir=args.maps_dir)
    device = env._sim.device
    path = resolve_model_path(args.model)
    ae_model, ae_cfg, dynamics, trained_tps = load_world_model(path, device)
    _validate_shapes(ae_cfg, env, path)
    print(f"Loaded world model: {path}  (latent_dim={ae_cfg['latent_dim']}, device={device})")

    # Resolve the physics-ticks-per-dynamics-step: honor an explicit --ticks-per-step, else use the
    # stride the head was trained on (recorded in the checkpoint), else fall back to the old 256 default
    # for checkpoints that predate the field. Warn on an explicit override that disagrees with training,
    # since a mismatched stride is exactly the drift this field exists to prevent.
    if args.ticks_per_step is None:
        if trained_tps is not None:
            args.ticks_per_step = int(trained_tps)
            print(f"ticks-per-step: {args.ticks_per_step} (from the checkpoint's training stride)")
        else:
            args.ticks_per_step = 256
            print("ticks-per-step: 256 (checkpoint predates the recorded training stride; pass "
                  "--ticks-per-step to match how the head was trained)")
    elif trained_tps is not None and int(trained_tps) != args.ticks_per_step:
        print(f"warning: --ticks-per-step {args.ticks_per_step} differs from the checkpoint's training "
              f"stride {int(trained_tps)}; the physics will not advance one training step per model step.")

    if args.data is not None:
        run_offline(args, config, env, ae_model, ae_cfg, dynamics)
    else:
        run_online(args, config, env, ae_model, ae_cfg, dynamics)


if __name__ == "__main__":
    main()
