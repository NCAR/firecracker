"""
rollout_viewer.py

Rollout visualiser for a latent world model (src/train_world_model.py): it rolls the dynamics head
forward in latent space and decodes each step, so you watch the neural world evolve and see where it
diverges from the physics. Two modes:

  ONLINE (default) -- roll against the LIVE physics simulation.
      python tools/rollout_viewer.py --model models/world_model_convbn256_16ch_best_bf16.pt
    The model was trained on jumps of `--ticks-per-step` sim-ticks (default 256 = the dataset
    stride), so one dynamics step means that many physics ticks. Rather than burst-simulate them
    (which freezes the UI for ~25 s), physics runs at `--ticks-per-second` (default 10) and the
    dynamics head advances once every `--ticks-per-step` ticks -- the model updates ~every 25 s while
    the physics panel animates smoothly. A re-anchors the latent to the current frame.

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
    """Load a world-model checkpoint -> (autoencoder, ae_config, dynamics_head), all in eval mode.

    Reuses model_viewer.load_model for the autoencoder half (it reads model_state/model_config),
    then rebuilds the LatentTransition head from dynamics_config and loads dynamics_state.
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
    dynamics = LatentTransition(
        dc["latent_dim"], hidden_dim=dc.get("hidden_dim"), depth=dc.get("depth", 2),
    ).to(device=device, dtype=dyn_dtype)
    dynamics.load_state_dict(ckpt["dynamics_state"])
    dynamics.eval()
    return ae_model, ae_cfg, dynamics


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

class RolloutViewer(ComparisonViewer):
    """ComparisonViewer whose single model panel shows an open-loop latent rollout against the live
    physics. Only the panel's *source* differs from the base class: instead of encode(current obs) ->
    decode, it decodes a latent advanced by the dynamics head. The decode is cached and only re-run
    when the latent actually changes (once per --ticks-per-step), so a slow host isn't decoding every
    physics tick."""

    def __init__(self, env: FirecrackerEnv, ae_model, ae_cfg: dict, dynamics: torch.nn.Module) -> None:
        super().__init__(env, [(ae_model, ae_cfg, "Neural rollout")])
        self.dynamics = dynamics
        self._z: torch.Tensor | None = None
        self._steps = 0
        self._z_version = 0          # bumped whenever the latent changes
        self._recon_version = -1     # version the cached decode was made at

    def anchor_to_current(self) -> None:
        """(Re)set the rollout latent to the encoding of the current physics observation."""
        p = self.panels[0]
        gt = self.env._observation()
        x = torch.from_numpy(gt).to(self.device)
        x_in = (x - p.offset) / p.scale if p.offset is not None else x
        md = next(p.model.parameters()).dtype
        with torch.no_grad():
            self._z = p.model.encode(x_in.unsqueeze(0).to(md))
        self._steps = 0
        self._z_version += 1
        self._gt_obs = None          # force the next render to recompute, even if the env isn't dirty

    def advance_rollout(self) -> None:
        """Step the latent one dynamics step (anchoring first if not yet anchored)."""
        if self._z is None:
            self.anchor_to_current()
            return
        md = next(self.dynamics.parameters()).dtype
        with torch.no_grad():
            self._z = self.dynamics(self._z.to(md))
        self._steps += 1
        self._z_version += 1
        self._gt_obs = None          # force the next render to recompute, even if the env isn't dirty

    def _compute_reconstruction(self) -> None:
        """Override: the model panel decodes the ROLLED latent, cached across physics ticks. The
        ground-truth obs is refreshed every call (cheap) so the TAB error map tracks the live physics
        as it advances toward the model's prediction."""
        p = self.panels[0]
        self._gt_obs = self.env._observation()                        # live physics, for the error map
        if self._z is None:
            self.anchor_to_current()
        if self._recon_version == self._z_version:
            return                                                    # latent unchanged -> reuse decode
        md = next(p.model.parameters()).dtype
        with torch.no_grad():
            x_hat = p.model.decode(self._z.to(md))[0].float()
        recon = x_hat * p.scale + p.offset if p.offset is not None else x_hat
        recon = recon.detach().cpu().numpy().astype(np.float32)
        p.recon_obs = recon
        p.recon_phys = recon * self.scales[:, None, None] + self.offsets[:, None, None]
        self._recon_version = self._z_version

    def _model_title(self, i: int, mode) -> str:
        base = f"Neural rollout  (t+{self._steps})"
        return base if mode in RECONSTRUCTABLE else f"{base} (n/a)"


def run_online(args, config, env, ae_model, ae_cfg, dynamics) -> None:
    viewer = RolloutViewer(env, ae_model, ae_cfg, dynamics)
    env.reset()
    viewer.anchor_to_current()

    tick_interval = 1.0 / max(args.ticks_per_second, 1e-6)
    print(f"online: physics at {args.ticks_per_second} ticks/s; dynamics step every "
          f"{args.ticks_per_step} ticks (~{args.ticks_per_step / args.ticks_per_second:.0f} s per model step)")

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

        if stepped and phys_ticks >= args.ticks_per_step:
            viewer.advance_rollout()                # a full stride elapsed -> advance the prediction
            phys_ticks = 0

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

        with torch.no_grad():
            gt_obs = affine                                              # (F, C, H, W) affine-OBS
            z = p.model.encode(model_in[0:1].to(md))                     # anchor on frame 0
            zs = [z]
            for _ in range(self.F - 1):
                z = self.dynamics(z.to(next(self.dynamics.parameters()).dtype))
                zs.append(z)
            preds = [unnorm(p.model.decode(zk.to(md))[0].float()) for zk in zs]
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
    parser.add_argument("--ticks-per-step", type=int, default=256,
                        help="physics ticks per dynamics step (default 256 = the training stride)")
    parser.add_argument("--ticks-per-second", type=float, default=10.0,
                        help="online physics tick rate; keeps the UI responsive (default 10)")
    parser.add_argument("--interval", type=float, default=0.3,
                        help="seconds between frames in offline autoplay (default 0.3)")
    args = parser.parse_args()

    config = load_config(args.config)
    env = FirecrackerEnv(config=config, render_mode="rgb_array",
                         map_name=args.map_name, maps_dir=args.maps_dir)
    device = env._sim.device
    path = resolve_model_path(args.model)
    ae_model, ae_cfg, dynamics = load_world_model(path, device)
    _validate_shapes(ae_cfg, env, path)
    print(f"Loaded world model: {path}  (latent_dim={ae_cfg['latent_dim']}, device={device})")

    if args.data is not None:
        run_offline(args, config, env, ae_model, ae_cfg, dynamics)
    else:
        run_online(args, config, env, ae_model, ae_cfg, dynamics)


if __name__ == "__main__":
    main()
