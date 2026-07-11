"""
rollout_viewer.py

Open-loop rollout visualiser for a latent world model (src/train_world_model.py) against the live
physics simulation. Where model_viewer.py autoencodes the *current* frame, this rolls the dynamics
head forward in latent space and decodes each step, so you watch the neural world evolve on its own
and see where it diverges from the physics.

Each step:
    z <- transition(z)                       # advance the latent one dynamics step (no re-encode)
    model panel = decode(z)                  # the imagined observation, in physical units
    physics     = sim stepped `--ticks-per-step` ticks   # the ground truth it's compared against

The latent is anchored once (z = encode(current obs)); thereafter it free-runs. Because the model
was trained on single-step jumps of `stride` sim-ticks, one neural step should be compared against
that many physics ticks (`--ticks-per-step`, default 256 = the training stride). The panels reuse
model_viewer.ComparisonViewer wholesale (same de-standardisation, channel layout, surface builders,
and the TAB error map = |physics_now - neural_rollout|).

Controls: A re-anchors the latent to the current physics obs (re-sync); SPACE pause, . single-step,
number/s/b switch view, TAB reconstruction<->error, ESC quit -- all inherited from the base viewer.

Note: trained open-loop error compounds (single-step training, ~8-step horizon), so expect drift
after a few steps -- visualising that drift is the point.

Usage:
    python tools/rollout_viewer.py --model runs/<wm_run>/run_00/best.pt
    python tools/rollout_viewer.py --model runs/<wm_run>/run_00/best.pt --ticks-per-step 256
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
from firecracker_env import FirecrackerEnv, TARGET_FPS
from dynamics import LatentTransition
from model_viewer import ComparisonViewer, RECONSTRUCTABLE, load_model, resolve_model_path


class RolloutViewer(ComparisonViewer):
    """ComparisonViewer whose single model panel shows an open-loop latent rollout, not a
    per-frame reconstruction. Inherits all rendering; only the panel's *source* changes: instead of
    encode(current obs) -> decode, it decodes a latent advanced by the dynamics head each step."""

    def __init__(self, env: FirecrackerEnv, ae_model, ae_cfg: dict, dynamics: torch.nn.Module) -> None:
        super().__init__(env, [(ae_model, ae_cfg, "Neural rollout")])
        self.dynamics = dynamics
        self._z: torch.Tensor | None = None     # current rollout latent (1, latent_dim)
        self._steps = 0                          # dynamics steps since the last anchor

    def anchor_to_current(self) -> None:
        """(Re)set the rollout latent to the encoding of the current physics observation."""
        p = self.panels[0]
        gt = self.env._observation()                                  # (C, H, W) normalized-OBS
        x = torch.from_numpy(gt).to(self.device)
        x_in = (x - p.mean) / p.std if p.mean is not None else x
        md = next(p.model.parameters()).dtype
        with torch.no_grad():
            self._z = p.model.encode(x_in.unsqueeze(0).to(md))
        self._steps = 0

    def advance_rollout(self) -> None:
        """Step the latent one dynamics step (anchoring first if not yet anchored)."""
        if self._z is None:
            self.anchor_to_current()
            return
        md = next(self.dynamics.parameters()).dtype
        with torch.no_grad():
            self._z = self.dynamics(self._z.to(md))
        self._steps += 1

    def _compute_reconstruction(self) -> None:
        """Override: the model panel decodes the ROLLED latent (self._z), not an encode of the
        current frame. self._gt_obs stays the live physics obs, so the TAB error panel reads as
        |physics_now - neural_rollout|. Mirrors the base class's de-standardisation exactly."""
        p = self.panels[0]
        self._gt_obs = self.env._observation()                        # ground truth for the error map
        if self._z is None:
            self.anchor_to_current()
        md = next(p.model.parameters()).dtype
        with torch.no_grad():
            x_hat = p.model.decode(self._z.to(md))[0].float()         # (C, H, W)
        recon = x_hat * p.std + p.mean if p.mean is not None else x_hat
        recon = recon.detach().cpu().numpy().astype(np.float32)
        p.recon_obs = recon
        p.recon_phys = recon * self.scales[:, None, None] + self.offsets[:, None, None]

    def _model_title(self, i: int, mode) -> str:
        base = f"Neural rollout  (t+{self._steps})"
        return base if mode in RECONSTRUCTABLE else f"{base} (n/a)"


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
    dynamics = LatentTransition(
        dc["latent_dim"], hidden_dim=dc.get("hidden_dim"), depth=dc.get("depth", 2),
    ).to(device=device)
    dynamics.load_state_dict(ckpt["dynamics_state"])
    dynamics.eval()
    return ae_model, ae_cfg, dynamics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Open-loop rollout of a latent world model vs. the live physics simulation."
    )
    parser.add_argument("--model", required=True, metavar="PATH",
                        help="world-model checkpoint (runs/<wm_run>/run_NN/best.pt)")
    parser.add_argument("--config", metavar="PATH", help="path to a TOML config file")
    parser.add_argument("--map", dest="map_name", default=None, help="specific map to load")
    parser.add_argument("--maps-dir", default=None, help="directory to sample the map from")
    parser.add_argument("--ticks-per-step", type=int, default=256,
                        help="physics ticks advanced per dynamics step (default 256 = the dataset "
                             "stride the model was trained on)")
    parser.add_argument("--interval", type=float, default=0.3,
                        help="seconds between dynamics steps when not paused (default 0.3)")
    args = parser.parse_args()

    config = load_config(args.config)
    env = FirecrackerEnv(config=config, render_mode="rgb_array",
                         map_name=args.map_name, maps_dir=args.maps_dir)
    device = env._sim.device

    path = resolve_model_path(args.model)
    ae_model, ae_cfg, dynamics = load_world_model(path, device)
    # Same shape guards run_comparison uses: refuse a config the checkpoint wasn't trained for.
    if ae_cfg["grid_size"] != env.grid_size:
        raise SystemExit(f"model '{path}' trained on grid_size={ae_cfg['grid_size']} but this "
                         f"config uses {env.grid_size}; pass a matching --config.")
    if ae_cfg["in_channels"] != len(env.obs_channels):
        raise SystemExit(f"model '{path}' expects {ae_cfg['in_channels']} channels but this config's "
                         f"fuel types give {len(env.obs_channels)}.")
    print(f"Loaded world model: {path}  (latent_dim={ae_cfg['latent_dim']}, "
          f"ticks/step={args.ticks_per_step})")

    viewer = RolloutViewer(env, ae_model, ae_cfg, dynamics)
    env.reset()
    viewer.anchor_to_current()

    last_step = time.monotonic()
    a_prev = False
    while env._running:
        now = time.monotonic()
        # A re-anchors the latent to the current physics frame (edge-detected). Read from the held-key
        # snapshot; render() pumps the event queue each frame, so it registers within a frame.
        pressed = pygame.key.get_pressed()
        a_down = pressed[pygame.K_a]
        if a_down and not a_prev:
            viewer.anchor_to_current()
        a_prev = a_down

        # Single-step (.) advances one dynamics step; otherwise auto-advance on the interval.
        if env._step_once:
            env._step_once = False
            for _ in range(args.ticks_per_step):
                env.step(None)
            viewer.advance_rollout()
            last_step = now
        elif not env._paused and now - last_step >= args.interval:
            for _ in range(args.ticks_per_step):
                env.step(None)
            viewer.advance_rollout()
            last_step = now

        if env._reset_requested:
            env.reset()
            viewer.anchor_to_current()

        viewer.render()

    viewer.close()


if __name__ == "__main__":
    main()
