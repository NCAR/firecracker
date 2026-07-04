"""
model_viewer.py

Side-by-side visualiser for a trained autoencoder against the live physics simulation.

Given a checkpoint (models/best.pt or similar), this loads the model, rebuilds its architecture
from the config embedded in the checkpoint, and runs the interactive world alongside the model's
reconstruction of it. Three panels are drawn per frame:

    Physics        the ground-truth simulation view (exactly what the normal viewer shows)
    Reconstruction the same view rebuilt from the autoencoder's output
    Error          per-cell |physics - reconstruction| for the shown field, as a heatmap

so you can eyeball how faithfully the model reproduces each observation channel.

The autoencoder only ever sees the world-model *observation* — the normalized stack of
fuel_temperature, one fuel_<type> mass channel per fuel type, terrain, and the two near-surface
wind components (see Simulation.build_observation). Only views backed by those channels can be
reconstructed (fuel temperature, terrain, wind speed, fire); other views (air temperature, oxygen,
pressure, radiant heat, ...) draw from state that is not in the observation, so the reconstruction
panel shows a "not in observation" placeholder for them while the physics panel still renders.

The observation is doubly normalized on the way into the model: build_observation applies the affine
OBS_NORM windows (roughly [0, 1] per channel), then training z-scores each channel to zero mean /
unit variance using stats carried in the checkpoint (channel_mean / channel_std). Both transforms
are inverted here so the reconstruction lands back in physical units for rendering.

Invoked via `python src/main.py --model models/best.pt`; see main.run_comparison.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pygame
import torch

import physics_constants as pc
from autoencoder import ConvAutoencoder
from strided_autoencoder import StridedConvAutoencoder
from strided_vae import StridedConvVAE
from vae import ConvVAE
from firecracker_env import FirecrackerEnv, ViewMode, TARGET_FPS
from simulation import OBS_CHANNELS_PRE, OBS_CHANNELS_POST, obs_channel_names
from rendering import (
    LEGEND_PANEL_WIDTH,
    build_color_surface,
    build_terrain_surface,
    build_wind_speed_surface,
    build_fire_surface,
    build_legend_panel,
    heat_colormap,
)

# The autoencoder and the strided variant share the (reconstruction, latent) forward contract and
# the same constructor signature; the checkpoint's model_config["arch"] picks one. Kept in step with
# train_autoencoder.ARCHITECTURES (imported directly here to avoid pulling in the training deps).
ARCHITECTURES = {
    "shuffle": ConvAutoencoder,
    "strided": StridedConvAutoencoder,
    "shuffle-vae": ConvVAE,
    "strided-vae": StridedConvVAE,
}

# Weight dtypes a checkpoint may be saved in (see train_autoencoder.WEIGHT_DTYPES). Checkpoints
# from before --weight-dtype existed have no such key and load as float32.
WEIGHT_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

# View modes whose field is fully recoverable from the observation channels, so the reconstruction
# panel can render them. Every other mode draws from non-observed state and gets a placeholder.
RECONSTRUCTABLE: frozenset[ViewMode] = frozenset({
    ViewMode.FUEL_TEMPERATURE,
    ViewMode.TERRAIN,
    ViewMode.WIND_SPEED,
    ViewMode.FIRE,
})

# Layout constants for the composited window.
PAD = 8
TITLE_H = 24
BG = (0, 0, 0)
TITLE_FG = (220, 220, 220)
PLACEHOLDER_FG = (120, 120, 120)


def resolve_model_path(spec: str) -> Path:
    """Resolve a --model spec to a checkpoint file. Accepts a .pt path, a directory (prefers
    best.pt, then last.pt, else the first *.pt), or a bare name resolved under models/."""
    p = Path(spec)
    if p.is_dir():
        for name in ("best.pt", "last.pt"):
            if (p / name).is_file():
                return p / name
        pts = sorted(p.glob("*.pt"))
        if pts:
            return pts[0]
        raise SystemExit(f"--model: no .pt checkpoint found in directory '{p}'.")
    if p.is_file():
        return p
    for cand in (Path("models") / spec, Path("models") / f"{spec}.pt", Path(f"{spec}.pt")):
        if cand.is_file():
            return cand
    raise SystemExit(f"--model: no checkpoint found for '{spec}' (looked for a file, a directory, "
                     f"and models/{spec}[.pt]).")


def load_model(path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    """Load a checkpoint and rebuild the trained autoencoder on `device`, in eval mode.

    Returns (model, model_config). The model_config carries the architecture (arch / channels /
    latent_dim / grid_size / in_channels) and, when the training dataset was standardized, the
    per-channel mean/std the input was z-scored by (channel_mean / channel_std).
    """
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        # Older torch has no weights_only kwarg.
        ckpt = torch.load(path, map_location=device)

    if "model_config" not in ckpt or "model_state" not in ckpt:
        raise SystemExit(f"'{path}' is not a Firecracker autoencoder checkpoint "
                         f"(missing model_config/model_state).")
    cfg = ckpt["model_config"]
    arch = cfg.get("arch", "shuffle")
    if arch not in ARCHITECTURES:
        raise SystemExit(f"'{path}' names unknown arch '{arch}'; expected one of {list(ARCHITECTURES)}.")

    # Rebuild in the dtype the weights were trained in, except on CPU: half precision is slow /
    # partially unimplemented there, so up-cast to float32 (lossless from bfloat16). load_state_dict
    # copies each saved tensor with Tensor.copy_, which casts dtype, so a bf16 checkpoint widens
    # cleanly into an f32 model.
    saved_dtype = WEIGHT_DTYPES.get(cfg.get("weight_dtype", "float32"), torch.float32)
    load_dtype = torch.float32 if device.type == "cpu" else saved_dtype

    model = ARCHITECTURES[arch](
        in_channels=cfg["in_channels"],
        grid_size=cfg["grid_size"],
        channels=tuple(cfg["channels"]),
        latent_dim=cfg["latent_dim"],
    ).to(device=device, dtype=load_dtype)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg


class ComparisonViewer:
    """Owns the pygame window and composites the physics / reconstruction / error panels.

    The env is driven in rgb_array mode (it builds no window of its own); this viewer reuses the
    env's surface builders and event handling for the physics panel, runs the model to fill the
    reconstruction panel, and draws the per-cell error between them.
    """

    def __init__(self, env: FirecrackerEnv, model: torch.nn.Module, model_config: dict) -> None:
        self.env = env
        self.model = model
        self.device = env._sim.device
        self.scale = env._pixel_scale
        self.sim_px = env.grid_size * self.scale

        n_fuel = env._sim.num_fuel_types
        self.n_fuel = n_fuel
        # Observation channel layout (see obs_channel_names): fuel_temperature, then one fuel_<type>
        # mass channel per fuel type, then terrain, wind_x, wind_y.
        self.i_fuel_temp = 0
        self.i_fuel = slice(1, 1 + n_fuel)
        self.i_terrain = 1 + n_fuel
        self.i_wind_x = 2 + n_fuel
        self.i_wind_y = 3 + n_fuel

        # Affine OBS_NORM windows used by build_observation, per channel, for inverting back to
        # physical units. Order and values must mirror Simulation.build_observation exactly.
        windows = (
            [pc.OBS_NORM[c] for c in OBS_CHANNELS_PRE]
            + [pc.OBS_NORM["fuel"]] * n_fuel
            + [pc.OBS_NORM[c] for c in OBS_CHANNELS_POST]
        )
        self.offsets = np.array([w[0] for w in windows], dtype=np.float32)
        self.scales = np.array([w[1] for w in windows], dtype=np.float32)

        # Per-channel z-score the model was trained under (if the dataset was standardized). Kept on
        # the device so the (de)standardisation is a cheap broadcast around the forward pass.
        mean = model_config.get("channel_mean")
        std = model_config.get("channel_std")
        if mean is not None and std is not None:
            self.mean = torch.tensor(mean, dtype=torch.float32, device=self.device)[:, None, None]
            self.std = torch.tensor(std, dtype=torch.float32, device=self.device)[:, None, None]
        else:
            self.mean = self.std = None

        # Fuel display params reused by the reconstruction FIRE surface.
        self.ignition_thresholds = env._sim.ignition_thresholds.detach().cpu().numpy().reshape(-1)
        self.fuel_burnt_threshold = env._sim.fuel_burnt_threshold
        self.fuel_type_names = list(env._sim.fuel_type_names)

        # Cached model outputs, refreshed whenever the world advances (see render()).
        self._gt_obs: np.ndarray | None = None       # (C, H, W) normalized-OBS ground truth
        self._recon_obs: np.ndarray | None = None    # (C, H, W) normalized-OBS reconstruction
        self._recon_phys: np.ndarray | None = None   # (C, H, W) reconstruction in physical units

        pygame.init()
        pygame.font.init()
        pygame.display.set_caption("Firecracker — model reconstruction")
        width = 3 * self.sim_px + 4 * PAD + LEGEND_PANEL_WIDTH
        height = TITLE_H + self.sim_px + PAD
        self._screen = pygame.display.set_mode((width, height))
        self._clock = pygame.time.Clock()
        self._title_font = pygame.font.Font(None, 26)
        self._legend_font = pygame.font.Font(None, 20)
        self._legend_title_font = pygame.font.Font(None, 22)
        self._placeholder_font = pygame.font.Font(None, 30)

    # -- model -----------------------------------------------------------------

    def _compute_reconstruction(self) -> None:
        """Run the model on the current observation, caching the reconstruction in both
        normalized-OBS space (for the error panel) and physical units (for rendering)."""
        gt = self.env._observation()                               # (C, H, W) normalized-OBS
        x = torch.from_numpy(gt).to(self.device)
        x_in = (x - self.mean) / self.std if self.mean is not None else x
        # Feed the model in its own weight dtype (bf16/f16 on GPU; f32 on CPU) then widen the
        # reconstruction back to float32 for the de-standardisation math and rendering.
        model_dtype = next(self.model.parameters()).dtype
        with torch.no_grad():
            # VAE forwards return (x_hat, z, mu, logvar); the plain AEs return (x_hat, z). Take
            # the reconstruction either way. In eval mode the VAE decodes the posterior mean, so
            # this stays deterministic.
            x_hat = self.model(x_in.unsqueeze(0).to(model_dtype))[0]
        x_hat = x_hat.squeeze(0).float()
        recon = x_hat * self.std + self.mean if self.mean is not None else x_hat
        recon = recon.detach().cpu().numpy().astype(np.float32)    # (C, H, W) normalized-OBS

        self._gt_obs = gt
        self._recon_obs = recon
        self._recon_phys = recon * self.scales[:, None, None] + self.offsets[:, None, None]

    # -- reconstruction / error surfaces --------------------------------------

    def _recon_surface(self, mode: ViewMode) -> pygame.Surface | None:
        """Build the reconstruction-panel surface for `mode` from the cached model output, reusing
        the physics panel's display windows so equal colors mean equal values. Returns None for
        modes not backed by the observation (the caller draws a placeholder)."""
        if mode not in RECONSTRUCTABLE:
            return None
        env, phys, scale = self.env, self._recon_phys, self.scale
        if mode == ViewMode.FUEL_TEMPERATURE:
            return build_color_surface(
                phys[self.i_fuel_temp], scale,
                env._fuel_temp_display_min, env._fuel_temp_display_max,
            )
        if mode == ViewMode.TERRAIN:
            return build_terrain_surface(phys[self.i_terrain], scale, env._terrain_display_max)
        if mode == ViewMode.WIND_SPEED:
            return build_wind_speed_surface(
                phys[self.i_wind_x], phys[self.i_wind_y], scale, env._wind_speed_display_max,
            )
        # FIRE: the observation carries per-type fuel mass and the hottest fuel temperature (but not
        # per-type fuel temps or oxygen). Vegetation coloring needs only the mass; the burning
        # overlay is approximated by broadcasting the hottest temperature across types with oxygen
        # assumed present, so the reconstructed flame footprint reads against the physics panel.
        fuel = phys[self.i_fuel]
        hottest = phys[self.i_fuel_temp]
        fuel_temps = np.broadcast_to(hottest, fuel.shape)
        oxygen = np.ones(hottest.shape, dtype=np.float32)
        return build_fire_surface(
            fuel_temps, fuel, oxygen, scale,
            self.ignition_thresholds, self.fuel_burnt_threshold, self.fuel_type_names,
            env._show_fire_overlay and env._sim.fire_enabled,
        )

    def _error_field(self, mode: ViewMode) -> np.ndarray:
        """Per-cell reconstruction error (normalized-OBS units) for the shown field: the channel(s)
        backing the mode when it is reconstructable, else the mean absolute error over all channels."""
        gt, rc = self._gt_obs, self._recon_obs
        if mode == ViewMode.FUEL_TEMPERATURE:
            return np.abs(gt[self.i_fuel_temp] - rc[self.i_fuel_temp])
        if mode == ViewMode.TERRAIN:
            return np.abs(gt[self.i_terrain] - rc[self.i_terrain])
        if mode == ViewMode.WIND_SPEED:
            return np.hypot(gt[self.i_wind_x] - rc[self.i_wind_x], gt[self.i_wind_y] - rc[self.i_wind_y])
        if mode == ViewMode.FIRE:
            return np.abs(gt[self.i_fuel] - rc[self.i_fuel]).mean(axis=0)
        return np.abs(gt - rc).mean(axis=0)

    def _placeholder_surface(self) -> pygame.Surface:
        surf = pygame.Surface((self.sim_px, self.sim_px))
        surf.fill((24, 24, 24))
        text = self._placeholder_font.render("not in observation", True, PLACEHOLDER_FG)
        surf.blit(text, text.get_rect(center=(self.sim_px // 2, self.sim_px // 2)))
        return surf

    # -- frame -----------------------------------------------------------------

    def render(self) -> None:
        """Advance input handling, refresh surfaces, and composite one frame. Mirrors the role of
        FirecrackerEnv.render in the normal viewer (events -> surfaces -> flip)."""
        env = self.env
        dirty = env._surfaces_dirty
        env._rebuild_surfaces_if_dirty()
        # Reuse the env's own event handling for the physics panel (mode keys, pause/step/reset,
        # right-click ignition). It reads the global pygame event queue and mouse, so it works
        # against this viewer's window with the physics panel anchored at the origin.
        env._running, env._current_mode, fire_click = env._handle_events()
        if fire_click is not None:
            env._spawn_fire_patch(*fire_click)
            env._surfaces_dirty = True
            dirty = True
            env._rebuild_surfaces_if_dirty()
        if dirty or self._recon_phys is None:
            self._compute_reconstruction()

        mode = env._current_mode
        self._screen.fill(BG)

        phys_surface = env._surface_for_mode()
        recon_surface = self._recon_surface(mode)
        if recon_surface is None:
            recon_surface = self._placeholder_surface()

        error = self._error_field(mode)
        err_max = float(error.max())
        err_surface = build_color_surface(error, self.scale, 0.0, err_max, heat_colormap)

        xs = [PAD + i * (self.sim_px + PAD) for i in range(3)]
        self._blit_panel(phys_surface, xs[0], "Physics")
        recon_title = "Reconstruction" if mode in RECONSTRUCTABLE else "Reconstruction (n/a)"
        self._blit_panel(recon_surface, xs[1], recon_title)
        self._blit_panel(
            err_surface, xs[2],
            f"Error   mean {float(error.mean()):.3f}  max {err_max:.3f}",
        )

        # Shared legend for the physics/reconstruction color scale (both use the same window).
        panel = build_legend_panel(
            env._legend_specs(), self.sim_px, self._legend_font, self._legend_title_font
        )
        self._screen.blit(panel, (xs[2] + self.sim_px + PAD, TITLE_H))

        pygame.display.flip()
        self._clock.tick(TARGET_FPS)

    def _blit_panel(self, surface: pygame.Surface, x: int, title: str) -> None:
        self._screen.blit(self._title_font.render(title, True, TITLE_FG), (x, 4))
        self._screen.blit(surface, (x, TITLE_H))

    def close(self) -> None:
        pygame.quit()
