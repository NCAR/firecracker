"""
model_viewer.py

Side-by-side visualiser for one or more trained autoencoders against the live physics simulation.

Given one or more checkpoints (models/best.pt or similar), this loads each model, rebuilds its
architecture from the config embedded in the checkpoint, and runs the interactive world alongside
the models' reconstructions of it. One panel is drawn per model, next to the physics view:

    Physics        the ground-truth simulation view (exactly what the normal viewer shows)
    Model          the same view rebuilt from an autoencoder's output (one panel per model)

so you can eyeball how faithfully each model reproduces the shown observation channel, and — with
two models — compare them against each other directly.

Press TAB to flip the model panels between their reconstruction and a per-cell |physics - model|
error heatmap for the shown field (the physics panel is unaffected). With more than one model the
error panels share a color scale so their magnitudes are directly comparable.

In the FIRE view, press M to overlay red, on each model panel, the cells the model predicts are
burning via its reconstructed "ignited" channel (>50% certainty) but whose reconstructed fuel
temperature is still below the ignition threshold (~300 C) — the cells the normal temperature-based
fire overlay leaves unlit, flagging where the model disagrees with the temperature it reconstructed.

The autoencoder only ever sees the world-model *observation* — the normalized stack of air
temperature, one fuel_temperature_<type> channel and one fuel_<type> mass channel per fuel type,
the two signed terrain-slope components, the two near-surface wind components, and a binary ignited
mask (see Simulation.build_observation). Only views backed by those channels can be reconstructed
(air temperature, fuel temperature, wind speed, fire); other views (terrain height, biome, oxygen,
pressure, radiant heat, ...) draw from state that is not in the observation — terrain enters only as
its slope, not its height, and the biome is not observed at all — so the model panel shows a "not in
observation" placeholder for them while the physics panel still renders.

The observation is doubly normalized on the way into the model: build_observation applies the affine
OBS_NORM windows (roughly [0, 1] per channel), then training min-max scales each channel to [0, 1]
using stats carried in the checkpoint (channel_min / channel_max; legacy checkpoints carry
channel_mean / channel_std). Both transforms are inverted here so the reconstruction lands back in
physical units for rendering.

Invoked via `python src/main.py --model models/best.pt [more.pt ...]`; see main.run_comparison.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pygame
import torch

import physics_constants as pc
from strided_autoencoder import ConvAutoencoder
from firecracker_env import FirecrackerEnv, ViewMode, TARGET_FPS
from simulation import (
    OBS_CHANNELS_PRE, OBS_CHANNELS_POST, OBS_CHANNEL_IGNITED, obs_channel_names,
)
from rendering import (
    LEGEND_PANEL_WIDTH,
    build_color_surface,
    build_wind_speed_surface,
    build_fire_surface,
    build_legend_panel,
    build_biome_legend_panel,
    heat_colormap,
)
from rollout import stats_affine

# The checkpoint's model_config["arch"] selects the architecture. Kept in step with
# train_autoencoder.ARCHITECTURES (imported directly here to avoid pulling in the training deps).
ARCHITECTURES = {
    "pooled": ConvAutoencoder,
}


def resolve_arch(cfg: dict, model_state: dict) -> str:
    """Resolve the architecture key for a checkpoint, accounting for legacy labels.

    Checkpoints written before the pooled/strided split all recorded arch='strided' but were in
    fact average-pool models (the class named "strided" used a 2x2 avg pool then; the name is
    historical). A real strided-conv model has per-stage 4x4 downsample convs at
    `encoder.blocks.<i>.downsample.weight`; an avg-pool model has none. So a checkpoint labelled
    'strided' with no such weights is a legacy pooled model and loads as 'pooled'.
    """
    arch = cfg.get("arch", "pooled")
    if arch == "strided" and not any(".downsample." in k for k in model_state):
        return "pooled"
    return arch

# Weight dtypes a checkpoint may be saved in (see train_autoencoder.WEIGHT_DTYPES). Checkpoints
# from before --weight-dtype existed have no such key and load as float32.
WEIGHT_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

# View modes whose field is fully recoverable from the observation channels, so the model panel
# can render them. Every other mode draws from non-observed state and gets a placeholder.
RECONSTRUCTABLE: frozenset[ViewMode] = frozenset({
    ViewMode.TEMPERATURE,
    ViewMode.FUEL_TEMPERATURE,
    ViewMode.WIND_SPEED,
    ViewMode.FIRE,
})
# Note: TERRAIN and BIOME are no longer reconstructable — the observation carries terrain *slope*
# (slope_x / slope_y), not height, and no biome channels at all, so the model panel shows the "not in
# observation" placeholder for both while the physics panel still renders true elevation and biome.

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
    latent_dim / grid_size / in_channels) and, when the training dataset was normalized, the
    per-channel min/max the input was scaled by (channel_min / channel_max; legacy checkpoints carry
    channel_mean / channel_std).
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
    arch = resolve_arch(cfg, ckpt["model_state"])
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
        # Absent in pre-normalization checkpoints; they were trained without it.
        normalize_latent=cfg.get("normalize_latent", False),
        # Absent in pre-bottleneck checkpoints; None reproduces the un-bottlenecked model.
        bottleneck_channels=cfg.get("bottleneck_channels"),
        # Absent in pre-latent-BN checkpoints; False reproduces the model without the latent BatchNorm.
        latent_bn=bool(cfg.get("latent_bn")),
    ).to(device=device, dtype=load_dtype)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg


class _ModelPanel:
    """One loaded autoencoder plus the per-model state the viewer keeps for it: its de-normalizing
    affine stats and the cached reconstruction (in both normalized-OBS and physical units)."""

    def __init__(self, model: torch.nn.Module, config: dict, name: str, device: torch.device) -> None:
        self.model = model
        self.config = config
        self.name = name
        # Per-channel affine normalization the model was trained under (offset = min, scale = max-min
        # for a min-max dataset; mean/std for a legacy z-scored one), if the dataset was normalized.
        # Kept on the device so (de)normalization is a cheap broadcast around the forward pass.
        if "channel_min" in config or "channel_mean" in config:
            offset, scale = stats_affine(config)
            self.offset = torch.tensor(offset, dtype=torch.float32, device=device)[:, None, None]
            self.scale = torch.tensor(scale, dtype=torch.float32, device=device)[:, None, None]
        else:
            self.offset = self.scale = None
        # A model trained with --rec-loss bce / --out-activation sigmoid has a linear decoder that
        # emits logits; its reconstruction is sigmoid(logits). Absent/"none" -> raw linear output.
        self.out_sigmoid = config.get("out_activation") == "sigmoid"
        self.recon_obs: np.ndarray | None = None   # (C, H, W) normalized-OBS reconstruction
        self.recon_phys: np.ndarray | None = None  # (C, H, W) reconstruction in physical units


class ComparisonViewer:
    """Owns the pygame window and composites the physics panel beside one panel per model.

    The env is driven in rgb_array mode (it builds no window of its own); this viewer reuses the
    env's surface builders and event handling for the physics panel, runs each model to fill its
    panel, and — when TAB is toggled — draws the per-cell error between each model and the physics.
    """

    def __init__(self, env: FirecrackerEnv, models: list[tuple[torch.nn.Module, dict, str]]) -> None:
        self.env = env
        self.device = env._sim.device
        self.scale = env._pixel_scale
        self.sim_px = env.grid_size * self.scale

        self.panels = [_ModelPanel(m, cfg, name, self.device) for (m, cfg, name) in models]
        self.n_models = len(self.panels)
        self.n_panels = 1 + self.n_models   # physics + one per model

        # TAB flips the model panels between reconstruction and error; edge-detected against the
        # held-key snapshot in render() so it toggles once per press rather than every frame held.
        self._show_error = False
        self._tab_prev = False

        # M overlays red, on the FIRE model panels, the cells the model predicts are burning via the
        # reconstructed ignited channel (>50% certainty) yet whose reconstructed fuel temperature
        # stays below its ignition threshold, so the temperature-based fire overlay leaves them
        # unlit. Edge-detected like TAB.
        self._show_burning_pred = False
        self._m_prev = False

        n_fuel = env._sim.num_fuel_types
        self.n_fuel = n_fuel
        # Observation channel layout (see obs_channel_names): air_temperature, then one
        # fuel_temperature_<type> channel per fuel type, then one fuel_<type> mass channel per fuel
        # type, then slope_x, slope_y, wind_x, wind_y, and the ignited mask (no biome channels).
        self.i_air_temp = 0
        self.i_fuel_temp = slice(1, 1 + n_fuel)
        self.i_fuel = slice(1 + n_fuel, 1 + 2 * n_fuel)
        self.i_slope_x = 1 + 2 * n_fuel
        self.i_slope_y = 2 + 2 * n_fuel
        self.i_wind_x = 3 + 2 * n_fuel
        self.i_wind_y = 4 + 2 * n_fuel
        self.i_ignited = 5 + 2 * n_fuel

        # Affine OBS_NORM windows used by build_observation, per channel, for inverting back to
        # physical units. Order and values must mirror Simulation.build_observation exactly.
        windows = (
            [pc.OBS_NORM[c] for c in OBS_CHANNELS_PRE]
            + [pc.OBS_NORM["fuel_temperature"]] * n_fuel
            + [pc.OBS_NORM["fuel"]] * n_fuel
            + [pc.OBS_NORM[c] for c in OBS_CHANNELS_POST]
            + [pc.OBS_NORM[OBS_CHANNEL_IGNITED]]
        )
        self.offsets = np.array([w[0] for w in windows], dtype=np.float32)
        self.scales = np.array([w[1] for w in windows], dtype=np.float32)

        # Fuel display params reused by each model panel's FIRE surface.
        self.ignition_thresholds = env._sim.ignition_thresholds.detach().cpu().numpy().reshape(-1)
        self.fuel_burnt_threshold = env._sim.fuel_burnt_threshold
        self.fuel_type_names = list(env._sim.fuel_type_names)

        # Cached ground-truth observation, refreshed whenever the world advances (see render()).
        self._gt_obs: np.ndarray | None = None       # (C, H, W) normalized-OBS ground truth

        pygame.init()
        pygame.font.init()
        pygame.display.set_caption("Firecracker — model reconstruction")
        width = self.n_panels * self.sim_px + (self.n_panels + 1) * PAD + LEGEND_PANEL_WIDTH
        height = TITLE_H + self.sim_px + PAD
        self._screen = pygame.display.set_mode((width, height))
        self._clock = pygame.time.Clock()
        self._title_font = pygame.font.Font(None, 26)
        self._legend_font = pygame.font.Font(None, 20)
        self._legend_title_font = pygame.font.Font(None, 22)
        self._placeholder_font = pygame.font.Font(None, 30)

    # -- model -----------------------------------------------------------------

    def _compute_reconstruction(self) -> None:
        """Run every model on the current observation, caching each reconstruction in both
        normalized-OBS space (for the error panels) and physical units (for rendering)."""
        gt = self.env._observation()                               # (C, H, W) normalized-OBS
        x = torch.from_numpy(gt).to(self.device)
        self._gt_obs = gt
        for p in self.panels:
            x_in = (x - p.offset) / p.scale if p.offset is not None else x
            # Feed the model in its own weight dtype (bf16/f16 on GPU; f32 on CPU) then widen the
            # reconstruction back to float32 for the de-normalization math and rendering.
            model_dtype = next(p.model.parameters()).dtype
            with torch.no_grad():
                # The autoencoder forward returns (x_hat, z); take the reconstruction.
                x_hat = p.model(x_in.unsqueeze(0).to(model_dtype))[0]
            x_hat = x_hat.squeeze(0).float()
            if p.out_sigmoid:                                        # logits -> [0,1] reconstruction
                x_hat = torch.sigmoid(x_hat)
            recon = x_hat * p.scale + p.offset if p.offset is not None else x_hat
            recon = recon.detach().cpu().numpy().astype(np.float32)  # (C, H, W) normalized-OBS
            p.recon_obs = recon
            p.recon_phys = recon * self.scales[:, None, None] + self.offsets[:, None, None]

    # -- reconstruction / error surfaces --------------------------------------

    def _recon_surface(self, mode: ViewMode, panel: _ModelPanel) -> pygame.Surface | None:
        """Build a model panel's reconstruction surface for `mode` from the cached model output,
        reusing the physics panel's display windows so equal colors mean equal values. Returns None
        for modes not backed by the observation (the caller draws a placeholder)."""
        if mode not in RECONSTRUCTABLE:
            return None
        env, phys, scale = self.env, panel.recon_phys, self.scale
        if mode == ViewMode.TEMPERATURE:
            return build_color_surface(
                phys[self.i_air_temp], scale,
                env._air_temp_display_min, env._air_temp_display_max,
            )
        if mode == ViewMode.FUEL_TEMPERATURE:
            return build_color_surface(
                phys[self.i_fuel_temp].max(axis=0), scale,
                env._fuel_temp_display_min, env._fuel_temp_display_max,
            )
        if mode == ViewMode.WIND_SPEED:
            return build_wind_speed_surface(
                phys[self.i_wind_x], phys[self.i_wind_y], scale, env._wind_speed_display_max,
            )
        # FIRE: the observation carries per-type fuel mass and per-type fuel temperature (but not
        # oxygen). Vegetation coloring needs only the mass; the burning overlay reads the per-type
        # temperatures directly with oxygen assumed present, so the reconstructed flame footprint
        # reads against the physics panel.
        fuel = phys[self.i_fuel]
        fuel_temps = phys[self.i_fuel_temp]
        oxygen = np.ones(fuel.shape[1:], dtype=np.float32)
        surface = build_fire_surface(
            fuel_temps, fuel, oxygen, scale,
            self.ignition_thresholds, self.fuel_burnt_threshold, self.fuel_type_names,
            env._show_fire_overlay and env._sim.fire_enabled,
        )
        # With M held on, paint red the cells the model's reconstructed ignited channel calls
        # burning with >50% certainty but whose reconstructed fuel temperature stays below the
        # ignition threshold — exactly the cells build_fire_surface's temperature overlay leaves
        # unlit, so the red flags where the model disagrees with the temperature it reconstructed.
        # The ignited window is the identity, so recon_phys reads straight as certainty.
        if self._show_burning_pred:
            ign = self.ignition_thresholds[:, None, None]
            below_ignition = (fuel_temps < ign).all(axis=0)        # (H, W) no fuel type alight
            pred_burning = (panel.recon_phys[self.i_ignited] > 0.5) & below_ignition   # (H, W)
            if pred_burning.any():
                mask = np.repeat(np.repeat(pred_burning, scale, axis=0), scale, axis=1)  # (H*s, W*s)
                pixels = pygame.surfarray.pixels3d(surface)         # (W*s, H*s, 3), x-major
                pixels[mask.T] = (191, 0, 0)
                del pixels   # release the surface lock before returning
        return surface

    def _error_field(self, mode: ViewMode, panel: _ModelPanel) -> np.ndarray:
        """Per-cell reconstruction error (normalized-OBS units) for the shown field: the channel(s)
        backing the mode when it is reconstructable, else the mean absolute error over all channels."""
        gt, rc = self._gt_obs, panel.recon_obs
        if mode == ViewMode.TEMPERATURE:
            return np.abs(gt[self.i_air_temp] - rc[self.i_air_temp])
        if mode == ViewMode.FUEL_TEMPERATURE:
            return np.abs(gt[self.i_fuel_temp] - rc[self.i_fuel_temp]).mean(axis=0)
        if mode == ViewMode.TERRAIN:
            # Height is not observed; report the reconstruction error of the slope it enters as.
            return np.hypot(gt[self.i_slope_x] - rc[self.i_slope_x],
                            gt[self.i_slope_y] - rc[self.i_slope_y])
        if mode == ViewMode.WIND_SPEED:
            return np.hypot(gt[self.i_wind_x] - rc[self.i_wind_x], gt[self.i_wind_y] - rc[self.i_wind_y])
        if mode == ViewMode.FIRE:
            return np.abs(gt[self.i_fuel] - rc[self.i_fuel]).mean(axis=0)
        # BIOME (not observed) and any other non-reconstructable mode fall to the mean over channels.
        return np.abs(gt - rc).mean(axis=0)

    def _placeholder_surface(self) -> pygame.Surface:
        surf = pygame.Surface((self.sim_px, self.sim_px))
        surf.fill((24, 24, 24))
        text = self._placeholder_font.render("not in observation", True, PLACEHOLDER_FG)
        surf.blit(text, text.get_rect(center=(self.sim_px // 2, self.sim_px // 2)))
        return surf

    def _model_title(self, i: int, mode: ViewMode) -> str:
        base = "Model" if self.n_models == 1 else f"Model {i + 1}"
        return base if mode in RECONSTRUCTABLE else f"{base} (n/a)"

    def _error_title(self, i: int, error: np.ndarray) -> str:
        base = "Error" if self.n_models == 1 else f"Error {i + 1}"
        return f"{base}   mean {float(error.mean()):.3f}  max {float(error.max()):.3f}"

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
        # env._handle_events drains and pumps the event queue but doesn't touch TAB; read TAB from
        # the (now-current) held-key snapshot and edge-detect it into the reconstruction/error flip.
        pressed = pygame.key.get_pressed()
        tab_down = pressed[pygame.K_TAB]
        if tab_down and not self._tab_prev:
            self._show_error = not self._show_error
        self._tab_prev = tab_down
        m_down = pressed[pygame.K_m]
        if m_down and not self._m_prev:
            self._show_burning_pred = not self._show_burning_pred
        self._m_prev = m_down
        if fire_click is not None:
            env._spawn_fire_patch(*fire_click)
            env._surfaces_dirty = True
            dirty = True
            env._rebuild_surfaces_if_dirty()
        if dirty or self._gt_obs is None:
            self._compute_reconstruction()

        mode = env._current_mode
        self._screen.fill(BG)

        xs = [PAD + i * (self.sim_px + PAD) for i in range(self.n_panels)]
        self._blit_panel(env._surface_for_mode(), xs[0], "Physics")

        if self._show_error:
            # Share one color scale across the model panels so their error magnitudes are comparable.
            errors = [self._error_field(mode, p) for p in self.panels]
            err_max = max((float(e.max()) for e in errors), default=0.0)
            for i, error in enumerate(errors):
                surface = build_color_surface(error, self.scale, 0.0, err_max, heat_colormap)
                self._blit_panel(surface, xs[i + 1], self._error_title(i, error))
        else:
            for i, panel in enumerate(self.panels):
                surface = self._recon_surface(mode, panel)
                if surface is None:
                    surface = self._placeholder_surface()
                self._blit_panel(surface, xs[i + 1], self._model_title(i, mode))

        # Shared legend for the physics/reconstruction color scale (both use the same window).
        # The biome view is categorical, so it gets labeled swatches instead of a gradient colorbar.
        if mode == ViewMode.BIOME:
            panel = build_biome_legend_panel(
                self.sim_px, self._legend_font, self._legend_title_font
            )
        else:
            panel = build_legend_panel(
                env._legend_specs(), self.sim_px, self._legend_font, self._legend_title_font
            )
        self._screen.blit(panel, (xs[-1] + self.sim_px + PAD, TITLE_H))

        pygame.display.flip()
        self._clock.tick(TARGET_FPS)

    def _blit_panel(self, surface: pygame.Surface, x: int, title: str) -> None:
        self._screen.blit(self._title_font.render(title, True, TITLE_FG), (x, 4))
        self._screen.blit(surface, (x, TITLE_H))

    def close(self) -> None:
        pygame.quit()
