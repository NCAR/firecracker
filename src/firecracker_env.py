"""
firecracker_env.py

Gymnasium environment wrapping the Firecracker heat-diffusion simulation.
Observation: the world-model observation — a normalized (C, grid_size, grid_size) float32
             stack of the observation channels (see Simulation.build_observation /
             obs_channel_names).
Action:      Discrete(action_grid_size) — an action-cell is selected over two steps (first a
             row, then a column). A completed selection goes through only when the cell or one
             of its 8 neighbour action-cells contains fire; over a fire-free neighbourhood it is
             a no-op (nothing selected).
"""

import math
import time
from enum import Enum

import numpy as np
import torch
import pygame
import gymnasium
from gymnasium import spaces

from simulation import Simulation, SimState, obs_channel_names
from config import boundary_pad
from gen_maps import MapGenerator
from map_loader import load_map, resolve_map, validate_against_config
# Presentation layer: the stateless field/colorbar drawing primitives and their display
# constants. This module orchestrates them against env state; it never draws a pixel itself.
from rendering import (
    DISPLAY_TEMP_FLOOR_K,
    DISPLAY_MIN_TEMP_SPAN_K,
    FUEL_TEMP_DISPLAY_MIN_K,
    FUEL_TEMP_DISPLAY_MAX_K,
    FIRE_COLOR_TEMP_SPAN_K,
    KELVIN_TO_CELSIUS,
    LEGEND_PANEL_WIDTH,
    LEGEND_FONT_SIZE,
    LEGEND_TITLE_FONT_SIZE,
    DISPLAY_OXYGEN_FLOOR,
    DISPLAY_OXYGEN_CEILING,
    ColorbarSpec,
    heat_colormap,
    oxygen_colormap,
    wind_speed_colormap,
    pressure_colormap,
    radiant_heat_colormap,
    terrain_colormap,
    sunlight_colormap,
    column_height_colormap,
    wind_temp_colormap,
    fire_overlay_colormap,
    wind_temp_window,
    wind_arrow_color,
    draw_wind_arrow,
    build_color_surface,
    build_wind_surface,
    build_wind_speed_surface,
    build_oxygen_surface,
    build_pressure_surface,
    build_radiant_heat_surface,
    build_terrain_surface,
    build_sunlight_surface,
    build_column_height_surface,
    build_fire_surface,
    build_biome_surface,
    build_legend_panel,
    build_biome_legend_panel,
)


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    """Move a (possibly GPU-resident) tensor to a host numpy array for rendering/info."""
    return t.detach().cpu().numpy()

# ---------------------------------------------------------------------------
# Display constants
#
# Window/frame-rate and action-overlay constants only. The field/colorbar drawing constants
# (color channels, wind-arrow geometry, temperature windows, legend layout) live with the
# primitives in rendering.py; the few this module needs are imported from there above.
# ---------------------------------------------------------------------------

DEFAULT_GRID_SIZE: int = 256
DEFAULT_WINDOW_SIZE: int = 512
WINDOW_TITLE: str = "Firecracker"
TARGET_FPS: int = 60

FIRE_SPAWN_RADIUS: int = 5

# Alpha of the white rectangle blended over the action-cell under the mouse (0-255):
# subtle enough that the underlying view still reads clearly.
ACTION_HIGHLIGHT_ALPHA: int = 80

# A valid action (one that goes through) flashes the selected square white: this is the peak
# alpha (0-255) and how long, in seconds, the flash fades back to nothing.
ACTION_FLASH_ALPHA: int = 230
ACTION_FLASH_SECONDS: float = 0.35

# ---------------------------------------------------------------------------
# View modes
# ---------------------------------------------------------------------------

class ViewMode(Enum):
    WIND_SPEED = 0
    TEMPERATURE = 1
    WIND = 2
    FIRE = 3
    OXYGEN = 4
    PRESSURE = 5
    FUEL_TEMPERATURE = 6
    RADIANT_HEAT = 7
    TERRAIN = 8
    COLUMN_HEIGHT = 9
    SUNLIGHT = 10        # off the number row -> bound to a letter key below (the digits are taken)
    BIOME = 11           # off the number row -> bound to a letter key below (the digits are taken)


# Modes 0-9 are picked up automatically from the number row; modes past it (the digits ran out)
# get an explicit letter binding.
MODE_KEYS: dict[int, ViewMode] = {
    getattr(pygame, f"K_{mode.value}"): mode
    for mode in ViewMode
    if 0 <= mode.value <= 9
}
MODE_KEYS[pygame.K_s] = ViewMode.SUNLIGHT
MODE_KEYS[pygame.K_b] = ViewMode.BIOME

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class FirecrackerEnv(gymnasium.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": TARGET_FPS}

    def __init__(
        self,
        config: dict | None = None,
        render_mode: str | None = None,
        map_name: str | None = None,
        maps_dir: str | None = None,
    ):
        super().__init__()
        assert render_mode is None or render_mode in self.metadata["render_modes"], (
            f"render_mode must be one of {self.metadata['render_modes']} or None"
        )

        self._sim = Simulation(config)

        # Reuses the map generator's biome definition (thresholds + classify_biomes) to label each
        # cell woodland / grassland / shrubland for the BIOME view, so the view and the fuel placement
        # share one source of truth. Only classify_biomes is used (no device/noise), and it is
        # recomputed from the loaded map's terrain + rest surface temperature at reset.
        self._biome_gen = MapGenerator(config)

        env_cfg  = (config or {}).get("environment", {})
        fire_cfg = (config or {}).get("fire", {})
        maps_cfg = (config or {}).get("maps", {})

        # Map selection: a name (specific map) or None (random pick at reset).
        # Explicit constructor args win over config.
        self._maps_dir: str = maps_dir if maps_dir is not None else str(maps_cfg.get("dir", "maps"))
        self._map_name: str | None = (
            map_name if map_name is not None else (str(maps_cfg.get("name", "")) or None)
        )

        # grid_size is the OBSERVED grid: what the observation, rendering, UI and action grid use.
        # The physics runs on a grid padded by _pad cells on every side (_sim_size = grid_size +
        # 2*_pad), so the open-boundary sponge sits in the ring and the observed interior is sponge-
        # free. The observation and the rendered surfaces are the inner [_pad:_pad+grid_size] window;
        # screen->cell coordinates add _pad to index the physics grid. _pad = 0 recovers the
        # un-padded grid (observed == physics, sponge on the observed edges).
        self.grid_size   = int(env_cfg.get("grid_size",   DEFAULT_GRID_SIZE))
        self._pad        = boundary_pad(config)
        self._sim_size   = self.grid_size + 2 * self._pad
        self.window_size = int(env_cfg.get("window_size", DEFAULT_WINDOW_SIZE))
        self.max_steps   = int(env_cfg.get("max_steps",   1000))
        self._fire_spawn_radius = int(fire_cfg.get("spawn_radius", FIRE_SPAWN_RADIUS))
        self._spawn_fire = bool(fire_cfg.get("spawn_fire", False))
        self.render_mode = render_mode
        self._pixel_scale = self.window_size // self.grid_size

        # Coarse action grid: the sim grid is divided into action_grid_size x action_grid_size
        # action-cells, each a square block of (grid_size / action_grid_size) sim cells. Used for
        # the hover-highlight overlay (and a future grid action space), so it must tile the grid
        # evenly. Precompute the action-cell side in sim cells and in display pixels.
        action_cfg = (config or {}).get("action", {})
        specified_action_grid = action_cfg.get("grid_size")
        if specified_action_grid is not None:
            # An explicit value must tile the grid evenly — surface a clear error if it doesn't.
            self.action_grid_size = int(specified_action_grid)
            if self.action_grid_size < 1 or self.grid_size % self.action_grid_size != 0:
                raise ValueError(
                    f"action.grid_size ({self.action_grid_size}) must be a positive divisor of "
                    f"environment.grid_size ({self.grid_size})."
                )
        else:
            # Unspecified: use the default unless it doesn't divide this grid (e.g. small test
            # grids), in which case fall back to a per-cell action grid so any grid_size works.
            default = 32
            self.action_grid_size = default if self.grid_size % default == 0 else self.grid_size
        self._action_cell_cells = self.grid_size // self.action_grid_size
        self._action_cell_px = self._action_cell_cells * self._pixel_scale
        # Action-cell (arow, acol) currently under the mouse, or None when the cursor is
        # outside the window. Drawn as a brightening overlay each frame (render).
        self._hovered_action_cell: tuple[int, int] | None = None
        # Two-step action state. _pending_action_row holds the row chosen on a row step while the
        # env waits for the column step (None means the next action is read as a row). On the
        # completing column step _selected_action_cell is the chosen (arow, acol) if the action goes
        # through (the cell or a neighbour is on fire), else None (the fire-free no-op). It is also
        # None on row steps and before the first selection. The world effect of a cell is TBD.
        self._pending_action_row: int | None = None
        self._selected_action_cell: tuple[int, int] | None = None

        # reset/step return the same world-model observation the rollout collector consumes:
        # the normalized multi-channel stack produced by Simulation.build_observation. The
        # channel count depends on the configured fuel types (one fuel_<name> channel each), so
        # it is derived from the sim's fuel types via obs_channel_names. Channels are mapped to
        # roughly [0, 1] by their affine windows, but the high side is intentionally not clamped
        # (flames/strong gusts ride above 1) and wind is signed, so the Box is left unbounded
        # rather than asserting a false finite range.
        self.obs_channels = obs_channel_names(self._sim.fuel_type_names)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(len(self.obs_channels), self.grid_size, self.grid_size), dtype=np.float32,
        )
        # An action-cell is selected over two consecutive steps (first a row, then a column), each a
        # Discrete index in [0, action_grid_size) -- no dedicated no-op action, so the space is
        # action_grid_size (= 32). A completed selection only "goes through" when the chosen cell or
        # one of its 8 neighbouring action-cells contains fire (_action_cell_near_fire); over a
        # fire-free neighbourhood it is implicitly a no-op (nothing selected). See _select_action.
        self.action_space = spaces.Discrete(self.action_grid_size)

        # Simulation state (populated by reset)
        self._air_temperatures: np.ndarray | None = None
        self._fuel_temperatures: np.ndarray | None = None
        self._x_wind_vel: np.ndarray | None = None
        self._y_wind_vel: np.ndarray | None = None
        self._fuel: np.ndarray | None = None
        self._initial_canopy_fuel: torch.Tensor | None = None  # per-cell original canopy load (for bole occlusion)
        self._oxygen: np.ndarray | None = None
        self._mass: np.ndarray | None = None   # boundary-layer areal mass [kg/m^2]
        self._mass_eq: torch.Tensor | None = None   # rest-state mass (open-boundary sponge target)
        self._terrain: np.ndarray | None = None
        self._sunlight: np.ndarray | None = None   # static average solar exposure [0,1] (SUNLIGHT view)
        self._biome_labels: np.ndarray | None = None   # static per-cell biome label 0/1/2 (BIOME view)
        self._u_amb_x: torch.Tensor | None = None   # synoptic ambient wind [m/s]
        self._u_amb_y: torch.Tensor | None = None
        # Phase 6 (Option 2): static terrain-channeling gain and the near-surface wind the fire reads.
        self._channel_gain: torch.Tensor | None = None
        self._x_wind_fire: torch.Tensor | None = None
        self._y_wind_fire: torch.Tensor | None = None
        # Rest-state targets (initial profiles): radiative-equilibrium temperature and oxygen.
        self._temp_eq: np.ndarray | None = None
        self._oxygen_eq: np.ndarray | None = None
        self._ground_temperature: torch.Tensor | None = None   # surface skin [K]
        self._terrain_stats: list[float] = [0.0, 0.0, 0.0]   # (mean, min, max), set at reset
        self._last_radiant_flux: np.ndarray | None = None
        self._current_map: str | None = None   # filename of the loaded map
        # Per-mode color-scale windows captured while building the surfaces, so each colorbar
        # labels the exact range its (cached) field was normalised over. Each is the live max
        # over all cells. Set in _rebuild_surfaces_if_dirty; the defaults keep a legend valid
        # before the first build.
        # The air-temperature window auto-ranges to the live field (floor = coldest cell, ceiling =
        # hottest, min-span clamped); its default holds until the first surface build. The fuel-
        # temperature window is fixed (see build), so it uses the fixed bounds from the start.
        self._air_temp_display_min: float = DISPLAY_TEMP_FLOOR_K
        self._air_temp_display_max: float = DISPLAY_TEMP_FLOOR_K + DISPLAY_MIN_TEMP_SPAN_K
        self._fuel_temp_display_min: float = FUEL_TEMP_DISPLAY_MIN_K
        self._fuel_temp_display_max: float = FUEL_TEMP_DISPLAY_MAX_K
        self._radiant_flux_display_max: float = 0.0
        self._pressure_display_max: float = 0.0
        self._terrain_display_max: float = 0.0
        self._column_height_display_max: float = 0.0
        self._wind_speed_display_max: float = 0.0
        # The wind/temperature fields the last surfaces were built from, kept on the host so the
        # cursor wind-vector probe can read the hovered cell without another device transfer.
        self._render_wind_x: np.ndarray | None = None
        self._render_wind_y: np.ndarray | None = None
        self._render_air_temp: np.ndarray | None = None
        self._wind_temp_window: tuple[float, float] = (
            DISPLAY_TEMP_FLOOR_K, DISPLAY_TEMP_FLOOR_K + DISPLAY_MIN_TEMP_SPAN_K
        )
        self._step_count: int = 0

        # Rendering state
        self._screen: pygame.Surface | None = None
        self._clock: pygame.time.Clock | None = None
        # Legend fonts: created only in human render mode (None otherwise -> no legend drawn).
        self._legend_font: pygame.font.Font | None = None
        self._legend_title_font: pygame.font.Font | None = None
        self._color_surface: pygame.Surface | None = None
        self._wind_surface: pygame.Surface | None = None
        self._wind_speed_surface: pygame.Surface | None = None
        self._fire_surface: pygame.Surface | None = None
        self._oxygen_surface: pygame.Surface | None = None
        self._pressure_surface: pygame.Surface | None = None
        self._fuel_temperature_surface: pygame.Surface | None = None
        self._radiant_flux_surface: pygame.Surface | None = None
        self._terrain_surface: pygame.Surface | None = None
        self._sunlight_surface: pygame.Surface | None = None
        self._biome_surface: pygame.Surface | None = None
        self._column_height_surface: pygame.Surface | None = None
        # Reusable translucent white fills for the action highlight, cached by (w, h) pixel size.
        self._alpha_rect_cache: dict[tuple[int, int], pygame.Surface] = {}
        # White-flash feedback for a valid action: the flashed action-cell and the monotonic time
        # the flash fades out at, plus a cached solid-white square reused (via set_alpha) each frame.
        self._action_flash_cell: tuple[int, int] | None = None
        self._action_flash_until: float = 0.0
        self._flash_square: pygame.Surface | None = None
        self._surfaces_dirty: bool = True
        self._current_mode: ViewMode = ViewMode.TEMPERATURE
        self._show_wind_overlay: bool = False
        self._show_fire_overlay: bool = True
        self._running: bool = True
        self._paused: bool = False
        self._reset_requested: bool = False
        self._step_once: bool = False

        if render_mode == "human":
            pygame.init()
            pygame.font.init()
            pygame.display.set_caption(WINDOW_TITLE)
            sim_px = self.grid_size * self._pixel_scale
            # Reserve a fixed legend panel on the right, so the window is permanently wider than
            # the simulation and the colorbar never overlaps the field.
            self._screen = pygame.display.set_mode((sim_px + LEGEND_PANEL_WIDTH, sim_px))
            self._clock = pygame.time.Clock()
            self._legend_font = pygame.font.Font(None, LEGEND_FONT_SIZE)
            self._legend_title_font = pygame.font.Font(None, LEGEND_TITLE_FONT_SIZE)

    def _crop(self, field):
        """The observed inner window of a padded (..., H, W) field: drops the _pad-cell sponge
        ring so the observation/render see only the interior. Works on torch tensors and numpy
        arrays alike; a no-op when _pad == 0. Use _pad-offset indices to map the other way."""
        if self._pad == 0:
            return field
        p, g = self._pad, self.grid_size
        return field[..., p:p + g, p:p + g]

    def _observation(self) -> np.ndarray:
        """The world-model observation for the current state: a normalized (C, H, W) float32
        stack of the observation channels over the observed interior (the padded sponge ring is
        cropped off).
        Identical to what BatchedRollout.observe collects, including the near-surface fire wind
        (prognostic wind x channeling gain) used by the spread physics."""
        obs = Simulation.build_observation(
            self._air_temperatures, self._fuel_temperatures, self._fuel, self._terrain,
            self._x_wind_fire if self._x_wind_fire is not None else self._x_wind_vel,
            self._y_wind_fire if self._y_wind_fire is not None else self._y_wind_vel,
            self._sim.ignition_thresholds, self._sim.cell_size_m,
        )
        return _to_numpy(self._crop(obs)).astype(np.float32)

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        # Load a fully-baked initial state from disk. `options={"map": name}` overrides
        # the configured selection for targeted (e.g. test) scenarios; otherwise a
        # named map is loaded, or a random one is picked when no name is set.
        map_name = (options or {}).get("map", self._map_name)
        map_path = resolve_map(self._maps_dir, map_name, self.np_random)
        m = load_map(map_path)
        # Maps are stored at the padded physics size (observed grid + 2*_pad sponge ring).
        validate_against_config(m, self._sim_size, self._sim.fuel_type_names)
        self._current_map = map_path.name

        # Maps load from disk as numpy (map_loader is source-agnostic); move every field
        # onto the simulation device so the per-step physics stays GPU-resident.
        self._terrain           = self._to_tensor(m.terrain)
        self._sunlight          = self._to_tensor(m.sunlight)   # static solar exposure (SUNLIGHT view)
        self._air_temperatures  = self._to_tensor(m.air_temperatures)
        self._temp_eq           = self._to_tensor(m.temp_eq)
        self._oxygen            = self._to_tensor(m.oxygen)
        self._oxygen_eq         = self._to_tensor(m.oxygen_eq)
        self._mass              = self._to_tensor(m.mass)      # developed (spun-up) initial mass
        self._mass_eq           = self._to_tensor(m.mass_eq)   # level-lid rest state -> sponge target
        self._fuel              = self._to_tensor(m.fuel)
        # Snapshot the original per-cell canopy load now, before any burning thins it; the fire
        # view uses it to reveal the bole once a cell's canopy drops below a fraction of this.
        canopy_axes = [i for i, n in enumerate(self._sim.fuel_type_names) if n == "tree_canopy"]
        self._initial_canopy_fuel = (
            self._fuel[canopy_axes].sum(dim=0).clone() if canopy_axes else None
        )
        self._fuel_temperatures = self._to_tensor(m.fuel_temperatures)
        # Surface skin starts at the rest temperature profile (radiative-equilibrium target).
        self._ground_temperature = self._temp_eq.clone()

        # Per-map synoptic ambient wind: the momentum drag relaxes toward it. The prognostic wind
        # is initialised to the map's developed (spun-up) orographic field, so the world starts in
        # its terrain-shaped state rather than relaxing into it over hundreds of ticks.
        self._u_amb_x = self._to_tensor(np.float32(m.ambient_wind_x))
        self._u_amb_y = self._to_tensor(np.float32(m.ambient_wind_y))
        self._x_wind_vel = self._to_tensor(m.x_wind_vel)
        self._y_wind_vel = self._to_tensor(m.y_wind_vel)

        # Static terrain-channeling gain (Phase 6, Option 2): precomputed once from the terrain and
        # the per-map synoptic wind direction. The fire/oxygen read -- and the WIND view shows -- the
        # prognostic wind sped up through gaps by this gain (the near-surface wind the model actually
        # estimates). Seed it now from the developed wind so the Venturi is visible on the first frame.
        self._channel_gain = self._sim.compute_channel_gain(self._terrain, self._u_amb_x, self._u_amb_y)
        self._x_wind_fire = self._x_wind_vel * self._channel_gain
        self._y_wind_fire = self._y_wind_vel * self._channel_gain

        # Terrain is static for the episode, so reduce its stats once here (one host transfer)
        # instead of re-syncing them every step in _build_info. Stats are over the observed
        # interior (what's rendered), not the padded sponge ring.
        terr_obs = self._crop(self._terrain)
        self._terrain_stats = torch.stack(
            [terr_obs.mean(), terr_obs.min(), terr_obs.max()]
        ).tolist()

        # Static biome label (0 woodland / 1 grassland / 2 shrubland) for the BIOME view, recomputed
        # over the observed interior from the same terrain + rest surface temperature (temp_eq) the
        # generator classified at bake time (temp_eq is the pre-spin-up surface temperature, so this
        # reproduces the biomes that placed the fuel). classify_biomes returns three exclusive masks.
        woodland, grassland, shrubland = self._biome_gen.classify_biomes(
            _to_numpy(terr_obs), _to_numpy(self._crop(self._temp_eq))
        )
        self._biome_labels = np.where(woodland, 0, np.where(grassland, 1, 2)).astype(np.int64)

        # Optional ignition overlay (random spawn). Maps describe the world at rest; fire is a
        # runtime concern applied on top. The patch is placed in the observed interior (coords are
        # offset into the padded grid by _spawn_fire_patch), so a spawned fire is always on-screen.
        if self._spawn_fire:
            r = int(self.np_random.integers(0, self.grid_size))
            c = int(self.np_random.integers(0, self.grid_size))
            self._spawn_fire_patch(r, c)

        # Radiant-flux buffer lives on the physics grid (it travels in SimState through the engine).
        self._last_radiant_flux = torch.zeros(
            (self._sim_size, self._sim_size), dtype=self._sim.dtype, device=self._sim.device
        )
        self._step_count = 0
        self._pending_action_row = None
        self._selected_action_cell = None
        self._action_flash_cell = None
        self._running = True
        self._paused = False
        self._reset_requested = False
        self._step_once = False
        self._surfaces_dirty = True

        return self._observation(), self._build_info()

    def _select_action(self, action: int) -> None:
        """Advance the two-step row-then-column selection with this tick's Discrete action.

        Each action is a row/column index in [0, action_grid_size). With no row pending the action
        is the row and is stashed; with a row pending it is the column that completes the cell. A
        completed selection only goes through when the cell or one of its 8 neighbouring action-cells
        is on fire (_action_cell_near_fire) -- then _selected_action_cell is set and, when rendering,
        the cell is flashed white; over a fire-free neighbourhood it is a no-op (nothing selected).
        Raises if the index is outside the action space.
        """
        action = int(action)
        if not 0 <= action < self.action_grid_size:
            raise ValueError(
                f"action {action} is out of range for action_space "
                f"Discrete({self.action_grid_size})."
            )
        self._selected_action_cell = None
        if self._pending_action_row is None:
            self._pending_action_row = action          # row step: stash the row, await the column
        else:
            arow, acol = self._pending_action_row, action
            self._pending_action_row = None             # column step completes the cell
            if self._action_cell_near_fire(arow, acol):
                # Fire in the cell or a neighbour -> the action goes through.
                self._selected_action_cell = (arow, acol)
                if self.render_mode is not None:
                    self._action_flash_cell = (arow, acol)
                    self._action_flash_until = time.monotonic() + ACTION_FLASH_SECONDS
            # else: fire-free neighbourhood -> implicit no-op (nothing selected, no flash).

    def _burning_per_type(self) -> torch.Tensor:
        """(N, H, W) boolean mask of cells currently burning: hot enough to pyrolyse, with fuel and
        oxygen present. All-False when fire is disabled. Shared by _build_info and the action gate."""
        if not self._sim.fire_enabled:
            return torch.zeros_like(self._fuel, dtype=torch.bool)
        ign = self._sim.ignition_thresholds   # (N, 1, 1)
        return (
            (self._fuel_temperatures >= ign)
            & (self._fuel > self._sim.fuel_burnt_threshold)
            & (self._oxygen[None] > 0.0)
        )

    def _action_cell_near_fire(self, arow: int, acol: int) -> bool:
        """True if the action-cell (arow, acol) or any of its 8 neighbours contains a burning sim
        cell. An action goes through only when this holds; a fire-free neighbourhood is a no-op."""
        G, c = self.action_grid_size, self._action_cell_cells
        # The action grid tiles the OBSERVED interior, so crop off the sponge ring before binning.
        burning = self._crop(self._burning_per_type().any(dim=0))   # (obs, obs) any-type burning
        box = burning.reshape(G, c, G, c).any(dim=(1, 3))      # (G, G) burning per action-cell
        r0, r1 = max(0, arow - 1), min(G, arow + 2)            # 3x3 Moore neighbourhood, clamped
        c0, c1 = max(0, acol - 1), min(G, acol + 2)
        return bool(box[r0:r1, c0:c1].any())

    def step(
        self, action: int | None
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        # Apply the action to the two-step (row, then column) selection, then advance the physics.
        # action is a Discrete index (row / column / no-op). The interactive runner passes None on
        # ticks where the user issued nothing, so the world keeps advancing while an in-progress
        # selection is *held* untouched -- a no-op would instead abort a pending row. There, the
        # selection is driven live by clicks / the 'n' key (see _handle_events); the agent drives it
        # through this argument. The selection's world effect is not wired in yet, so the chosen cell
        # is just recorded for reward/intervention logic to read later.
        if action is not None:
            self._select_action(action)

        # Advance every field one tick via the shared physics step (the same routine the
        # batched rollout collector uses). The convective transport carries mass and energy
        # (E = m*T) together down the transport potential, fire/radiant/relaxation forcings
        # follow, and the diagnostic wind is refreshed for the next tick's oxygen advection.
        self._store_field_state(self._sim.step_fields(self._field_state()))
        self._step_count += 1
        self._surfaces_dirty = True

        reward = 0.0
        terminated = False
        truncated = self._step_count >= self.max_steps
        return self._observation(), reward, terminated, truncated, self._build_info()

    def render(self) -> np.ndarray | None:
        if self.render_mode == "human":
            self._rebuild_surfaces_if_dirty()
            self._running, self._current_mode, fire_click = self._handle_events()
            if fire_click is not None:
                self._spawn_fire_patch(*fire_click)
                self._surfaces_dirty = True
                self._rebuild_surfaces_if_dirty()
            surface = self._surface_for_mode()
            self._screen.fill((0, 0, 0))
            self._screen.blit(surface, (0, 0))
            # The wind surface is transparent (black colorkey) except for the arrows, so it
            # overlays any view. Skip WIND, where the arrows are already the primary view.
            if self._show_wind_overlay and self._current_mode is not ViewMode.WIND:
                self._screen.blit(self._wind_surface, (0, 0))
            if self._current_mode in (ViewMode.WIND, ViewMode.WIND_SPEED):
                self._blit_cursor_wind_vector()
            self._blit_action_highlight()
            self._blit_action_flash()
            self._blit_legend()
            pygame.display.flip()
            self._clock.tick(TARGET_FPS)
            return None

        if self.render_mode == "rgb_array":
            if not pygame.get_init():
                pygame.init()
            self._rebuild_surfaces_if_dirty()
            surface = self._surface_for_mode()
            # surfarray returns (W, H, 3); transpose to (H, W, 3)
            return pygame.surfarray.array3d(surface).transpose(1, 0, 2)

        return None

    def close(self) -> None:
        if self._screen is not None:
            pygame.quit()
            self._screen = None
            self._clock = None

    # ---------------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------------

    def _alpha_rect(self, w: int, h: int) -> pygame.Surface:
        """A translucent white w x h fill (cached by size) for the action highlight."""
        surf = self._alpha_rect_cache.get((w, h))
        if surf is None:
            surf = pygame.Surface((w, h), pygame.SRCALPHA)
            surf.fill((255, 255, 255, ACTION_HIGHLIGHT_ALPHA))
            self._alpha_rect_cache[(w, h)] = surf
        return surf

    def _blit_action_highlight(self) -> None:
        """Show what the next left-click will commit in the two-step action selection.

        The two halves of a cell are chosen on separate steps, so the highlight is phase-aware: in
        the row phase a horizontal band marks the hovered row (a click commits that row), and in the
        column phase a vertical band marks the hovered column while the already-committed row stays
        lit (their brighter intersection previews the resulting cell). Translucent white over the
        active view, so the field still reads through. No-op when the cursor is off the window.
        """
        # get_focused() is true only while the cursor is over the focused window, so this clears
        # the highlight the moment the mouse leaves — more reliable than a WINDOWLEAVE event, which
        # can be missed (leaving a stale cell lit, or jumping to a stray last-motion position).
        if self._hovered_action_cell is None or not pygame.mouse.get_focused():
            return
        arow, acol = self._hovered_action_cell
        side = self._action_cell_px
        win = self.grid_size * self._pixel_scale
        if self._pending_action_row is None:
            # Row phase: a click commits the hovered row, so band the whole row.
            self._screen.blit(self._alpha_rect(win, side), (0, arow * side))
        else:
            # Column phase: the row is locked; band it plus the hovered column (overlap = the cell).
            self._screen.blit(self._alpha_rect(win, side), (0, self._pending_action_row * side))
            self._screen.blit(self._alpha_rect(side, win), (acol * side, 0))

    def _blit_action_flash(self) -> None:
        """Flash the selected square white for a moment after a valid action goes through.

        Brightest on the tick the action lands, fading to nothing over ACTION_FLASH_SECONDS, so the
        user sees which selections were valid (near fire) versus a no-op (fire-free neighbourhood).
        """
        if self._action_flash_cell is None:
            return
        remaining = self._action_flash_until - time.monotonic()
        if remaining <= 0.0:
            self._action_flash_cell = None
            return
        side = self._action_cell_px
        if self._flash_square is None:
            self._flash_square = pygame.Surface((side, side))
            self._flash_square.fill((255, 255, 255))
        self._flash_square.set_alpha(int(ACTION_FLASH_ALPHA * remaining / ACTION_FLASH_SECONDS))
        arow, acol = self._action_flash_cell
        self._screen.blit(self._flash_square, (acol * side, arow * side))

    def _legend_specs(self) -> list[ColorbarSpec]:
        """Colorbar specs for the active view mode, drawn left to right. Returning several renders
        adjacent bars (e.g. a future multi-scale view). Built fresh each frame so dynamic scales
        (the data-driven ceilings captured in _rebuild_surfaces_if_dirty) and mode switches are
        always reflected, even while paused. Each spec's window matches what the surface normalised
        over, so the bar's tick labels read the field's true range."""
        celsius = lambda k: f"{k - KELVIN_TO_CELSIUS:.0f}°C"
        meters  = lambda v: f"{v:.0f} m"
        mode = self._current_mode
        if mode == ViewMode.WIND_SPEED:
            return [ColorbarSpec("Wind spd", wind_speed_colormap,
                                 0.0, self._wind_speed_display_max, lambda v: f"{v:.1f} m/s")]
        if mode == ViewMode.TEMPERATURE:
            return [ColorbarSpec("Air temp", heat_colormap,
                                 self._air_temp_display_min, self._air_temp_display_max, celsius)]
        if mode == ViewMode.WIND:
            # The arrows are tinted by air temperature; the bar shows that same window.
            lo, hi = self._wind_temp_window
            return [ColorbarSpec("Wind temp", wind_temp_colormap, lo, hi, celsius)]
        if mode == ViewMode.FIRE:
            # Burning cells are tinted by fuel temperature over a fixed span above ignition.
            min_ign = float(self._sim.ignition_thresholds.min())
            return [ColorbarSpec("Fire temp", fire_overlay_colormap,
                                 min_ign, min_ign + FIRE_COLOR_TEMP_SPAN_K, celsius)]
        if mode == ViewMode.OXYGEN:
            return [ColorbarSpec("Oxygen", oxygen_colormap,
                                 DISPLAY_OXYGEN_FLOOR, DISPLAY_OXYGEN_CEILING, lambda v: f"{v:.2f}")]
        if mode == ViewMode.PRESSURE:
            return [ColorbarSpec("Mass", pressure_colormap,
                                 0.0, self._pressure_display_max, lambda v: f"{v:.2g}")]
        if mode == ViewMode.RADIANT_HEAT:
            return [ColorbarSpec("Radiant", radiant_heat_colormap,
                                 0.0, self._radiant_flux_display_max, lambda v: f"{v:.0f}")]
        if mode == ViewMode.FUEL_TEMPERATURE:
            return [ColorbarSpec("Fuel temp", heat_colormap,
                                 self._fuel_temp_display_min, self._fuel_temp_display_max, celsius)]
        if mode == ViewMode.TERRAIN:
            return [ColorbarSpec("Terrain", terrain_colormap,
                                 0.0, self._terrain_display_max, meters)]
        if mode == ViewMode.SUNLIGHT:
            return [ColorbarSpec("Sunlight", sunlight_colormap,
                                 0.0, 1.0, lambda v: f"{v * 100:.0f}%")]
        if mode == ViewMode.COLUMN_HEIGHT:
            return [ColorbarSpec("Column", column_height_colormap,
                                 0.0, self._column_height_display_max, meters)]
        return []

    def _blit_legend(self) -> None:
        """Draw the right-side colorbar panel for the active mode. The panel space is always
        reserved (the window is wider than the sim); modes with no spec just leave it black."""
        if self._legend_font is None:
            return
        sim_px = self.grid_size * self._pixel_scale
        # The biome view is categorical, so it gets labeled swatches instead of a gradient colorbar.
        if self._current_mode == ViewMode.BIOME:
            panel = build_biome_legend_panel(sim_px, self._legend_font, self._legend_title_font)
        else:
            panel = build_legend_panel(
                self._legend_specs(), sim_px, self._legend_font, self._legend_title_font
            )
        self._screen.blit(panel, (sim_px, 0))

    def _blit_cursor_wind_vector(self) -> None:
        """In the wind views, draw one wind arrow — styled exactly like the field arrows — at the
        sim cell under the mouse, so a specific point can be probed/compared. No-op when the cursor
        is off the map or the wind data isn't ready yet."""
        if self._render_wind_x is None or self._wind_speed_display_max <= 0.0:
            return
        sim_w = self.grid_size * self._pixel_scale
        px, py = pygame.mouse.get_pos()
        if not (0 <= px < sim_w and 0 <= py < sim_w):
            return
        col = min(px // self._pixel_scale, self.grid_size - 1)
        row = min(py // self._pixel_scale, self.grid_size - 1)
        vx = float(self._render_wind_x[row, col])
        vy = float(self._render_wind_y[row, col])
        temp_min, temp_max = self._wind_temp_window
        color = wind_arrow_color(float(self._render_air_temp[row, col]), temp_min, temp_max)
        # Base the arrow at the cursor (the measured cell) so it points outward from the pointer tip.
        draw_wind_arrow(
            self._screen, px, py, vx, vy,
            math.hypot(vx, vy), self._wind_speed_display_max, color, from_base=True,
        )

    def _surface_for_mode(self) -> pygame.Surface:
        if self._current_mode == ViewMode.WIND_SPEED:
            return self._wind_speed_surface
        if self._current_mode == ViewMode.TEMPERATURE:
            return self._color_surface
        if self._current_mode == ViewMode.WIND:
            return self._wind_surface
        if self._current_mode == ViewMode.FIRE:
            return self._fire_surface
        if self._current_mode == ViewMode.OXYGEN:
            return self._oxygen_surface
        if self._current_mode == ViewMode.PRESSURE:
            return self._pressure_surface
        if self._current_mode == ViewMode.FUEL_TEMPERATURE:
            return self._fuel_temperature_surface
        if self._current_mode == ViewMode.TERRAIN:
            return self._terrain_surface
        if self._current_mode == ViewMode.SUNLIGHT:
            return self._sunlight_surface
        if self._current_mode == ViewMode.BIOME:
            return self._biome_surface
        if self._current_mode == ViewMode.COLUMN_HEIGHT:
            return self._column_height_surface
        return self._radiant_flux_surface

    def _rebuild_surfaces_if_dirty(self) -> None:
        if not self._surfaces_dirty:
            return
        # The surface builders are CPU/pygame numpy code, so pull the state to the host once here --
        # the numpy/tensor boundary lives at the render edge. Every spatial field is cropped to the
        # observed interior first (drop the sponge ring), so what's drawn is exactly the observation.
        temp = _to_numpy(self._crop(self._air_temperatures))
        # The WIND view shows the near-surface wind the fire/oxygen actually read: the prognostic
        # wind sped up through gaps by the terrain-channeling gain (so the Venturi is visible).
        # Falls back to the prognostic wind when channeling produced no fire-wind field.
        wx = _to_numpy(self._crop(self._x_wind_fire if self._x_wind_fire is not None else self._x_wind_vel))
        wy = _to_numpy(self._crop(self._y_wind_fire if self._y_wind_fire is not None else self._y_wind_vel))
        pressure_field = _to_numpy(self._crop(self._mass))
        fuel_temps = _to_numpy(self._crop(self._fuel_temperatures))
        fuel = _to_numpy(self._crop(self._fuel))
        oxygen = _to_numpy(self._crop(self._oxygen))
        column_height = _to_numpy(self._crop(
            self._sim.column_height(self._mass, self._terrain, self._air_temperatures)
        ))
        ignition_thresholds = _to_numpy(self._sim.ignition_thresholds).reshape(-1)
        terrain = _to_numpy(self._crop(self._terrain))
        radiant_flux = _to_numpy(self._crop(self._last_radiant_flux))
        wind_ref = float(np.sqrt(wx ** 2 + wy ** 2).max())

        # Capture each mode's color-scale window once, then feed the same values to the surface and
        # (via _legend_specs) its colorbar, so the bar's labels match the rendered field. The
        # temperature views auto-range: floor = the coldest cell, ceiling = the hottest, with the
        # span clamped to a minimum so a near-uniform (no-fire) field isn't over-stretched. The
        # floor sits at ambient and stays there during a fire (combustion only heats), so a flame
        # lifts the ceiling without moving the floor.
        hottest_fuel = fuel_temps.max(axis=0)                     # per-cell hottest fuel (the fuel view)
        self._air_temp_display_min = float(temp.min())
        self._air_temp_display_max = max(self._air_temp_display_min + DISPLAY_MIN_TEMP_SPAN_K, float(temp.max()))
        # The fuel-temperature view uses a fixed window (unlike the auto-ranging air-temp view above)
        # so a flame's color reads the same temperature every frame and across models in the viewer.
        self._fuel_temp_display_min = FUEL_TEMP_DISPLAY_MIN_K
        self._fuel_temp_display_max = FUEL_TEMP_DISPLAY_MAX_K
        self._radiant_flux_display_max = float(radiant_flux.max())
        self._pressure_display_max = float(pressure_field.max())
        self._terrain_display_max = float(terrain.max())
        self._column_height_display_max = float(column_height.max())
        self._wind_speed_display_max = wind_ref
        self._wind_temp_window = wind_temp_window(wx, wy, temp, wind_ref)
        self._render_wind_x, self._render_wind_y, self._render_air_temp = wx, wy, temp

        self._color_surface = build_color_surface(
            temp, self._pixel_scale, self._air_temp_display_min, self._air_temp_display_max,
        )
        self._wind_surface = build_wind_surface(
            wx, wy, temp, self._pixel_scale, wind_ref,
        )
        self._wind_speed_surface = build_wind_speed_surface(
            wx, wy, self._pixel_scale, self._wind_speed_display_max
        )
        initial_canopy_fuel = (
            _to_numpy(self._crop(self._initial_canopy_fuel))
            if self._initial_canopy_fuel is not None else None
        )
        self._fire_surface = build_fire_surface(
            fuel_temps, fuel, oxygen, self._pixel_scale,
            ignition_thresholds, self._sim.fuel_burnt_threshold,
            self._sim.fuel_type_names,
            self._show_fire_overlay and self._sim.fire_enabled,
            initial_canopy_fuel,
        )
        self._oxygen_surface = build_oxygen_surface(oxygen, self._pixel_scale)
        self._pressure_surface = build_pressure_surface(
            pressure_field, self._pixel_scale, self._pressure_display_max
        )
        # Fuel-temperature view: the same auto-ranged window over the per-cell hottest fuel.
        self._fuel_temperature_surface = build_color_surface(
            hottest_fuel, self._pixel_scale, self._fuel_temp_display_min, self._fuel_temp_display_max,
        )
        self._radiant_flux_surface = build_radiant_heat_surface(
            radiant_flux, self._pixel_scale, upper_bound=self._radiant_flux_display_max,
        )
        # Static after reset, but rebuilt with the batch for consistency (cost is negligible).
        self._terrain_surface = build_terrain_surface(
            terrain, self._pixel_scale, self._terrain_display_max
        )
        # Static after reset (like terrain); the [0,1] exposure maps straight onto its ramp.
        self._sunlight_surface = build_sunlight_surface(
            _to_numpy(self._crop(self._sunlight)), self._pixel_scale
        )
        # Static after reset: the per-cell biome labels computed at reset, painted with a flat
        # per-biome color (the vegetation hues from the fire view).
        self._biome_surface = build_biome_surface(self._biome_labels, self._pixel_scale)
        self._column_height_surface = build_column_height_surface(
            column_height, self._pixel_scale, self._column_height_display_max
        )
        self._surfaces_dirty = False

    def _to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        """Move a host numpy field onto the simulation device as a float32 tensor."""
        return torch.as_tensor(arr, dtype=self._sim.dtype, device=self._sim.device)

    def _field_state(self) -> SimState:
        """Bundle this (single-world) env's fields into a SimState for step_fields."""
        return SimState(
            mass=self._mass,
            air_temperatures=self._air_temperatures,
            ground_temperature=self._ground_temperature,
            fuel_temperatures=self._fuel_temperatures,
            fuel=self._fuel,
            oxygen=self._oxygen,
            terrain=self._terrain,
            temp_eq=self._temp_eq,
            oxygen_eq=self._oxygen_eq,
            mass_eq=self._mass_eq,
            x_wind_vel=self._x_wind_vel,
            y_wind_vel=self._y_wind_vel,
            u_amb_x=self._u_amb_x,
            u_amb_y=self._u_amb_y,
            radiant_flux=self._last_radiant_flux,
            channel_gain=self._channel_gain,
            x_wind_fire=self._x_wind_fire,
            y_wind_fire=self._y_wind_fire,
        )

    def _store_field_state(self, s: SimState) -> None:
        """Write a stepped SimState's mutable fields back onto the env attributes."""
        self._mass              = s.mass
        self._air_temperatures  = s.air_temperatures
        self._ground_temperature = s.ground_temperature
        self._fuel_temperatures = s.fuel_temperatures
        self._fuel              = s.fuel
        self._oxygen            = s.oxygen
        self._x_wind_vel        = s.x_wind_vel
        self._y_wind_vel        = s.y_wind_vel
        self._x_wind_fire       = s.x_wind_fire
        self._y_wind_fire       = s.y_wind_fire
        self._last_radiant_flux = s.radiant_flux

    def _spawn_fire_patch(self, row: int, col: int) -> None:
        # (row, col) are observed-grid coordinates (from a click or a random pick); offset by _pad
        # to index the padded physics grid, and build the patch mask at the physics size.
        row, col = row + self._pad, col + self._pad
        rows_idx = torch.arange(self._sim_size, device=self._sim.device).view(-1, 1)
        cols_idx = torch.arange(self._sim_size, device=self._sim.device).view(1, -1)
        patch = (rows_idx - row) ** 2 + (cols_idx - col) ** 2 <= self._fire_spawn_radius ** 2
        self._air_temperatures[patch] = float(self._sim.ignition_thresholds.max()) * 2.0
        for n in range(self._sim.num_fuel_types):
            self._fuel_temperatures[n][patch] = float(self._sim.ignition_thresholds[n]) * 2.0
            self._fuel[n][patch] = 1.0

    def _handle_events(self) -> tuple[bool, ViewMode, tuple[int, int] | None]:
        running = self._running
        mode = self._current_mode
        fire_click: tuple[int, int] | None = None
        sim_w = self.grid_size * self._pixel_scale   # the simulation occupies [0, sim_w); the legend is to its right
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEMOTION:
                # Track which coarse action-cell the cursor sits over (for the hover overlay).
                # Over the legend panel there is no cell to hover, so clear the highlight.
                px, py = event.pos
                if px >= sim_w:
                    self._hovered_action_cell = None
                    continue
                arow = min(py // self._action_cell_px, self.action_grid_size - 1)
                acol = min(px // self._action_cell_px, self.action_grid_size - 1)
                self._hovered_action_cell = (arow, acol)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                px, py = event.pos
                if px >= sim_w:
                    continue   # clicks on the legend panel are inert
                if event.button == 1:
                    # Left-click drives the action space live: commit this phase's index from the
                    # hovered action-cell (its row in the row phase, its column in the column phase),
                    # walking the same Discrete(action_grid_size + 1) selection the agent does.
                    # Applied at once so the phase flips on the click itself (works even while
                    # paused); the physics ticks pass None to hold it between clicks.
                    arow = min(py // self._action_cell_px, self.action_grid_size - 1)
                    acol = min(px // self._action_cell_px, self.action_grid_size - 1)
                    self._select_action(arow if self._pending_action_row is None else acol)
                elif event.button == 3:
                    # Right-click ignites a fire patch (sim-cell resolution), as left-click used to.
                    row = min(py // self._pixel_scale, self.grid_size - 1)
                    col = min(px // self._pixel_scale, self.grid_size - 1)
                    fire_click = (row, col)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    self._paused = not self._paused
                elif event.key == pygame.K_PERIOD:
                    self._step_once = True
                elif event.key == pygame.K_f:
                    self._show_fire_overlay = not self._show_fire_overlay
                    self._surfaces_dirty = True
                elif event.key == pygame.K_w:
                    self._show_wind_overlay = not self._show_wind_overlay
                elif event.key == pygame.K_r:
                    self._reset_requested = True
                elif event.key in MODE_KEYS:
                    mode = MODE_KEYS[event.key]
        return running, mode, fire_click

    def _build_info(self) -> dict:
        # Metrics describe the observed world, so reduce over the interior (drop the sponge ring).
        air     = self._crop(self._air_temperatures)
        oxygen  = self._crop(self._oxygen)
        fuel     = self._crop(self._fuel)
        ftemps   = self._crop(self._fuel_temperatures)
        wind_speeds = torch.sqrt(self._crop(self._x_wind_vel) ** 2 + self._crop(self._y_wind_vel) ** 2)
        burning_per_type = self._crop(self._burning_per_type())

        # Reduce every metric on-device, then pull the whole batch back in a SINGLE
        # host transfer (.tolist()). Doing per-scalar float() instead would force a
        # separate device sync for each of the ~5*N+10 values, stalling every step.
        fuel_flat  = fuel.flatten(1)                       # (N, obs*obs)
        ftemp_flat = ftemps.flatten(1)                     # (N, obs*obs)
        scalars = torch.stack([
            air.mean(),
            air.amin(),
            air.amax(),
            air.std(),
            wind_speeds.mean(),
            wind_speeds.amax(),
            wind_speeds.std(),
            oxygen.mean(),
            oxygen.amin(),
            burning_per_type.any(dim=0).sum().to(self._sim.dtype),
        ])
        per_type = torch.cat([
            fuel_flat.mean(dim=1),
            fuel_flat.amax(dim=1),
            ftemp_flat.mean(dim=1),
            ftemp_flat.amax(dim=1),
            burning_per_type.flatten(1).sum(dim=1).to(self._sim.dtype),
        ])
        vals = torch.cat([scalars, per_type]).tolist()     # one device->host sync

        (air_mean, air_min, air_max, air_std,
         wind_mean, wind_max, wind_std, ox_mean, ox_min, cells_burning) = vals[:10]
        n = self._sim.num_fuel_types
        block = vals[10:]   # five N-length blocks: fuel mean/max, fuel-temp mean/max, burn count
        terrain_mean, terrain_min, terrain_max = self._terrain_stats

        info = {
            "air_temperature": air_mean,
            "air_temp_min":    air_min,
            "air_temp_max":    air_max,
            "air_temp_std":    air_std,
            "wind_mean":       wind_mean,
            "wind_max":        wind_max,
            "wind_std":        wind_std,
            "oxygen_mean":     ox_mean,
            "oxygen_min":      ox_min,
            "cells_burning":   int(cells_burning),
            "terrain_mean":    terrain_mean,
            "terrain_min":     terrain_min,
            "terrain_max":     terrain_max,
            "map":             self._current_map,
            "step":            self._step_count,
            # Two-step action phase: with a row pending the next action is read as the column
            # (awaiting_column True), otherwise as a row. pending_action_row is that stashed row
            # (None on a row step). The policy must condition its action on this.
            "awaiting_column":    self._pending_action_row is not None,
            "pending_action_row": self._pending_action_row,
        }
        for i, name in enumerate(self._sim.fuel_type_names):
            info[f"fuel_{name}_mean"]      = block[i]
            info[f"fuel_{name}_max"]       = block[n + i]
            info[f"fuel_temp_{name}_mean"] = block[2 * n + i]
            info[f"fuel_temp_{name}_max"]  = block[3 * n + i]
            info[f"cells_burning_{name}"]  = int(block[4 * n + i])
        return info
