"""
firecracker_env.py

Gymnasium environment wrapping the Firecracker heat-diffusion simulation.
Observation: (grid_size, grid_size) float32 air_temperature field in [0, 1].
Action:      Discrete(1) — single no-op placeholder for future RL action spaces.
"""

import math
from enum import Enum

import numpy as np
import torch
import pygame
import gymnasium
from gymnasium import spaces

from simulation import Simulation, SimState
from map_loader import load_map, resolve_map, validate_against_config


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    """Move a (possibly GPU-resident) tensor to a host numpy array for rendering/info."""
    return t.detach().cpu().numpy()

# ---------------------------------------------------------------------------
# Display constants
# ---------------------------------------------------------------------------

DEFAULT_GRID_SIZE: int = 256
DEFAULT_WINDOW_SIZE: int = 512
WINDOW_TITLE: str = "Firecracker"
TARGET_FPS: int = 60

MAX_CHANNEL_VALUE: int = 255
RED_CHANNEL: int = 0
GREEN_CHANNEL: int = 1
BLUE_CHANNEL: int = 2

# ---------------------------------------------------------------------------
# Wind arrow visualization parameters
# ---------------------------------------------------------------------------

_WIND_DISPLAY_THRESHOLD:       float = 1e-4

# Temperature views map a Kelvin window onto the color ramp. The air-temperature view
# uses a fixed ambient fire-weather window (fire pushes air above the ceiling and simply
# saturates to the hot color); the fuel-temperature view shares the floor but tracks its
# ceiling to the (EMA-smoothed) peak so flame structure stays visible.
DISPLAY_TEMP_FLOOR_K:    float = 290.0
DISPLAY_AIR_TEMP_CEIL_K: float = 330.0
DISPLAY_MIN_TEMP_SPAN_K: float = 50.0
# Kelvin above ignition that spans the fire-overlay color ramp (ignition -> +span).
FIRE_COLOR_TEMP_SPAN_K:  float = 600.0

# EMA weight for the upper bound of the fuel-temperature and radiant-heat color
# scales. Smaller = steadier (slower to track the peak); larger = more responsive.
_DISPLAY_SCALE_EMA_ALPHA: float = 0.1
# Fraction of the frame max the radiant-heat EMA tracks toward, so cells at/above this
# fraction of the smoothed peak saturate to the top color.
_DISPLAY_SCALE_MAX_FRACTION: float = 0.5

WIND_ARROW_STRIDE: int = 16
WIND_ARROW_LENGTH: int = 24
WIND_ARROW_LINE_WIDTH: int = 1
WIND_ARROW_HEAD_SIZE: int = 4
WIND_ARROW_HEAD_ANGLE: float = math.pi / 6  # 30 degrees

FIRE_SPAWN_RADIUS: int = 5

# ---------------------------------------------------------------------------
# View modes
# ---------------------------------------------------------------------------

class ViewMode(Enum):
    TEMPERATURE = 1
    WIND = 2
    FIRE = 3
    OXYGEN = 4
    PRESSURE = 5
    FUEL_TEMPERATURE = 6
    RADIANT_HEAT = 7
    TERRAIN = 8
    COLUMN_HEIGHT = 9


# Populated from ViewMode values so new modes are picked up automatically.
MODE_KEYS: dict[int, ViewMode] = {
    getattr(pygame, f"K_{mode.value}"): mode
    for mode in ViewMode
}

# ---------------------------------------------------------------------------
# Surface builders
# ---------------------------------------------------------------------------

def build_color_surface(
    field: np.ndarray, scale: int, lo: float, hi: float
) -> pygame.Surface:
    """Map a temperature field [K] linearly over the window [lo, hi] onto the heat ramp.

    Values at/below lo render dark; at/above hi saturate to the hot color.
    """
    rows, cols = field.shape
    span = hi - lo
    normalized = (
        np.clip((field - lo) / span, 0.0, 1.0) if span > 0.0 else np.zeros_like(field)
    )
    r = (normalized * 255).astype(np.uint8)
    g = ((1.0 - normalized) * 15).astype(np.uint8)
    b = ((1.0 - normalized) * 31).astype(np.uint8)
    rgb = np.stack([
        np.repeat(np.repeat(r, scale, axis=0), scale, axis=1),
        np.repeat(np.repeat(g, scale, axis=0), scale, axis=1),
        np.repeat(np.repeat(b, scale, axis=0), scale, axis=1),
    ], axis=-1)
    surface = pygame.Surface((cols * scale, rows * scale))
    # surfarray expects (width, height, 3); numpy is (height, width, 3)
    pygame.surfarray.blit_array(surface, rgb.transpose(1, 0, 2))
    return surface


def build_wind_surface(
    x_wind_vel: np.ndarray,
    y_wind_vel: np.ndarray,
    air_temperatures: np.ndarray,
    scale: int,
    reference_magnitude: float,
) -> pygame.Surface:
    rows, cols = x_wind_vel.shape
    surface = pygame.Surface((cols * scale, rows * scale))
    surface.set_colorkey((0, 0, 0))

    if reference_magnitude == 0.0:
        return surface

    t = min(reference_magnitude / _WIND_DISPLAY_THRESHOLD, 1.0)
    magnitudes = np.sqrt(x_wind_vel ** 2 + y_wind_vel ** 2)
    max_half_len = WIND_ARROW_LENGTH / 2.0
    grid_offset = WIND_ARROW_STRIDE // 2

    visible_temps = air_temperatures[magnitudes >= 0.1 * reference_magnitude]
    temp_min = float(visible_temps.min()) if visible_temps.size > 0 else float(air_temperatures.min())
    temp_max = float(visible_temps.max()) if visible_temps.size > 0 else float(air_temperatures.max())
    temp_range = temp_max - temp_min

    for row in range(grid_offset, rows, WIND_ARROW_STRIDE):
        for col in range(grid_offset, cols, WIND_ARROW_STRIDE):
            vx = float(x_wind_vel[row, col])
            vy = float(y_wind_vel[row, col])
            magnitude = float(magnitudes[row, col])
            if magnitude < 0.1 * reference_magnitude:
                continue

            relative = min(magnitude / reference_magnitude, 1.0)
            absolute = min(magnitude / _WIND_DISPLAY_THRESHOLD, 1.0)
            half_len = (t * relative + (1.0 - t) * absolute) * max_half_len
            if half_len < 1.0:
                continue
            ux, uy = vx / magnitude, vy / magnitude
            cx = (col + 0.5) * scale
            cy = (row + 0.5) * scale

            heat = (float(air_temperatures[row, col]) - temp_min) / temp_range if temp_range > 0.0 else 0.0
            color = (int(heat * MAX_CHANNEL_VALUE), int((1.0 - heat) * 127), int((1.0 - heat) * MAX_CHANNEL_VALUE))

            start = (int(cx - ux * half_len), int(cy - uy * half_len))
            tip = (int(cx + ux * half_len), int(cy + uy * half_len))
            pygame.draw.line(surface, color, start, tip, WIND_ARROW_LINE_WIDTH)

            angle = math.atan2(uy, ux)
            for sign in (+1, -1):
                barb = (
                    int(tip[0] - WIND_ARROW_HEAD_SIZE * math.cos(angle + sign * WIND_ARROW_HEAD_ANGLE)),
                    int(tip[1] - WIND_ARROW_HEAD_SIZE * math.sin(angle + sign * WIND_ARROW_HEAD_ANGLE)),
                )
                pygame.draw.line(surface, color, tip, barb, WIND_ARROW_LINE_WIDTH)

    return surface


def build_oxygen_surface(oxygen: np.ndarray, scale: int) -> pygame.Surface:
    rows, cols = oxygen.shape
    # O2 is now a partial density [kg/m^3], so normalise against the current max:
    # black = most oxygen, white = least.
    o_max = float(oxygen.max())
    fraction = oxygen / o_max if o_max > 0.0 else np.zeros_like(oxygen)
    brightness = ((1.0 - np.clip(fraction, 0.0, 1.0)) * MAX_CHANNEL_VALUE).astype(np.uint8)
    brightness_scaled = np.repeat(np.repeat(brightness, scale, axis=0), scale, axis=1)
    rgb = np.stack([brightness_scaled] * 3, axis=-1)
    surface = pygame.Surface((cols * scale, rows * scale))
    pygame.surfarray.blit_array(surface, rgb.transpose(1, 0, 2))
    return surface


def build_pressure_surface(pressure: np.ndarray, scale: int) -> pygame.Surface:
    rows, cols = pressure.shape
    # Anchored at zero, normalised by the current max (like the temperature view): a
    # uniform field renders uniformly bright, and a gradient's spread visibly shrinks as
    # it equalises — unlike a min-max scale, which re-stretches the residual every frame.
    p_max = float(pressure.max())
    normalized = np.clip(pressure / p_max, 0.0, 1.0) if p_max > 0.0 else np.zeros_like(pressure)
    blue = (normalized * MAX_CHANNEL_VALUE).astype(np.uint8)
    blue_scaled = np.repeat(np.repeat(blue, scale, axis=0), scale, axis=1)
    rgb = np.zeros((rows * scale, cols * scale, 3), dtype=np.uint8)
    rgb[:, :, BLUE_CHANNEL] = blue_scaled
    surface = pygame.Surface((cols * scale, rows * scale))
    pygame.surfarray.blit_array(surface, rgb.transpose(1, 0, 2))
    return surface


def build_radiant_heat_surface(
    radiant_flux: np.ndarray, scale: int, upper_bound: float | None = None
) -> pygame.Surface:
    rows, cols = radiant_flux.shape
    # EMA-smoothed upper bound when provided; otherwise fall back to the frame max.
    bound = upper_bound if upper_bound is not None else float(radiant_flux.max())
    normalized = (np.clip(radiant_flux / bound, 0.0, 1.0) if bound > 0.0 else np.zeros_like(radiant_flux))
    r = (normalized * MAX_CHANNEL_VALUE).astype(np.uint8)
    g = (normalized * 80).astype(np.uint8)
    b = (normalized * MAX_CHANNEL_VALUE).astype(np.uint8)
    rgb = np.stack([
        np.repeat(np.repeat(r, scale, axis=0), scale, axis=1),
        np.repeat(np.repeat(g, scale, axis=0), scale, axis=1),
        np.repeat(np.repeat(b, scale, axis=0), scale, axis=1),
    ], axis=-1)
    surface = pygame.Surface((cols * scale, rows * scale))
    pygame.surfarray.blit_array(surface, rgb.transpose(1, 0, 2))
    return surface


def build_terrain_surface(terrain: np.ndarray, scale: int) -> pygame.Surface:
    rows, cols = terrain.shape
    # Terrain is elevation in metres; normalise against the current max for the relief.
    t_max = float(terrain.max())
    h = np.clip(terrain / t_max, 0.0, 1.0) if t_max > 0.0 else np.zeros_like(terrain)
    # Elevation relief: dark green valleys -> white peaks.
    r = (h * MAX_CHANNEL_VALUE).astype(np.uint8)
    g = (60.0 + h * (MAX_CHANNEL_VALUE - 60.0)).astype(np.uint8)
    b = (h * MAX_CHANNEL_VALUE).astype(np.uint8)
    rgb = np.stack([
        np.repeat(np.repeat(r, scale, axis=0), scale, axis=1),
        np.repeat(np.repeat(g, scale, axis=0), scale, axis=1),
        np.repeat(np.repeat(b, scale, axis=0), scale, axis=1),
    ], axis=-1)
    surface = pygame.Surface((cols * scale, rows * scale))
    pygame.surfarray.blit_array(surface, rgb.transpose(1, 0, 2))
    return surface


def build_column_height_surface(height: np.ndarray, scale: int) -> pygame.Surface:
    rows, cols = height.shape
    # Relief of the air-column top, anchored at zero and normalised by the current max
    # (like the temperature view): dark = short columns (cold/low terrain), bright cyan =
    # tall columns (warm/high terrain). Column height is always non-negative.
    h_max = float(height.max())
    normalized = np.clip(height / h_max, 0.0, 1.0) if h_max > 0.0 else np.zeros_like(height)
    r = (normalized * 40).astype(np.uint8)
    g = (normalized * MAX_CHANNEL_VALUE).astype(np.uint8)
    b = (normalized * MAX_CHANNEL_VALUE).astype(np.uint8)
    rgb = np.stack([
        np.repeat(np.repeat(r, scale, axis=0), scale, axis=1),
        np.repeat(np.repeat(g, scale, axis=0), scale, axis=1),
        np.repeat(np.repeat(b, scale, axis=0), scale, axis=1),
    ], axis=-1)
    surface = pygame.Surface((cols * scale, rows * scale))
    pygame.surfarray.blit_array(surface, rgb.transpose(1, 0, 2))
    return surface


def build_fire_surface(
    fuel_temperatures: np.ndarray,       # (N, H, W)
    fuel: np.ndarray,                    # (N, H, W)
    oxygen: np.ndarray,                  # (H, W)
    scale: int,
    ignition_thresholds: np.ndarray,     # (N,)
    fuel_burnt_threshold: float,
    show_fire_overlay: bool = True,
) -> pygame.Surface:
    rows, cols = fuel_temperatures.shape[1], fuel_temperatures.shape[2]
    rgb = np.zeros((rows, cols, 3), dtype=np.uint8)

    # Visual "burning": hot enough to pyrolyse, with fuel and oxygen present. Combustion is a
    # smooth Arrhenius rate (no extinction threshold), so this is a display/diagnostic cutoff only.
    ign = ignition_thresholds[:, np.newaxis, np.newaxis]
    burning_per_type = (
        (fuel_temperatures >= ign) &
        (fuel > fuel_burnt_threshold) &
        (oxygen[np.newaxis] > 0.0)
    )
    any_burning = burning_per_type.any(axis=0)  # (H, W)

    # Green brightness tracks total fuel, normalised so the cell with the most fuel
    # on the map is full brightness. This keeps the image scale-invariant: uniformly
    # scaling all fuel leaves the ratios (and thus the rendered intensities) unchanged.
    total_fuel = fuel.sum(axis=0)
    max_fuel = float(total_fuel.max())
    normalized_fuel = total_fuel / max_fuel if max_fuel > 0.0 else total_fuel
    rgb[:, :, GREEN_CHANNEL] = (normalized_fuel * MAX_CHANNEL_VALUE).astype(np.uint8)

    if show_fire_overlay:
        # Color gradient based on the hottest fuel type; a fixed Kelvin span above the
        # minimum ignition temperature spans the ramp.
        max_fuel_temp = fuel_temperatures.max(axis=0)
        min_ign = float(ignition_thresholds.min())
        t = np.clip((max_fuel_temp - min_ign) / FIRE_COLOR_TEMP_SPAN_K, 0.0, 1.0)
        rgb[any_burning, RED_CHANNEL]   = MAX_CHANNEL_VALUE
        rgb[any_burning, GREEN_CHANNEL] = (t[any_burning] * MAX_CHANNEL_VALUE).astype(np.uint8)

    rgb_scaled = np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)
    surface = pygame.Surface((cols * scale, rows * scale))
    pygame.surfarray.blit_array(surface, rgb_scaled.transpose(1, 0, 2))
    return surface


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

        env_cfg  = (config or {}).get("environment", {})
        fire_cfg = (config or {}).get("fire", {})
        maps_cfg = (config or {}).get("maps", {})

        # Map selection: a name (specific map) or None (random pick at reset).
        # Explicit constructor args win over config.
        self._maps_dir: str = maps_dir if maps_dir is not None else str(maps_cfg.get("dir", "maps"))
        self._map_name: str | None = (
            map_name if map_name is not None else (str(maps_cfg.get("name", "")) or None)
        )

        self.grid_size   = int(env_cfg.get("grid_size",   DEFAULT_GRID_SIZE))
        self.window_size = int(env_cfg.get("window_size", DEFAULT_WINDOW_SIZE))
        self.max_steps   = int(env_cfg.get("max_steps",   1000))
        self._fire_spawn_radius = int(fire_cfg.get("spawn_radius", FIRE_SPAWN_RADIUS))
        self._spawn_fire = bool(fire_cfg.get("spawn_fire", False))
        self.render_mode = render_mode
        self._pixel_scale = self.window_size // self.grid_size

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.grid_size, self.grid_size), dtype=np.float32
        )
        # Single no-op action; replace with the real action space when designing the agent.
        self.action_space = spaces.Discrete(1)

        # Simulation state (populated by reset)
        self._air_temperatures: np.ndarray | None = None
        self._fuel_temperatures: np.ndarray | None = None
        self._x_wind_vel: np.ndarray | None = None
        self._y_wind_vel: np.ndarray | None = None
        self._fuel: np.ndarray | None = None
        self._oxygen: np.ndarray | None = None
        self._mass: np.ndarray | None = None   # boundary-layer areal mass [kg/m^2]
        self._mass_eq: torch.Tensor | None = None   # rest-state mass (open-boundary sponge target)
        self._terrain: np.ndarray | None = None
        self._u_amb_x: torch.Tensor | None = None   # synoptic ambient wind [m/s]
        self._u_amb_y: torch.Tensor | None = None
        # Rest-state targets (initial profiles): radiative-equilibrium temperature and oxygen.
        self._temp_eq: np.ndarray | None = None
        self._oxygen_eq: np.ndarray | None = None
        self._ground_temperature: torch.Tensor | None = None   # surface skin [K]
        self._terrain_stats: list[float] = [0.0, 0.0, 0.0]   # (mean, min, max), set at reset
        self._last_radiant_flux: np.ndarray | None = None
        self._current_map: str | None = None   # filename of the loaded map
        # EMA-smoothed upper bounds for the mode 6 / mode 7 color scales.
        self._fuel_temp_display_scale: float = 0.0
        self._radiant_flux_display_scale: float = 0.0
        self._step_count: int = 0

        # Rendering state
        self._screen: pygame.Surface | None = None
        self._clock: pygame.time.Clock | None = None
        self._color_surface: pygame.Surface | None = None
        self._wind_surface: pygame.Surface | None = None
        self._fire_surface: pygame.Surface | None = None
        self._oxygen_surface: pygame.Surface | None = None
        self._pressure_surface: pygame.Surface | None = None
        self._fuel_temperature_surface: pygame.Surface | None = None
        self._radiant_flux_surface: pygame.Surface | None = None
        self._terrain_surface: pygame.Surface | None = None
        self._column_height_surface: pygame.Surface | None = None
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
            pygame.display.set_caption(WINDOW_TITLE)
            self._screen = pygame.display.set_mode(
                (self.grid_size * self._pixel_scale, self.grid_size * self._pixel_scale)
            )
            self._clock = pygame.time.Clock()

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
        validate_against_config(m, self.grid_size, self._sim.fuel_type_names)
        self._current_map = map_path.name

        # Maps load from disk as numpy (map_loader is source-agnostic); move every field
        # onto the simulation device so the per-step physics stays GPU-resident.
        self._terrain           = self._to_tensor(m.terrain)
        self._air_temperatures  = self._to_tensor(m.air_temperatures)
        self._temp_eq           = self._to_tensor(m.temp_eq)
        self._oxygen            = self._to_tensor(m.oxygen)
        self._oxygen_eq         = self._to_tensor(m.oxygen_eq)
        self._mass              = self._to_tensor(m.mass)
        self._mass_eq           = self._mass.clone()   # level-lid rest state -> sponge target
        self._fuel              = self._to_tensor(m.fuel)
        self._fuel_temperatures = self._to_tensor(m.fuel_temperatures)
        # Surface skin starts at the rest temperature profile (radiative-equilibrium target).
        self._ground_temperature = self._temp_eq.clone()

        # Per-map synoptic ambient wind: the momentum drag relaxes toward it, and the
        # prognostic wind is initialised to it so the world starts near its steady state.
        self._u_amb_x = self._to_tensor(np.float32(m.ambient_wind_x))
        self._u_amb_y = self._to_tensor(np.float32(m.ambient_wind_y))
        self._x_wind_vel = torch.full_like(self._mass, float(m.ambient_wind_x))
        self._y_wind_vel = torch.full_like(self._mass, float(m.ambient_wind_y))

        # Terrain is static for the episode, so reduce its stats once here (one host
        # transfer) instead of re-syncing them every step in _build_info.
        self._terrain_stats = torch.stack(
            [self._terrain.mean(), self._terrain.min(), self._terrain.max()]
        ).tolist()

        # Optional ignition overlay (random spawn). Maps describe the world at rest;
        # fire is a runtime concern applied on top of the loaded state.
        if self._spawn_fire:
            r = int(self.np_random.integers(0, self.grid_size))
            c = int(self.np_random.integers(0, self.grid_size))
            self._spawn_fire_patch(r, c)

        self._last_radiant_flux = torch.zeros(
            (self.grid_size, self.grid_size), dtype=self._sim.dtype, device=self._sim.device
        )
        # Seed the display scales from the initial state so the first frames are scaled
        # sanely. These drive the mode 6/7 color scales only, so they (and their host
        # syncs) are skipped entirely on the headless training path (render_mode=None).
        if self.render_mode is not None:
            # Fuel-temp scale tracks the actual peak (it is used as the color ceiling);
            # radiant flux is anchored at 0, so it tracks a fraction of its peak.
            self._fuel_temp_display_scale = float(self._fuel_temperatures.max())
            self._radiant_flux_display_scale = _DISPLAY_SCALE_MAX_FRACTION * float(self._last_radiant_flux.max())
        self._step_count = 0
        self._running = True
        self._paused = False
        self._reset_requested = False
        self._step_once = False
        self._surfaces_dirty = True

        return _to_numpy(self._air_temperatures), self._build_info()

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        # Advance every field one tick via the shared physics step (the same routine the
        # batched rollout collector uses). The convective transport carries mass and energy
        # (E = m*T) together down the transport potential, fire/radiant/relaxation forcings
        # follow, and the diagnostic wind is refreshed for the next tick's oxygen advection.
        self._store_field_state(self._sim.step_fields(self._field_state()))
        # Track EMA-smoothed peaks (toward a fraction of the max) so the mode 6 / mode 7
        # color scales don't flicker. Display-only, so the host syncs are skipped on the
        # headless training path (render_mode=None).
        if self.render_mode is not None:
            a = _DISPLAY_SCALE_EMA_ALPHA
            f = _DISPLAY_SCALE_MAX_FRACTION
            self._fuel_temp_display_scale = (
                a * float(self._fuel_temperatures.max()) + (1.0 - a) * self._fuel_temp_display_scale
            )
            self._radiant_flux_display_scale = (
                a * f * float(self._last_radiant_flux.max()) + (1.0 - a) * self._radiant_flux_display_scale
            )
        self._step_count += 1
        self._surfaces_dirty = True

        reward = 0.0
        terminated = False
        truncated = self._step_count >= self.max_steps
        return _to_numpy(self._air_temperatures), reward, terminated, truncated, self._build_info()

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
            if self._show_wind_overlay and self._current_mode in (ViewMode.TEMPERATURE, ViewMode.PRESSURE):
                self._screen.blit(self._wind_surface, (0, 0))
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

    def _surface_for_mode(self) -> pygame.Surface:
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
        if self._current_mode == ViewMode.COLUMN_HEIGHT:
            return self._column_height_surface
        return self._radiant_flux_surface

    def _rebuild_surfaces_if_dirty(self) -> None:
        if not self._surfaces_dirty:
            return
        # The surface builders are CPU/pygame numpy code, so pull the state to the host
        # once here — the numpy/tensor boundary lives at the render edge.
        temp = _to_numpy(self._air_temperatures)
        wx, wy = _to_numpy(self._x_wind_vel), _to_numpy(self._y_wind_vel)
        pressure_field = _to_numpy(self._mass)
        fuel_temps = _to_numpy(self._fuel_temperatures)
        fuel = _to_numpy(self._fuel)
        oxygen = _to_numpy(self._oxygen)
        column_height = _to_numpy(
            self._sim.column_height(self._mass, self._terrain, self._air_temperatures)
        )
        ignition_thresholds = _to_numpy(self._sim.ignition_thresholds).reshape(-1)
        wind_ref = float(np.sqrt(wx ** 2 + wy ** 2).max())

        self._color_surface = build_color_surface(
            temp, self._pixel_scale, DISPLAY_TEMP_FLOOR_K, DISPLAY_AIR_TEMP_CEIL_K,
        )
        self._wind_surface = build_wind_surface(
            wx, wy, temp, self._pixel_scale, wind_ref,
        )
        self._fire_surface = build_fire_surface(
            fuel_temps, fuel, oxygen, self._pixel_scale,
            ignition_thresholds, self._sim.fuel_burnt_threshold,
            self._show_fire_overlay and self._sim.fire_enabled,
        )
        self._oxygen_surface = build_oxygen_surface(oxygen, self._pixel_scale)
        self._pressure_surface = build_pressure_surface(pressure_field, self._pixel_scale)
        # Fuel-temperature view: floor at ambient, ceiling tracks the smoothed peak
        # (clamped to a minimum span so a cold map doesn't over-stretch the ramp).
        fuel_temp_ceil = max(
            DISPLAY_TEMP_FLOOR_K + DISPLAY_MIN_TEMP_SPAN_K, self._fuel_temp_display_scale
        )
        self._fuel_temperature_surface = build_color_surface(
            fuel_temps.max(axis=0), self._pixel_scale, DISPLAY_TEMP_FLOOR_K, fuel_temp_ceil,
        )
        self._radiant_flux_surface = build_radiant_heat_surface(
            _to_numpy(self._last_radiant_flux), self._pixel_scale,
            upper_bound=self._radiant_flux_display_scale,
        )
        # Static after reset, but rebuilt with the batch for consistency (cost is negligible).
        self._terrain_surface = build_terrain_surface(_to_numpy(self._terrain), self._pixel_scale)
        self._column_height_surface = build_column_height_surface(column_height, self._pixel_scale)
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
        self._last_radiant_flux = s.radiant_flux

    def _spawn_fire_patch(self, row: int, col: int) -> None:
        rows_idx = torch.arange(self.grid_size, device=self._sim.device).view(-1, 1)
        cols_idx = torch.arange(self.grid_size, device=self._sim.device).view(1, -1)
        patch = (rows_idx - row) ** 2 + (cols_idx - col) ** 2 <= self._fire_spawn_radius ** 2
        self._air_temperatures[patch] = float(self._sim.ignition_thresholds.max()) * 2.0
        for n in range(self._sim.num_fuel_types):
            self._fuel_temperatures[n][patch] = float(self._sim.ignition_thresholds[n]) * 2.0
            self._fuel[n][patch] = 1.0

    def _handle_events(self) -> tuple[bool, ViewMode, tuple[int, int] | None]:
        running = self._running
        mode = self._current_mode
        fire_click: tuple[int, int] | None = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    px, py = event.pos
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
        wind_speeds = torch.sqrt(self._x_wind_vel ** 2 + self._y_wind_vel ** 2)
        ign = self._sim.ignition_thresholds   # already (N, 1, 1)
        burning_per_type = (
            (self._fuel_temperatures >= ign) &
            (self._fuel > self._sim.fuel_burnt_threshold) &
            (self._oxygen[None] > 0.0)
        ) if self._sim.fire_enabled else torch.zeros_like(self._fuel, dtype=torch.bool)

        # Reduce every metric on-device, then pull the whole batch back in a SINGLE
        # host transfer (.tolist()). Doing per-scalar float() instead would force a
        # separate device sync for each of the ~5*N+10 values, stalling every step.
        fuel_flat  = self._fuel.flatten(1)                 # (N, H*W)
        ftemp_flat = self._fuel_temperatures.flatten(1)    # (N, H*W)
        scalars = torch.stack([
            self._air_temperatures.mean(),
            self._air_temperatures.amin(),
            self._air_temperatures.amax(),
            self._air_temperatures.std(),
            wind_speeds.mean(),
            wind_speeds.amax(),
            wind_speeds.std(),
            self._oxygen.mean(),
            self._oxygen.amin(),
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
        }
        for i, name in enumerate(self._sim.fuel_type_names):
            info[f"fuel_{name}_mean"]      = block[i]
            info[f"fuel_{name}_max"]       = block[n + i]
            info[f"fuel_temp_{name}_mean"] = block[2 * n + i]
            info[f"fuel_temp_{name}_max"]  = block[3 * n + i]
            info[f"cells_burning_{name}"]  = int(block[4 * n + i])
        return info
