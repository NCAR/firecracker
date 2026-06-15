"""
firecracker_env.py

Gymnasium environment wrapping the Firecracker heat-diffusion simulation.
Observation: (grid_size, grid_size) float32 air_temperature field in [0, 1].
Action:      Discrete(1) — single no-op placeholder for future RL action spaces.
"""

import math
from enum import Enum

import numpy as np
import pygame
import gymnasium
from gymnasium import spaces

from simulation import Simulation

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

_AIR_TEMP_DISPLAY_THRESHOLD:       float = 1e-2
_WIND_DISPLAY_THRESHOLD:       float = 1e-4
_PRESSURE_DISPLAY_THRESHOLD:   float = 1e-2

# EMA weight for the upper bound of the fuel-temperature and radiant-heat color
# scales. Smaller = steadier (slower to track the peak); larger = more responsive.
_DISPLAY_SCALE_EMA_ALPHA: float = 0.1
# Fraction of the frame max the EMA tracks toward, so cells at/above this fraction
# of the smoothed peak saturate to the top color.
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
    air_temperatures: np.ndarray, scale: int, upper_bound: float | None = None
) -> pygame.Surface:
    rows, cols = air_temperatures.shape
    if upper_bound is not None:
        # Fixed (EMA-smoothed) upper bound: normalize against it and clamp values
        # above it to the top color, so the scale doesn't flicker with the frame max.
        normalized = (
            np.clip(air_temperatures / upper_bound, 0.0, 1.0)
            if upper_bound > 0.0 else np.zeros_like(air_temperatures)
        )
    else:
        max_temp = float(air_temperatures.max())
        if max_temp == 0.0:
            normalized = np.zeros_like(air_temperatures)
        else:
            t        = min(max_temp / _AIR_TEMP_DISPLAY_THRESHOLD, 1.0)
            relative = np.clip(air_temperatures / max_temp, 0.0, 1.0)
            absolute = np.clip(air_temperatures / _AIR_TEMP_DISPLAY_THRESHOLD, 0.0, 1.0)
            normalized = t * relative + (1.0 - t) * absolute
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
    # black = full oxygen (1.0), white = no oxygen (0.0)
    brightness = ((1.0 - np.clip(oxygen, 0.0, 1.0)) * MAX_CHANNEL_VALUE).astype(np.uint8)
    brightness_scaled = np.repeat(np.repeat(brightness, scale, axis=0), scale, axis=1)
    rgb = np.stack([brightness_scaled] * 3, axis=-1)
    surface = pygame.Surface((cols * scale, rows * scale))
    pygame.surfarray.blit_array(surface, rgb.transpose(1, 0, 2))
    return surface


def build_pressure_surface(pressure: np.ndarray, scale: int) -> pygame.Surface:
    rows, cols = pressure.shape
    p_min = float(pressure.min())
    p_max = float(pressure.max())
    p_range = p_max - p_min
    absolute = np.clip(pressure, 0.0, 1.0)
    minmax = (pressure - p_min) / p_range if p_range > 0.0 else absolute
    t = min(p_range / _PRESSURE_DISPLAY_THRESHOLD, 1.0)
    normalized = t * minmax + (1.0 - t) * absolute
    fade = min(p_max / _PRESSURE_DISPLAY_THRESHOLD, 1.0)
    blue = (normalized * fade * MAX_CHANNEL_VALUE).astype(np.uint8)
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
    h = np.clip(terrain, 0.0, 1.0)
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
    # Min-max normalised relief of the air-column top: dark = short columns (cold/low
    # terrain), bright cyan = tall columns (warm/high terrain).
    h_min = float(height.min())
    h_max = float(height.max())
    h_range = h_max - h_min
    normalized = (height - h_min) / h_range if h_range > 0.0 else np.zeros_like(height)
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
    oxygen_extinction_threshold: float,
    show_fire_overlay: bool = True,
) -> pygame.Surface:
    rows, cols = fuel_temperatures.shape[1], fuel_temperatures.shape[2]
    rgb = np.zeros((rows, cols, 3), dtype=np.uint8)

    ign = ignition_thresholds[:, np.newaxis, np.newaxis]
    burning_per_type = (
        (fuel_temperatures >= ign) &
        (fuel > fuel_burnt_threshold) &
        (oxygen[np.newaxis] > oxygen_extinction_threshold)
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
        # Color gradient based on the hottest fuel type; scale from min ignition threshold.
        max_fuel_temp = fuel_temperatures.max(axis=0)
        min_ign = float(ignition_thresholds.min())
        t = np.clip((max_fuel_temp - min_ign) / (24.0 * min_ign), 0.0, 1.0)
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
    ):
        super().__init__()
        assert render_mode is None or render_mode in self.metadata["render_modes"], (
            f"render_mode must be one of {self.metadata['render_modes']} or None"
        )

        self._sim = Simulation(config)

        env_cfg  = (config or {}).get("environment", {})
        fire_cfg = (config or {}).get("fire", {})

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
        self._reference_wind_magnitude: float = 0.0
        self._fuel: np.ndarray | None = None
        self._oxygen: np.ndarray | None = None
        self._mass: np.ndarray | None = None   # column mass = surface pressure
        self._terrain: np.ndarray | None = None
        # Relaxation targets (initial profiles) and per-cell temperature rate.
        self._temp_eq: np.ndarray | None = None
        self._oxygen_eq: np.ndarray | None = None
        self._temp_relax_rate: np.ndarray | None = None
        self._last_radiant_flux: np.ndarray | None = None
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
        # Elevation height map drives initial temperature, oxygen, and vegetation.
        self._terrain = self._sim.create_terrain(
            self.grid_size, base=int(self.np_random.integers(0, 256))
        )

        # Air temperature falls off exponentially with elevation (1.0 at sea level).
        self._air_temperatures = self._sim.elevation_falloff(
            self._terrain, self._sim.temperature_lapse_rate
        )
        # Radiative-equilibrium target: the clean elevation profile, captured before
        # any spawned fire perturbs the field.
        self._temp_eq = self._air_temperatures.copy()

        N = self._sim.num_fuel_types
        self._fuel = np.zeros((N, self.grid_size, self.grid_size), dtype=np.float32)

        # Grass (type 0): continuous Perlin noise scaled to [0, 0.5], thinned by elevation.
        grass_noise = self._sim.create_grid(
            self.grid_size,
            scale=float(self._sim.fuel_noise_scales[0]),
            base=int(self.np_random.integers(0, 256)),
        )
        grass_alt = self._sim.altitude_vegetation_multiplier(self._terrain, 0)
        self._fuel[0] = (grass_noise * float(self._sim.spawn_densities[0]) * grass_alt).astype(np.float32)

        # Trees (type 1): per-cell counts drawn from an exponential whose mean is the
        # elevation-thinned noise density, capped at max_trees_per_cell, then scaled
        # to fuel mass. Higher density -> more trees on average.
        if N > 1:
            tree_noise = self._sim.create_grid(
                self.grid_size,
                scale=float(self._sim.fuel_noise_scales[1]),
                base=int(self.np_random.integers(0, 256)),
            )
            tree_alt = self._sim.altitude_vegetation_multiplier(self._terrain, 1)
            density = tree_noise * float(self._sim.spawn_densities[1]) * tree_alt
            tree_counts = self._sim.sample_tree_counts(density, self.np_random)
            self._fuel[1] = (tree_counts * self._sim.fuel_per_tree).astype(np.float32)

        # Each fuel type starts at ambient air temperature.
        self._fuel_temperatures = np.tile(self._air_temperatures[np.newaxis], (N, 1, 1)).astype(np.float32)
        if self._spawn_fire:
            r = int(self.np_random.integers(0, self.grid_size))
            c = int(self.np_random.integers(0, self.grid_size))
            self._spawn_fire_patch(r, c)
        # Column mass (surface pressure), initialised at its leveled rest state so the
        # mass field starts balanced (uses the clean elevation temperature profile).
        self._mass = self._sim.equilibrium_mass(self._terrain, self._temp_eq)
        self._x_wind_vel, self._y_wind_vel = self._sim.compute_wind_from_pressure(self._mass)
        self._reference_wind_magnitude = float(
            np.sqrt(self._x_wind_vel ** 2 + self._y_wind_vel ** 2).max()
        )
        # Oxygen also thins exponentially with elevation (1.0 at sea level).
        self._oxygen = self._sim.elevation_falloff(self._terrain, self._sim.oxygen_lapse_rate)
        # Oxygen replenishment target and the elevation-based temperature rate field
        # (static within an episode, cached once).
        self._oxygen_eq = self._oxygen.copy()
        self._temp_relax_rate = self._sim.temperature_relax_rate(self._terrain)
        self._last_radiant_flux = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        # Seed the display scales from the initial state so the first frames are scaled sanely.
        self._fuel_temp_display_scale = _DISPLAY_SCALE_MAX_FRACTION * float(self._fuel_temperatures.max())
        self._radiant_flux_display_scale = _DISPLAY_SCALE_MAX_FRACTION * float(self._last_radiant_flux.max())
        self._step_count = 0
        self._running = True
        self._paused = False
        self._reset_requested = False
        self._step_once = False
        self._surfaces_dirty = True

        return self._air_temperatures.copy(), self._build_info()

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        # Convective transport: mass and energy (E = m*T) flow together down the
        # transport potential Phi = c_p*m + c_b*H, so heat is carried by the air mass
        # (convection) with no compression term and no gravity waves. Temperature is the
        # diagnostic T = E/m recovered afterward.
        energy = self._mass * self._air_temperatures
        self._mass, energy = self._sim.transport_mass_energy(
            self._mass, energy, self._terrain, self._air_temperatures
        )
        self._air_temperatures = (energy / np.maximum(self._mass, 1e-6)).astype(np.float32)
        # Surface radiative relaxation (sun warms, space cools); a sink/source forcing.
        if self._sim.relaxation_enabled:
            self._air_temperatures = self._sim.relax_to_equilibrium(
                self._air_temperatures, self._temp_eq, self._temp_relax_rate
            )
        self._air_temperatures, self._fuel_temperatures = self._sim.exchange_fuel_air_heat(
            self._air_temperatures, self._fuel_temperatures, self._fuel
        )
        self._oxygen = self._sim.diffuse_and_advect_oxygen(
            self._oxygen, self._x_wind_vel, self._y_wind_vel, self._reference_wind_magnitude
        )
        # Fresh-air replenishment toward the elevation oxygen profile (combustion below
        # still draws this down, so a vigorous fire can outpace it locally).
        if self._sim.relaxation_enabled:
            self._oxygen = self._sim.relax_to_equilibrium(
                self._oxygen, self._oxygen_eq, self._sim.oxygen_rate
            )
        if self._sim.fire_enabled:
            self._air_temperatures, self._fuel_temperatures, self._fuel, self._oxygen = self._sim.update_fire(
                self._air_temperatures, self._fuel_temperatures, self._fuel, self._oxygen
            )
            self._fuel_temperatures, self._last_radiant_flux = self._sim.apply_radiant_heat(
                self._fuel_temperatures, self._fuel, self._oxygen
            )
        # Refresh the surface wind (-grad pressure) for display and next tick's oxygen
        # advection. The wind is purely diagnostic; mass moves via the transport above.
        self._x_wind_vel, self._y_wind_vel = self._sim.update_wind(
            self._mass, self._x_wind_vel, self._y_wind_vel
        )
        self._reference_wind_magnitude = float(
            np.sqrt(self._x_wind_vel ** 2 + self._y_wind_vel ** 2).max()
        )
        # Track EMA-smoothed peaks (toward a fraction of the max) so the mode 6 / mode 7
        # color scales don't flicker.
        a = _DISPLAY_SCALE_EMA_ALPHA
        f = _DISPLAY_SCALE_MAX_FRACTION
        self._fuel_temp_display_scale = (
            a * f * float(self._fuel_temperatures.max()) + (1.0 - a) * self._fuel_temp_display_scale
        )
        self._radiant_flux_display_scale = (
            a * f * float(self._last_radiant_flux.max()) + (1.0 - a) * self._radiant_flux_display_scale
        )
        self._step_count += 1
        self._surfaces_dirty = True

        reward = 0.0
        terminated = False
        truncated = self._step_count >= self.max_steps
        return self._air_temperatures.copy(), reward, terminated, truncated, self._build_info()

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
        temp = self._air_temperatures
        wx, wy = self._x_wind_vel, self._y_wind_vel
        pressure_field = self._mass
        wind_ref = float(np.sqrt(wx ** 2 + wy ** 2).max())

        self._color_surface = build_color_surface(temp, self._pixel_scale)
        self._wind_surface = build_wind_surface(
            wx, wy, temp, self._pixel_scale, wind_ref,
        )
        self._fire_surface = build_fire_surface(
            self._fuel_temperatures, self._fuel, self._oxygen, self._pixel_scale,
            self._sim.ignition_thresholds, self._sim.fuel_burnt_threshold,
            self._sim.oxygen_extinction_threshold,
            self._show_fire_overlay and self._sim.fire_enabled,
        )
        self._oxygen_surface = build_oxygen_surface(self._oxygen, self._pixel_scale)
        self._pressure_surface = build_pressure_surface(pressure_field, self._pixel_scale)
        self._fuel_temperature_surface = build_color_surface(
            self._fuel_temperatures.max(axis=0), self._pixel_scale,
            upper_bound=self._fuel_temp_display_scale,
        )
        self._radiant_flux_surface = build_radiant_heat_surface(
            self._last_radiant_flux, self._pixel_scale,
            upper_bound=self._radiant_flux_display_scale,
        )
        # Static after reset, but rebuilt with the batch for consistency (cost is negligible).
        self._terrain_surface = build_terrain_surface(self._terrain, self._pixel_scale)
        self._column_height_surface = build_column_height_surface(
            self._sim.column_height(self._mass, self._terrain, self._air_temperatures),
            self._pixel_scale,
        )
        self._surfaces_dirty = False

    def _spawn_fire_patch(self, row: int, col: int) -> None:
        rows_idx, cols_idx = np.ogrid[:self.grid_size, :self.grid_size]
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
        wind_speeds = np.sqrt(self._x_wind_vel ** 2 + self._y_wind_vel ** 2)
        ign = self._sim.ignition_thresholds[:, np.newaxis, np.newaxis]
        burning_per_type = (
            (self._fuel_temperatures >= ign) &
            (self._fuel > self._sim.fuel_burnt_threshold) &
            (self._oxygen[np.newaxis] > self._sim.oxygen_extinction_threshold)
        ) if self._sim.fire_enabled else np.zeros_like(self._fuel, dtype=bool)
        info = {
            "air_temperature": self._sim.compute_air_temperature(self._air_temperatures),
            "air_temp_min":    float(self._air_temperatures.min()),
            "air_temp_max":    float(self._air_temperatures.max()),
            "air_temp_std":    float(self._air_temperatures.std()),
            "wind_mean":       float(wind_speeds.mean()),
            "wind_max":        float(wind_speeds.max()),
            "wind_std":        float(wind_speeds.std()),
            "oxygen_mean":     float(self._oxygen.mean()),
            "oxygen_min":      float(self._oxygen.min()),
            "cells_burning":   int(burning_per_type.any(axis=0).sum()),
            "terrain_mean":    float(self._terrain.mean()),
            "terrain_min":     float(self._terrain.min()),
            "terrain_max":     float(self._terrain.max()),
            "step":            self._step_count,
        }
        for i, name in enumerate(self._sim.fuel_type_names):
            info[f"fuel_{name}_mean"]      = float(self._fuel[i].mean())
            info[f"fuel_{name}_max"]       = float(self._fuel[i].max())
            info[f"fuel_temp_{name}_mean"] = float(self._fuel_temperatures[i].mean())
            info[f"fuel_temp_{name}_max"]  = float(self._fuel_temperatures[i].max())
            info[f"cells_burning_{name}"]  = int(burning_per_type[i].sum())
        return info
