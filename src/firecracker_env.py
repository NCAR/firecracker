"""
firecracker_env.py

Gymnasium environment wrapping the Firecracker heat-diffusion simulation.
Observation: (grid_size, grid_size) float32 temperature field in [0, 1].
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

WIND_ARROW_STRIDE: int = 16
WIND_ARROW_LENGTH: int = 14
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


# Populated from ViewMode values so new modes are picked up automatically.
MODE_KEYS: dict[int, ViewMode] = {
    getattr(pygame, f"K_{mode.value}"): mode
    for mode in ViewMode
}

# ---------------------------------------------------------------------------
# Surface builders
# ---------------------------------------------------------------------------

def build_color_surface(temperatures: np.ndarray, scale: int) -> pygame.Surface:
    rows, cols = temperatures.shape
    red = (np.clip(temperatures, 0.0, 1.0) * MAX_CHANNEL_VALUE).astype(np.uint8)
    red_scaled = np.repeat(np.repeat(red, scale, axis=0), scale, axis=1)
    rgb = np.zeros((rows * scale, cols * scale, 3), dtype=np.uint8)
    rgb[:, :, RED_CHANNEL] = red_scaled
    surface = pygame.Surface((cols * scale, rows * scale))
    # surfarray expects (width, height, 3); numpy is (height, width, 3)
    pygame.surfarray.blit_array(surface, rgb.transpose(1, 0, 2))
    return surface


def build_wind_surface(
    x_wind_vel: np.ndarray,
    y_wind_vel: np.ndarray,
    scale: int,
    reference_magnitude: float,
) -> pygame.Surface:
    rows, cols = x_wind_vel.shape
    surface = pygame.Surface((cols * scale, rows * scale))

    if reference_magnitude == 0.0:
        return surface

    magnitudes = np.sqrt(x_wind_vel ** 2 + y_wind_vel ** 2)
    half_len = WIND_ARROW_LENGTH / 2.0
    grid_offset = WIND_ARROW_STRIDE // 2

    for row in range(grid_offset, rows, WIND_ARROW_STRIDE):
        for col in range(grid_offset, cols, WIND_ARROW_STRIDE):
            vx = float(x_wind_vel[row, col])
            vy = float(y_wind_vel[row, col])
            magnitude = float(magnitudes[row, col])
            if magnitude == 0.0:
                continue

            brightness = int(min(magnitude / reference_magnitude, 1.0) * MAX_CHANNEL_VALUE)
            color = (brightness, brightness, brightness)
            ux, uy = vx / magnitude, vy / magnitude
            cx = (col + 0.5) * scale
            cy = (row + 0.5) * scale

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


_COLOR_BURNT    = (  0,   0,   0)
_COLOR_BURNING  = (255, 140,   0)
_COLOR_DAMAGED  = (  0,  60,   0)
_COLOR_HEALTHY  = (  0, 100,   0)


def build_fire_surface(
    temperatures: np.ndarray,
    fuel: np.ndarray,
    damaged: np.ndarray,
    scale: int,
    ignition_threshold: float,
    fuel_burnt_threshold: float,
) -> pygame.Surface:
    rows, cols = temperatures.shape
    rgb = np.zeros((rows, cols, 3), dtype=np.uint8)

    burnt    = fuel <= fuel_burnt_threshold
    burning  = ~burnt & (temperatures >= ignition_threshold)
    damaged_ = ~burnt & ~burning & (damaged == 1)
    healthy  = ~burnt & ~burning & (damaged == 0)

    rgb[healthy]  = _COLOR_HEALTHY
    rgb[damaged_] = _COLOR_DAMAGED
    rgb[burning]  = _COLOR_BURNING
    # burnt stays (0, 0, 0)

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
        self.render_mode = render_mode
        self._pixel_scale = self.window_size // self.grid_size

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.grid_size, self.grid_size), dtype=np.float32
        )
        # Single no-op action; replace with the real action space when designing the agent.
        self.action_space = spaces.Discrete(1)

        # Simulation state (populated by reset)
        self._temperatures: np.ndarray | None = None
        self._x_wind_vel: np.ndarray | None = None
        self._y_wind_vel: np.ndarray | None = None
        self._reference_wind_magnitude: float = 0.0
        self._fuel: np.ndarray | None = None
        self._damaged: np.ndarray | None = None
        self._oxygen: np.ndarray | None = None
        self._step_count: int = 0

        # Rendering state
        self._screen: pygame.Surface | None = None
        self._clock: pygame.time.Clock | None = None
        self._color_surface: pygame.Surface | None = None
        self._wind_surface: pygame.Surface | None = None
        self._fire_surface: pygame.Surface | None = None
        self._oxygen_surface: pygame.Surface | None = None
        self._surfaces_dirty: bool = True
        self._current_mode: ViewMode = ViewMode.TEMPERATURE
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
        temp_raw = self._sim.create_grid(self.grid_size, base=int(self.np_random.integers(0, 256)))
        self._temperatures = (temp_raw * 0.1).astype(np.float32)
        fuel_raw = self._sim.create_grid(self.grid_size, scale=self._sim.fuel_noise_scale, base=int(self.np_random.integers(0, 256)))
        self._fuel = (fuel_raw * (1.0 - self._sim.fuel_min) + self._sim.fuel_min).astype(np.float32)
        r = int(self.np_random.integers(0, self.grid_size))
        c = int(self.np_random.integers(0, self.grid_size))
        rows_idx, cols_idx = np.ogrid[:self.grid_size, :self.grid_size]
        patch = (rows_idx - r) ** 2 + (cols_idx - c) ** 2 <= self._fire_spawn_radius ** 2
        self._temperatures[patch] = 1.0
        self._fuel[patch] = 1.0
        self._x_wind_vel, self._y_wind_vel = self._sim.compute_wind_from_temperature(self._temperatures)
        self._reference_wind_magnitude = float(
            np.sqrt(self._x_wind_vel ** 2 + self._y_wind_vel ** 2).max()
        )
        self._damaged = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        self._oxygen = np.ones((self.grid_size, self.grid_size), dtype=np.float32)
        self._step_count = 0
        self._running = True
        self._paused = False
        self._reset_requested = False
        self._step_once = False
        self._surfaces_dirty = True

        return self._temperatures.copy(), self._build_info()

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        self._temperatures = self._sim.diffuse_and_advect(
            self._temperatures, self._x_wind_vel, self._y_wind_vel, self._reference_wind_magnitude
        )
        self._temperatures = self._sim.apply_atmospheric_cooling(self._temperatures)
        self._oxygen = self._sim.diffuse_and_advect_oxygen(
            self._oxygen, self._x_wind_vel, self._y_wind_vel, self._reference_wind_magnitude
        )
        self._temperatures, self._fuel, self._damaged, self._oxygen = self._sim.update_fire(
            self._temperatures, self._fuel, self._damaged, self._oxygen
        )
        self._x_wind_vel, self._y_wind_vel = self._sim.update_wind(
            self._temperatures, self._x_wind_vel, self._y_wind_vel
        )
        self._reference_wind_magnitude = float(
            np.sqrt(self._x_wind_vel ** 2 + self._y_wind_vel ** 2).max()
        )
        self._step_count += 1
        self._surfaces_dirty = True

        reward = 0.0
        terminated = False
        truncated = self._step_count >= self.max_steps
        return self._temperatures.copy(), reward, terminated, truncated, self._build_info()

    def render(self) -> np.ndarray | None:
        if self.render_mode == "human":
            self._rebuild_surfaces_if_dirty()
            self._running, self._current_mode = self._handle_events()
            surface = self._surface_for_mode()
            self._screen.blit(surface, (0, 0))
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
        return self._oxygen_surface

    def _rebuild_surfaces_if_dirty(self) -> None:
        if not self._surfaces_dirty:
            return
        self._color_surface = build_color_surface(self._temperatures, self._pixel_scale)
        self._wind_surface = build_wind_surface(
            self._x_wind_vel, self._y_wind_vel,
            self._pixel_scale, self._reference_wind_magnitude,
        )
        self._fire_surface = build_fire_surface(
            self._temperatures, self._fuel, self._damaged, self._pixel_scale,
            self._sim.ignition_threshold, self._sim.fuel_burnt_threshold,
        )
        self._oxygen_surface = build_oxygen_surface(self._oxygen, self._pixel_scale)
        self._surfaces_dirty = False

    def _handle_events(self) -> tuple[bool, ViewMode]:
        running = self._running
        mode = self._current_mode
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    self._paused = not self._paused
                elif event.key == pygame.K_PERIOD:
                    self._step_once = True
                elif event.key == pygame.K_r:
                    self._reset_requested = True
                elif event.key in MODE_KEYS:
                    mode = MODE_KEYS[event.key]
        return running, mode

    def _build_info(self) -> dict:
        wind_speeds = np.sqrt(self._x_wind_vel ** 2 + self._y_wind_vel ** 2)
        burning = (self._temperatures >= self._sim.ignition_threshold) & (self._fuel > self._sim.fuel_burnt_threshold)
        return {
            "temperature": self._sim.compute_temperature(self._temperatures),
            "temp_min": float(self._temperatures.min()),
            "temp_max": float(self._temperatures.max()),
            "temp_std": float(self._temperatures.std()),
            "wind_mean": float(wind_speeds.mean()),
            "wind_max": float(wind_speeds.max()),
            "wind_std": float(wind_speeds.std()),
            "fuel_mean": float(self._fuel.mean()),
            "fuel_min": float(self._fuel.min()),
            "fuel_max": float(self._fuel.max()),
            "cells_burning": int(burning.sum()),
            "cells_burnt": int((self._fuel <= self._sim.fuel_burnt_threshold).sum()),
            "oxygen_mean": float(self._oxygen.mean()),
            "oxygen_min": float(self._oxygen.min()),
            "step": self._step_count,
        }
