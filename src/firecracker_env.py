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

from simulation import (
    create_grid,
    compute_wind_from_temperature,
    diffuse_and_advect,
    compute_temperature,
    WIND_TEMPORAL_SMOOTHING,
)

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

# ---------------------------------------------------------------------------
# View modes
# ---------------------------------------------------------------------------

class ViewMode(Enum):
    TEMPERATURE = 1
    WIND = 2


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

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class FirecrackerEnv(gymnasium.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": TARGET_FPS}

    def __init__(
        self,
        grid_size: int = DEFAULT_GRID_SIZE,
        window_size: int = DEFAULT_WINDOW_SIZE,
        render_mode: str | None = None,
        max_steps: int = 1000,
    ):
        super().__init__()
        assert render_mode is None or render_mode in self.metadata["render_modes"], (
            f"render_mode must be one of {self.metadata['render_modes']} or None"
        )

        self.grid_size = grid_size
        self.window_size = window_size
        self.render_mode = render_mode
        self.max_steps = max_steps
        self._pixel_scale = window_size // grid_size

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(grid_size, grid_size), dtype=np.float32
        )
        # Single no-op action; replace with the real action space when designing the agent.
        self.action_space = spaces.Discrete(1)

        # Simulation state (populated by reset)
        self._temperatures: np.ndarray | None = None
        self._x_wind_vel: np.ndarray | None = None
        self._y_wind_vel: np.ndarray | None = None
        self._reference_wind_magnitude: float = 0.0
        self._step_count: int = 0

        # Rendering state
        self._screen: pygame.Surface | None = None
        self._clock: pygame.time.Clock | None = None
        self._color_surface: pygame.Surface | None = None
        self._wind_surface: pygame.Surface | None = None
        self._surfaces_dirty: bool = True
        self._current_mode: ViewMode = ViewMode.TEMPERATURE
        self._running: bool = True

        if render_mode == "human":
            pygame.init()
            pygame.display.set_caption(WINDOW_TITLE)
            self._screen = pygame.display.set_mode(
                (grid_size * self._pixel_scale, grid_size * self._pixel_scale)
            )
            self._clock = pygame.time.Clock()

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        self._temperatures = create_grid(self.grid_size)
        self._x_wind_vel, self._y_wind_vel = compute_wind_from_temperature(self._temperatures)
        self._reference_wind_magnitude = float(
            np.sqrt(self._x_wind_vel ** 2 + self._y_wind_vel ** 2).max()
        )
        self._step_count = 0
        self._running = True
        self._surfaces_dirty = True

        return self._temperatures.copy(), self._build_info()

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        self._temperatures = diffuse_and_advect(
            self._temperatures, self._x_wind_vel, self._y_wind_vel, self._reference_wind_magnitude
        )
        new_x_wind, new_y_wind = compute_wind_from_temperature(self._temperatures)
        self._x_wind_vel = (
            WIND_TEMPORAL_SMOOTHING * new_x_wind
            + (1.0 - WIND_TEMPORAL_SMOOTHING) * self._x_wind_vel
        )
        self._y_wind_vel = (
            WIND_TEMPORAL_SMOOTHING * new_y_wind
            + (1.0 - WIND_TEMPORAL_SMOOTHING) * self._y_wind_vel
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
            surface = (
                self._color_surface
                if self._current_mode == ViewMode.TEMPERATURE
                else self._wind_surface
            )
            self._screen.blit(surface, (0, 0))
            pygame.display.flip()
            self._clock.tick(TARGET_FPS)
            return None

        if self.render_mode == "rgb_array":
            if not pygame.get_init():
                pygame.init()
            self._rebuild_surfaces_if_dirty()
            surface = (
                self._color_surface
                if self._current_mode == ViewMode.TEMPERATURE
                else self._wind_surface
            )
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

    def _rebuild_surfaces_if_dirty(self) -> None:
        if not self._surfaces_dirty:
            return
        self._color_surface = build_color_surface(self._temperatures, self._pixel_scale)
        self._wind_surface = build_wind_surface(
            self._x_wind_vel, self._y_wind_vel,
            self._pixel_scale, self._reference_wind_magnitude,
        )
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
                elif event.key in MODE_KEYS:
                    mode = MODE_KEYS[event.key]
        return running, mode

    def _build_info(self) -> dict:
        wind_speeds = np.sqrt(self._x_wind_vel ** 2 + self._y_wind_vel ** 2)
        return {
            "temperature": compute_temperature(self._temperatures),
            "temp_min": float(self._temperatures.min()),
            "temp_max": float(self._temperatures.max()),
            "temp_std": float(self._temperatures.std()),
            "wind_mean": float(wind_speeds.mean()),
            "wind_max": float(wind_speeds.max()),
            "wind_std": float(wind_speeds.std()),
            "step": self._step_count,
        }
