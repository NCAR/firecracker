"""
main.py

Displays an NxN grid of pixels using pygame, where each pixel value is a float
in [0, 1] sourced from a 2D numpy array. Zero maps to black, one maps to pure
red, and intermediate values are proportional shades of red.

The grid is initialized with Perlin noise and then evolved in real time using
repeated Gaussian blurring to simulate heat diffusion. The mean pixel value
(system temperature) is conserved across every diffusion step.
"""

import math
import sys
from enum import Enum
import numpy as np
import pygame
import noise
from scipy.ndimage import gaussian_filter

# ---------------------------------------------------------------------------
# View modes
# ---------------------------------------------------------------------------

class ViewMode(Enum):
    """
    Selectable display modes cycled via number keys at runtime.

    Each member's integer value corresponds to the key used to activate it
    (e.g. ViewMode.TEMPERATURE is activated by pressing 1).  Add new members
    here and a matching branch in render_frame to extend the system.
    """
    TEMPERATURE = 1
    WIND = 2


# Maps pygame key constants to their corresponding ViewMode.
# Populated from ViewMode values so new modes are picked up automatically.
MODE_KEYS: dict[int, ViewMode] = {
    getattr(pygame, f"K_{mode.value}"): mode
    for mode in ViewMode
}


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_GRID_SIZE: int = 256      # Default side length of the square grid (N)
DEFAULT_WINDOW_SIZE: int = 512    # Default side length of the square window in pixels
WINDOW_TITLE: str = "Firecracker" # Title shown in the pygame window title bar
TARGET_FPS: int = 60              # Frame-rate cap (rendering; decoupled from simulation)

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------

# Number of diffusion steps advanced per second.
SIMULATION_STEPS_PER_SECOND: int = 10

# Milliseconds between diffusion steps (derived from above).
MS_PER_STEP: int = 1000 // SIMULATION_STEPS_PER_SECOND

# Standard deviation (in grid cells) of the Gaussian kernel used for diffusion.
# Larger values spread heat faster per step.
BLUR_SIGMA: float = 1.0

# Tolerance passed to math.isclose() when checking energy conservation.
ENERGY_CONSERVATION_REL_TOL: float = 1e-4
ENERGY_CONSERVATION_ABS_TOL: float = 1e-6

# Color channel indices within an (R, G, B) tuple
RED_CHANNEL: int = 0
GREEN_CHANNEL: int = 1
BLUE_CHANNEL: int = 2

# Maximum value of an 8-bit color channel
MAX_CHANNEL_VALUE: int = 255

# ---------------------------------------------------------------------------
# Perlin noise parameters
# ---------------------------------------------------------------------------

# Coordinate scale applied before sampling pnoise2.  Smaller values zoom in,
# producing larger, smoother blobs; larger values zoom out for finer detail.
NOISE_SCALE: float = 64.0

# Each additional octave adds a finer layer of detail at the cost of compute.
NOISE_OCTAVES: int = 4

# How quickly each successive octave's amplitude decreases (0–1).
NOISE_PERSISTENCE: float = 0.5

# How quickly each successive octave's frequency increases (>1).
NOISE_LACUNARITY: float = 2.0

# ---------------------------------------------------------------------------
# Wind field parameters
# ---------------------------------------------------------------------------

# Independent noise scale for wind — slightly larger than temperature to
# produce broader, more coherent wind patterns.
WIND_NOISE_SCALE: float = 80.0

WIND_NOISE_OCTAVES: int = 4
WIND_NOISE_PERSISTENCE: float = 0.5
WIND_NOISE_LACUNARITY: float = 2.0

# ---------------------------------------------------------------------------
# Wind arrow visualization parameters
# ---------------------------------------------------------------------------

# Draw one arrow for every this many grid cells in each axis.
WIND_ARROW_STRIDE: int = 16

# Length of each arrow shaft in screen pixels (constant regardless of speed).
WIND_ARROW_LENGTH: int = 14

# Stroke width in pixels for the shaft and arrowhead lines.
WIND_ARROW_LINE_WIDTH: int = 1

# Length of each arrowhead barb in screen pixels.
WIND_ARROW_HEAD_SIZE: int = 4

# Half-angle (radians) between each barb and the shaft at the arrow tip.
WIND_ARROW_HEAD_ANGLE: float = math.pi / 6  # 30 degrees


# ---------------------------------------------------------------------------
# Grid initialisation
# ---------------------------------------------------------------------------

def sample_perlin_grid(
    size: int,
    base: int,
    scale: float = NOISE_SCALE,
    octaves: int = NOISE_OCTAVES,
    persistence: float = NOISE_PERSISTENCE,
    lacunarity: float = NOISE_LACUNARITY,
) -> np.ndarray:
    """
    Sample 2D Perlin noise on an NxN grid and return raw (un-normalized) values.

    Each cell (row, col) is sampled at coordinates (col / scale, row / scale)
    so that adjacent cells are spatially coherent.  The scale parameter
    controls the apparent size of features: larger values produce bigger blobs.

    Parameters
    ----------
    size : int
        Side length N of the square grid.
    base : int
        Seed offset passed to pnoise2; changing this produces a different map.
    scale : float, optional
        Coordinate divisor controlling feature size.  Defaults to NOISE_SCALE.
    octaves : int, optional
        Number of noise layers.  Defaults to NOISE_OCTAVES.
    persistence : float, optional
        Amplitude falloff per octave.  Defaults to NOISE_PERSISTENCE.
    lacunarity : float, optional
        Frequency multiplier per octave.  Defaults to NOISE_LACUNARITY.

    Returns
    -------
    np.ndarray
        2-D array of shape (size, size) with dtype float32 containing raw
        Perlin noise values (range approximately [-0.5, 0.5] for one octave).
    """
    grid = np.empty((size, size), dtype=np.float32)
    for row in range(size):
        for col in range(size):
            grid[row, col] = noise.pnoise2(
                col / scale,
                row / scale,
                octaves=octaves,
                persistence=persistence,
                lacunarity=lacunarity,
                base=base,
            )
    return grid


def normalize_grid(grid: np.ndarray) -> np.ndarray:
    """
    Linearly rescale all values in *grid* to the range [0, 1].

    Uses the grid's own observed minimum and maximum so the full color range
    is always utilized regardless of the raw noise amplitude.

    Parameters
    ----------
    grid : np.ndarray
        2-D float32 array of arbitrary value range.

    Returns
    -------
    np.ndarray
        A new float32 array of the same shape with values in [0, 1].
    """
    lo = grid.min()
    hi = grid.max()
    # Guard against a degenerate flat grid (all values identical).
    if hi == lo:
        return np.zeros_like(grid)
    return ((grid - lo) / (hi - lo)).astype(np.float32)


def normalize_to_symmetric(grid: np.ndarray) -> np.ndarray:
    """
    Linearly rescale all values in *grid* to the range [-1, 1].

    Used for signed quantities such as wind velocity components, where the sign
    encodes direction and the magnitude encodes speed.

    Parameters
    ----------
    grid : np.ndarray
        2-D float32 array of arbitrary value range.

    Returns
    -------
    np.ndarray
        A new float32 array of the same shape with values in [-1, 1].
    """
    return (normalize_grid(grid) * 2.0 - 1.0).astype(np.float32)


def create_grid(size: int) -> np.ndarray:
    """
    Build an NxN float32 grid of Perlin noise values normalized to [0, 1].

    A random seed is chosen each call so every run produces a different map.
    Values near 0 represent cold (black) regions; values near 1 represent hot
    (red) regions, with spatially coherent variation between them.

    Parameters
    ----------
    size : int
        Side length N of the square grid.

    Returns
    -------
    np.ndarray
        2-D array of shape (size, size) with dtype float32, values in [0, 1].
    """
    base = np.random.randint(0, 256)
    raw = sample_perlin_grid(size, base)
    return normalize_grid(raw)


def create_wind_field(size: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Build x and y wind velocity arrays from independent Perlin noise maps.

    Each component is sampled with its own random seed so the x and y fields
    are uncorrelated.  Values are normalized to [-1, 1]: negative x means wind
    blowing west, positive x means east; negative y means north, positive y
    means south.

    Parameters
    ----------
    size : int
        Side length N of the square grid.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A (x_wind_vel, y_wind_vel) pair, each a float32 array of shape
        (size, size) with values in [-1, 1].
    """
    base_x = np.random.randint(0, 256)
    base_y = np.random.randint(0, 256)
    raw_x = sample_perlin_grid(
        size, base_x,
        scale=WIND_NOISE_SCALE,
        octaves=WIND_NOISE_OCTAVES,
        persistence=WIND_NOISE_PERSISTENCE,
        lacunarity=WIND_NOISE_LACUNARITY,
    )
    raw_y = sample_perlin_grid(
        size, base_y,
        scale=WIND_NOISE_SCALE,
        octaves=WIND_NOISE_OCTAVES,
        persistence=WIND_NOISE_PERSISTENCE,
        lacunarity=WIND_NOISE_LACUNARITY,
    )
    return normalize_to_symmetric(raw_x), normalize_to_symmetric(raw_y)


# ---------------------------------------------------------------------------
# Heat diffusion
# ---------------------------------------------------------------------------

def compute_temperature(grid: np.ndarray) -> float:
    """
    Return the mean pixel value of the grid, treated as the system temperature.

    Parameters
    ----------
    grid : np.ndarray
        2-D float32 array of shape (rows, cols).

    Returns
    -------
    float
        Arithmetic mean of all pixel values.
    """
    return float(grid.mean())


def diffuse(grid: np.ndarray) -> np.ndarray:
    """
    Advance the simulation by one heat-diffusion step via Gaussian blurring.

    Border padding is set to the current system temperature so that edge pixels
    exchange heat with a virtual environment at the mean value rather than with
    zeros or reflections, preserving the total energy of the system.

    Parameters
    ----------
    grid : np.ndarray
        2-D float32 array of shape (rows, cols) with values in [0, 1].

    Returns
    -------
    np.ndarray
        A new float32 array of the same shape after one diffusion step.
    """
    temperature = compute_temperature(grid)
    blurred = gaussian_filter(
        grid,
        sigma=BLUR_SIGMA,
        mode='constant',
        cval=temperature,
    ).astype(np.float32)
    # Constant padding causes asymmetric energy exchange at the borders: hot
    # edge pixels leak more energy to the phantom region than cold ones, so the
    # mean drifts unless the edge values happen to cancel exactly.  Adding a
    # uniform offset restores the mean without distorting the spatial structure.
    correction = temperature - compute_temperature(blurred)
    return (blurred + correction).astype(np.float32)


def check_energy_conservation(temp_before: float, temp_after: float) -> None:
    """
    Print a WARNING if the system temperature changed between diffusion steps.

    Floating-point tolerances are used so that harmless rounding noise does not
    trigger false alarms.

    Parameters
    ----------
    temp_before : float
        Mean pixel value before the diffusion step.
    temp_after : float
        Mean pixel value after the diffusion step.
    """
    if not math.isclose(
        temp_before,
        temp_after,
        rel_tol=ENERGY_CONSERVATION_REL_TOL,
        abs_tol=ENERGY_CONSERVATION_ABS_TOL,
    ):
        print(
            f"WARNING: Energy not conserved — temperature changed from "
            f"{temp_before:.8f} to {temp_after:.8f} "
            f"(delta={temp_after - temp_before:.2e})"
        )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

# Header/footer bar drawn around the environment info block.
INFO_SEPARATOR: str = "-" * 36


def print_environment_info(grid: np.ndarray, label: str, window_size: int) -> None:
    """
    Print a labeled summary of the current simulation state to stdout.

    Displays the grid dimensions, window resolution, simulation rate, and key
    statistics of the pixel values (temperature, range, and spread).  Called
    at startup and shutdown so the initial and final states are both visible.

    Parameters
    ----------
    grid : np.ndarray
        Current 2-D float32 grid of shape (rows, cols).
    label : str
        Short context label shown in the header (e.g. "Startup", "Shutdown").
    window_size : int
        Side length of the square display window in pixels.
    """
    rows, cols = grid.shape
    print(INFO_SEPARATOR)
    print(f" {label}")
    print(INFO_SEPARATOR)
    print(f"  Grid size   : {rows} x {cols}  ({rows * cols} cells)")
    print(f"  Window size : {window_size} x {window_size} px")
    print(f"  Sim rate    : {SIMULATION_STEPS_PER_SECOND} steps/sec")
    print(f"  Temperature : {compute_temperature(grid):.6f}")
    print(f"  Min value   : {float(grid.min()):.6f}")
    print(f"  Max value   : {float(grid.max()):.6f}")
    print(f"  Std dev     : {float(grid.std()):.6f}")
    print(INFO_SEPARATOR)


# ---------------------------------------------------------------------------
# Color mapping
# ---------------------------------------------------------------------------

def value_to_color(value: float) -> tuple[int, int, int]:
    """
    Map a scalar in [0, 1] to an (R, G, B) color.

    The mapping is linear: 0 produces black (0, 0, 0) and 1 produces full red
    (255, 0, 0).  Values between these extremes are intermediate shades of red.

    Parameters
    ----------
    value : float
        Scalar intensity, expected to lie in [0, 1].  Values outside this range
        are clamped silently by the integer cast (negative → 0, >1 → >255 but
        visually saturated by pygame).

    Returns
    -------
    tuple[int, int, int]
        An (R, G, B) tuple suitable for pygame drawing calls.
    """
    red = int(value * MAX_CHANNEL_VALUE)
    return (red, 0, 0)


def build_color_surface(grid: np.ndarray, scale: int) -> pygame.Surface:
    """
    Construct a pygame Surface that visualizes every cell in *grid* as a
    solid-colored square of side *scale* pixels.

    Parameters
    ----------
    grid : np.ndarray
        2-D float32 array of shape (rows, cols) with values in [0, 1].
    scale : int
        Number of screen pixels per grid cell on each axis.

    Returns
    -------
    pygame.Surface
        A surface whose pixel dimensions are (cols * scale, rows * scale).
    """
    rows, cols = grid.shape
    surface = pygame.Surface((cols * scale, rows * scale))

    for row in range(rows):
        for col in range(cols):
            color = value_to_color(grid[row, col])
            rect = cell_rect(row, col, scale)
            pygame.draw.rect(surface, color, rect)

    return surface


def build_wind_surface(
    x_wind_vel: np.ndarray,
    y_wind_vel: np.ndarray,
    scale: int,
) -> pygame.Surface:
    """
    Construct a pygame Surface that visualizes the wind velocity field as arrows.

    One arrow is drawn every WIND_ARROW_STRIDE cells in each axis to avoid
    clutter.  Each arrow is centered on its cell and points in the wind
    direction.  Shaft length is constant (WIND_ARROW_LENGTH pixels); brightness
    encodes speed, with brighter gray meaning faster wind.

    Parameters
    ----------
    x_wind_vel : np.ndarray
        2-D float32 array of x velocity components, values in [-1, 1].
    y_wind_vel : np.ndarray
        2-D float32 array of y velocity components, values in [-1, 1].
    scale : int
        Number of screen pixels per grid cell on each axis.

    Returns
    -------
    pygame.Surface
        A black surface with gray arrows, pixel dimensions (cols*scale, rows*scale).
    """
    rows, cols = x_wind_vel.shape
    surface = pygame.Surface((cols * scale, rows * scale))  # black by default

    # Maximum possible magnitude when both components are ±1.
    max_magnitude = math.sqrt(2.0)
    half_len = WIND_ARROW_LENGTH / 2.0

    # Start half a stride from each edge so the arrow grid is centered and
    # the margin is equal on opposite sides.
    grid_offset = WIND_ARROW_STRIDE // 2

    for row in range(grid_offset, rows, WIND_ARROW_STRIDE):
        for col in range(grid_offset, cols, WIND_ARROW_STRIDE):
            vx = float(x_wind_vel[row, col])
            vy = float(y_wind_vel[row, col])
            magnitude = math.sqrt(vx ** 2 + vy ** 2)

            if magnitude == 0.0:
                continue

            brightness = int(magnitude / max_magnitude * MAX_CHANNEL_VALUE)
            color = (brightness, brightness, brightness)

            # Unit direction vector.
            ux = vx / magnitude
            uy = vy / magnitude

            # Cell center in screen pixels.
            cx = (col + 0.5) * scale
            cy = (row + 0.5) * scale

            # Shaft: centered on the cell.
            start = (int(cx - ux * half_len), int(cy - uy * half_len))
            tip   = (int(cx + ux * half_len), int(cy + uy * half_len))
            pygame.draw.line(surface, color, start, tip, WIND_ARROW_LINE_WIDTH)

            # Arrowhead: two barbs angled back from the tip.
            angle = math.atan2(uy, ux)
            for sign in (+1, -1):
                barb = (
                    int(tip[0] - WIND_ARROW_HEAD_SIZE * math.cos(angle + sign * WIND_ARROW_HEAD_ANGLE)),
                    int(tip[1] - WIND_ARROW_HEAD_SIZE * math.sin(angle + sign * WIND_ARROW_HEAD_ANGLE)),
                )
                pygame.draw.line(surface, color, tip, barb, WIND_ARROW_LINE_WIDTH)

    return surface


def cell_rect(row: int, col: int, scale: int) -> pygame.Rect:
    """
    Return the pygame.Rect that corresponds to a single grid cell.

    Parameters
    ----------
    row : int
        Row index of the cell (0-based, top-to-bottom).
    col : int
        Column index of the cell (0-based, left-to-right).
    scale : int
        Number of screen pixels per grid cell on each axis.

    Returns
    -------
    pygame.Rect
        Rect with position (col * scale, row * scale) and size (scale, scale).
    """
    x = col * scale
    y = row * scale
    return pygame.Rect(x, y, scale, scale)


# ---------------------------------------------------------------------------
# Window / display helpers
# ---------------------------------------------------------------------------

def create_window(grid_size: int, scale: int, title: str) -> pygame.Surface:
    """
    Initialize pygame and open a display window sized to fit the grid.

    Parameters
    ----------
    grid_size : int
        Side length N of the square grid.
    scale : int
        Number of screen pixels per grid cell on each axis.
    title : str
        Text shown in the window title bar.

    Returns
    -------
    pygame.Surface
        The main display surface returned by pygame.display.set_mode.
    """
    pygame.display.set_caption(title)
    window_size = (grid_size * scale, grid_size * scale)
    return pygame.display.set_mode(window_size)


def handle_events(current_mode: ViewMode) -> tuple[bool, ViewMode]:
    """
    Process the pygame event queue for the current frame.

    Parameters
    ----------
    current_mode : ViewMode
        The active view mode at the start of this frame.

    Returns
    -------
    tuple[bool, ViewMode]
        A (running, mode) pair.  running is False if a quit event was received;
        mode is the updated ViewMode after processing any number-key presses.
    """
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False, current_mode
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return False, current_mode
            if event.key in MODE_KEYS:
                current_mode = MODE_KEYS[event.key]
    return True, current_mode


def render_frame(
    screen: pygame.Surface,
    mode: ViewMode,
    color_surface: pygame.Surface,
    wind_surface: pygame.Surface,
) -> None:
    """
    Draw one frame to *screen* according to the active *mode*.

    Parameters
    ----------
    screen : pygame.Surface
        The main display surface to draw onto.
    mode : ViewMode
        The currently active view mode.
    color_surface : pygame.Surface
        Pre-built temperature heat-map surface (used by TEMPERATURE mode).
    wind_surface : pygame.Surface
        Pre-built wind velocity surface (used by WIND mode).
    """
    if mode == ViewMode.TEMPERATURE:
        screen.blit(color_surface, (0, 0))
    elif mode == ViewMode.WIND:
        screen.blit(wind_surface, (0, 0))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(grid_size: int = DEFAULT_GRID_SIZE, window_size: int = DEFAULT_WINDOW_SIZE) -> None:
    """
    Initialize the grid and pygame, then enter the main display loop.

    The simulation advances at SIMULATION_STEPS_PER_SECOND via Gaussian
    blurring.  The color surface is rebuilt after every diffusion step and
    energy conservation is verified each step.  The window is displayed until
    the user closes it or presses Escape.

    The pixel scale (screen pixels per grid cell) is derived by dividing
    window_size by grid_size using integer division, so the actual rendered
    window may be slightly smaller than window_size if it is not evenly
    divisible.

    Parameters
    ----------
    grid_size : int, optional
        Side length N of the square grid.  Defaults to DEFAULT_GRID_SIZE.
    window_size : int, optional
        Desired side length of the display window in pixels.  Defaults to
        DEFAULT_WINDOW_SIZE.
    """
    pixel_scale = window_size // grid_size

    pygame.init()

    grid = create_grid(grid_size)
    x_wind_vel, y_wind_vel = create_wind_field(grid_size)
    print_environment_info(grid, "Startup", window_size)
    screen = create_window(grid_size, pixel_scale, WINDOW_TITLE)
    clock = pygame.time.Clock()
    color_surface = build_color_surface(grid, pixel_scale)
    wind_surface = build_wind_surface(x_wind_vel, y_wind_vel, pixel_scale)

    last_step_ms: int = pygame.time.get_ticks()
    current_mode: ViewMode = ViewMode.TEMPERATURE

    running = True
    while running:
        running, current_mode = handle_events(current_mode)

        current_ms = pygame.time.get_ticks()
        if current_ms - last_step_ms >= MS_PER_STEP:
            temp_before = compute_temperature(grid)
            grid = diffuse(grid)
            temp_after = compute_temperature(grid)
            check_energy_conservation(temp_before, temp_after)
            color_surface = build_color_surface(grid, pixel_scale)
            last_step_ms = current_ms

        render_frame(screen, current_mode, color_surface, wind_surface)
        pygame.display.flip()
        clock.tick(TARGET_FPS)

    print_environment_info(grid, "Shutdown", window_size)
    pygame.quit()


if __name__ == "__main__":
    # Optional positional arguments:  python main.py [N [window_size]]
    try:
        n = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_GRID_SIZE
        w = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_WINDOW_SIZE
    except ValueError:
        print("Usage: python main.py [N [window_size]]  (both must be positive integers)")
        sys.exit(1)
    run(grid_size=n, window_size=w)
