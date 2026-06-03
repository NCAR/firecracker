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
from scipy.ndimage import convolve, gaussian_filter, map_coordinates

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

# Standard deviation (in grid cells) of the total diffusion spread per simulation step.
# Larger values spread heat faster per step.
BLUR_SIGMA: float = 1.0

# Number of 3x3 kernel applications used to approximate one full diffusion step.
# Variances add across substeps, so each substep uses sigma = BLUR_SIGMA / sqrt(N).
# More substeps produce a more isotropic result at the cost of N convolutions per tick.
DIFFUSION_SUBSTEPS: int = 4

# Controls the balance between advection and diffusion per substep.  At 0 the
# advection trace is disabled and only diffusion occurs.  At 1 the wind displaces
# heat by BLUR_SIGMA cells per substep at the reference wind speed, making
# advection dominant.  Must be in [0, 1].
WIND_ADVECTION_STRENGTH: float = 0.8

# Standard deviation (in grid cells) of the Gaussian applied to both wind
# components after deriving them from the temperature gradient.  Approximates
# the spatial integration that the pressure Poisson equation performs in a real
# fluid, suppressing cell-scale gradient spikes at sharp temperature fronts.
WIND_SMOOTH_SIGMA: float = 2.0

# Fraction of the newly computed wind field blended in each tick (0, 1].
# The remainder is carried over from the previous tick, giving the wind field
# inertia: direction rotates gradually rather than snapping, and transient
# spikes from passing fronts are damped before they fully materialise.
# Approximates the momentum term in the Navier-Stokes equations.
WIND_TEMPORAL_SMOOTHING: float = 0.2

# Fractional energy correction (relative to mean temperature) above which a
# warning is printed.  0.05 = warn when the correction exceeds 5% of the mean.
ENERGY_CORRECTION_WARN_FRACTION: float = 0.05

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


# ---------------------------------------------------------------------------
# Heat diffusion
# ---------------------------------------------------------------------------

def _build_diffusion_kernel(sigma: float) -> np.ndarray:
    offsets = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    gy, gx = np.meshgrid(offsets, offsets, indexing='ij')
    kernel = np.exp(-(gx ** 2 + gy ** 2) / (2.0 * sigma ** 2))
    return (kernel / kernel.sum()).astype(np.float32)


_DIFFUSION_STEP_SIGMA: float = BLUR_SIGMA / math.sqrt(DIFFUSION_SUBSTEPS)
_DIFFUSION_KERNEL: np.ndarray = _build_diffusion_kernel(_DIFFUSION_STEP_SIGMA)


def _check_energy_correction(correction: float, temperature: float) -> None:
    if temperature > 0.0 and abs(correction) / temperature > ENERGY_CORRECTION_WARN_FRACTION:
        pct = abs(correction) / temperature * 100.0
        print(
            f"WARNING: Large energy correction applied — {correction:.2e} "
            f"({pct:.2f}% of system temperature); wind divergence may be causing "
            f"significant energy non-conservation in the backward-trace advection"
        )


def compute_temperature(temperatures: np.ndarray) -> float:
    """
    Return the mean pixel value of the grid, treated as the system temperature.

    Parameters
    ----------
    temperatures : np.ndarray
        2-D float32 array of shape (rows, cols).

    Returns
    -------
    float
        Arithmetic mean of all pixel values.
    """
    return float(temperatures.mean())


def diffuse_and_advect(
    temperatures: np.ndarray,
    x_wind_vel: np.ndarray,
    y_wind_vel: np.ndarray,
    reference_wind_magnitude: float,
) -> np.ndarray:
    """
    Advance the simulation by one combined advection-diffusion step.

    Runs DIFFUSION_SUBSTEPS interleaved substeps.  Each substep first traces
    each cell backward along the local wind direction by a sub-cell distance
    (bilinear interpolation across up to four neighbours handles fractional and
    diagonal wind), then applies the isotropic 3x3 Gaussian diffusion kernel.
    Long-range heat transport emerges from the chain of local steps rather than
    any single non-local operation.

    The per-substep backward displacement is:
        WIND_ADVECTION_STRENGTH * BLUR_SIGMA / reference_wind_magnitude

    so that at WIND_ADVECTION_STRENGTH=1 and the reference (startup-maximum)
    wind speed each substep traces back exactly BLUR_SIGMA cells.

    A uniform additive correction is applied after all substeps to restore the
    mean temperature exactly, compensating for the small energy asymmetry that
    the non-symmetric backward-trace kernel introduces at the grid boundaries.

    Parameters
    ----------
    temperatures : np.ndarray
        2-D float32 array of shape (rows, cols) with values in [0, 1].
    x_wind_vel : np.ndarray
        2-D float32 array of x (eastward) wind velocity components.
    y_wind_vel : np.ndarray
        2-D float32 array of y (southward) wind velocity components.
    reference_wind_magnitude : float
        Maximum wind speed at simulation startup, used to normalise the
        advection displacement so WIND_ADVECTION_STRENGTH has consistent units.

    Returns
    -------
    np.ndarray
        A new float32 array of the same shape after one combined step.
    """
    temperature = compute_temperature(temperatures)
    rows, cols = temperatures.shape

    row_coords = np.arange(rows, dtype=np.float32)
    col_coords = np.arange(cols, dtype=np.float32)
    row_grid, col_grid = np.meshgrid(row_coords, col_coords, indexing='ij')

    if reference_wind_magnitude > 0.0:
        advection_scale = WIND_ADVECTION_STRENGTH * BLUR_SIGMA / reference_wind_magnitude
    else:
        advection_scale = 0.0

    # Source positions are fixed for the whole tick: wind does not change
    # between substeps.  Subtracting the wind vector traces backward (upwind).
    source_rows = row_grid - y_wind_vel * advection_scale
    source_cols = col_grid - x_wind_vel * advection_scale

    result = temperatures
    for _ in range(DIFFUSION_SUBSTEPS):
        result = map_coordinates(
            result,
            [source_rows, source_cols],
            order=1,
            mode='nearest',
        ).astype(np.float32)
        result = convolve(
            result,
            _DIFFUSION_KERNEL,
            mode='nearest',
        ).astype(np.float32)

    correction = temperature - compute_temperature(result)
    _check_energy_correction(correction, temperature)
    return (result + correction).astype(np.float32)


def compute_wind_from_temperature(temperatures: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Derive wind velocity vectors from the spatial gradient of the temperature field.

    Wind moves from cool (high pressure) toward warm (low pressure) regions,
    so the wind direction at each cell is the direction of steepest temperature
    increase — i.e., the gradient vector.  Wind speed is proportional to
    gradient magnitude: a flat temperature field produces no wind, while a
    steep gradient produces strong wind.

    np.gradient uses second-order central differences for interior cells and
    first-order one-sided differences at the borders, which is an adequate
    approximation for a smoothly varying Perlin-noise temperature field.

    Parameters
    ----------
    temperatures : np.ndarray
        2-D float32 array of shape (rows, cols) with values in [0, 1].

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A (x_wind_vel, y_wind_vel) pair of float32 arrays, each with shape
        (rows, cols).  Positive x means blowing east; positive y means south.
        Magnitudes are in units of temperature-change per cell.
    """
    # np.gradient returns gradients as [d/d_row, d/d_col], i.e. [y, x].
    grad_y, grad_x = np.gradient(temperatures)
    x_wind = gaussian_filter(grad_x, sigma=WIND_SMOOTH_SIGMA).astype(np.float32)
    y_wind = gaussian_filter(grad_y, sigma=WIND_SMOOTH_SIGMA).astype(np.float32)
    return x_wind, y_wind


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

# Header/footer bar drawn around the environment info block.
INFO_SEPARATOR: str = "-" * 36


def print_environment_info(
    temperatures: np.ndarray,
    label: str,
    window_size: int,
    x_wind_vel: np.ndarray,
    y_wind_vel: np.ndarray,
) -> None:
    """
    Print a labeled summary of the current simulation state to stdout.

    Displays the grid dimensions, window resolution, simulation rate, and key
    statistics of both the temperature field and the wind velocity field.
    Called at startup and shutdown so the initial and final states are both visible.

    Parameters
    ----------
    temperatures : np.ndarray
        Current 2-D float32 temperature grid of shape (rows, cols).
    label : str
        Short context label shown in the header (e.g. "Startup", "Shutdown").
    window_size : int
        Side length of the square display window in pixels.
    x_wind_vel : np.ndarray
        2-D float32 array of x wind velocity components, shape (rows, cols).
    y_wind_vel : np.ndarray
        2-D float32 array of y wind velocity components, shape (rows, cols).
    """
    rows, cols = temperatures.shape
    wind_speeds = np.sqrt(x_wind_vel ** 2 + y_wind_vel ** 2)
    print(INFO_SEPARATOR)
    print(f" {label}")
    print(INFO_SEPARATOR)
    print(f"  Grid size   : {rows} x {cols}  ({rows * cols} cells)")
    print(f"  Window size : {window_size} x {window_size} px")
    print(f"  Sim rate    : {SIMULATION_STEPS_PER_SECOND} steps/sec")
    print(f"  Temperature : {compute_temperature(temperatures):.6f}")
    print(f"  Temp min    : {float(temperatures.min()):.6f}")
    print(f"  Temp max    : {float(temperatures.max()):.6f}")
    print(f"  Temp std    : {float(temperatures.std()):.6f}")
    print(f"  Wind mean   : {float(wind_speeds.mean()):.6f}")
    print(f"  Wind max    : {float(wind_speeds.max()):.6f}")
    print(f"  Wind std    : {float(wind_speeds.std()):.6f}")
    print(INFO_SEPARATOR)


# ---------------------------------------------------------------------------
# Color mapping
# ---------------------------------------------------------------------------

def build_color_surface(temperatures: np.ndarray, scale: int) -> pygame.Surface:
    """
    Construct a pygame Surface that visualizes every cell in *temperatures* as a
    solid-colored square of side *scale* pixels.

    Parameters
    ----------
    temperatures : np.ndarray
        2-D float32 array of shape (rows, cols) with values in [0, 1].
    scale : int
        Number of screen pixels per grid cell on each axis.

    Returns
    -------
    pygame.Surface
        A surface whose pixel dimensions are (cols * scale, rows * scale).
    """
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

    # Brightness is normalised against reference_magnitude (the max gradient at
    # startup) so arrows begin at full brightness and dim proportionally as the
    # temperature field flattens.  When reference_magnitude is 0 the field was
    # already uniform at startup, so there is nothing to show.
    if reference_magnitude == 0.0:
        return surface

    magnitudes = np.sqrt(x_wind_vel ** 2 + y_wind_vel ** 2)

    half_len = WIND_ARROW_LENGTH / 2.0

    # Start half a stride from each edge so the arrow grid is centered and
    # the margin is equal on opposite sides.
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

    The simulation advances at SIMULATION_STEPS_PER_SECOND via combined
    advection-diffusion steps.  Both render surfaces are rebuilt after every
    simulation step.  The window is displayed until
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

    temperatures = create_grid(grid_size)
    x_wind_vel, y_wind_vel = compute_wind_from_temperature(temperatures)
    # Capture the initial gradient strength as a fixed brightness reference so
    # arrows dim naturally over time rather than always rescaling to full brightness.
    reference_wind_magnitude = float(np.sqrt(x_wind_vel ** 2 + y_wind_vel ** 2).max())
    print_environment_info(temperatures, "Startup", window_size, x_wind_vel, y_wind_vel)
    screen = create_window(grid_size, pixel_scale, WINDOW_TITLE)
    clock = pygame.time.Clock()
    color_surface = build_color_surface(temperatures, pixel_scale)
    wind_surface = build_wind_surface(x_wind_vel, y_wind_vel, pixel_scale, reference_wind_magnitude)

    last_step_ms: int = pygame.time.get_ticks()
    current_mode: ViewMode = ViewMode.TEMPERATURE

    running = True
    while running:
        running, current_mode = handle_events(current_mode)

        current_ms = pygame.time.get_ticks()
        if current_ms - last_step_ms >= MS_PER_STEP:
            step_start_ms = pygame.time.get_ticks()
            temperatures = diffuse_and_advect(
                temperatures, x_wind_vel, y_wind_vel, reference_wind_magnitude
            )
            new_x_wind, new_y_wind = compute_wind_from_temperature(temperatures)
            x_wind_vel = WIND_TEMPORAL_SMOOTHING * new_x_wind + (1.0 - WIND_TEMPORAL_SMOOTHING) * x_wind_vel
            y_wind_vel = WIND_TEMPORAL_SMOOTHING * new_y_wind + (1.0 - WIND_TEMPORAL_SMOOTHING) * y_wind_vel
            color_surface = build_color_surface(temperatures, pixel_scale)
            wind_surface = build_wind_surface(x_wind_vel, y_wind_vel, pixel_scale, reference_wind_magnitude)
            last_step_ms = current_ms
            step_elapsed_ms = pygame.time.get_ticks() - step_start_ms
            if step_elapsed_ms > MS_PER_STEP:
                print(f"WARNING: Step took {step_elapsed_ms}ms (budget: {MS_PER_STEP}ms)")

        render_frame(screen, current_mode, color_surface, wind_surface)
        pygame.display.flip()
        clock.tick(TARGET_FPS)

    print_environment_info(temperatures, "Shutdown", window_size, x_wind_vel, y_wind_vel)
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
