"""
main.py

Displays an NxN grid of pixels using pygame, where each pixel value is a float
in [0, 1] sourced from a 2D numpy array. Zero maps to black, one maps to pure
red, and intermediate values are proportional shades of red.
"""

import sys
import numpy as np
import pygame
import noise

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_GRID_SIZE: int = 64       # Default side length of the square grid (N)
PIXEL_SCALE: int = 8              # Screen pixels rendered per grid cell
WINDOW_TITLE: str = "Firecracker" # Title shown in the pygame window title bar
TARGET_FPS: int = 60              # Frame-rate cap (visual-only; data is static)

# Colour channel indices within an (R, G, B) tuple
RED_CHANNEL: int = 0
GREEN_CHANNEL: int = 1
BLUE_CHANNEL: int = 2

# Maximum value of an 8-bit colour channel
MAX_CHANNEL_VALUE: int = 255

# ---------------------------------------------------------------------------
# Perlin noise parameters
# ---------------------------------------------------------------------------

# Coordinate scale applied before sampling pnoise2.  Smaller values zoom in,
# producing larger, smoother blobs; larger values zoom out for finer detail.
NOISE_SCALE: float = 16.0

# Each additional octave adds a finer layer of detail at the cost of compute.
NOISE_OCTAVES: int = 4

# How quickly each successive octave's amplitude decreases (0–1).
NOISE_PERSISTENCE: float = 0.5

# How quickly each successive octave's frequency increases (>1).
NOISE_LACUNARITY: float = 2.0


# ---------------------------------------------------------------------------
# Grid initialisation
# ---------------------------------------------------------------------------

def sample_perlin_grid(size: int, base: int) -> np.ndarray:
    """
    Sample 2D Perlin noise on an NxN grid and return raw (un-normalised) values.

    Each cell (row, col) is sampled at coordinates (col / NOISE_SCALE,
    row / NOISE_SCALE) so that adjacent cells are spatially coherent and the
    NOISE_SCALE constant controls the apparent size of hot/cold regions.

    Parameters
    ----------
    size : int
        Side length N of the square grid.
    base : int
        Seed offset passed to pnoise2; changing this produces a different map.

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
                col / NOISE_SCALE,
                row / NOISE_SCALE,
                octaves=NOISE_OCTAVES,
                persistence=NOISE_PERSISTENCE,
                lacunarity=NOISE_LACUNARITY,
                base=base,
            )
    return grid


def normalise_grid(grid: np.ndarray) -> np.ndarray:
    """
    Linearly rescale all values in *grid* to the range [0, 1].

    Uses the grid's own observed minimum and maximum so the full colour range
    is always utilised regardless of the raw noise amplitude.

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
    Build an NxN float32 grid of Perlin noise values normalised to [0, 1].

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
    return normalise_grid(raw)


# ---------------------------------------------------------------------------
# Colour mapping
# ---------------------------------------------------------------------------

def value_to_colour(value: float) -> tuple[int, int, int]:
    """
    Map a scalar in [0, 1] to an (R, G, B) colour.

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


def build_colour_surface(grid: np.ndarray, scale: int) -> pygame.Surface:
    """
    Construct a pygame Surface that visualises every cell in *grid* as a
    solid-coloured square of side *scale* pixels.

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
            colour = value_to_colour(grid[row, col])
            rect = cell_rect(row, col, scale)
            pygame.draw.rect(surface, colour, rect)

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
    Initialise pygame and open a display window sized to fit the grid.

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


def handle_events() -> bool:
    """
    Process the pygame event queue for the current frame.

    Returns
    -------
    bool
        True if the application should continue running, False if a quit event
        (window close button or Escape key) was received.
    """
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            return False
    return True


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(grid_size: int = DEFAULT_GRID_SIZE) -> None:
    """
    Initialise the grid and pygame, then enter the main display loop.

    The grid is generated once at startup and displayed until the user closes
    the window or presses Escape.

    Parameters
    ----------
    grid_size : int, optional
        Side length N of the square grid.  Defaults to DEFAULT_GRID_SIZE.
    """
    pygame.init()

    grid = create_grid(grid_size)
    screen = create_window(grid_size, PIXEL_SCALE, WINDOW_TITLE)
    clock = pygame.time.Clock()

    # Build the colour surface once — the grid is static.
    colour_surface = build_colour_surface(grid, PIXEL_SCALE)

    running = True
    while running:
        running = handle_events()
        screen.blit(colour_surface, (0, 0))
        pygame.display.flip()
        clock.tick(TARGET_FPS)

    pygame.quit()


if __name__ == "__main__":
    # Allow an optional command-line argument to override the grid size:
    #   python main.py [N]
    if len(sys.argv) > 1:
        try:
            n = int(sys.argv[1])
        except ValueError:
            print(f"Usage: python main.py [N]  (N must be a positive integer)")
            sys.exit(1)
        run(grid_size=n)
    else:
        run()
