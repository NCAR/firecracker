"""
simulation.py

Pure-numpy simulation core for the Firecracker heat-diffusion model.
No pygame dependency; safe to import in headless training environments.
"""

import math
import numpy as np
import noise
from scipy.ndimage import convolve, gaussian_filter, map_coordinates

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------

SIMULATION_STEPS_PER_SECOND: int = 10
MS_PER_STEP: int = 1000 // SIMULATION_STEPS_PER_SECOND

BLUR_SIGMA: float = 1.0
DIFFUSION_SUBSTEPS: int = 4

WIND_ADVECTION_STRENGTH: float = 0.8
WIND_SMOOTH_SIGMA: float = 2.0
WIND_TEMPORAL_SMOOTHING: float = 0.2

ENERGY_CORRECTION_WARN_FRACTION: float = 0.05

# ---------------------------------------------------------------------------
# Perlin noise parameters
# ---------------------------------------------------------------------------

NOISE_SCALE: float = 64.0
NOISE_OCTAVES: int = 4
NOISE_PERSISTENCE: float = 0.5
NOISE_LACUNARITY: float = 2.0

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
    lo = grid.min()
    hi = grid.max()
    if hi == lo:
        return np.zeros_like(grid)
    return ((grid - lo) / (hi - lo)).astype(np.float32)


def create_grid(size: int) -> np.ndarray:
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
    return float(temperatures.mean())


def diffuse_and_advect(
    temperatures: np.ndarray,
    x_wind_vel: np.ndarray,
    y_wind_vel: np.ndarray,
    reference_wind_magnitude: float,
) -> np.ndarray:
    temperature = compute_temperature(temperatures)
    rows, cols = temperatures.shape

    row_coords = np.arange(rows, dtype=np.float32)
    col_coords = np.arange(cols, dtype=np.float32)
    row_grid, col_grid = np.meshgrid(row_coords, col_coords, indexing='ij')

    if reference_wind_magnitude > 0.0:
        advection_scale = WIND_ADVECTION_STRENGTH * BLUR_SIGMA / reference_wind_magnitude
    else:
        advection_scale = 0.0

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
    grad_y, grad_x = np.gradient(temperatures)
    x_wind = gaussian_filter(grad_x, sigma=WIND_SMOOTH_SIGMA).astype(np.float32)
    y_wind = gaussian_filter(grad_y, sigma=WIND_SMOOTH_SIGMA).astype(np.float32)
    return x_wind, y_wind
