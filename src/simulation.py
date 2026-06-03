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

WIND_ADVECTION_STRENGTH: float = 0.4
WIND_SMOOTH_SIGMA: float = 2.0
WIND_TEMPORAL_SMOOTHING: float = 0.2

OXYGEN_DIFFUSION_SIGMA: float = 3.0
OXYGEN_ADVECTION_STRENGTH: float = 3.0

ATMOSPHERIC_COOLING_RATE: float = 0.01

# ---------------------------------------------------------------------------
# Perlin noise parameters
# ---------------------------------------------------------------------------

NOISE_SCALE: float = 64.0
NOISE_OCTAVES: int = 4
NOISE_PERSISTENCE: float = 0.5
NOISE_LACUNARITY: float = 2.0

FUEL_NOISE_SCALE: float = 32.0

# ---------------------------------------------------------------------------
# Fire parameters
# ---------------------------------------------------------------------------

IGNITION_THRESHOLD: float = 0.2
FUEL_CONSUMPTION_RATE: float = 0.3
OXYGEN_CONSUMPTION_RATE: float = 0.3
FUEL_BURN_HEAT_SCALE: float = 5.0
FUEL_BURNT_THRESHOLD: float = 0.01
FUEL_MIN: float = 0.1

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


def create_grid(size: int, scale: float = NOISE_SCALE, base: int | None = None) -> np.ndarray:
    if base is None:
        base = np.random.randint(0, 256)
    raw = sample_perlin_grid(size, base, scale=scale)
    return normalize_grid(raw)


# ---------------------------------------------------------------------------
# Diffusion / advection
# ---------------------------------------------------------------------------

def _build_diffusion_kernel(sigma: float) -> np.ndarray:
    offsets = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    gy, gx = np.meshgrid(offsets, offsets, indexing='ij')
    kernel = np.exp(-(gx ** 2 + gy ** 2) / (2.0 * sigma ** 2))
    return (kernel / kernel.sum()).astype(np.float32)


_DIFFUSION_STEP_SIGMA: float = BLUR_SIGMA / math.sqrt(DIFFUSION_SUBSTEPS)
_DIFFUSION_KERNEL: np.ndarray = _build_diffusion_kernel(_DIFFUSION_STEP_SIGMA)

_OXYGEN_DIFFUSION_STEP_SIGMA: float = OXYGEN_DIFFUSION_SIGMA / math.sqrt(DIFFUSION_SUBSTEPS)
_OXYGEN_DIFFUSION_KERNEL: np.ndarray = _build_diffusion_kernel(_OXYGEN_DIFFUSION_STEP_SIGMA)


def _advect_and_diffuse_field(
    field: np.ndarray,
    source_rows: np.ndarray,
    source_cols: np.ndarray,
    boundary_mode: str,
    boundary_cval: float,
    kernel: np.ndarray,
) -> np.ndarray:
    result = field
    for _ in range(DIFFUSION_SUBSTEPS):
        result = map_coordinates(
            result,
            [source_rows, source_cols],
            order=1,
            mode=boundary_mode,
            cval=boundary_cval,
        ).astype(np.float32)
        result = convolve(
            result,
            kernel,
            mode=boundary_mode,
            cval=boundary_cval,
        ).astype(np.float32)
    return result


def _compute_source_coords(
    field: np.ndarray,
    x_wind_vel: np.ndarray,
    y_wind_vel: np.ndarray,
    reference_wind_magnitude: float,
    advection_strength: float,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = field.shape
    row_coords = np.arange(rows, dtype=np.float32)
    col_coords = np.arange(cols, dtype=np.float32)
    row_grid, col_grid = np.meshgrid(row_coords, col_coords, indexing='ij')

    if reference_wind_magnitude > 0.0:
        advection_scale = advection_strength * sigma / reference_wind_magnitude
    else:
        advection_scale = 0.0

    return row_grid - y_wind_vel * advection_scale, col_grid - x_wind_vel * advection_scale


def compute_temperature(temperatures: np.ndarray) -> float:
    return float(temperatures.mean())


def diffuse_and_advect(
    temperatures: np.ndarray,
    x_wind_vel: np.ndarray,
    y_wind_vel: np.ndarray,
    reference_wind_magnitude: float,
) -> np.ndarray:
    source_rows, source_cols = _compute_source_coords(
        temperatures, x_wind_vel, y_wind_vel, reference_wind_magnitude,
        WIND_ADVECTION_STRENGTH, BLUR_SIGMA,
    )
    return _advect_and_diffuse_field(temperatures, source_rows, source_cols, 'constant', 0.0, _DIFFUSION_KERNEL)


def apply_atmospheric_cooling(temperatures: np.ndarray) -> np.ndarray:
    return (temperatures * (1.0 - ATMOSPHERIC_COOLING_RATE)).astype(np.float32)


def diffuse_and_advect_oxygen(
    oxygen: np.ndarray,
    x_wind_vel: np.ndarray,
    y_wind_vel: np.ndarray,
    reference_wind_magnitude: float,
) -> np.ndarray:
    # Boundary cval=1.0 means the outside world acts as an infinite fresh-air source.
    # No energy correction: oxygen is not conserved — it flows in freely from outside.
    source_rows, source_cols = _compute_source_coords(
        oxygen, x_wind_vel, y_wind_vel, reference_wind_magnitude,
        OXYGEN_ADVECTION_STRENGTH, OXYGEN_DIFFUSION_SIGMA,
    )
    result = _advect_and_diffuse_field(oxygen, source_rows, source_cols, 'constant', 1.0, _OXYGEN_DIFFUSION_KERNEL)
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def compute_wind_from_temperature(temperatures: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    grad_y, grad_x = np.gradient(temperatures)
    x_wind = gaussian_filter(grad_x, sigma=WIND_SMOOTH_SIGMA).astype(np.float32)
    y_wind = gaussian_filter(grad_y, sigma=WIND_SMOOTH_SIGMA).astype(np.float32)
    return x_wind, y_wind


def update_wind(
    temperatures: np.ndarray,
    x_wind_vel: np.ndarray,
    y_wind_vel: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    new_x, new_y = compute_wind_from_temperature(temperatures)
    x_wind = (WIND_TEMPORAL_SMOOTHING * new_x + (1.0 - WIND_TEMPORAL_SMOOTHING) * x_wind_vel).astype(np.float32)
    y_wind = (WIND_TEMPORAL_SMOOTHING * new_y + (1.0 - WIND_TEMPORAL_SMOOTHING) * y_wind_vel).astype(np.float32)
    return x_wind, y_wind


# ---------------------------------------------------------------------------
# Fire
# ---------------------------------------------------------------------------

def update_fire(
    temperatures: np.ndarray,
    fuel: np.ndarray,
    damaged: np.ndarray,
    oxygen: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    burning = (temperatures >= IGNITION_THRESHOLD) & (fuel > FUEL_BURNT_THRESHOLD)

    fuel_consumed = np.where(burning, fuel * FUEL_CONSUMPTION_RATE, 0.0).astype(np.float32)
    oxygen_consumed = np.where(burning, oxygen * OXYGEN_CONSUMPTION_RATE, 0.0).astype(np.float32)

    fuel = (fuel - fuel_consumed).astype(np.float32)
    oxygen = np.clip(oxygen - oxygen_consumed, 0.0, 1.0).astype(np.float32)
    temperatures = np.maximum(temperatures + oxygen_consumed * FUEL_BURN_HEAT_SCALE, 0.0).astype(np.float32)
    damaged = np.where(burning, 1, damaged).astype(np.uint8)

    return temperatures, fuel, damaged, oxygen
