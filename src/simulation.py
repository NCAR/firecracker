"""
simulation.py

Pure-numpy simulation core for the Firecracker heat-diffusion model.
No pygame dependency; safe to import in headless training environments.
"""

import math
import numpy as np
import noise
from scipy.ndimage import convolve, gaussian_filter, map_coordinates


class Simulation:
    def __init__(self, cfg: dict | None = None):
        sim    = (cfg or {}).get("simulation", {})
        wind   = (cfg or {}).get("wind", {})
        oxygen = (cfg or {}).get("oxygen", {})
        noise  = (cfg or {}).get("noise", {})
        fire   = (cfg or {}).get("fire", {})

        self.simulation_steps_per_second: int   = int(sim.get("steps_per_second",        10))
        self.ms_per_step:                 int   = 1000 // self.simulation_steps_per_second
        self.blur_sigma:                  float = float(sim.get("blur_sigma",              1.0))
        self.diffusion_substeps:          int   = int(sim.get("diffusion_substeps",        4))
        self.atmospheric_cooling_rate:    float = float(sim.get("atmospheric_cooling_rate", 0.01))

        self.wind_advection_strength: float = float(wind.get("advection_strength", 0.4))
        self.wind_smooth_sigma:       float = float(wind.get("smooth_sigma",       2.0))
        self.wind_temporal_smoothing: float = float(wind.get("temporal_smoothing", 0.2))

        self.oxygen_diffusion_sigma:    float = float(oxygen.get("diffusion_sigma",    3.0))
        self.oxygen_advection_strength: float = float(oxygen.get("advection_strength", 3.0))

        self.noise_scale:       float = float(noise.get("scale",      64.0))
        self.noise_octaves:     int   = int(noise.get("octaves",       4))
        self.noise_persistence: float = float(noise.get("persistence", 0.5))
        self.noise_lacunarity:  float = float(noise.get("lacunarity",  2.0))
        self.fuel_noise_scale:  float = float(noise.get("fuel_scale",  32.0))

        self.ignition_threshold:      float = float(fire.get("ignition_threshold",      0.2))
        self.fuel_consumption_rate:   float = float(fire.get("fuel_consumption_rate",   0.3))
        self.oxygen_consumption_rate: float = float(fire.get("oxygen_consumption_rate", 0.3))
        self.fuel_burn_heat_scale:    float = float(fire.get("burn_heat_scale",         5.0))
        self.fuel_burnt_threshold:    float = float(fire.get("burnt_threshold",         0.01))
        self.fuel_min:                float = float(fire.get("fuel_min",                0.1))

        self._diffusion_kernel        = self._build_diffusion_kernel(
            self.blur_sigma / math.sqrt(self.diffusion_substeps)
        )
        self._oxygen_diffusion_kernel = self._build_diffusion_kernel(
            self.oxygen_diffusion_sigma / math.sqrt(self.diffusion_substeps)
        )

    # ---------------------------------------------------------------------------
    # Grid initialisation
    # ---------------------------------------------------------------------------

    def sample_perlin_grid(self, size: int, base: int, scale: float | None = None) -> np.ndarray:
        if scale is None:
            scale = self.noise_scale
        grid = np.empty((size, size), dtype=np.float32)
        for row in range(size):
            for col in range(size):
                grid[row, col] = noise.pnoise2(
                    col / scale,
                    row / scale,
                    octaves=self.noise_octaves,
                    persistence=self.noise_persistence,
                    lacunarity=self.noise_lacunarity,
                    base=base,
                )
        return grid

    @staticmethod
    def normalize_grid(grid: np.ndarray) -> np.ndarray:
        lo = grid.min()
        hi = grid.max()
        if hi == lo:
            return np.zeros_like(grid)
        return ((grid - lo) / (hi - lo)).astype(np.float32)

    def create_grid(self, size: int, scale: float | None = None, base: int | None = None) -> np.ndarray:
        if scale is None:
            scale = self.noise_scale
        if base is None:
            base = np.random.randint(0, 256)
        raw = self.sample_perlin_grid(size, base, scale=scale)
        return self.normalize_grid(raw)

    # ---------------------------------------------------------------------------
    # Diffusion / advection
    # ---------------------------------------------------------------------------

    @staticmethod
    def _build_diffusion_kernel(sigma: float) -> np.ndarray:
        offsets = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        gy, gx = np.meshgrid(offsets, offsets, indexing='ij')
        kernel = np.exp(-(gx ** 2 + gy ** 2) / (2.0 * sigma ** 2))
        return (kernel / kernel.sum()).astype(np.float32)

    def _advect_and_diffuse_field(
        self,
        field: np.ndarray,
        source_rows: np.ndarray,
        source_cols: np.ndarray,
        boundary_mode: str,
        boundary_cval: float,
        kernel: np.ndarray,
    ) -> np.ndarray:
        result = field
        for _ in range(self.diffusion_substeps):
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

    @staticmethod
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

    @staticmethod
    def compute_temperature(temperatures: np.ndarray) -> float:
        return float(temperatures.mean())

    def diffuse_and_advect(
        self,
        temperatures: np.ndarray,
        x_wind_vel: np.ndarray,
        y_wind_vel: np.ndarray,
        reference_wind_magnitude: float,
    ) -> np.ndarray:
        source_rows, source_cols = self._compute_source_coords(
            temperatures, x_wind_vel, y_wind_vel, reference_wind_magnitude,
            self.wind_advection_strength, self.blur_sigma,
        )
        return self._advect_and_diffuse_field(
            temperatures, source_rows, source_cols, 'constant', 0.0, self._diffusion_kernel
        )

    def apply_atmospheric_cooling(self, temperatures: np.ndarray) -> np.ndarray:
        return (temperatures * (1.0 - self.atmospheric_cooling_rate)).astype(np.float32)

    def diffuse_and_advect_oxygen(
        self,
        oxygen: np.ndarray,
        x_wind_vel: np.ndarray,
        y_wind_vel: np.ndarray,
        reference_wind_magnitude: float,
    ) -> np.ndarray:
        # Boundary cval=1.0 means the outside world acts as an infinite fresh-air source.
        # No energy correction: oxygen is not conserved — it flows in freely from outside.
        source_rows, source_cols = self._compute_source_coords(
            oxygen, x_wind_vel, y_wind_vel, reference_wind_magnitude,
            self.oxygen_advection_strength, self.oxygen_diffusion_sigma,
        )
        result = self._advect_and_diffuse_field(
            oxygen, source_rows, source_cols, 'constant', 1.0, self._oxygen_diffusion_kernel
        )
        return np.clip(result, 0.0, 1.0).astype(np.float32)

    def compute_wind_from_temperature(
        self, temperatures: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        grad_y, grad_x = np.gradient(temperatures)
        x_wind = gaussian_filter(grad_x, sigma=self.wind_smooth_sigma).astype(np.float32)
        y_wind = gaussian_filter(grad_y, sigma=self.wind_smooth_sigma).astype(np.float32)
        return x_wind, y_wind

    def update_wind(
        self,
        temperatures: np.ndarray,
        x_wind_vel: np.ndarray,
        y_wind_vel: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        new_x, new_y = self.compute_wind_from_temperature(temperatures)
        x_wind = (self.wind_temporal_smoothing * new_x + (1.0 - self.wind_temporal_smoothing) * x_wind_vel).astype(np.float32)
        y_wind = (self.wind_temporal_smoothing * new_y + (1.0 - self.wind_temporal_smoothing) * y_wind_vel).astype(np.float32)
        return x_wind, y_wind

    # ---------------------------------------------------------------------------
    # Fire
    # ---------------------------------------------------------------------------

    def update_fire(
        self,
        temperatures: np.ndarray,
        fuel: np.ndarray,
        damaged: np.ndarray,
        oxygen: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        burning = (temperatures >= self.ignition_threshold) & (fuel > self.fuel_burnt_threshold)

        fuel_consumed     = np.where(burning, fuel   * self.fuel_consumption_rate,   0.0).astype(np.float32)
        oxygen_consumed   = np.where(burning, oxygen * self.oxygen_consumption_rate, 0.0).astype(np.float32)

        fuel         = (fuel - fuel_consumed).astype(np.float32)
        oxygen       = np.clip(oxygen - oxygen_consumed, 0.0, 1.0).astype(np.float32)
        temperatures = np.maximum(temperatures + oxygen_consumed * self.fuel_burn_heat_scale, 0.0).astype(np.float32)
        damaged      = np.where(burning, 1, damaged).astype(np.uint8)

        return temperatures, fuel, damaged, oxygen
