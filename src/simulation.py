"""
simulation.py

Pure-numpy simulation core for the Firecracker heat-diffusion model.
No pygame dependency; safe to import in headless training environments.
"""

import math
import numpy as np
import noise
from scipy.ndimage import gaussian_filter, laplace, map_coordinates
from scipy.signal import fftconvolve, oaconvolve


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

        pressure = (cfg or {}).get("pressure", {})

        self.pressure_temp_scale:        float = float(pressure.get("temp_scale",        0.2))
        self.pressure_relaxation_rate:   float = float(pressure.get("relaxation_rate",   0.1))
        self.pressure_equalization_rate: float = float(pressure.get("equalization_rate", 0.1))

        self.oxygen_consumption_rate:     float = float(fire.get("oxygen_consumption_rate",     0.01))
        self.fuel_burnt_threshold:        float = float(fire.get("burnt_threshold",             0.01))
        self.oxygen_extinction_threshold: float = float(fire.get("oxygen_extinction_threshold", 0.05))
        self.burn_heat_fuel_fraction:     float = float(fire.get("burn_heat_fuel_fraction",     0.2))

        # ---------------------------------------------------------------------------
        # Fuel types — parsed in config order; add subtables to expand.
        # ---------------------------------------------------------------------------

        fuel_types_cfg = (cfg or {}).get("fuel_types", {})
        self.fuel_type_names: list[str] = list(fuel_types_cfg.keys())
        self.num_fuel_types:  int       = len(self.fuel_type_names)

        def _ft(name: str, key: str, default: float) -> float:
            return float(fuel_types_cfg.get(name, {}).get(key, default))

        self.heat_capacities     = np.array([_ft(n, "heat_capacity",          1.0)   for n in self.fuel_type_names], dtype=np.float32)
        self.ignition_thresholds = np.array([_ft(n, "ignition_threshold",     2.0)   for n in self.fuel_type_names], dtype=np.float32)
        self.consumption_rates   = np.array([_ft(n, "consumption_rate",       0.001) for n in self.fuel_type_names], dtype=np.float32)
        self.burn_heat_scales    = np.array([_ft(n, "burn_heat_scale",        500.0) for n in self.fuel_type_names], dtype=np.float32)
        self.fuel_transfer_rates = np.array([_ft(n, "fuel_air_transfer_rate", 0.01)  for n in self.fuel_type_names], dtype=np.float32)
        self.fuel_noise_scales   = np.array([_ft(n, "noise_scale",            32.0)  for n in self.fuel_type_names], dtype=np.float32)

        radiant = (cfg or {}).get("radiant_heat", {})
        self.radiant_kernel_radius:     int   = int(radiant.get("kernel_radius",      20))
        self.radiant_emission_scale:    float = float(radiant.get("emission_scale",   0.05))
        self.radiant_emission_exponent: float = float(radiant.get("emission_exponent", 2.0))

        self._diffusion_kernel        = self._build_diffusion_kernel(
            self.blur_sigma / math.sqrt(self.diffusion_substeps)
        )
        self._oxygen_diffusion_kernel = self._build_diffusion_kernel(
            self.oxygen_diffusion_sigma / math.sqrt(self.diffusion_substeps)
        )
        self._radiant_kernel = self._build_radiant_kernel(self.radiant_kernel_radius)

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

    @staticmethod
    def _build_radiant_kernel(radius: int) -> np.ndarray:
        size = 2 * radius + 1
        rows, cols = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
        r_sq = ((rows - radius) ** 2 + (cols - radius) ** 2).astype(np.float32)
        # 1/r² inverse-square law; center cell contributes nothing (self-heating handled by combustion).
        # Avoid dividing by zero at the center by substituting 1.0 before masking it out.
        safe_r_sq = np.where(r_sq > 0, r_sq, 1.0)
        kernel = np.where(r_sq > 0, 1.0 / safe_r_sq, 0.0).astype(np.float32)
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
        pad_r = kernel.shape[0] // 2
        pad_c = kernel.shape[1] // 2
        for _ in range(self.diffusion_substeps):
            result = map_coordinates(
                result,
                [source_rows, source_cols],
                order=1,
                mode=boundary_mode,
                cval=boundary_cval,
            ).astype(np.float32)
            if boundary_mode == 'reflect':
                padded = np.pad(result, ((pad_r, pad_r), (pad_c, pad_c)), mode='reflect')
            else:
                padded = np.pad(result, ((pad_r, pad_r), (pad_c, pad_c)), mode='constant', constant_values=boundary_cval)
            result = oaconvolve(padded, kernel, mode='valid').astype(np.float32)
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
    def compute_air_temperature(air_temperatures: np.ndarray) -> float:
        return float(air_temperatures.mean())

    def diffuse_and_advect(
        self,
        air_temperatures: np.ndarray,
        x_wind_vel: np.ndarray,
        y_wind_vel: np.ndarray,
        reference_wind_magnitude: float,
    ) -> np.ndarray:
        source_rows, source_cols = self._compute_source_coords(
            air_temperatures, x_wind_vel, y_wind_vel, reference_wind_magnitude,
            self.wind_advection_strength, self.blur_sigma,
        )
        return self._advect_and_diffuse_field(
            air_temperatures, source_rows, source_cols, 'reflect', 0.0, self._diffusion_kernel
        )

    def apply_atmospheric_cooling(self, air_temperatures: np.ndarray) -> np.ndarray:
        return (air_temperatures * (1.0 - self.atmospheric_cooling_rate)).astype(np.float32)

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

    def update_pressure(
        self,
        pressure: np.ndarray,
        air_temperatures: np.ndarray,
    ) -> np.ndarray:
        lap   = laplace(pressure, mode='reflect').astype(np.float32)
        p_eq  = np.exp(-self.pressure_temp_scale * air_temperatures).astype(np.float32)
        new_p = (
            pressure
            + self.pressure_equalization_rate * lap
            + self.pressure_relaxation_rate * (p_eq - pressure)
        )
        return np.clip(new_p, 0.0, 1.0).astype(np.float32)

    def compute_wind_from_pressure(
        self, pressure: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        grad_y, grad_x = np.gradient(pressure)
        # Wind flows from high to low pressure, so negate the gradient.
        x_wind = gaussian_filter(-grad_x, sigma=self.wind_smooth_sigma).astype(np.float32)
        y_wind = gaussian_filter(-grad_y, sigma=self.wind_smooth_sigma).astype(np.float32)
        return x_wind, y_wind

    def update_wind(
        self,
        pressure: np.ndarray,
        x_wind_vel: np.ndarray,
        y_wind_vel: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        new_x, new_y = self.compute_wind_from_pressure(pressure)
        x_wind = (self.wind_temporal_smoothing * new_x + (1.0 - self.wind_temporal_smoothing) * x_wind_vel).astype(np.float32)
        y_wind = (self.wind_temporal_smoothing * new_y + (1.0 - self.wind_temporal_smoothing) * y_wind_vel).astype(np.float32)
        return x_wind, y_wind

    # ---------------------------------------------------------------------------
    # Fire
    # ---------------------------------------------------------------------------

    def exchange_fuel_air_heat(
        self,
        air_temperatures: np.ndarray,   # (H, W)
        fuel_temperatures: np.ndarray,  # (N, H, W)
        fuel: np.ndarray,               # (N, H, W)
    ) -> tuple[np.ndarray, np.ndarray]:
        h = self.heat_capacities[:, np.newaxis, np.newaxis]    # (N, 1, 1)
        k = self.fuel_transfer_rates[:, np.newaxis, np.newaxis]  # (N, 1, 1)

        mask    = fuel > self.fuel_burnt_threshold              # (N, H, W)
        dT      = air_temperatures[np.newaxis] - fuel_temperatures  # (N, H, W)
        C_fuel  = h * fuel                                      # (N, H, W)

        # Avoid dividing by zero for non-participating cells.
        safe_C = np.where(mask, C_fuel, 1.0)

        # Unconstrained Newton step and equilibrium step (maximum without overshoot).
        dT_fuel_raw = k * dT / safe_C
        dT_fuel_eq  = dT / (1.0 + safe_C)

        # Take the smaller magnitude — whichever doesn't cross equilibrium.
        dT_fuel = np.where(
            np.abs(dT_fuel_raw) <= np.abs(dT_fuel_eq),
            dT_fuel_raw,
            dT_fuel_eq,
        )
        dT_fuel = dT_fuel.astype(np.float32)

        # Air loses the sum of heat transferred to all fuel types (energy conserving).
        # Depleted cells (mask=False) contribute ~0 because C_fuel = h*fuel ≈ 0.
        dT_air = -np.sum(C_fuel * dT_fuel, axis=0).astype(np.float32)

        return (air_temperatures + dT_air).astype(np.float32), (fuel_temperatures + dT_fuel).astype(np.float32)

    def update_fire(
        self,
        air_temperatures: np.ndarray,   # (H, W)
        fuel_temperatures: np.ndarray,  # (N, H, W)
        fuel: np.ndarray,               # (N, H, W)
        oxygen: np.ndarray,             # (H, W)
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ign = self.ignition_thresholds[:, np.newaxis, np.newaxis]  # (N, 1, 1)
        cr  = self.consumption_rates[:, np.newaxis, np.newaxis]    # (N, 1, 1)
        bhs = self.burn_heat_scales[:, np.newaxis, np.newaxis]     # (N, 1, 1)
        h   = self.heat_capacities[:, np.newaxis, np.newaxis]      # (N, 1, 1)

        burning = (
            (fuel_temperatures >= ign) &
            (fuel > self.fuel_burnt_threshold) &
            (oxygen[np.newaxis] > self.oxygen_extinction_threshold)
        )  # (N, H, W)

        fuel_consumed = np.where(
            burning,
            fuel * np.minimum(fuel_temperatures * cr, 1.0),
            0.0,
        ).astype(np.float32)  # (N, H, W)

        # Oxygen consumed by all burning types combined, capped at available oxygen.
        oxygen_consumed = np.minimum(
            np.sum(
                np.where(burning, oxygen[np.newaxis] * np.minimum(fuel_temperatures * self.oxygen_consumption_rate, 1.0), 0.0),
                axis=0,
            ),
            oxygen,
        ).astype(np.float32)  # (H, W)

        fuel   = (fuel - fuel_consumed).astype(np.float32)
        oxygen = np.clip(oxygen - oxygen_consumed, 0.0, 1.0).astype(np.float32)

        # Heat released per type, split between air and fuel.
        burn_heat_per_type = (fuel_consumed * bhs).astype(np.float32)  # (N, H, W)
        total_burn_heat    = burn_heat_per_type.sum(axis=0)             # (H, W)

        air_temperatures = np.maximum(
            air_temperatures + total_burn_heat * (1.0 - self.burn_heat_fuel_fraction), 0.0
        ).astype(np.float32)

        # Each type's fuel temperature rises by its share of burn heat divided by thermal mass.
        C_fuel    = h * fuel
        safe_C    = np.where(fuel > self.fuel_burnt_threshold, C_fuel, 1.0)
        dT_fuel   = np.where(
            fuel > self.fuel_burnt_threshold,
            burn_heat_per_type * self.burn_heat_fuel_fraction / safe_C,
            0.0,
        )
        fuel_temperatures = np.maximum(fuel_temperatures + dT_fuel, 0.0).astype(np.float32)

        return air_temperatures, fuel_temperatures, fuel, oxygen

    def apply_radiant_heat(
        self,
        fuel_temperatures: np.ndarray,  # (N, H, W)
        fuel: np.ndarray,               # (N, H, W)
        oxygen: np.ndarray,             # (H, W)
    ) -> np.ndarray:
        ign = self.ignition_thresholds[:, np.newaxis, np.newaxis]

        burning = (
            (fuel_temperatures >= ign) &
            (fuel > self.fuel_burnt_threshold) &
            (oxygen[np.newaxis] > self.oxygen_extinction_threshold)
        )  # (N, H, W)

        # Total emission summed across all burning types.
        emission = np.where(
            burning,
            self.radiant_emission_scale * fuel_temperatures ** self.radiant_emission_exponent,
            0.0,
        ).astype(np.float32)
        total_emission = emission.sum(axis=0)  # (H, W)

        radiant_flux = np.clip(
            fftconvolve(total_emission, self._radiant_kernel, mode='same'), 0.0, None
        ).astype(np.float32)  # (H, W)

        # Distribute flux to each type proportional to its fuel mass fraction,
        # then divide by thermal mass to get temperature change.
        total_fuel = fuel.sum(axis=0)                             # (H, W)
        safe_total = np.where(total_fuel > 0, total_fuel, 1.0)
        fuel_frac  = fuel / safe_total[np.newaxis]                # (N, H, W)

        h      = self.heat_capacities[:, np.newaxis, np.newaxis]
        C_fuel = h * fuel
        safe_C = np.where(fuel > self.fuel_burnt_threshold, C_fuel, 1.0)

        dT_fuel = np.where(
            fuel > self.fuel_burnt_threshold,
            radiant_flux[np.newaxis] * fuel_frac / safe_C,
            0.0,
        ).astype(np.float32)

        return (fuel_temperatures + dT_fuel).astype(np.float32)
