"""
simulation.py

Pure-numpy simulation core for the Firecracker heat-diffusion model.
No pygame dependency; safe to import in headless training environments.
"""

import math
import numpy as np
import noise
from scipy.ndimage import gaussian_filter, laplace
from scipy.signal import fftconvolve

# Stability cap for the explicit Laplacian diffusion coefficient (2D: must be < 0.25).
_MAX_DIFFUSION_COEFF: float = 0.2


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

        self.wind_advection_strength: float = float(wind.get("advection_strength", 0.4))
        self.wind_smooth_sigma:       float = float(wind.get("smooth_sigma",       2.0))
        self.wind_temporal_smoothing: float = float(wind.get("temporal_smoothing", 0.2))

        self.oxygen_diffusion_sigma:    float = float(oxygen.get("diffusion_sigma",    3.0))
        self.oxygen_advection_strength: float = float(oxygen.get("advection_strength", 3.0))

        self.noise_scale:       float = float(noise.get("scale",      64.0))
        self.noise_octaves:     int   = int(noise.get("octaves",       4))
        self.noise_persistence: float = float(noise.get("persistence", 0.5))
        self.noise_lacunarity:  float = float(noise.get("lacunarity",  2.0))

        terrain = (cfg or {}).get("terrain", {})

        self.terrain_scale:       float = float(terrain.get("scale",      96.0))
        self.terrain_octaves:     int   = int(terrain.get("octaves",       6))
        self.terrain_persistence: float = float(terrain.get("persistence", 0.5))
        self.terrain_lacunarity:  float = float(terrain.get("lacunarity",  2.0))
        self.temperature_lapse_rate: float = float(terrain.get("temperature_lapse_rate", 3.0))
        self.oxygen_lapse_rate:      float = float(terrain.get("oxygen_lapse_rate",      1.5))

        relaxation = (cfg or {}).get("relaxation", {})

        # Newtonian relaxation toward the elevation equilibrium profiles (radiative
        # forcing for temperature, fresh-air replenishment for oxygen).
        self.temperature_rate_low:  float = float(relaxation.get("temperature_rate_low",  0.02))
        self.temperature_rate_high: float = float(relaxation.get("temperature_rate_high", 0.30))
        self.oxygen_rate:           float = float(relaxation.get("oxygen_rate",           0.05))

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
        self.altitude_falloffs   = np.array([_ft(n, "altitude_falloff",        1.0)  for n in self.fuel_type_names], dtype=np.float32)
        self.spawn_densities     = np.array([_ft(n, "spawn_density",           0.5)  for n in self.fuel_type_names], dtype=np.float32)

        # Tree-specific discretisation: per-cell tree counts are drawn from an
        # exponential whose mean is the local density, then capped.
        tree_cfg = fuel_types_cfg.get("tree", {})
        self.fuel_per_tree:      float = float(tree_cfg.get("fuel_per_tree",      0.33))
        self.max_trees_per_cell: int   = int(tree_cfg.get("max_trees_per_cell",   3))

        radiant = (cfg or {}).get("radiant_heat", {})
        self.radiant_kernel_radius:     int   = int(radiant.get("kernel_radius",      20))
        self.radiant_emission_scale:    float = float(radiant.get("emission_scale",   0.05))
        self.radiant_emission_exponent: float = float(radiant.get("emission_exponent", 2.0))

        self._radiant_kernel = self._build_radiant_kernel(self.radiant_kernel_radius)

    # ---------------------------------------------------------------------------
    # Grid initialisation
    # ---------------------------------------------------------------------------

    def sample_perlin_grid(
        self,
        size: int,
        base: int,
        scale: float | None = None,
        octaves: int | None = None,
        persistence: float | None = None,
        lacunarity: float | None = None,
    ) -> np.ndarray:
        if scale is None:
            scale = self.noise_scale
        if octaves is None:
            octaves = self.noise_octaves
        if persistence is None:
            persistence = self.noise_persistence
        if lacunarity is None:
            lacunarity = self.noise_lacunarity
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

    @staticmethod
    def normalize_grid(grid: np.ndarray) -> np.ndarray:
        lo = grid.min()
        hi = grid.max()
        if hi == lo:
            return np.zeros_like(grid)
        return ((grid - lo) / (hi - lo)).astype(np.float32)

    def create_grid(
        self,
        size: int,
        scale: float | None = None,
        base: int | None = None,
        octaves: int | None = None,
        persistence: float | None = None,
        lacunarity: float | None = None,
    ) -> np.ndarray:
        if scale is None:
            scale = self.noise_scale
        if base is None:
            base = np.random.randint(0, 256)
        raw = self.sample_perlin_grid(
            size, base, scale=scale,
            octaves=octaves, persistence=persistence, lacunarity=lacunarity,
        )
        return self.normalize_grid(raw)

    def create_terrain(self, size: int, base: int | None = None) -> np.ndarray:
        """Normalised [0, 1] elevation height map from octave Perlin noise.

        The normalised map is squared to bias toward flat low-elevation
        terrain with sharper, less frequent high-elevation peaks.
        """
        height = self.create_grid(
            size,
            scale=self.terrain_scale,
            base=base,
            octaves=self.terrain_octaves,
            persistence=self.terrain_persistence,
            lacunarity=self.terrain_lacunarity,
        )
        return (height ** 2).astype(np.float32)

    def altitude_vegetation_multiplier(self, terrain: np.ndarray, fuel_index: int) -> np.ndarray:
        """Per-cell vegetation density scale from elevation: (1 - h)^falloff in [0, 1]."""
        return ((1.0 - terrain) ** float(self.altitude_falloffs[fuel_index])).astype(np.float32)

    @staticmethod
    def elevation_falloff(terrain: np.ndarray, rate: float) -> np.ndarray:
        """Field that decays exponentially with elevation: exp(-rate * h), 1.0 at sea level."""
        return np.exp(-rate * terrain).astype(np.float32)

    def sample_tree_counts(self, density: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Stochastic per-cell tree counts.

        Each cell's count is floor(Exp(mean=density)) capped at max_trees_per_cell,
        so cells with higher density average more trees (mean 0 -> always 0).
        """
        samples = rng.exponential(np.maximum(density, 0.0))
        return np.minimum(np.floor(samples), self.max_trees_per_cell).astype(np.float32)

    # ---------------------------------------------------------------------------
    # Diffusion / advection
    # ---------------------------------------------------------------------------

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
        x_vel: np.ndarray,
        y_vel: np.ndarray,
        sigma: float,
        saturate: bool = False,
    ) -> np.ndarray:
        # Operator splitting: conservative flux advection, then conservative diffusion.
        result = self._advect_field(field, x_vel, y_vel, saturate)
        return self._diffuse(result, sigma)

    def _diffuse(self, field: np.ndarray, sigma: float) -> np.ndarray:
        # Explicit Laplacian diffusion with no-flux (reflect) boundaries. The
        # discrete Laplacian sums to zero under reflect, so this conserves the total
        # exactly (same scheme as the pressure equalisation). Each substep is a
        # convex combination (coefficient <= 0.25), so values stay within bounds.
        if sigma <= 0.0:
            return field
        variance = sigma * sigma
        n = max(1, int(math.ceil(variance / (2.0 * _MAX_DIFFUSION_COEFF))))
        coeff = variance / (2.0 * n)   # exact: total variance = 2 * coeff * n = sigma^2
        result = field
        for _ in range(n):
            result = (result + coeff * laplace(result, mode='reflect')).astype(np.float32)
        return result

    @staticmethod
    def _advection_velocity(
        x_wind_vel: np.ndarray,
        y_wind_vel: np.ndarray,
        reference_wind_magnitude: float,
        advection_strength: float,
        sigma: float,
        diffusion_substeps: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        # Per-tick displacement field in grid cells. The diffusion_substeps factor
        # preserves the previous total transport (the old loop applied the full
        # displacement once per diffusion substep).
        if reference_wind_magnitude > 0.0:
            scale = advection_strength * sigma / reference_wind_magnitude * diffusion_substeps
        else:
            scale = 0.0
        return (x_wind_vel * scale).astype(np.float32), (y_wind_vel * scale).astype(np.float32)

    @staticmethod
    def _advect_upwind(
        field: np.ndarray, x_vel: np.ndarray, y_vel: np.ndarray, saturate: bool = False
    ) -> np.ndarray:
        # One CFL-safe (<= 1 cell) first-order upwind step with no-flux walls.
        # Each shared face flux is added to one neighbour and subtracted from the
        # other, so the interior total is conserved exactly (telescoping sum).
        uf = 0.5 * (x_vel[:, :-1] + x_vel[:, 1:])   # x-velocity on interior vertical faces
        fx = np.maximum(uf, 0.0) * field[:, :-1] + np.minimum(uf, 0.0) * field[:, 1:]
        vf = 0.5 * (y_vel[:-1, :] + y_vel[1:, :])   # y-velocity on interior horizontal faces
        fy = np.maximum(vf, 0.0) * field[:-1, :] + np.minimum(vf, 0.0) * field[1:, :]

        if saturate:
            # Saturation: a cell at capacity (1.0) cannot accept more; the rejected
            # flux stays in the upstream cell. Scale each face flux by the acceptance
            # ratio of its destination cell. This conserves (flux still added once /
            # subtracted once) while keeping every cell <= 1.0.
            inflow = np.zeros_like(field)
            inflow[:, 1:]  += np.maximum(fx, 0.0)    # fx > 0 flows into the right cell
            inflow[:, :-1] += np.maximum(-fx, 0.0)   # fx < 0 flows into the left cell
            inflow[1:, :]  += np.maximum(fy, 0.0)
            inflow[:-1, :] += np.maximum(-fy, 0.0)
            capacity = np.maximum(1.0 - field, 0.0)
            accept = np.where(inflow > 0.0, np.minimum(1.0, capacity / np.maximum(inflow, 1e-12)), 1.0)
            fx = fx * np.where(fx > 0.0, accept[:, 1:], accept[:, :-1])
            fy = fy * np.where(fy > 0.0, accept[1:, :], accept[:-1, :])

        out = field.copy()
        out[:, :-1] -= fx
        out[:, 1:]  += fx
        out[:-1, :] -= fy
        out[1:, :]  += fy
        return out.astype(np.float32)

    def _advect_field(
        self, field: np.ndarray, x_vel: np.ndarray, y_vel: np.ndarray, saturate: bool = False
    ) -> np.ndarray:
        # CFL-adaptive substepping: split the displacement so each upwind substep
        # moves at most one cell, then apply it that many times.
        max_disp = float(np.abs(x_vel).max() + np.abs(y_vel).max())
        n = max(1, int(math.ceil(max_disp)))
        sx, sy = x_vel / n, y_vel / n
        result = field
        for _ in range(n):
            result = self._advect_upwind(result, sx, sy, saturate)
        return result

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
        x_vel, y_vel = self._advection_velocity(
            x_wind_vel, y_wind_vel, reference_wind_magnitude,
            self.wind_advection_strength, self.blur_sigma, self.diffusion_substeps,
        )
        return self._advect_and_diffuse_field(
            air_temperatures, x_vel, y_vel, self.blur_sigma
        )

    def temperature_relax_rate(self, terrain: np.ndarray) -> np.ndarray:
        """Per-cell radiative relaxation rate: slow in valleys, fast at altitude."""
        return (
            self.temperature_rate_low
            + (self.temperature_rate_high - self.temperature_rate_low) * terrain
        ).astype(np.float32)

    @staticmethod
    def relax_to_equilibrium(
        field: np.ndarray, target: np.ndarray, rate: np.ndarray | float
    ) -> np.ndarray:
        """Newtonian relaxation toward target, exact-exponential form (stable for any rate).

        rate may be a scalar or a per-cell array.
        """
        return (target + (field - target) * np.exp(-rate)).astype(np.float32)

    def diffuse_and_advect_oxygen(
        self,
        oxygen: np.ndarray,
        x_wind_vel: np.ndarray,
        y_wind_vel: np.ndarray,
        reference_wind_magnitude: float,
    ) -> np.ndarray:
        # Closed domain (no-flux walls): oxygen is neither lost nor replenished at
        # the edges. Conservative flux advection with saturation keeps the total
        # constant while capping each cell at 1.0 (saturated cells reject inflow).
        x_vel, y_vel = self._advection_velocity(
            x_wind_vel, y_wind_vel, reference_wind_magnitude,
            self.oxygen_advection_strength, self.oxygen_diffusion_sigma, self.diffusion_substeps,
        )
        return self._advect_and_diffuse_field(
            oxygen, x_vel, y_vel, self.oxygen_diffusion_sigma, saturate=True
        )

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
        # Burnt-out cells (mask=False) are ~massless: snap them straight to air
        # temperature (dT_fuel = dT). They carry no thermal mass, so the dT_air term
        # below leaves the air essentially unchanged — massless ash holds no heat.
        dT_fuel = np.where(mask, dT_fuel, dT).astype(np.float32)

        # Air loses the sum of heat transferred to all fuel types (energy conserving).
        # Depleted cells contribute ~0 because C_fuel = h*fuel ≈ 0.
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

        # Oxygen consumed is stoichiometric: proportional to fuel actually burned this
        # tick (summed over types), capped at available oxygen. Tying it to fuel_consumed
        # rather than ambient oxygen avoids a hot cell suffocating itself in one tick.
        oxygen_consumed = np.minimum(
            np.sum(fuel_consumed * self.oxygen_consumption_rate, axis=0),
            oxygen,
        ).astype(np.float32)  # (H, W)

        fuel   = (fuel - fuel_consumed).astype(np.float32)
        # Combustion is a real oxygen sink. oxygen_consumed <= oxygen and incoming
        # oxygen is already <= 1 (saturated advection), so no clip is needed here —
        # clipping would silently destroy/create oxygen and break conservation.
        oxygen = (oxygen - oxygen_consumed).astype(np.float32)

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
    ) -> tuple[np.ndarray, np.ndarray]:
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

        return (fuel_temperatures + dT_fuel).astype(np.float32), radiant_flux
