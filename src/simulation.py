"""
simulation.py

Pure-numpy simulation core for the Firecracker heat-diffusion model.
No pygame dependency; safe to import in headless training environments.
"""

import math
import numpy as np
from scipy.ndimage import gaussian_filter, laplace
from scipy.signal import fftconvolve

# Stability cap for the explicit Laplacian diffusion coefficient (2D: must be < 0.25).
_MAX_DIFFUSION_COEFF: float = 0.2


class Simulation:
    def __init__(self, cfg: dict | None = None):
        sim    = (cfg or {}).get("simulation", {})
        wind   = (cfg or {}).get("wind", {})
        oxygen = (cfg or {}).get("oxygen", {})
        fire   = (cfg or {}).get("fire", {})

        self.simulation_steps_per_second: int   = int(sim.get("steps_per_second",        10))
        self.ms_per_step:                 int   = 1000 // self.simulation_steps_per_second
        self.blur_sigma:                  float = float(sim.get("blur_sigma",              1.0))

        self.wind_smooth_sigma:       float = float(wind.get("smooth_sigma",       2.0))
        self.wind_temporal_smoothing: float = float(wind.get("temporal_smoothing", 0.2))

        self.oxygen_diffusion_sigma:    float = float(oxygen.get("diffusion_sigma",    3.0))
        self.oxygen_advection_strength: float = float(oxygen.get("advection_strength", 3.0))

        relaxation = (cfg or {}).get("relaxation", {})

        # Newtonian relaxation toward the elevation equilibrium profiles (radiative
        # forcing for temperature, fresh-air replenishment for oxygen).
        self.relaxation_enabled:    bool  = bool(relaxation.get("enabled", True))
        self.temperature_rate_low:  float = float(relaxation.get("temperature_rate_low",  0.02))
        self.temperature_rate_high: float = float(relaxation.get("temperature_rate_high", 0.30))
        self.oxygen_rate:           float = float(relaxation.get("oxygen_rate",           0.05))

        convection = (cfg or {}).get("convection", {})

        # Single-layer mass/energy convection. Surface pressure is the column mass m;
        # the column height H = gamma*terrain + m*(1 + alpha*T) is taller for warm/high
        # columns. Mass and energy flow down the transport potential Phi = c_p*m + c_b*H,
        # so both the pressure gradient and the height gradient drive the air.
        self.thermal_expansion:    float = float(convection.get("thermal_expansion",    0.5))
        self.terrain_height_scale: float = float(convection.get("terrain_height_scale", 0.5))
        self.buoyancy_transport_rate: float = float(convection.get("buoyancy_transport_rate", 0.1))
        self.pressure_transport_rate: float = float(convection.get("pressure_transport_rate", 0.1))

        self.fire_enabled:                bool  = bool(fire.get("enabled", True))
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

        radiant = (cfg or {}).get("radiant_heat", {})
        self.radiant_kernel_radius:     int   = int(radiant.get("kernel_radius",      20))
        self.radiant_emission_scale:    float = float(radiant.get("emission_scale",   0.05))
        self.radiant_emission_exponent: float = float(radiant.get("emission_exponent", 4.0))
        # Fraction of each cell's emitted energy that escapes upward to space (lost from
        # the grid). The rest is redistributed to neighbours by the inverse-square kernel.
        self.sky_escape_fraction:       float = float(radiant.get("sky_escape_fraction", 0.2))

        self._radiant_kernel = self._build_radiant_kernel(self.radiant_kernel_radius)

    # ---------------------------------------------------------------------------
    # Diffusion / advection
    # ---------------------------------------------------------------------------

    @staticmethod
    def _build_radiant_kernel(radius: int) -> np.ndarray:
        size = 2 * radius + 1
        rows, cols = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
        r_sq = ((rows - radius) ** 2 + (cols - radius) ** 2).astype(np.float32)
        # 1/r² inverse-square law; the center cell contributes nothing (a cell does not
        # reabsorb its own radiation — that loss is the escape/redistribution in apply_radiant_heat).
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

    @staticmethod
    def _phi_diffuse_step(
        mass: np.ndarray, energy: np.ndarray, phi: np.ndarray, dt: float
    ) -> tuple[np.ndarray, np.ndarray]:
        # One conservative substep of relaxational diffusion: each face carries a
        # down-gradient flux F = -q_face * (Phi[R] - Phi[L]) of mass and energy, with the
        # face quantity q_face taken as the *arithmetic mean* of the two cells. Centered
        # (not upwind), because this is diffusion, not advection — the "velocity" is the
        # potential gradient itself. Each face flux is added to one neighbour and
        # subtracted from the other (telescoping), so both totals are conserved under
        # no-flux walls.
        dphix = phi[:, 1:] - phi[:, :-1]    # Phi[R] - Phi[L] across vertical faces
        dphiy = phi[1:, :] - phi[:-1, :]
        mfx = 0.5 * (mass[:, :-1] + mass[:, 1:]) * dphix * dt    # flux L -> R (sign folded in)
        efx = 0.5 * (energy[:, :-1] + energy[:, 1:]) * dphix * dt
        mfy = 0.5 * (mass[:-1, :] + mass[1:, :]) * dphiy * dt
        efy = 0.5 * (energy[:-1, :] + energy[1:, :]) * dphiy * dt

        m, e = mass.copy(), energy.copy()
        # Down-gradient: a cell loses to lower-Phi neighbours. dphix>0 means the right cell
        # is higher, so flux runs R -> L: the left cell gains (+), the right loses (-).
        m[:, :-1] += mfx; m[:, 1:] -= mfx
        e[:, :-1] += efx; e[:, 1:] -= efx
        m[:-1, :] += mfy; m[1:, :] -= mfy
        e[:-1, :] += efy; e[1:, :] -= efy
        return m.astype(np.float32), e.astype(np.float32)

    def transport_mass_energy(
        self,
        mass: np.ndarray,
        energy: np.ndarray,
        terrain: np.ndarray,
        air_temperatures: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convective transport: relax mass and energy down the transport-potential gradient.

        Phi = c_p*m + c_b*H, H = gamma*terrain + m*(1 + alpha*T). Since m*(1+alpha*T) =
        m + alpha*E, Phi is linear in the conserved fields: Phi = (c_p+c_b)*m + c_b*alpha*E
        + c_b*gamma*terrain. Mass and energy are carried by the same down-gradient flux
        F = -q * grad(Phi). This is relaxational *diffusion* (the flow vanishes where Phi is
        uniform), so it is discretised with a conservative centered finite-volume flux —
        face quantities are arithmetic means, which couples neighbours directly without the
        odd/even checkerboard a cell-centred central-difference velocity produces.

        Stability is the explicit-diffusion limit (diffusion number <~ 0.25 in 2D), which is
        far stricter than an advective CFL: with the local diffusivity D = q * dPhi/dq =
        (c_p+c_b)*m + c_b*alpha*E, the substep count is sized so dt*D stays well under the
        limit at every cell. Both totals are conserved exactly (no-flux walls); under that
        step the centered diffusion is monotone, so mass stays non-negative.
        """
        c_p, c_b = self.pressure_transport_rate, self.buoyancy_transport_rate
        gamma, alpha = self.terrain_height_scale, self.thermal_expansion
        terrain_term = (c_b * gamma * terrain).astype(np.float32)

        def phi_of(m: np.ndarray, e: np.ndarray) -> np.ndarray:
            return ((c_p + c_b) * m + c_b * alpha * e + terrain_term).astype(np.float32)

        # Local diffusivity D = q * dPhi/dq = (c_p+c_b)*m + c_b*alpha*E. Explicit 2D
        # diffusion is stable for dt*D below ~0.25 per axis; use a safety factor of 8 so a
        # cell's worst-case coupling across all four faces stays well within bounds.
        diffusivity = float(((c_p + c_b) * mass + c_b * alpha * energy).max(initial=0.0))
        n = max(1, int(math.ceil(8.0 * diffusivity)))
        dt = 1.0 / n

        for _ in range(n):
            mass, energy = self._phi_diffuse_step(mass, energy, phi_of(mass, energy), dt)

        # A little conservative thermal mixing on the energy (intensive T smoothing would
        # not conserve, so diffuse the extensive energy instead).
        energy = self._diffuse(energy, self.blur_sigma)
        return mass.astype(np.float32), energy.astype(np.float32)

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
    ) -> np.ndarray:
        # Closed domain (no-flux walls): oxygen is neither lost nor replenished at the
        # edges. Oxygen rides the *actual* wind (scaled by advection_strength), then
        # diffuses; both are conservative, and the saturating advection caps each cell
        # at 1.0 (saturated cells reject inflow). Advecting with the real wind — rather
        # than a wind-magnitude-normalised displacement — keeps oxygen transport tied to
        # the wind speed, so diffusion can balance it instead of it snapping to a piled
        # steady state.
        x_vel = (x_wind_vel * self.oxygen_advection_strength).astype(np.float32)
        y_vel = (y_wind_vel * self.oxygen_advection_strength).astype(np.float32)
        return self._advect_and_diffuse_field(
            oxygen, x_vel, y_vel, self.oxygen_diffusion_sigma, saturate=True
        )

    def column_height(self, mass: np.ndarray, terrain: np.ndarray, air_temperatures: np.ndarray) -> np.ndarray:
        """Column-top geopotential: terrain raises the base, temperature expands the column."""
        return (
            self.terrain_height_scale * terrain
            + mass * (1.0 + self.thermal_expansion * air_temperatures)
        ).astype(np.float32)

    def compute_wind_from_pressure(
        self, surface_pressure: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        grad_y, grad_x = np.gradient(surface_pressure)
        # Wind flows from high to low pressure, so negate the gradient.
        x_wind = gaussian_filter(-grad_x, sigma=self.wind_smooth_sigma).astype(np.float32)
        y_wind = gaussian_filter(-grad_y, sigma=self.wind_smooth_sigma).astype(np.float32)
        return x_wind, y_wind

    def update_wind(
        self,
        surface_pressure: np.ndarray,
        x_wind_vel: np.ndarray,
        y_wind_vel: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        new_x, new_y = self.compute_wind_from_pressure(surface_pressure)
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
        air_mass: np.ndarray,           # (H, W)
    ) -> tuple[np.ndarray, np.ndarray]:
        h = self.heat_capacities[:, np.newaxis, np.newaxis]    # (N, 1, 1)
        k = self.fuel_transfer_rates[:, np.newaxis, np.newaxis]  # (N, 1, 1)

        C_fuel = h * fuel                                      # (N, H, W) fuel thermal mass
        # Air thermal mass is the column mass (surface pressure): a dense/high-pressure
        # column changes temperature less for the same heat, a thin column more.
        C_air  = np.maximum(air_mass, 1e-6)[np.newaxis]        # (1, H, W)
        total  = C_air + C_fuel                                # >= C_air > 0, always safe

        # Mass-weighted equilibrium temperature of each air/fuel pair (the conserved
        # mean the pair relaxes toward).
        T_eq = (C_air * air_temperatures[np.newaxis] + C_fuel * fuel_temperatures) / total

        # Exact two-body relaxation over one tick: the air/fuel gap decays by
        # exp(-k / C_red), where C_red = C_air*C_fuel/total is the reduced heat capacity.
        # This folds the conduction rate and BOTH masses into a single factor in (0, 1],
        # so each side lands on the equilibrium side without overshoot for any rate or
        # masses — no clamping and no fictitious unit capacities needed.
        # Massless ash (C_fuel -> 0) drives C_red -> 0 and T_eq -> air temp, so the
        # exponent -> -inf and decay -> 0: the fuel equilibrates instantly to the air and
        # returns ~no energy to it. The transfer rate k is always positive, so the only
        # division by zero here is this benign C_fuel -> 0 limit (exp(-inf) = 0, no NaN).
        with np.errstate(divide="ignore"):
            decay = np.exp(-k * total / (C_air * C_fuel)).astype(np.float32)

        dT_fuel = ((T_eq - fuel_temperatures) * (1.0 - decay)).astype(np.float32)  # (N, H, W)

        # Air loses exactly the energy each fuel type gained (summed over types, divided
        # by the air's own thermal mass), so total air+fuel energy is conserved.
        dT_air = -(np.sum(C_fuel * dT_fuel, axis=0) / C_air[0]).astype(np.float32)

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

        h      = self.heat_capacities[:, np.newaxis, np.newaxis]
        C_fuel = h * fuel                                         # (N, H, W) thermal mass
        safe_C = np.where(fuel > self.fuel_burnt_threshold, C_fuel, 1.0)

        # Energy each burning cell radiates this tick (Stefan-Boltzmann-like, T^exponent).
        # Capped at the cell's available thermal energy so the explicit emitter cooling
        # below can never drive temperature negative, even at large emission_scale.
        emission = np.where(
            burning,
            self.radiant_emission_scale * fuel_temperatures ** self.radiant_emission_exponent,
            0.0,
        ).astype(np.float32)
        emission = np.minimum(emission, np.maximum(C_fuel * fuel_temperatures, 0.0))  # (N, H, W)

        # The emitter loses exactly what it radiates (conservation): radiation is a sink,
        # not a free source. This T^exponent loss self-limits flame temperature.
        fuel_temperatures = np.maximum(
            fuel_temperatures - np.where(fuel > self.fuel_burnt_threshold, emission / safe_C, 0.0),
            0.0,
        ).astype(np.float32)

        # A fraction escapes upward to space (lost from the grid); the rest is spread to
        # neighbours by the inverse-square kernel. fftconvolve in 'same' mode also drops
        # energy that would land off the grid edges, an additional (boundary) loss.
        total_emission = emission.sum(axis=0)                    # (H, W) energy leaving each cell
        to_neighbours  = (1.0 - self.sky_escape_fraction) * total_emission
        radiant_flux = np.clip(
            fftconvolve(to_neighbours, self._radiant_kernel, mode='same'), 0.0, None
        ).astype(np.float32)  # (H, W) energy absorbed by each cell

        # Distribute absorbed flux to each type proportional to its fuel mass fraction,
        # then divide by thermal mass to get the temperature rise.
        total_fuel = fuel.sum(axis=0)                            # (H, W)
        safe_total = np.where(total_fuel > 0, total_fuel, 1.0)
        fuel_frac  = fuel / safe_total[np.newaxis]               # (N, H, W)

        dT_fuel = np.where(
            fuel > self.fuel_burnt_threshold,
            radiant_flux[np.newaxis] * fuel_frac / safe_C,
            0.0,
        ).astype(np.float32)

        return (fuel_temperatures + dT_fuel).astype(np.float32), radiant_flux
