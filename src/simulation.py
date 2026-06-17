"""
simulation.py

Pure-torch simulation core for the Firecracker heat-diffusion model.

All state is held in torch tensors on a single device (CUDA when available, CPU
otherwise), so the per-step physics runs on the GPU during training and falls
back to the CPU on machines without one — same code path either way. No pygame
dependency; safe to import in headless training environments.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

import physics_constants as pc

# Stability cap for the explicit Laplacian diffusion coefficient (2D: must be < 0.25).
_MAX_DIFFUSION_COEFF: float = 0.2

# Observation channels, in order, produced by Simulation.build_observation. Each is a
# single (H, W) field reduced across fuel types. Extend this (and build_observation) to
# grow the channel count C.
OBS_CHANNELS: tuple[str, ...] = ("fuel_temperature", "fuel", "terrain")


@dataclass
class SimState:
    """The full set of per-tick simulation fields, shared by the single-world env and the
    batched rollout collector. Every field is a torch tensor whose trailing two axes are
    (H, W); a leading batch axis B is optional and handled transparently by the
    rank-agnostic Simulation ops. The *_eq / terrain / temp_relax_rate fields are constant
    for an episode but travel with the state so step_fields stays a pure function of it.
    """

    mass:              torch.Tensor   # column mass = surface pressure
    air_temperatures:  torch.Tensor
    fuel_temperatures: torch.Tensor   # per-type stack: trailing axes (..., N_fuel, H, W)
    fuel:              torch.Tensor   # per-type stack
    oxygen:            torch.Tensor
    terrain:           torch.Tensor
    temp_eq:           torch.Tensor   # temperature relaxation target
    oxygen_eq:         torch.Tensor   # oxygen replenishment target
    temp_relax_rate:   torch.Tensor   # per-cell radiative relaxation rate
    x_wind_vel:        torch.Tensor
    y_wind_vel:        torch.Tensor
    radiant_flux:      torch.Tensor   # last per-cell absorbed radiant flux (diagnostic)


def _resolve_device(name: str | None) -> torch.device:
    """Pick the compute device: an explicit config value wins, else CUDA if present."""
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Simulation:
    def __init__(self, cfg: dict | None = None):
        sim    = (cfg or {}).get("simulation", {})
        wind   = (cfg or {}).get("wind", {})
        oxygen = (cfg or {}).get("oxygen", {})
        fire   = (cfg or {}).get("fire", {})
        env    = (cfg or {}).get("environment", {})

        # Single device for all state. float64 throughout: the SI state spans ~1e-1..1e9
        # (mass ~1e4 kg/m^2, energy ~1e9 J, T ~3e2 K), which erodes float32 and the tight
        # conservation tolerances, so the physical model runs in double precision.
        self.device: torch.device = _resolve_device(env.get("device"))
        self.dtype:  torch.dtype  = torch.float64

        self.simulation_steps_per_second: int   = int(sim.get("steps_per_second",        10))
        self.ms_per_step:                 int   = 1000 // self.simulation_steps_per_second
        self.blur_sigma:                  float = float(sim.get("blur_sigma",              1.0))

        # Physical discretisation scales. dx is the cell edge length; dt is the wall-clock
        # duration of one tick. Both are needed by the SI physics (Phase 2+); they are wired
        # in here so every subsystem reads them from one place.
        units = (cfg or {}).get("units", {})
        self.cell_size_m: float = float(units.get("cell_size_m", pc.DEFAULT_CELL_SIZE_M))
        self.dt:          float = 1.0 / self.simulation_steps_per_second

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

        def _ft_tensor(key: str, default: float) -> torch.Tensor:
            # Per-type physics constants live on-device, shaped (N, 1, 1) so they
            # broadcast against the (N, H, W) fuel stacks.
            values = [_ft(n, key, default) for n in self.fuel_type_names]
            return torch.tensor(values, dtype=self.dtype, device=self.device).view(-1, 1, 1)

        self.heat_capacities     = _ft_tensor("heat_capacity",          1.0)
        self.ignition_thresholds = _ft_tensor("ignition_threshold",     2.0)
        self.consumption_rates   = _ft_tensor("consumption_rate",       0.001)
        self.burn_heat_scales    = _ft_tensor("burn_heat_scale",        500.0)
        self.fuel_transfer_rates = _ft_tensor("fuel_air_transfer_rate", 0.01)

        radiant = (cfg or {}).get("radiant_heat", {})
        self.radiant_kernel_radius:     int   = int(radiant.get("kernel_radius",      20))
        self.radiant_emission_scale:    float = float(radiant.get("emission_scale",   0.05))
        self.radiant_emission_exponent: float = float(radiant.get("emission_exponent", 4.0))
        # Fraction of each cell's emitted energy that escapes upward to space (lost from
        # the grid). The rest is redistributed to neighbours by the inverse-square kernel.
        self.sky_escape_fraction:       float = float(radiant.get("sky_escape_fraction", 0.2))

        self._radiant_kernel = self._build_radiant_kernel(self.radiant_kernel_radius)
        # Fixed 5-point Laplacian stencil for explicit diffusion (1,1,3,3) on-device.
        self._laplace_kernel = torch.tensor(
            [[[[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]]],
            dtype=self.dtype, device=self.device,
        )

    # ---------------------------------------------------------------------------
    # Stencil helpers (torch replacements for the old scipy ops)
    # ---------------------------------------------------------------------------

    def _laplace(self, field: torch.Tensor) -> torch.Tensor:
        # 5-point Laplacian with no-flux walls. scipy.ndimage.laplace(mode='reflect')
        # duplicates the edge sample, which for this 1-pixel stencil is exactly torch's
        # 'replicate' padding: the ghost cell equals the edge, so the boundary stencil
        # sums to zero and the total is conserved.
        # Rank-agnostic: flatten every leading dim (none, (N,), (B,), or (B, N)) into the
        # conv batch axis so this works on 2-D fields and batched/per-type stacks alike.
        *lead, h, w = field.shape
        x = F.pad(field.reshape(-1, 1, h, w), (1, 1, 1, 1), mode="replicate")
        return F.conv2d(x, self._laplace_kernel).reshape(*lead, h, w)

    def _gaussian_blur(self, field: torch.Tensor, sigma: float) -> torch.Tensor:
        # Separable Gaussian smoothing (used only for the diagnostic wind field). The
        # 1-D kernel is truncated at 4 sigma to match scipy.ndimage.gaussian_filter's
        # default radius; reflect padding keeps the edges smooth.
        if sigma <= 0.0:
            return field
        radius = max(1, int(4.0 * sigma + 0.5))
        coords = torch.arange(-radius, radius + 1, dtype=self.dtype, device=self.device)
        k = torch.exp(-0.5 * (coords / sigma) ** 2)
        k = k / k.sum()
        # Rank-agnostic: fold any leading dims into the conv batch axis (see _laplace).
        *lead, h, w = field.shape
        x = field.reshape(-1, 1, h, w)
        x = F.conv2d(F.pad(x, (radius, radius, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
        x = F.conv2d(F.pad(x, (0, 0, radius, radius), mode="reflect"), k.view(1, 1, -1, 1))
        return x.reshape(*lead, h, w)

    def _radiant_convolve(self, field: torch.Tensor) -> torch.Tensor:
        # Inverse-square redistribution. conv2d with zero padding sized to keep the
        # output the same shape drops energy that lands off the grid — the same
        # boundary loss scipy.signal.fftconvolve(mode='same') produces. The kernel is
        # radially symmetric, so cross-correlation equals true convolution.
        # Rank-agnostic: fold any leading dims into the conv batch axis (see _laplace).
        *lead, h, w = field.shape
        x = field.reshape(-1, 1, h, w)
        out = F.conv2d(x, self._radiant_kernel, padding=self.radiant_kernel_radius)
        return out.reshape(*lead, h, w)

    # ---------------------------------------------------------------------------
    # Diffusion / advection
    # ---------------------------------------------------------------------------

    def _build_radiant_kernel(self, radius: int) -> torch.Tensor:
        size = 2 * radius + 1
        coords = torch.arange(size, dtype=self.dtype, device=self.device)
        rows, cols = torch.meshgrid(coords, coords, indexing="ij")
        r_sq = (rows - radius) ** 2 + (cols - radius) ** 2
        # 1/r² inverse-square law; the center cell contributes nothing (a cell does not
        # reabsorb its own radiation — that loss is the escape/redistribution in apply_radiant_heat).
        # Avoid dividing by zero at the center by substituting 1.0 before masking it out.
        safe_r_sq = torch.where(r_sq > 0, r_sq, torch.ones_like(r_sq))
        kernel = torch.where(r_sq > 0, 1.0 / safe_r_sq, torch.zeros_like(r_sq))
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, size, size)

    def _advect_and_diffuse_field(
        self,
        field: torch.Tensor,
        x_vel: torch.Tensor,
        y_vel: torch.Tensor,
        sigma: float,
        saturate: bool = False,
    ) -> torch.Tensor:
        # Operator splitting: conservative flux advection, then conservative diffusion.
        result = self._advect_field(field, x_vel, y_vel, saturate)
        return self._diffuse(result, sigma)

    def _diffuse(self, field: torch.Tensor, sigma: float) -> torch.Tensor:
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
            result = result + coeff * self._laplace(result)
        return result

    @staticmethod
    def _advect_upwind(
        field: torch.Tensor, x_vel: torch.Tensor, y_vel: torch.Tensor, saturate: bool = False
    ) -> torch.Tensor:
        # One CFL-safe (<= 1 cell) first-order upwind step with no-flux walls.
        # Each shared face flux is added to one neighbour and subtracted from the
        # other, so the interior total is conserved exactly (telescoping sum).
        # All slices index the trailing (row, col) axes via ellipsis, so this works on
        # 2-D fields and on batched (B, H, W) stacks identically.
        uf = 0.5 * (x_vel[..., :-1] + x_vel[..., 1:])   # x-velocity on interior vertical faces
        fx = uf.clamp(min=0.0) * field[..., :-1] + uf.clamp(max=0.0) * field[..., 1:]
        vf = 0.5 * (y_vel[..., :-1, :] + y_vel[..., 1:, :])   # y-velocity on interior horizontal faces
        fy = vf.clamp(min=0.0) * field[..., :-1, :] + vf.clamp(max=0.0) * field[..., 1:, :]

        if saturate:
            # Saturation: a cell at capacity (1.0) cannot accept more; the rejected
            # flux stays in the upstream cell. Scale each face flux by the acceptance
            # ratio of its destination cell. This conserves (flux still added once /
            # subtracted once) while keeping every cell <= 1.0.
            inflow = torch.zeros_like(field)
            inflow[..., 1:]  += fx.clamp(min=0.0)     # fx > 0 flows into the right cell
            inflow[..., :-1] += (-fx).clamp(min=0.0)  # fx < 0 flows into the left cell
            inflow[..., 1:, :]  += fy.clamp(min=0.0)
            inflow[..., :-1, :] += (-fy).clamp(min=0.0)
            capacity = (1.0 - field).clamp(min=0.0)
            accept = torch.where(
                inflow > 0.0,
                (capacity / inflow.clamp(min=1e-12)).clamp(max=1.0),
                torch.ones_like(inflow),
            )
            fx = fx * torch.where(fx > 0.0, accept[..., 1:], accept[..., :-1])
            fy = fy * torch.where(fy > 0.0, accept[..., 1:, :], accept[..., :-1, :])

        out = field.clone()
        out[..., :-1] -= fx
        out[..., 1:]  += fx
        out[..., :-1, :] -= fy
        out[..., 1:, :]  += fy
        return out

    def _advect_field(
        self, field: torch.Tensor, x_vel: torch.Tensor, y_vel: torch.Tensor, saturate: bool = False
    ) -> torch.Tensor:
        # CFL-adaptive substepping: split the displacement so each upwind substep
        # moves at most one cell, then apply it that many times.
        max_disp = float(x_vel.abs().max() + y_vel.abs().max())
        n = max(1, int(math.ceil(max_disp)))
        sx, sy = x_vel / n, y_vel / n
        result = field
        for _ in range(n):
            result = self._advect_upwind(result, sx, sy, saturate)
        return result

    @staticmethod
    def compute_air_temperature(air_temperatures: torch.Tensor) -> float:
        return float(air_temperatures.mean())

    @staticmethod
    def _phi_diffuse_step(
        mass: torch.Tensor, energy: torch.Tensor, phi: torch.Tensor, dt: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # One conservative substep of relaxational diffusion: each face carries a
        # down-gradient flux F = -q_face * (Phi[R] - Phi[L]) of mass and energy, with the
        # face quantity q_face taken as the *arithmetic mean* of the two cells. Centered
        # (not upwind), because this is diffusion, not advection — the "velocity" is the
        # potential gradient itself. Each face flux is added to one neighbour and
        # subtracted from the other (telescoping), so both totals are conserved under
        # no-flux walls.
        # All slices index the trailing (row, col) axes via ellipsis, so this works on
        # 2-D fields and on batched (B, H, W) stacks identically.
        dphix = phi[..., 1:] - phi[..., :-1]    # Phi[R] - Phi[L] across vertical faces
        dphiy = phi[..., 1:, :] - phi[..., :-1, :]
        mfx = 0.5 * (mass[..., :-1] + mass[..., 1:]) * dphix * dt    # flux L -> R (sign folded in)
        efx = 0.5 * (energy[..., :-1] + energy[..., 1:]) * dphix * dt
        mfy = 0.5 * (mass[..., :-1, :] + mass[..., 1:, :]) * dphiy * dt
        efy = 0.5 * (energy[..., :-1, :] + energy[..., 1:, :]) * dphiy * dt

        m, e = mass.clone(), energy.clone()
        # Down-gradient: a cell loses to lower-Phi neighbours. dphix>0 means the right cell
        # is higher, so flux runs R -> L: the left cell gains (+), the right loses (-).
        m[..., :-1] += mfx; m[..., 1:] -= mfx
        e[..., :-1] += efx; e[..., 1:] -= efx
        m[..., :-1, :] += mfy; m[..., 1:, :] -= mfy
        e[..., :-1, :] += efy; e[..., 1:, :] -= efy
        return m, e

    def transport_mass_energy(
        self,
        mass: torch.Tensor,
        energy: torch.Tensor,
        terrain: torch.Tensor,
        air_temperatures: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        terrain_term = c_b * gamma * terrain

        def phi_of(m: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
            return (c_p + c_b) * m + c_b * alpha * e + terrain_term

        # Local diffusivity D = q * dPhi/dq = (c_p+c_b)*m + c_b*alpha*E. Explicit 2D
        # diffusion is stable for dt*D below ~0.25 per axis; use a safety factor of 8 so a
        # cell's worst-case coupling across all four faces stays well within bounds.
        diffusivity = max(0.0, float(((c_p + c_b) * mass + c_b * alpha * energy).max()))
        n = max(1, int(math.ceil(8.0 * diffusivity)))
        dt = 1.0 / n

        for _ in range(n):
            mass, energy = self._phi_diffuse_step(mass, energy, phi_of(mass, energy), dt)

        # A little conservative thermal mixing on the energy (intensive T smoothing would
        # not conserve, so diffuse the extensive energy instead).
        energy = self._diffuse(energy, self.blur_sigma)
        return mass, energy

    def temperature_relax_rate(self, terrain: torch.Tensor) -> torch.Tensor:
        """Per-cell radiative relaxation rate: slow in valleys, fast at altitude."""
        return (
            self.temperature_rate_low
            + (self.temperature_rate_high - self.temperature_rate_low) * terrain
        )

    @staticmethod
    def relax_to_equilibrium(
        field: torch.Tensor, target: torch.Tensor, rate: torch.Tensor | float
    ) -> torch.Tensor:
        """Newtonian relaxation toward target, exact-exponential form (stable for any rate).

        rate may be a scalar or a per-cell tensor.
        """
        decay = torch.exp(-rate) if torch.is_tensor(rate) else math.exp(-rate)
        return target + (field - target) * decay

    def diffuse_and_advect_oxygen(
        self,
        oxygen: torch.Tensor,
        x_wind_vel: torch.Tensor,
        y_wind_vel: torch.Tensor,
    ) -> torch.Tensor:
        # Closed domain (no-flux walls): oxygen is neither lost nor replenished at the
        # edges. Oxygen rides the *actual* wind (scaled by advection_strength), then
        # diffuses; both are conservative, and the saturating advection caps each cell
        # at 1.0 (saturated cells reject inflow). Advecting with the real wind — rather
        # than a wind-magnitude-normalised displacement — keeps oxygen transport tied to
        # the wind speed, so diffusion can balance it instead of it snapping to a piled
        # steady state.
        x_vel = x_wind_vel * self.oxygen_advection_strength
        y_vel = y_wind_vel * self.oxygen_advection_strength
        return self._advect_and_diffuse_field(
            oxygen, x_vel, y_vel, self.oxygen_diffusion_sigma, saturate=True
        )

    def column_height(self, mass: torch.Tensor, terrain: torch.Tensor, air_temperatures: torch.Tensor) -> torch.Tensor:
        """Column-top geopotential: terrain raises the base, temperature expands the column."""
        return (
            self.terrain_height_scale * terrain
            + mass * (1.0 + self.thermal_expansion * air_temperatures)
        )

    def compute_wind_from_pressure(
        self, surface_pressure: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Differentiate only the trailing (row, col) axes so a leading batch dim is left
        # untouched (each world gets its own gradient).
        grad_y, grad_x = torch.gradient(surface_pressure, dim=(-2, -1))
        # Wind flows from high to low pressure, so negate the gradient.
        x_wind = self._gaussian_blur(-grad_x, self.wind_smooth_sigma)
        y_wind = self._gaussian_blur(-grad_y, self.wind_smooth_sigma)
        return x_wind, y_wind

    def update_wind(
        self,
        surface_pressure: torch.Tensor,
        x_wind_vel: torch.Tensor,
        y_wind_vel: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        new_x, new_y = self.compute_wind_from_pressure(surface_pressure)
        s = self.wind_temporal_smoothing
        x_wind = s * new_x + (1.0 - s) * x_wind_vel
        y_wind = s * new_y + (1.0 - s) * y_wind_vel
        return x_wind, y_wind

    # ---------------------------------------------------------------------------
    # Fire
    # ---------------------------------------------------------------------------

    def exchange_fuel_air_heat(
        self,
        air_temperatures: torch.Tensor,   # (H, W)
        fuel_temperatures: torch.Tensor,  # (N, H, W)
        fuel: torch.Tensor,               # (N, H, W)
        air_mass: torch.Tensor,           # (H, W)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.heat_capacities      # (N, 1, 1)
        k = self.fuel_transfer_rates  # (N, 1, 1)

        C_fuel = h * fuel                                      # (N, H, W) fuel thermal mass
        # Air thermal mass is the column mass (surface pressure): a dense/high-pressure
        # column changes temperature less for the same heat, a thin column more.
        # unsqueeze(-3) inserts the fuel-type axis whether or not a batch dim is present.
        C_air  = air_mass.clamp(min=1e-6).unsqueeze(-3)       # (1, H, W) / (B, 1, H, W)
        total  = C_air + C_fuel                                # >= C_air > 0, always safe

        # Mass-weighted equilibrium temperature of each air/fuel pair (the conserved
        # mean the pair relaxes toward).
        T_eq = (C_air * air_temperatures.unsqueeze(-3) + C_fuel * fuel_temperatures) / total

        # Exact two-body relaxation over one tick: the air/fuel gap decays by
        # exp(-k / C_red), where C_red = C_air*C_fuel/total is the reduced heat capacity.
        # This folds the conduction rate and BOTH masses into a single factor in (0, 1],
        # so each side lands on the equilibrium side without overshoot for any rate or
        # masses — no clamping and no fictitious unit capacities needed.
        # Massless ash (C_fuel -> 0) drives C_red -> 0 and T_eq -> air temp, so the
        # exponent -> -inf and decay -> 0: the fuel equilibrates instantly to the air and
        # returns ~no energy to it. The transfer rate k is always positive, so the only
        # division by zero here is this benign C_fuel -> 0 limit (exp(-inf) = 0, no NaN).
        decay = torch.exp(-k * total / (C_air * C_fuel))

        dT_fuel = (T_eq - fuel_temperatures) * (1.0 - decay)  # (N, H, W)

        # Air loses exactly the energy each fuel type gained (summed over types, divided
        # by the air's own thermal mass), so total air+fuel energy is conserved.
        dT_air = -((C_fuel * dT_fuel).sum(dim=-3) / C_air.squeeze(-3))

        return air_temperatures + dT_air, fuel_temperatures + dT_fuel

    def update_fire(
        self,
        air_temperatures: torch.Tensor,   # (H, W)
        fuel_temperatures: torch.Tensor,  # (N, H, W)
        fuel: torch.Tensor,               # (N, H, W)
        oxygen: torch.Tensor,             # (H, W)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ign = self.ignition_thresholds  # (N, 1, 1)
        cr  = self.consumption_rates    # (N, 1, 1)
        bhs = self.burn_heat_scales     # (N, 1, 1)
        h   = self.heat_capacities      # (N, 1, 1)

        burning = (
            (fuel_temperatures >= ign) &
            (fuel > self.fuel_burnt_threshold) &
            (oxygen.unsqueeze(-3) > self.oxygen_extinction_threshold)
        )  # (N, H, W)

        fuel_consumed = torch.where(
            burning,
            fuel * (fuel_temperatures * cr).clamp(max=1.0),
            torch.zeros_like(fuel),
        )  # (N, H, W)

        # Oxygen consumed is stoichiometric: proportional to fuel actually burned this
        # tick (summed over types), capped at available oxygen. Tying it to fuel_consumed
        # rather than ambient oxygen avoids a hot cell suffocating itself in one tick.
        oxygen_consumed = torch.minimum(
            (fuel_consumed * self.oxygen_consumption_rate).sum(dim=-3),
            oxygen,
        )  # (H, W)

        fuel   = fuel - fuel_consumed
        # Combustion is a real oxygen sink. oxygen_consumed <= oxygen and incoming
        # oxygen is already <= 1 (saturated advection), so no clip is needed here —
        # clipping would silently destroy/create oxygen and break conservation.
        oxygen = oxygen - oxygen_consumed

        # Heat released per type, split between air and fuel.
        burn_heat_per_type = fuel_consumed * bhs            # (N, H, W)
        total_burn_heat    = burn_heat_per_type.sum(dim=-3)  # (H, W)

        air_temperatures = (
            air_temperatures + total_burn_heat * (1.0 - self.burn_heat_fuel_fraction)
        ).clamp(min=0.0)

        # Each type's fuel temperature rises by its share of burn heat divided by thermal mass.
        C_fuel    = h * fuel
        safe_C    = torch.where(fuel > self.fuel_burnt_threshold, C_fuel, torch.ones_like(C_fuel))
        dT_fuel   = torch.where(
            fuel > self.fuel_burnt_threshold,
            burn_heat_per_type * self.burn_heat_fuel_fraction / safe_C,
            torch.zeros_like(C_fuel),
        )
        fuel_temperatures = (fuel_temperatures + dT_fuel).clamp(min=0.0)

        return air_temperatures, fuel_temperatures, fuel, oxygen

    def apply_radiant_heat(
        self,
        fuel_temperatures: torch.Tensor,  # (N, H, W)
        fuel: torch.Tensor,               # (N, H, W)
        oxygen: torch.Tensor,             # (H, W)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ign = self.ignition_thresholds

        burning = (
            (fuel_temperatures >= ign) &
            (fuel > self.fuel_burnt_threshold) &
            (oxygen.unsqueeze(-3) > self.oxygen_extinction_threshold)
        )  # (N, H, W)

        h      = self.heat_capacities
        C_fuel = h * fuel                                         # (N, H, W) thermal mass
        safe_C = torch.where(fuel > self.fuel_burnt_threshold, C_fuel, torch.ones_like(C_fuel))

        # Energy each burning cell radiates this tick (Stefan-Boltzmann-like, T^exponent).
        # Capped at the cell's available thermal energy so the explicit emitter cooling
        # below can never drive temperature negative, even at large emission_scale.
        emission = torch.where(
            burning,
            self.radiant_emission_scale * fuel_temperatures ** self.radiant_emission_exponent,
            torch.zeros_like(fuel_temperatures),
        )
        emission = torch.minimum(emission, (C_fuel * fuel_temperatures).clamp(min=0.0))  # (N, H, W)

        # The emitter loses exactly what it radiates (conservation): radiation is a sink,
        # not a free source. This T^exponent loss self-limits flame temperature.
        fuel_temperatures = (
            fuel_temperatures
            - torch.where(fuel > self.fuel_burnt_threshold, emission / safe_C, torch.zeros_like(emission))
        ).clamp(min=0.0)

        # A fraction escapes upward to space (lost from the grid); the rest is spread to
        # neighbours by the inverse-square kernel. The 'same'-mode convolution also drops
        # energy that would land off the grid edges, an additional (boundary) loss.
        total_emission = emission.sum(dim=-3)                    # (H, W) energy leaving each cell
        to_neighbours  = (1.0 - self.sky_escape_fraction) * total_emission
        radiant_flux = self._radiant_convolve(to_neighbours).clamp(min=0.0)  # (H, W) absorbed per cell

        # Distribute absorbed flux to each type proportional to its fuel mass fraction,
        # then divide by thermal mass to get the temperature rise.
        total_fuel = fuel.sum(dim=-3)                            # (H, W)
        safe_total = torch.where(total_fuel > 0, total_fuel, torch.ones_like(total_fuel))
        fuel_frac  = fuel / safe_total.unsqueeze(-3)             # (N, H, W)

        dT_fuel = torch.where(
            fuel > self.fuel_burnt_threshold,
            radiant_flux.unsqueeze(-3) * fuel_frac / safe_C,
            torch.zeros_like(C_fuel),
        )

        return fuel_temperatures + dT_fuel, radiant_flux

    # ---------------------------------------------------------------------------
    # Full per-tick step + observation
    # ---------------------------------------------------------------------------

    def step_fields(self, s: SimState) -> SimState:
        """Advance every field one tick (the pure physics of one env step).

        This is the single source of truth for the per-tick sequence — both
        FirecrackerEnv.step (single world) and BatchedRollout (B worlds) call it, so they
        can never drift apart. It carries no rendering/EMA bookkeeping. Because every op is
        rank-agnostic, the same call advances a 2-D world or a batched (B, H, W) stack.

        The state is mutated in place and returned for convenience.
        """
        # Convective transport: mass and energy (E = m*T) flow together down the transport
        # potential, so heat is carried by the air mass. T = E/m is recovered afterward.
        energy = s.mass * s.air_temperatures
        s.mass, energy = self.transport_mass_energy(s.mass, energy, s.terrain, s.air_temperatures)
        s.air_temperatures = energy / s.mass.clamp(min=1e-6)

        # Surface radiative relaxation (sun warms, space cools); a sink/source forcing.
        if self.relaxation_enabled:
            s.air_temperatures = self.relax_to_equilibrium(
                s.air_temperatures, s.temp_eq, s.temp_relax_rate
            )

        s.air_temperatures, s.fuel_temperatures = self.exchange_fuel_air_heat(
            s.air_temperatures, s.fuel_temperatures, s.fuel, s.mass
        )

        s.oxygen = self.diffuse_and_advect_oxygen(s.oxygen, s.x_wind_vel, s.y_wind_vel)
        # Fresh-air replenishment toward the elevation oxygen profile (combustion below
        # still draws this down, so a vigorous fire can outpace it locally).
        if self.relaxation_enabled:
            s.oxygen = self.relax_to_equilibrium(s.oxygen, s.oxygen_eq, self.oxygen_rate)

        if self.fire_enabled:
            s.air_temperatures, s.fuel_temperatures, s.fuel, s.oxygen = self.update_fire(
                s.air_temperatures, s.fuel_temperatures, s.fuel, s.oxygen
            )
            s.fuel_temperatures, s.radiant_flux = self.apply_radiant_heat(
                s.fuel_temperatures, s.fuel, s.oxygen
            )

        # Refresh the surface wind (-grad pressure) for display and next tick's oxygen
        # advection. The wind is purely diagnostic; mass moves via the transport above.
        s.x_wind_vel, s.y_wind_vel = self.update_wind(s.mass, s.x_wind_vel, s.y_wind_vel)
        return s

    @staticmethod
    def build_observation(
        fuel_temperatures: torch.Tensor,  # (..., N_fuel, H, W)
        fuel: torch.Tensor,               # (..., N_fuel, H, W)
        terrain: torch.Tensor,            # (..., H, W)
    ) -> torch.Tensor:
        """Stack the OBS_CHANNELS into a (..., C, H, W) observation.

        Each channel is reduced across fuel types: fuel temperature is the hottest type
        (amax), fuel/vegetation is the total mass (sum), terrain is passed through. With a
        leading batch axis the result is the B x C x N x N tensor the world model trains on.
        """
        channels = [
            fuel_temperatures.amax(dim=-3),  # fuel_temperature: hottest fuel type
            fuel.sum(dim=-3),                # fuel: total vegetation mass
            terrain,                         # terrain: elevation
        ]
        return torch.stack(channels, dim=-3)
