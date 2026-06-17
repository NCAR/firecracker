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
    rank-agnostic Simulation ops. The *_eq / terrain fields are constant for an episode but
    travel with the state so step_fields stays a pure function of it.
    """

    mass:              torch.Tensor   # column mass = surface pressure
    air_temperatures:  torch.Tensor
    ground_temperature: torch.Tensor  # surface (soil/litter) skin temperature [K]
    fuel_temperatures: torch.Tensor   # per-type stack: trailing axes (..., N_fuel, H, W)
    fuel:              torch.Tensor   # per-type stack
    oxygen:            torch.Tensor
    terrain:           torch.Tensor
    temp_eq:           torch.Tensor   # rest temperature profile (radiative-equilibrium + sponge target)
    oxygen_eq:         torch.Tensor   # oxygen replenishment target
    mass_eq:           torch.Tensor   # boundary-layer rest mass (the open-boundary sponge target)
    x_wind_vel:        torch.Tensor   # prognostic wind velocity u [m/s]
    y_wind_vel:        torch.Tensor   # prognostic wind velocity v [m/s]
    u_amb_x:           torch.Tensor   # per-world synoptic ambient wind [m/s], broadcastable to (..., H, W)
    u_amb_y:           torch.Tensor
    radiant_flux:      torch.Tensor   # last per-cell absorbed radiant flux (diagnostic)


def _resolve_device(name: str | None) -> torch.device:
    """Pick the compute device: an explicit config value wins, else CUDA if present."""
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Simulation:
    def __init__(self, cfg: dict | None = None):
        sim    = (cfg or {}).get("simulation", {})
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

        self.oxygen_diffusion_sigma:    float = float(oxygen.get("diffusion_sigma",    3.0))

        # Surface radiative energy balance (Phase 3). Sunlight is absorbed at the surface, not
        # by the (shortwave-transparent) air: bare ground and fuel each absorb solar S_net and
        # emit longwave eps*sigma*T^4, split by a vegetation cover fraction. The air is heated
        # only by sensible exchange with the ground. S_net is anchored per-cell to the rest
        # profile (S_net = eps*sigma*temp_eq^4), so radiative equilibrium sits at temp_eq
        # (= T_REF - Gamma*z); the fast (small heat-capacity) fuel/ground lead, the deep air lags.
        radiation = (cfg or {}).get("radiation", {})
        self.radiation_enabled:   bool  = bool(radiation.get("enabled", True))
        self.emissivity:          float = float(radiation.get("emissivity",          pc.EMISSIVITY))
        self.ground_heat_capacity: float = float(radiation.get("ground_heat_capacity", 1.0e5))
        self.ground_air_exchange:  float = float(radiation.get("ground_air_exchange",  10.0))
        self.cover_fuel_scale:     float = float(radiation.get("cover_fuel_scale",     1.0))

        # Oxygen replenishment toward the elevation profile (fresh-air mixing).
        relaxation = (cfg or {}).get("relaxation", {})
        self.relaxation_enabled: bool  = bool(relaxation.get("enabled", True))
        self.oxygen_rate:        float = float(relaxation.get("oxygen_rate", 0.05))

        # Phase 2 shallow-water momentum core. Wind is prognostic [m/s]; drag relaxes it
        # toward the per-map synoptic ambient wind, the column-top height gradient
        # (H = terrain + eta, eta = m*R_d*T/p_ref) forces it, eddy viscosity smooths it.
        momentum = (cfg or {}).get("momentum", {})
        self.layer_depth_ref: float = float(momentum.get("layer_depth_ref", 1000.0))
        self.reduced_gravity: float = float(momentum.get("reduced_gravity", 0.2))
        self.drag_coeff:      float = float(momentum.get("drag_coeff",      0.0005))
        self.viscosity:       float = float(momentum.get("viscosity",       50.0))
        self.sponge_width:    int   = int(momentum.get("sponge_width",      8))
        self.sponge_strength: float = float(momentum.get("sponge_strength", 10.0))
        self.cfl_target:      float = float(momentum.get("cfl_target",      0.5))
        self._sponge_cache: dict[tuple[int, int], torch.Tensor] = {}

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
        # SI specific heat [J/(kg*K)] used by the surface radiation balance (Phase 3). Kept
        # separate from the pre-SI "heat_capacity" above, which the legacy fire code still uses
        # until Phase 4 unifies them (grass ~1800, wood ~2300).
        self.fuel_specific_heat  = _ft_tensor("specific_heat",          2000.0)
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
    def compute_air_temperature(air_temperatures: torch.Tensor) -> float:
        return float(air_temperatures.mean())

    # ---------------------------------------------------------------------------
    # Shallow-water momentum core (Phase 2) — periodic stencils + solver
    # ---------------------------------------------------------------------------

    @staticmethod
    def _periodic_grad(field: torch.Tensor, dx: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Centered first derivatives [per metre] with periodic wrap on the trailing axes."""
        gx = (torch.roll(field, -1, dims=-1) - torch.roll(field, 1, dims=-1)) / (2.0 * dx)
        gy = (torch.roll(field, -1, dims=-2) - torch.roll(field, 1, dims=-2)) / (2.0 * dx)
        return gx, gy

    @staticmethod
    def _periodic_laplacian(field: torch.Tensor, dx: float) -> torch.Tensor:
        """5-point Laplacian [per m^2] with periodic wrap on the trailing axes."""
        return (
            torch.roll(field, 1, dims=-1) + torch.roll(field, -1, dims=-1)
            + torch.roll(field, 1, dims=-2) + torch.roll(field, -1, dims=-2)
            - 4.0 * field
        ) / (dx * dx)

    @staticmethod
    def _advect_periodic(phi: torch.Tensor, dispx: torch.Tensor, dispy: torch.Tensor) -> torch.Tensor:
        """One conservative first-order upwind advection substep of a density field, periodic.

        dispx/dispy are the per-substep displacements in CELLS (velocity * dt / dx) and must
        be <= 1 in magnitude for CFL stability. Each signed face flux is added to one
        neighbour and subtracted from the other (a telescoping sum over the periodic ring),
        so the total of phi is conserved exactly. Rank-agnostic via trailing-axis rolls.
        """
        # x-faces: face i sits between cell i and its +1 neighbour (wrapping at the edge).
        uf = 0.5 * (dispx + torch.roll(dispx, -1, dims=-1))
        flux = uf * torch.where(uf > 0, phi, torch.roll(phi, -1, dims=-1))   # signed flux i -> i+1
        phi = phi + torch.roll(flux, 1, dims=-1) - flux                       # gain from left, lose to right
        # y-faces.
        vf = 0.5 * (dispy + torch.roll(dispy, -1, dims=-2))
        flux = vf * torch.where(vf > 0, phi, torch.roll(phi, -1, dims=-2))
        phi = phi + torch.roll(flux, 1, dims=-2) - flux
        return phi

    def _sponge_rate(self, shape: tuple[int, int]) -> torch.Tensor:
        """Boundary relaxation-rate field [1/s], (H, W): peak at the edges, 0 in the interior.

        Implements the open domain: within sponge_width cells of any edge the wind/mass are
        relaxed toward the free-stream (inflow) and outgoing disturbances are absorbed (outflow),
        with a smooth quadratic taper so there is no sharp interface to reflect off. Cached per
        grid shape.
        """
        key = shape
        cached = self._sponge_cache.get(key)
        if cached is not None:
            return cached
        h, w = shape
        rows = torch.arange(h, device=self.device).view(-1, 1)
        cols = torch.arange(w, device=self.device).view(1, -1)
        d = torch.minimum(torch.minimum(rows, (h - 1) - rows), torch.minimum(cols, (w - 1) - cols))
        width = max(1, self.sponge_width)
        taper = (1.0 - d.to(self.dtype) / width).clamp(min=0.0) ** 2   # 1 at edge -> 0 at width
        rate = (self.sponge_strength * taper).to(self.dtype)
        self._sponge_cache[key] = rate
        return rate

    def step_dynamics(self, s: SimState) -> SimState:
        """Advance the prognostic wind, boundary-layer mass and air energy one tick.

        Shallow boundary-layer atmosphere over terrain: the wind is forced by the free-surface
        gradient (s = terrain + eta, eta = m*R_d*T/p_ref) and smoothed by viscosity, with only
        weak interior surface friction; mass, energy (E = m*c_p*T) and momentum are advected
        conservatively by that wind. Because the layer is shallow, terrain squeezes it, so
        continuity speeds the wind over crests and channels it through gaps. A boundary sponge
        relaxes the edges toward the per-map synoptic wind and the rest-state mass, making the
        domain open (inflow upwind, outflow downwind). CFL-substepped on the advective +
        gravity-wave speed. Rank-agnostic (single world or (B, H, W)).
        """
        cp, R_d, p_ref = pc.CP_AIR, pc.GAS_CONSTANT_DRY_AIR, pc.P_REF
        dx, dt = self.cell_size_m, self.dt
        g_prime, C_d, nu = self.reduced_gravity, self.drag_coeff, self.viscosity

        m, T = s.mass, s.air_temperatures
        u, v = s.x_wind_vel, s.y_wind_vel

        # Substep count from the worst-case Courant number: advective speed |u| plus the
        # gravity-wave speed c = sqrt(g'*eta).
        eta = m * R_d * T / p_ref
        wave = float(torch.sqrt((g_prime * eta).clamp(min=0.0)).max())
        flow = float((u.abs() + v.abs()).max())
        courant = (flow + wave) * dt / dx
        n = max(1, int(math.ceil(courant / self.cfl_target)))
        dts = dt / n

        for _ in range(n):
            eta = m * R_d * T / p_ref
            surface = s.terrain + eta
            gx, gy = self._periodic_grad(surface, dx)
            # Forcing: -g'*grad(s) (terrain + buoyancy), weak surface friction, eddy viscosity.
            u = u + dts * (-g_prime * gx - C_d * u + nu * self._periodic_laplacian(u, dx))
            v = v + dts * (-g_prime * gy - C_d * v + nu * self._periodic_laplacian(v, dx))

            # Conservative transport of mass, energy and momentum by the updated wind.
            dispx, dispy = u * dts / dx, v * dts / dx
            energy = m * cp * T
            mom_x, mom_y = m * u, m * v
            m      = self._advect_periodic(m,      dispx, dispy)
            energy = self._advect_periodic(energy, dispx, dispy)
            mom_x  = self._advect_periodic(mom_x,  dispx, dispy)
            mom_y  = self._advect_periodic(mom_y,  dispx, dispy)

            m_safe = m.clamp(min=1e-9)
            u, v = mom_x / m_safe, mom_y / m_safe
            T = energy / (cp * m_safe)

        # Open-boundary sponge: relax the edge belt toward the free-stream (synoptic wind and
        # rest-state mass/temperature) so the wind enters upwind and leaves downwind without
        # piling or reflecting. Exact-exponential, so stable for any sponge strength.
        decay = torch.exp(-self._sponge_rate(m.shape[-2:]) * dt)
        u = s.u_amb_x + (u - s.u_amb_x) * decay
        v = s.u_amb_y + (v - s.u_amb_y) * decay
        m = s.mass_eq + (m - s.mass_eq) * decay
        T = s.temp_eq + (T - s.temp_eq) * decay

        s.mass, s.air_temperatures, s.x_wind_vel, s.y_wind_vel = m, T, u, v
        return s

    def apply_radiation(
        self,
        air_temperatures: torch.Tensor,    # air column temperature T_a [K]      (..., H, W)
        ground_temperature: torch.Tensor,  # surface skin temperature T_g [K]    (..., H, W)
        fuel_temperatures: torch.Tensor,   # per-type fuel temperature T_f [K]   (..., N, H, W)
        fuel: torch.Tensor,                # per-type biomass [kg/m^2]           (..., N, H, W)
        mass: torch.Tensor,                # boundary-layer areal mass m [kg/m^2](..., H, W)
        temp_eq: torch.Tensor,             # rest temperature profile [K] (sets S_net)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Surface radiative energy balance over one tick (canopy-split model).

        Sunlight is absorbed at the surface -- the air is transparent to shortwave -- and split
        between bare ground and fuel by a vegetation cover fraction. Each surface body absorbs
        solar S_net and emits longwave eps*sigma*T^4, both weighted by its area fraction; the
        air is warmed only by sensible exchange with the ground (fuel<->air conduction stays in
        exchange_fuel_air_heat). The net flux S_net is fixed per-cell to the rest profile,
        S_net = eps*sigma*temp_eq^4, so radiative equilibrium sits exactly at temp_eq (= T_REF
        over flat ground): area-weighting both solar and longwave makes every body's balance
        vanish at temp_eq. Heat capacity sets the response speed -- fine fuel (~1e3 J/m^2/K)
        leads, the ground skin (~1e5) follows, the deep air column (~1e6) lags far behind, which
        is why fuels heat and dry in the sun well ahead of the air. Explicit forward Euler; all
        fluxes use the pre-step temperatures.

        Returns (air_temperatures, fuel_temperatures, ground_temperature).
        """
        eps, sigma, cp_air = self.emissivity, pc.STEFAN_BOLTZMANN, pc.CP_AIR
        dt = self.dt
        C_g, k_ga, kappa = self.ground_heat_capacity, self.ground_air_exchange, self.cover_fuel_scale

        s_net = eps * sigma * temp_eq ** 4                         # (..., H, W) per-cell anchor

        # Cover fraction (Beer-Lambert canopy interception) and per-body area shares.
        fuel_total = fuel.sum(dim=-3)                              # (..., H, W)
        cover = 1.0 - torch.exp(-fuel_total / kappa)               # (..., H, W) in [0, 1)
        a_ground = 1.0 - cover                                     # bare-ground area fraction
        safe_total = fuel_total.clamp(min=1e-12).unsqueeze(-3)
        a_fuel = cover.unsqueeze(-3) * fuel / safe_total           # (..., N, H, W), sums to cover

        # Ground: absorbs a_ground*S_net, emits a_ground*sigma*T^4, sheds sensible heat to air.
        q_ground = (
            a_ground * s_net
            - a_ground * eps * sigma * ground_temperature ** 4
            - k_ga * (ground_temperature - air_temperatures)
        )
        ground_new = ground_temperature + dt * q_ground / C_g

        # Air: warmed only by the ground sensible flux (transparent to solar/longwave).
        air_new = air_temperatures + dt * k_ga * (ground_temperature - air_temperatures) / (mass * cp_air).clamp(min=1e-9)

        # Fuel (per type): absorbs a_fuel*S_net, emits a_fuel*sigma*T^4. Only where fuel is
        # present -- a_fuel and the thermal mass both vanish as fuel -> 0, so guard the divide.
        present = fuel > self.fuel_burnt_threshold
        C_fuel = (fuel * self.fuel_specific_heat).clamp(min=1e-9)
        q_fuel = a_fuel * s_net - a_fuel * eps * sigma * fuel_temperatures ** 4
        dT_fuel = torch.where(present, dt * q_fuel / C_fuel, torch.zeros_like(fuel_temperatures))
        fuel_new = (fuel_temperatures + dT_fuel).clamp(min=0.0)

        return air_new, fuel_new, ground_new.clamp(min=0.0)

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
        # Oxygen [kg/m^3] rides the actual wind (conservative periodic advection), then mixes
        # by a small Gaussian diffusion. CFL-substepped on the displacement. The pre-SI [0,1]
        # saturation cap is dropped here; the full oxygen physics is reworked in Phase 4.
        dx, dt = self.cell_size_m, self.dt
        dispx, dispy = x_wind_vel * dt / dx, y_wind_vel * dt / dx
        n = max(1, int(math.ceil(float((dispx.abs() + dispy.abs()).max()) / self.cfl_target)))
        sx, sy = dispx / n, dispy / n
        for _ in range(n):
            oxygen = self._advect_periodic(oxygen, sx, sy)
        return self._diffuse(oxygen, self.oxygen_diffusion_sigma)

    def column_height(self, mass: torch.Tensor, terrain: torch.Tensor, air_temperatures: torch.Tensor) -> torch.Tensor:
        """Column-top geopotential height [m]: H = terrain + eta, eta = m*R_d*T/p_ref."""
        return terrain + mass * pc.GAS_CONSTANT_DRY_AIR * air_temperatures / pc.P_REF

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
        # Shallow-water momentum core: advance the prognostic wind, then advect mass and
        # energy (E = m*c_p*T) by it. This refreshes s.x/y_wind_vel too — the wind is now
        # prognostic, not a diagnostic of the pressure field.
        self.step_dynamics(s)

        # Surface radiative energy balance: sun warms the ground/fuel skin, longwave cools it,
        # and the ground sheds sensible heat to the air (which is transparent to radiation).
        if self.radiation_enabled:
            s.air_temperatures, s.fuel_temperatures, s.ground_temperature = self.apply_radiation(
                s.air_temperatures, s.ground_temperature, s.fuel_temperatures, s.fuel, s.mass, s.temp_eq
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
