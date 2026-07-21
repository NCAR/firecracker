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

# Observation channels, in order, produced by Simulation.build_observation. Each is a single
# (H, W) field. The stack is the near-surface air temperature, then one temperature channel per
# configured fuel type (fuel_temperature_<name>), then one vegetation-mass channel per fuel type
# (fuel_<name>), then the two signed terrain-slope components (slope_x/slope_y), the two near-surface
# wind components, the one-hot vegetation biome (one biome_<name> channel per biome), and finally a
# binary "ignited" mask. It is slope, not terrain height, that skews fire spread (Rothermel slope
# effect), and a conv stack struggles to synthesize that spatial derivative from a raw height field,
# so the slope is supplied directly. Because the per-fuel-type channels make the count depend on the
# configured fuel types, callers derive the ordered names via obs_channel_names(fuel_type_names)
# rather than reading a fixed tuple. OBS_CHANNELS_PRE/POST are the type-independent channels that
# bracket the two fuel blocks; every fuel_temperature_<name> channel shares the "fuel_temperature"
# window and every fuel_<name> channel the "fuel" window in physics_constants.OBS_NORM. The biome and
# ignited channels are already 0/1, so they share the identity "biome" / "ignited" windows.
OBS_CHANNELS_PRE:  tuple[str, ...] = ("air_temperature",)
OBS_CHANNELS_POST: tuple[str, ...] = ("slope_x", "slope_y", "wind_x", "wind_y")
# The three non-overlapping vegetation biomes, in the label order gen_maps.classify_biomes returns
# (0 woodland / 1 grassland / 2 shrubland). The biome is static per-cell state that drives fuel
# placement but is NOT an observation channel -- the per-type fuel-mass channels already carry the
# vegetation the model needs; the labels survive only for the (non-reconstructable) BIOME view.
BIOME_NAMES: tuple[str, ...] = ("woodland", "grassland", "shrubland")
# Trailing binary mask: 1 where any fuel type at the cell is at/above its own ignition threshold
# (i.e. the cell is burning), 0 otherwise. Matches the "burning" test used for rendering and stats.
OBS_CHANNEL_IGNITED: str = "ignited"


def obs_channel_names(fuel_type_names: list[str]) -> tuple[str, ...]:
    """The ordered observation channel names for a run with these fuel types: the fixed pre
    channels (air temperature), one fuel_temperature_<name> channel per fuel type, then one
    fuel_<name> mass channel per fuel type (both in fuel_type_names order), then the fixed post
    channels (slope + wind) and the trailing ignited mask. Matches the channel order produced by
    Simulation.build_observation."""
    return (
        OBS_CHANNELS_PRE
        + tuple(f"fuel_temperature_{n}" for n in fuel_type_names)
        + tuple(f"fuel_{n}" for n in fuel_type_names)
        + OBS_CHANNELS_POST
        + (OBS_CHANNEL_IGNITED,)
    )


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
    # Phase 6 (Option 2): the terrain-channeling surface wind. channel_gain is the static per-world
    # Venturi speedup (>= 1, precomputed from terrain + the synoptic wind); x/y_wind_fire is the
    # near-surface wind the fire and oxygen actually read (prognostic wind x gain). All three are
    # optional so hand-built states (and the single-layer path) keep working unchanged.
    channel_gain:      torch.Tensor | None = None
    x_wind_fire:       torch.Tensor | None = None
    y_wind_fire:       torch.Tensor | None = None


# --- Vertical air levels (Phase 1) -------------------------------------------------------------
# air_temperatures and temp_eq carry a level axis on -3 (mirroring the fuel stack): (..., L, H, W).
# Level 0 is the surface air. Until the fuel stages and dynamics go per-level (Phases 2-3), every
# stage still reads/writes only the surface level through these two helpers, so the physics is
# unchanged for L = 1.
def _surface_level(stack: torch.Tensor) -> torch.Tensor:
    """The surface (level-0) (..., H, W) view of a (..., L, H, W) air/temp stack."""
    return stack.select(-3, 0)


def _with_surface(stack: torch.Tensor, new_surface: torch.Tensor) -> torch.Tensor:
    """Return `stack` with its surface level replaced by `new_surface` (..., H, W)."""
    if stack.shape[-3] == 1:
        return new_surface.unsqueeze(-3)
    return torch.cat(
        [new_surface.unsqueeze(-3), stack.narrow(-3, 1, stack.shape[-3] - 1)], dim=-3
    )


def lift_air_levels(field: torch.Tensor, n_levels: int) -> torch.Tensor:
    """Replicate a single (..., H, W) air/temperature field across `n_levels` levels on axis -3
    (uniform init, no lapse yet): returns (..., L, H, W). Used to seed the multi-level air state
    from a single stored/loaded field."""
    return torch.stack([field] * n_levels, dim=-3)


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

        # Physical discretisation scales. dx is the cell edge length; dt is the amount of
        # physical time one tick advances. Both are needed by the SI physics (Phase 2+); they are
        # wired in here so every subsystem reads them from one place. dt is decoupled from the
        # display tick-rate: seconds_per_tick sets how much physics a tick covers (default 0.1 s,
        # the legacy 1/steps_per_second), while steps_per_second above is only the real-time
        # display throttle. A larger seconds_per_tick simulates more physics per tick (faster
        # than real time) at the cost of temporal resolution; the wind core CFL-substeps and the
        # combustion/heat-exchange integrators are exact-exponential, so it stays stable.
        units = (cfg or {}).get("units", {})
        self.cell_size_m: float = float(units.get("cell_size_m", pc.DEFAULT_CELL_SIZE_M))
        self.dt:          float = float(units.get("seconds_per_tick", 1.0 / self.simulation_steps_per_second))

        self.oxygen_diffusion_sigma:    float = float(oxygen.get("diffusion_sigma",    3.0))

        # Surface radiative energy balance (Phase 3). Sunlight is absorbed at the surface, not
        # by the (shortwave-transparent) air: bare ground and fuel each absorb solar S_net and
        # emit longwave eps*sigma*T^4, split by a vegetation cover fraction. The air is heated
        # only by sensible exchange with the ground. S_net is anchored per-cell to the rest
        # profile (S_net = eps*sigma*temp_eq^4), so radiative equilibrium sits at temp_eq -- the
        # baked surface temperature (elevation profile warmed by sun-facing exposure, cooled in
        # shadow), so sunlit slopes settle warmer and shaded valleys cooler; the fast (small
        # heat-capacity) fuel/ground lead, the deep air lags.
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

        # Plume vertical venting (the air-temperature sink). The combustion air-share heat injects
        # into the shallow plume slab (update_fire), but the air field otherwise has no vertical
        # export -- it is radiatively transparent and only loses heat by weak ground exchange and
        # horizontal advection -- so over a large or sustained fire the plume heat accumulates without
        # bound and conducts back into the fuel, a temperature runaway. A real plume rises buoyantly
        # and carries that heat up and out of the modelled slab while fresh air entrains; we model that
        # as an exact-exponential relaxation of the super-ambient air toward the rest profile at rate
        # lambda ~ updraft / d_plume (the inverse plume-residence time). See vent_plume_heat.
        venting = (cfg or {}).get("venting", {})
        self.venting_enabled: bool  = bool(venting.get("enabled", True))
        self.plume_vent_rate: float = float(venting.get("rate",    0.05))

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

        # Phase 6 (Option 2): diagnostic terrain-channeling surface wind. The prognostic core above
        # is terrain-responsive (windward blocking, lee/foehn acceleration) but its level-lid layer
        # *deepens* in valleys, so it makes gaps slow -- it cannot produce the gap/Venturi speedup.
        # Rather than a second prognostic layer, we impose the continuity Venturi as a rule: a static
        # per-map gain (computed from the cross-stream relief and the fixed synoptic wind direction)
        # speeds up -- direction unchanged -- the wind the fire and oxygen read. The prognostic field
        # is untouched. See compute_channel_gain and docs/units.md.
        channeling = (cfg or {}).get("channeling", {})
        self.channeling_enabled:    bool  = bool(channeling.get("enabled", True))
        chan_radius_m                     = float(channeling.get("radius_m", 300.0))
        self.channel_radius:        int   = max(1, round(chan_radius_m / self.cell_size_m))
        self.channel_height_scale_m: float = float(channeling.get("height_scale_m", 100.0))
        self.channel_gain_max:      float = float(channeling.get("gain_max", 2.0))
        self.channel_smooth_sigma:  float = float(channeling.get("smooth_sigma", 1.0))

        # Combustion (Phase 4). The reaction rate is a smooth Arrhenius law limited by oxygen,
        # so there is no hard ignition threshold -- cold fuel is inert because the exponential
        # is vanishingly small at ambient temperature. The oxygen field is the near-surface
        # combustion-layer O2 density [kg/m^3]; combustion_mixing_depth is that layer's depth, so
        # the areal burn s*Dfuel [kg/m^2] draws from a d_mix-deep slab and a vigorous fire can
        # locally deplete its own oxygen. Resupply is modelled separately: advection (the wind
        # feeding fresh air, step_dynamics) horizontally and the oxygen relaxation (fresh air
        # mixing down from aloft) vertically. This is the ventilation-limited regime -- sparse
        # fuel breathes freely, dense fuel chokes and leans on wind/mixing to keep burning.
        self.fire_enabled:                bool  = bool(fire.get("enabled", True))
        self.fuel_burnt_threshold:        float = float(fire.get("burnt_threshold",          0.01))
        self.burn_heat_fuel_fraction:     float = float(fire.get("burn_heat_fuel_fraction",  0.05))
        self.combustion_mixing_depth:     float = float(fire.get("combustion_mixing_depth",  30.0))
        # O2 film-transport coefficient h [kg/(m^2*s)] (Phase 5.5): the diffusive oxygen mass flux
        # reaching a fuel surface, which caps the burn rate once the fuel is flame-hot (the kinetics
        # are no longer limiting). Sets the transport-limited surface flux psi_dif = h*(O2/O2_ref).
        self.surface_mass_transfer:       float = float(fire.get("surface_mass_transfer",    0.02))
        # Plume depth d_plume [m] (Phase 5.5b): the shallow near-surface layer the combustion
        # air-share heat is injected into (rather than the full ~1 km column), so the burn raises
        # a strong, advectable air-temperature signal that lifts eta and drives the indraft.
        self.plume_mixing_depth:          float = float(fire.get("plume_mixing_depth",        100.0))
        # Vertical air levels: the near-surface air resolves into stacked slabs -- surface (0-5 m) and
        # canopy (5-15 m), plus the above-canopy remainder. air_level_depths are the two resolved slab
        # depths [m]; the third level is the rest of the boundary-layer column. Level 0 is the surface.
        # Phase 1 keeps the dynamics/buoyancy and fuel coupling on the surface level; Phase 2 splits the
        # fuel<->air coupling across levels (see fuel_levels below).
        air_levels = (cfg or {}).get("air_levels", {})
        self.air_level_depths: tuple[float, ...] = tuple(
            float(x) for x in air_levels.get("depths", (5.0, 10.0))
        )
        self.num_air_levels: int = len(self.air_level_depths) + 1
        # Above-canopy slab depth [m] (the deep remainder that vents to space). Only fuel-free levels
        # use it, so its exact value affects the vent's diagnostic temperature, not any fuel coupling.
        self.air_above_depth:  float = float(air_levels.get("above_depth", 100.0))
        # Inter-level upward convection: gap-relaxation rate lambda [1/s] and its toggle (convect_air_levels).
        self.air_transfer_rate: float = float(air_levels.get("transfer_rate", 0.5))
        self.air_convection_enabled: bool = bool(air_levels.get("enabled", True))
        # Bole-segment vertical conduction (conduct_bole_segments): the trunk carries fire between its
        # stacked segments (tree_bole_surface <-> tree_bole_canopy). Anisotropic gap-relaxation rates
        # [1/s]: k_up (surface segment hotter -> fire climbs the trunk) >> k_down (canopy hotter -> fire
        # reluctant to descend). Phase-4-tunable placeholders.
        bole = (cfg or {}).get("bole", {})
        self.bole_conduction_up:   float = float(bole.get("conduction_up",   1.0))
        self.bole_conduction_down: float = float(bole.get("conduction_down", 0.1))
        # Crown-initiation preheat: a fraction of the surface combustion air-share conducts straight
        # into the co-located surface bole (the trunk base basking in the surface flames) instead of
        # warming the surface air. Decoupled from convective_fraction (which sets ROS), so a heavy
        # trunk crosses ignition only under a SUSTAINED surface fire -- crowning is intensity-gated by
        # fire duration. See the bole-preheat term in update_fire.
        self.bole_preheat_fraction: float = float(bole.get("preheat_fraction", 0.6))
        # Convective fire spread (Phase 5.5c). Radiation preheats the fuel ahead but is too weak to
        # ignite a neighbour at realistic flame temperatures, so ignition is carried by convection: a
        # fraction of the plume's air-share heat is convected to nearby fuel to ignite it (hot plume
        # gas / flame contact) -- the mechanism that propagates the front (convection-dominant spread,
        # realistic for grass). convective_fraction is that share; it is deposited only into
        # sub-flaming fuel (below flame_gate_temperature -- flaming fuel is combustion-controlled, so
        # this bounds the front temperature) over a short convective_radius_m reach, skewed by a von
        # Mises angular kernel whose bias vector blends the unit wind (convective_wind_bias -> lee
        # bias, leg c) with the upslope terrain gradient (convective_slope_bias -> Rothermel slope
        # effect, leg d). The slope term scales as tan^2(slope) so spread skews uphill ever harder on
        # steeper ground; the bias magnitude is the forward concentration, so wind and slope aligning
        # focuses the deposit into a tighter, faster head (no saturation).
        self.convective_fraction:         float = float(fire.get("convective_fraction",       0.12))
        self.flame_gate_temperature:      float = float(fire.get("flame_gate_temperature",    700.0))
        self.convective_wind_bias:        float = float(fire.get("convective_wind_bias",      1.0))
        # Reference wind speed [m/s] that the convective bias normalises by (Phase 6, Option 2): the
        # wind term is convective_wind_bias*(wind/u_ref), so wind *speed* (not just direction) drives
        # the forward concentration -- a faster wind throws the ignition heat harder downwind. At
        # u_ref the term matches the old unit-wind form; below it the forcing is gentler, above it
        # stronger. Set near a light prevailing wind so 5 m/s already bites and 10-15 m/s is strong.
        self.convective_wind_ref:         float = float(fire.get("convective_wind_ref",       10.0))
        self.convective_slope_bias:       float = float(fire.get("convective_slope_bias",     2.5))
        conv_radius_m                           = float(fire.get("convective_radius_m",       10.0))
        self.convective_radius:           int   = max(1, round(conv_radius_m / self.cell_size_m))
        # The convective deposit uses a von Mises angular kernel exp(b.offset_hat) (see
        # _convective_deposit): the bias vector b = wind + slope acts as concentration*direction,
        # non-negative for any |b|, so wind and slope aligning keeps focusing the deposit forward
        # (no saturation). convective_concentration_max caps |b| only to keep exp() from overflowing
        # at the spurious terrain-wrap gradient -- a numerical guard, not a physical limit.
        self._conv_concentration_max:     float = float(fire.get("convective_concentration_max", 12.0))

        # ---------------------------------------------------------------------------
        # Fuel types — parsed in config order; add subtables to expand.
        # ---------------------------------------------------------------------------

        fuel_types_cfg = (cfg or {}).get("fuel_types", {})
        self.fuel_type_names: list[str] = list(fuel_types_cfg.keys())
        self.num_fuel_types:  int       = len(self.fuel_type_names)
        # Fuel -> air-level index (Phase 2 coupling): canopy-height fuels coupled to the canopy air
        # (level 1), everything else to the surface air (level 0). A tree's foliage and its upper
        # (canopy) bole segment sit in the crown; grass, shrub and the lower bole are at the surface.
        self.fuel_levels: list[int] = [
            1 if "canopy" in n else 0 for n in self.fuel_type_names
        ]
        # Precompute for the per-level fuel<->air coupling (exchange_fuel_air_heat): the fuel->level
        # map as an index tensor for index_select/index_add_, and each air level's heat capacity
        # C = rho_ref*depth*cp_air [J/(m^2*K)] over the thin slab it represents. The surface (5 m) and
        # canopy (10 m) slabs are ~5.83 and ~11.65 kg/m^2 -- ~300x lighter than the full ~1747 kg/m^2
        # boundary-layer column the old single-level coupling used -- so a fuel warms *its* level's air
        # strongly and directly; the above-canopy level holds the deep remainder and only vents.
        self._fuel_level_index: torch.Tensor = torch.tensor(
            self.fuel_levels, dtype=torch.long, device=self.device
        )
        level_depths = torch.tensor(
            list(self.air_level_depths) + [self.air_above_depth],
            dtype=self.dtype, device=self.device,
        )
        self._air_level_capacity: torch.Tensor = (
            pc.RHO_REF * level_depths * pc.CP_AIR
        ).view(-1, 1, 1)   # (L, 1, 1) [J/(m^2*K)]
        # The distinct air levels that actually carry fuel (Stage 2 per-level combustion): the
        # convective-ignition deposit runs once per such level so a burning cell's level-L plume ignites
        # only neighbours' level-L fuels (the ladder during spread). Above-canopy (fuel-free) is skipped.
        self._levels_with_fuel: list[int] = sorted(set(self.fuel_levels))
        # Bole-segment indices for conduct_bole_segments (None when a config has no split trunk, e.g.
        # the single-fuel test worlds -> the stage is then a no-op).
        self._bole_surface_idx: int | None = (
            self.fuel_type_names.index("tree_bole_surface")
            if "tree_bole_surface" in self.fuel_type_names else None
        )
        self._bole_canopy_idx: int | None = (
            self.fuel_type_names.index("tree_bole_canopy")
            if "tree_bole_canopy" in self.fuel_type_names else None
        )

        def _ft(name: str, key: str, default: float) -> float:
            return float(fuel_types_cfg.get(name, {}).get(key, default))

        def _ft_tensor(key: str, default: float) -> torch.Tensor:
            # Per-type physics constants live on-device, shaped (N, 1, 1) so they
            # broadcast against the (N, H, W) fuel stacks.
            values = [_ft(n, key, default) for n in self.fuel_type_names]
            return torch.tensor(values, dtype=self.dtype, device=self.device).view(-1, 1, 1)

        # SI specific heat [J/(kg*K)]: the single thermal-mass constant used by the radiation
        # balance, the fuel<->air conduction, and combustion (grass ~1800, wood ~2300).
        self.fuel_specific_heat  = _ft_tensor("specific_heat",          2000.0)
        self.ignition_thresholds = _ft_tensor("ignition_threshold",     573.0)
        # Combustion constants. HHV is the heat of combustion [J/kg]; stoich_oxygen is the O2
        # demand [kg O2/kg fuel].
        self.heat_of_combustion  = _ft_tensor("heat_of_combustion",     1.6e7)
        self.stoich_oxygen       = _ft_tensor("stoich_oxygen",          1.4)
        # Surface-area combustion (Phase 5.5). Fuel reacts at its surface, not in bulk, so the burn
        # rate scales with the reactive surface area a_s = sigma/rho_p [m^2/kg]: sigma is the
        # surface-area-to-volume ratio (cured grass ~12000/m, a 10 cm log ~40/m) and rho_p the solid
        # particle density (~500 kg/m^3 -- wood and grass are the same material). High-SAV grass
        # flashes, low-SAV logs smoulder, from one geometric property. arrhenius_pre is now the
        # surface pre-exponential B [kg/(m^2*s)] of the kinetic flux psi_kin = B*exp(-E_a/(R*T_fuel)).
        self.arrhenius_pre       = _ft_tensor("arrhenius_pre",          1.0e8)
        self.activation_energy   = _ft_tensor("activation_energy",      1.0e5)
        sigma_sav                = _ft_tensor("surface_area_to_volume", 12000.0)  # [1/m]
        particle_density         = _ft_tensor("particle_density",       500.0)    # [kg/m^3]
        self.fuel_specific_surface = sigma_sav / particle_density                 # a_s [m^2/kg]
        # Fuel<->air convective conductance k [W/(m^2*K)] for exchange_fuel_air_heat.
        self.fuel_transfer_rates = _ft_tensor("fuel_air_transfer_rate", 12.0)

        radiant = (cfg or {}).get("radiant_heat", {})
        # Flame radiant transfer toggle: the convective-vs-radiant spread split. Off isolates
        # convective ignition (no flame preheat of neighbours) -- preheat alone cannot sustain a
        # front, but combined with convection it speeds spread (see tests/test_spread.py).
        self.radiant_heat_enabled:      bool  = bool(radiant.get("enabled", True))
        # Flame radiative reach: the inverse-square kernel spans this radius, set in meters and
        # converted to cells by the grid spacing (so the physical range is resolution-independent).
        radius_m = float(radiant.get("kernel_radius_m", 100.0))
        self.radiant_kernel_radius:     int   = max(1, round(radius_m / self.cell_size_m))
        # Fraction of each cell's super-ambient flame emission that escapes upward to space (lost
        # from the grid). The rest is redistributed to neighbours by the inverse-square kernel.
        self.sky_escape_fraction:       float = float(radiant.get("sky_escape_fraction", 0.2))

        self._radiant_kernel = self._build_radiant_kernel(self.radiant_kernel_radius)
        # rfft2 of the radiant kernel, cached per convolution size (see _radiant_convolve).
        self._radiant_kernel_fft_cache: dict[tuple[int, int], torch.Tensor] = {}
        # Neighbour-offset table for the convective-ignition deposit (von Mises angular kernel,
        # Phase 5.5c/d): a Gaussian-weighted neighbourhood skewed per-cell by the wind+slope bias.
        self._conv_offsets = self._build_convective_offsets(self.convective_radius)
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
        # Inverse-square redistribution, done as an FFT linear convolution: for the wide
        # (21x21 at kernel_radius_m=100) kernel this is ~20x faster than the equivalent
        # spatial conv2d and matches it to float64 round-off. Zero-padding to H+2r then
        # cropping the central HxW reproduces the same off-grid energy loss conv2d's
        # padding=r gives (scipy.signal.fftconvolve(mode='same')); the kernel is radially
        # symmetric, so cross-correlation equals convolution. Rank-agnostic via leading dims.
        *lead, h, w = field.shape
        r = self.radiant_kernel_radius
        full_h, full_w = h + 2 * r, w + 2 * r
        spec = torch.fft.rfft2(field, s=(full_h, full_w)) * self._radiant_kernel_fft(full_h, full_w)
        out = torch.fft.irfft2(spec, s=(full_h, full_w))
        return out[..., r:r + h, r:r + w]

    def _radiant_kernel_fft(self, full_h: int, full_w: int) -> torch.Tensor:
        """Cached rfft2 of the inverse-square kernel, padded to the convolution size."""
        key = (full_h, full_w)
        cached = self._radiant_kernel_fft_cache.get(key)
        if cached is None:
            ker = self._radiant_kernel.reshape(self._radiant_kernel.shape[-2:])
            cached = torch.fft.rfft2(ker, s=(full_h, full_w))
            self._radiant_kernel_fft_cache[key] = cached
        return cached

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

    def _build_convective_offsets(self, radius: int) -> list[tuple[int, int, float, float, float]]:
        """Neighbour-offset table for the convective deposit (von Mises angular kernel).

        Returns, for every neighbour within the radius (centre excluded), a tuple
        (drow, dcol, k0_weight, dhat_x, dhat_y): the integer offset, a normalised Gaussian weight
        over the neighbourhood, and the unit offset direction (x = cols, y = rows). The deposit
        loops this table, so a small convective_radius (1 by default) is a handful of offsets.
        """
        size = 2 * radius + 1
        coords = torch.arange(size, dtype=self.dtype, device=self.device) - radius
        rows, cols = torch.meshgrid(coords, coords, indexing="ij")
        r_sq = rows ** 2 + cols ** 2
        k0 = torch.where(r_sq > 0, torch.exp(-r_sq / (2.0 * (radius / 1.5) ** 2)), torch.zeros_like(r_sq))
        k0 = k0 / k0.sum()
        r = torch.sqrt(r_sq.clamp(min=1e-9))
        offsets: list[tuple[int, int, float, float, float]] = []
        for i in range(size):
            for j in range(size):
                if float(r_sq[i, j]) > 0:
                    offsets.append((
                        int(rows[i, j]), int(cols[i, j]), float(k0[i, j]),
                        float(cols[i, j] / r[i, j]), float(rows[i, j] / r[i, j]),
                    ))
        return offsets

    def _convective_deposit(self, source: torch.Tensor, bx: torch.Tensor, by: torch.Tensor) -> torch.Tensor:
        """Redistribute `source` [.., H, W] to neighbours, skewed along the per-cell bias (bx, by).

        A von Mises angular kernel: each source cell sheds k0(delta)*exp(b . delta_hat) to neighbour
        offset delta, normalised per source so it conserves the total it convects. The bias vector b
        acts as concentration*direction: as |b| grows the deposit focuses ever more sharply forward,
        with every weight >= 0, so wind and slope aligning keeps concentrating the deposit (no
        saturation). |b| is clamped only to keep exp() from overflowing (a numerical ceiling, not a
        physical limit); at the edge grad(z) is spuriously huge, and that guard keeps it finite (it
        sits in the sponge, where there is no fire). Uses zero-fill shifts, so heat convected past
        the domain edge leaves the grid rather than wrapping to the far side (the domain is a finite
        2.5 km patch, not a torus). Rank-agnostic via the trailing axes.
        """
        mag = torch.sqrt(bx ** 2 + by ** 2)
        scale = (self._conv_concentration_max / mag.clamp(min=1e-12)).clamp(max=1.0)
        bx, by = bx * scale, by * scale
        exps = [torch.exp(bx * dhx + by * dhy) for (_, _, _, dhx, dhy) in self._conv_offsets]
        z = sum(k0w * e for (_, _, k0w, _, _), e in zip(self._conv_offsets, exps))
        g = source / z.clamp(min=1e-30)
        out = torch.zeros_like(source)
        for (drow, dcol, k0w, _, _), e in zip(self._conv_offsets, exps):
            out = out + k0w * self._shift_zero(g * e, drow, dcol)
        return out

    @staticmethod
    def _shift_zero(field: torch.Tensor, drow: int, dcol: int) -> torch.Tensor:
        """Shift the trailing two axes by (drow, dcol) like torch.roll, but fill the vacated band
        with zeros instead of wrapping. Cells shifted in from beyond the domain edge contribute
        nothing, so a deposit aimed off-grid is lost rather than reappearing on the opposite side.
        """
        out = torch.roll(field, shifts=(drow, dcol), dims=(-2, -1))
        if drow > 0:
            out[..., :drow, :] = 0.0
        elif drow < 0:
            out[..., drow:, :] = 0.0
        if dcol > 0:
            out[..., :, :dcol] = 0.0
        elif dcol < 0:
            out[..., :, dcol:] = 0.0
        return out

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

    def compute_channel_gain(
        self, terrain: torch.Tensor, u_amb_x: torch.Tensor, u_amb_y: torch.Tensor
    ) -> torch.Tensor:
        """Static per-map terrain-channeling gain (>= 1): the continuity Venturi the fire reads.

        The prognostic core (step_dynamics) gives a deep, *slow* layer in low ground, so it cannot
        speed wind up through a gap. We impose that speedup as a rule. For every cell, scan the
        cross-stream profile -- the line perpendicular to the fixed synoptic wind -- out to
        channel_radius cells on each side and measure how *open* the channel stays, graded by relief:

          rise_k     = max(terrain_k - terrain_self, 0)              relief above me at step k
          openness_k = clamp(1 - rise_k / channel_height_scale_m, 0, 1)   1 fully open, 0 fully walled
          O_run      = running min of openness outward                a wall shadows everything beyond
          W_side     = sum of O_run over the R steps                  the open width on that side (cells)

        A low bump barely closes the channel; a rise of channel_height_scale_m fully closes it; and a
        wall at distance d caps W_side ~ d, so both wall *height* and *distance* feed the gain
        continuously (no hard threshold). The two sides combine as `W_eff = max(W_plus, W_minus)` --
        the more-open side limits the boost, so a true valley (confined both ways) gets a big gain
        while a hillside, mountain base or one side of a saddle (open on one side) stays ~1. Then
        `gain = clamp(R / W_eff, 1, gain_max)`: fully open -> W_eff = R -> gain 1, a tight gap pinches
        W_eff -> gain > 1. The scan samples along the *true* cross-stream direction by bilinear
        interpolation (grid_sample), so it is correct for any wind bearing, not just grid-aligned
        ones. A light Gaussian smooth removes per-cell speckle (convex -> stays >= 1). Returned
        per-cell with terrain's trailing shape; all-ones when channeling is disabled. The synoptic
        wind is a per-map constant, so this is computed once.

        Rank-agnostic: any leading dims (e.g. a batch B) are folded into grid_sample's batch axis,
        each world using its own synoptic direction.
        """
        if not self.channeling_enabled:
            return torch.ones_like(terrain)

        R, height_scale = self.channel_radius, self.channel_height_scale_m
        *lead, h, w = terrain.shape
        n_worlds = 1
        for d in lead:
            n_worlds *= d
        terr = terrain.reshape(n_worlds, 1, h, w)
        own = terr.reshape(n_worlds, h, w)

        # Per-world synoptic direction (the ambient wind is uniform per world -> reduce to a scalar).
        ax = u_amb_x.expand(*lead, h, w).reshape(n_worlds, h, w).mean(dim=(-2, -1))   # (n_worlds,)
        ay = u_amb_y.expand(*lead, h, w).reshape(n_worlds, h, w).mean(dim=(-2, -1))
        speed = torch.sqrt(ax ** 2 + ay ** 2).clamp(min=1e-12)
        # Cross-stream unit normal n_hat = wind rotated 90 deg: (wx, wy) -> (-wy, wx). x = cols, y = rows.
        nx = (-ay / speed).view(n_worlds, 1, 1)
        ny = ( ax / speed).view(n_worlds, 1, 1)

        device = terrain.device
        rows = torch.arange(h, device=device, dtype=self.dtype).view(-1, 1)
        cols = torch.arange(w, device=device, dtype=self.dtype).view(1, -1)
        # Normalised base coords for grid_sample (align_corners=True): idx -> 2*idx/(size-1) - 1.
        step_x = 2.0 / max(w - 1, 1)
        step_y = 2.0 / max(h - 1, 1)
        base_x = cols * step_x - 1.0   # (1, w)
        base_y = rows * step_y - 1.0   # (h, 1)

        def sample(sign: float, k: int) -> torch.Tensor:
            sx = base_x + sign * k * nx * step_x   # (n_worlds, h, w) via broadcast
            sy = base_y + sign * k * ny * step_y
            sx = sx.expand(n_worlds, h, w)
            sy = sy.expand(n_worlds, h, w)
            grid = torch.stack([sx, sy], dim=-1)   # (n_worlds, h, w, 2): last dim (x, y)
            return F.grid_sample(
                terr, grid, mode="bilinear", padding_mode="border", align_corners=True
            ).reshape(n_worlds, h, w)

        def open_width(sign: float) -> torch.Tensor:
            """Graded open width (in cells) on one side: integral of the shadowed openness."""
            o_run = torch.ones((n_worlds, h, w), dtype=self.dtype, device=device)   # fully open at the cell
            width = torch.zeros((n_worlds, h, w), dtype=self.dtype, device=device)
            for k in range(1, R + 1):
                rise = (sample(sign, k) - own).clamp(min=0.0)
                openness = (1.0 - rise / height_scale).clamp(min=0.0, max=1.0)
                o_run = torch.minimum(o_run, openness)   # a wall shadows everything beyond it
                width = width + o_run
            return width

        # The more-open side limits the boost: a valley is confined both ways, a slope only one way.
        w_eff = torch.maximum(open_width(+1.0), open_width(-1.0)).clamp(min=1e-6)
        gain = (float(R) / w_eff).clamp(min=1.0, max=self.channel_gain_max)
        gain = self._diffuse(gain, self.channel_smooth_sigma)   # convex combination -> stays >= 1
        return gain.reshape(*lead, h, w)

    def step_dynamics(self, s: SimState) -> SimState:
        """Advance the prognostic wind, boundary-layer mass and per-level air energy one tick.

        Shallow boundary-layer atmosphere over terrain: the wind is forced by the free-surface
        gradient (s = terrain + eta, eta = m*R_d*T/p_ref, surface level) and smoothed by viscosity,
        with only weak interior surface friction; mass, momentum, and EVERY air level's energy
        (E_i = m*c_p*T_i) are advected conservatively by that shared wind, so the canopy/above plumes
        drift downwind with the surface (Phase 3). Because the layer is shallow, terrain squeezes it, so
        continuity speeds the wind over crests and channels it through gaps. A boundary sponge
        relaxes the edges toward the per-map synoptic wind and the rest-state mass, making the
        domain open (inflow upwind, outflow downwind). CFL-substepped on the advective +
        gravity-wave speed. Rank-agnostic (single world or (B, H, W)).
        """
        cp, R_d, p_ref = pc.CP_AIR, pc.GAS_CONSTANT_DRY_AIR, pc.P_REF
        dx, dt = self.cell_size_m, self.dt
        g_prime, C_d, nu = self.reduced_gravity, self.drag_coeff, self.viscosity

        # Multi-level buoyancy, lean form (Phase 1, sub-area 2). Every air level is advected/conserved
        # with the shared mass and wind (Phase 3, below), but only the SURFACE level's shallow
        # combustion-plume slab of a super-ambient anomaly is *buoyant*: a fire lifts a
        # rho*plume_mixing_depth slab, not the full ~1.5 km boundary-layer column. So the anomaly's
        # contribution to the column thickness uses m_plume, while the rest-temperature baseline
        # keeps the full mass -- eta = (R_d/p_ref)*(m_plume*(T - T_rest) + m_total*T_rest). This
        # removes the ~15x fire-wind over-count (Phase 0 spike / design doc) without touching the
        # orographic wind (uniform-T, mass-driven: T == T_rest so eta == m_total*R_d*T_rest/p_ref,
        # unchanged) or mass/energy conservation. The per-level temperatures drive fuel/ladder behavior,
        # not the wind: buoyancy reads only the surface level (a deliberate lean choice), and per-level
        # masses stay deferred -- s.mass is one shared boundary-layer mass carrying every level.
        m   = s.mass
        air = s.air_temperatures                       # (L, H, W) full per-level stack
        u, v = s.x_wind_vel, s.y_wind_vel
        T_rest_all = s.temp_eq                          # (L, H, W) per-level rest profile
        T_rest  = _surface_level(T_rest_all)            # surface rest, the buoyancy baseline
        m_plume = pc.RHO_REF * self.plume_mixing_depth
        eta_of  = lambda m_, T_: (m_plume * (T_ - T_rest) + m_ * T_rest) * R_d / p_ref

        # Substep count from the worst-case Courant number: advective speed |u| plus the
        # gravity-wave speed c = sqrt(g'*eta), eta the total column thickness (surface-level buoyancy).
        eta = eta_of(m, _surface_level(air))
        wave = float(torch.sqrt((g_prime * eta).clamp(min=0.0)).max())
        flow = float((u.abs() + v.abs()).max())
        courant = (flow + wave) * dt / dx
        n = max(1, int(math.ceil(courant / self.cfl_target)))
        dts = dt / n

        for _ in range(n):
            eta = eta_of(m, _surface_level(air))
            surface = s.terrain + eta
            gx, gy = self._periodic_grad(surface, dx)
            # Forcing: -g'*grad(s) (terrain + buoyancy), weak surface friction, eddy viscosity.
            u = u + dts * (-g_prime * gx - C_d * u + nu * self._periodic_laplacian(u, dx))
            v = v + dts * (-g_prime * gy - C_d * v + nu * self._periodic_laplacian(v, dx))

            # Conservative transport of mass, momentum and every air level's energy by the updated
            # wind. Each level's energy E_i = m*c_p*T_i rides the shared mass/wind (Phase 3): one wind
            # moves all levels identically, so per-level advection just carries the vertical temperature
            # structure with the flow (the canopy/above plumes tilt downwind like the surface). The
            # energy stack takes a level axis on the displacement so the same (H,W) shift applies to
            # every level (rank-agnostic for a batched stack too). Sum_i m*c_p*T_i is conserved to
            # round-off, so the closed-core conservation is unaffected.
            dispx, dispy = u * dts / dx, v * dts / dx
            m_lvl = m.unsqueeze(-3)
            energy = m_lvl * cp * air                             # (L, H, W) each level's energy
            mom_x, mom_y = m * u, m * v
            m      = self._advect_periodic(m,      dispx, dispy)
            energy = self._advect_periodic(energy, dispx.unsqueeze(-3), dispy.unsqueeze(-3))
            mom_x  = self._advect_periodic(mom_x,  dispx, dispy)
            mom_y  = self._advect_periodic(mom_y,  dispx, dispy)

            m_safe = m.clamp(min=1e-9)
            u, v = mom_x / m_safe, mom_y / m_safe
            air = energy / (cp * m_safe.unsqueeze(-3))            # recover each level's temperature

        # Open-boundary sponge: relax the edge belt toward the free-stream (synoptic wind and
        # rest-state mass/temperature) so the wind enters upwind and leaves downwind without
        # piling or reflecting. Exact-exponential, so stable for any sponge strength. Every air level
        # relaxes toward its own rest profile (the temp_eq stack).
        decay = torch.exp(-self._sponge_rate(m.shape[-2:]) * dt)
        u = s.u_amb_x + (u - s.u_amb_x) * decay
        v = s.u_amb_y + (v - s.u_amb_y) * decay
        m = s.mass_eq + (m - s.mass_eq) * decay
        air = T_rest_all + (air - T_rest_all) * decay

        s.mass, s.x_wind_vel, s.y_wind_vel = m, u, v
        s.air_temperatures = air
        return s

    def _cover_fractions(self, fuel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Beer-Lambert vegetation cover and per-body area shares from the fuel load.

        Returns (a_ground, a_fuel): the bare-ground area fraction (..., H, W) and the per-type
        fuel area fractions (..., N, H, W), which sum over types to the cover fraction
        cover = 1 - exp(-fuel_total/kappa). Shared by the surface radiative balance
        (apply_radiation) and the flame radiant transfer (apply_radiant_heat) so both partition
        the surface the same way.
        """
        fuel_total = fuel.sum(dim=-3)                              # (..., H, W)
        cover = 1.0 - torch.exp(-fuel_total / self.cover_fuel_scale)   # (..., H, W) in [0, 1)
        a_ground = 1.0 - cover                                     # bare-ground area fraction
        safe_total = fuel_total.clamp(min=1e-12).unsqueeze(-3)
        a_fuel = cover.unsqueeze(-3) * fuel / safe_total           # (..., N, H, W), sums to cover
        return a_ground, a_fuel

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
        S_net = eps*sigma*temp_eq^4, so radiative equilibrium sits exactly at temp_eq. temp_eq is
        the baked surface (rest) temperature -- the elevation profile already warmed by sun-facing
        exposure and cooled in shadow (gen_maps.surface_temperature) -- so a sunlit slope's ground
        and fuel equilibrate warmer and a shaded valley's cooler. Heat capacity sets the response
        speed -- fine fuel (~1e3 J/m^2/K) leads, the ground skin (~1e5) follows, the deep air column
        (~1e6) lags far behind, which is why fuels heat and dry in the sun well ahead of the air.
        Explicit forward Euler; all fluxes use the pre-step temperatures.

        Returns (air_temperatures, fuel_temperatures, ground_temperature).
        """
        eps, sigma, cp_air = self.emissivity, pc.STEFAN_BOLTZMANN, pc.CP_AIR
        dt = self.dt
        C_g, k_ga = self.ground_heat_capacity, self.ground_air_exchange

        s_net = eps * sigma * temp_eq ** 4                         # (..., H, W) per-cell anchor

        # Cover fraction (Beer-Lambert canopy interception) and per-body area shares.
        a_ground, a_fuel = self._cover_fractions(fuel)

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
        q_fuel = a_fuel * s_net.unsqueeze(-3) - a_fuel * eps * sigma * fuel_temperatures ** 4
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

    def convect_air_levels(self, air_temperatures: torch.Tensor) -> torch.Tensor:
        """Upward-only inter-level heat convection (warm air rises; the reverse does not).

        Combustion injects its air-share into the surface level and fuels warm their own levels, but
        the vertical split gives the surface/canopy slabs no sink of their own (only the above-canopy
        level vents). This stage carries that heat *up*: for each adjacent pair (surface->canopy, then
        canopy->above) it runs the exact-exponential two-body relaxation of exchange_fuel_air_heat,
        but gated on instability -- it acts only where the lower level is hotter than the one above
        (super-adiabatic), never mixing a warm upper level down. Energy is conserved (what leaves a
        level enters the one above, weighted by each slab's capacity), it is self-limiting (stops once
        the gap closes), and it is a no-op on a quiescent world (all levels at rest -> no gap). Heat
        cascades surface -> canopy -> above in one tick, where the top vent finally sheds it.
        Rank-agnostic (single world or (B, L, H, W)).
        """
        L = air_temperatures.shape[-3]
        if not self.air_convection_enabled or self.air_transfer_rate <= 0.0 or L < 2:
            return air_temperatures
        cap = self._air_level_capacity                       # (L, 1, 1)
        # air_transfer_rate is the temperature-gap relaxation rate lambda [1/s]: the (T_lower-T_upper)
        # gap decays by exp(-lambda*dt) per tick regardless of the slab capacities, so the knob is
        # capacity-independent and reads directly as a residence time 1/lambda (much easier to tune
        # than a raw conductance against the tiny thin-slab capacities). The heat still splits
        # conservatively by capacity, so the pair relaxes toward its capacity-weighted mean.
        decay = math.exp(-self.air_transfer_rate * self.dt)
        air = air_temperatures.clone()
        for lo in range(L - 1):
            hi = lo + 1
            T_l, T_u = air.select(-3, lo), air.select(-3, hi)   # views into `air`
            C_l, C_u = cap[lo], cap[hi]                          # (1, 1) each
            # Two-body relaxation toward the capacity-weighted mean, then keep only the upward part:
            # where the lower level is hotter, it cools and the upper warms; elsewhere a no-op (no
            # downward mixing through this term).
            T_eq  = (C_l * T_l + C_u * T_u) / (C_l + C_u)
            dT_l  = torch.where(T_l > T_u, (T_eq - T_l) * (1.0 - decay), torch.zeros_like(T_l))
            dT_u  = -(C_l / C_u) * dT_l                          # energy-conserving back-reaction
            T_l.add_(dT_l)
            T_u.add_(dT_u)
        return air

    def conduct_bole_segments(
        self, fuel_temperatures: torch.Tensor, fuel: torch.Tensor
    ) -> torch.Tensor:
        """Anisotropic vertical conduction along the trunk, between its two stacked bole segments.

        The tree bole is split into co-located fuel types -- tree_bole_surface (0-5 m) and
        tree_bole_canopy (5-15 m) -- on the surface and canopy air levels. This term is the trunk
        itself carrying fire between them: an exact-exponential two-body relaxation of the two
        segments' *fuel temperatures*, energy-conserving over their thermal masses fuel*cp, with a
        **direction-dependent rate** -- k_up (fast) when the surface segment is hotter, so fire climbs
        the trunk into the crown, and k_down (slow) when the canopy segment is hotter, so a crown fire
        does not readily run back down. This is the ladder's dedicated conduit: once the surface bole
        catches from the ground fire, fire runs up it and ignites the canopy bole, which then lights the
        crown. A no-op where either segment has burnt away (no trunk mass to conduct) and when the
        config has no split bole (the single-fuel test worlds). Rank-agnostic.
        """
        si, ci = self._bole_surface_idx, self._bole_canopy_idx
        if si is None or ci is None:
            return fuel_temperatures
        cp = self.fuel_specific_heat                            # (N, 1, 1)
        Ts, Tc = fuel_temperatures.select(-3, si), fuel_temperatures.select(-3, ci)   # (..., H, W)
        Cs = cp[si] * fuel.select(-3, si)                       # (..., H, W) surface-segment thermal mass
        Cc = cp[ci] * fuel.select(-3, ci)                       # (..., H, W) canopy-segment thermal mass
        # Only conduct where BOTH segments carry mass; otherwise there is no trunk to carry fire (and the
        # capacity-weighted mean / back-reaction would divide by ~zero).
        active = (fuel.select(-3, si) > self.fuel_burnt_threshold) & (fuel.select(-3, ci) > self.fuel_burnt_threshold)
        # Gap-relaxation rate [1/s], anisotropic by the sign of the gradient: fire climbs fast, descends
        # slow (k_up >> k_down). Same exact-exponential gap decay as convect_air_levels.
        rate  = torch.where(Ts > Tc, torch.full_like(Ts, self.bole_conduction_up),
                            torch.full_like(Ts, self.bole_conduction_down))
        decay = torch.exp(-rate * self.dt)
        denom = (Cs + Cc).clamp(min=1e-12)
        T_eq  = (Cs * Ts + Cc * Tc) / denom
        dTs   = torch.where(active, (T_eq - Ts) * (1.0 - decay), torch.zeros_like(Ts))
        dTc   = -(Cs / Cc.clamp(min=1e-12)) * dTs              # energy-conserving back-reaction
        ft = fuel_temperatures.clone()
        ft.select(-3, si).add_(dTs)
        ft.select(-3, ci).add_(dTc)
        return ft

    def vent_plume_heat(self, air_temperatures: torch.Tensor, temp_eq: torch.Tensor) -> torch.Tensor:
        """Above-canopy energy-anomaly vent -- the single sink for all fire heat (the air-temp sink).

        Fuels and combustion warm the near-surface levels, and convect_air_levels cascades that heat
        upward into the deep above-canopy (top) level; the air is otherwise radiatively transparent
        with no vertical export, so without a sink the plume heat accumulates without bound and
        conducts back into the fuel (a temperature runaway). A real plume rises buoyantly and carries
        the heat up and out to space while fresh air entrains. We model that as an exact-exponential
        relaxation of the top level's *super-ambient* air toward its rest profile temp_eq at the vent
        rate lambda (~ updraft / depth, the inverse residence time). Venting only the top level makes
        this an energy-anomaly sink -- heat riding up from any level is shed at the same rate once it
        reaches the top, independent of which slab held it. One-directional (only cools air hotter than
        temp_eq -- strictly a sink, never injects energy), so a quiescent world is a no-op. Only the
        top level changes; the surface/canopy levels are returned untouched. Rank-agnostic.
        """
        if self.plume_vent_rate <= 0.0:
            return air_temperatures
        top = air_temperatures.shape[-3] - 1
        T_top   = air_temperatures.select(-3, top)
        eq_top  = temp_eq.select(-3, top)
        excess  = (T_top - eq_top).clamp(min=0.0)
        vented  = T_top - excess * (1.0 - math.exp(-self.plume_vent_rate * self.dt))
        air = air_temperatures.clone()
        air.select(-3, top).copy_(vented)
        return air

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
        air_temperatures: torch.Tensor,   # (L, H, W)  per-level air temperature
        fuel_temperatures: torch.Tensor,  # (N, H, W)
        fuel: torch.Tensor,               # (N, H, W)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Conductive fuel<->air heat exchange, each fuel coupled to *its own* air level.

        Every fuel type trades heat only with the thin air slab at its height (fuel_levels):
        grass/shrub/surface-bole against the surface level, canopy foliage/canopy-bole against the
        canopy level. The air thermal mass is that level's slab capacity C = rho_ref*depth*cp_air
        (_air_level_capacity), ~300x lighter than the old full-column coupling, so a fuel warms its
        level's air strongly and directly. The two-body relaxation math is unchanged (exact-
        exponential, conserving each pair's energy); only the air side is now per-level: each fuel
        reads its level's temperature and its level's capacity, and the back-reaction is scattered
        back into the level it came from so total air+fuel energy is conserved level by level.
        Rank-agnostic (single world or (B, L/N, H, W)).
        """
        cp_fuel = self.fuel_specific_heat  # (N, 1, 1)  [J/(kg*K)]
        k       = self.fuel_transfer_rates # (N, 1, 1)  [W/(m^2*K)]
        idx     = self._fuel_level_index   # (N,) long: each fuel's air-level index

        C_fuel = cp_fuel * fuel                                # (N, H, W) fuel thermal mass [J/(m^2*K)]
        # Gather each fuel's air level: its temperature (from the (..., L, H, W) stack) and its slab
        # heat capacity (from the (L, 1, 1) per-level capacity). Both index on the level axis by the
        # fuel->level map, yielding one air partner per fuel type.
        air_f  = air_temperatures.index_select(-3, idx)       # (N, H, W) each fuel's level air temp
        C_air  = self._air_level_capacity.index_select(0, idx)  # (N, 1, 1) each fuel's slab capacity
        total  = C_air + C_fuel                                # >= C_air > 0, always safe

        # Mass-weighted equilibrium temperature of each air/fuel pair (the conserved
        # mean the pair relaxes toward).
        T_eq = (C_air * air_f + C_fuel * fuel_temperatures) / total

        # Exact two-body relaxation over one tick: the air/fuel gap decays by
        # exp(-k / C_red), where C_red = C_air*C_fuel/total is the reduced heat capacity.
        # This folds the conduction rate and BOTH masses into a single factor in (0, 1],
        # so each side lands on the equilibrium side without overshoot for any rate or
        # masses — no clamping and no fictitious unit capacities needed.
        # Massless ash (C_fuel -> 0) drives C_red -> 0 and T_eq -> air temp, so the
        # exponent -> -inf and decay -> 0: the fuel equilibrates instantly to the air and
        # returns ~no energy to it. The transfer rate k is always positive, so the only
        # division by zero here is this benign C_fuel -> 0 limit (exp(-inf) = 0, no NaN).
        # k/C_red has units 1/s, so the gap decays by exp(-k*dt/C_red) over one tick.
        decay = torch.exp(-k * self.dt * total / (C_air * C_fuel))

        dT_fuel = (T_eq - fuel_temperatures) * (1.0 - decay)  # (N, H, W)

        # Air loses exactly the energy each fuel type gained; scatter that energy back into the
        # level each fuel drew it from (fuels sharing a level sum), then divide each level by its
        # own capacity to get its temperature change -- so air+fuel energy is conserved per level.
        air_energy_gain = torch.zeros_like(air_temperatures)          # (..., L, H, W) [J/m^2]
        air_energy_gain.index_add_(-3, idx, -(C_fuel * dT_fuel))
        air_new = air_temperatures + air_energy_gain / self._air_level_capacity

        return air_new, fuel_temperatures + dT_fuel

    def update_fire(
        self,
        air_temperatures: torch.Tensor,   # (L, H, W)     per-level air temperature T_a [K]
        fuel_temperatures: torch.Tensor,  # (N, H, W)     per-type fuel temperature T_f [K]
        fuel: torch.Tensor,               # (N, H, W)     per-type biomass [kg/m^2]
        oxygen: torch.Tensor,             # (H, W)        O2 partial density [kg/m^3]
        air_mass: torch.Tensor,           # (H, W)        boundary-layer areal mass m [kg/m^2]
        x_wind_vel: torch.Tensor | None = None,  # (H, W) wind u [m/s] -- skews convective ignition
        y_wind_vel: torch.Tensor | None = None,  # (H, W) wind v [m/s]
        terrain: torch.Tensor | None = None,     # (H, W) elevation [m] -- skews ignition upslope
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Surface-area-controlled, oxygen-limited combustion over one tick.

        Fuel reacts at its surface, not in bulk, so the burn rate is the reactive surface area
        a_s*fuel times a surface mass flux psi [kg/(m^2*s)]. psi puts chemistry in series with
        oxygen transport: a kinetic (Arrhenius) flux psi_kin = B*exp(-E_a/(R*T_fuel)) and an O2
        film-diffusion flux psi_dif = h*(O2/O2_ref), combined as resistances (1/psi = 1/psi_kin +
        1/psi_dif) so the slower one wins -- kinetic when cold (the ignition gate: cold fuel is
        inert because the exponential underflows, no hard threshold), the O2 supply when flame-hot
        (the transport cap). The flaming residence time tau ~ 1/(a_s*psi) thus emerges from the
        physics: high-SAV grass burns out in seconds, a low-SAV log over minutes. The burned mass
        releases its heat of combustion HHV, split between fuel and air by burn_heat_fuel_fraction.
        Oxygen is a partial density [kg/m^3] while the burn is areal [kg/m^2], so they also couple
        through the shallow combustion mixing depth d_mix (the near-surface air the fire entrains),
        giving a whole-cell inventory backstop on top of the per-surface film transport. The air
        share of each fuel's heat is deposited into a shallow plume slab of depth plume_mixing_depth
        (Phase 5.5b), not the full column, so a burn warms the air strongly enough to drive a
        convective signal. **The air-share and its convective ignition deposit are routed per air
        level (Stage 2):** each fuel warms its own level's plume, and a burning cell's level-L plume
        ignites only neighbours' level-L fuels -- so a ground grass fire deposits into neighbours'
        surface fuels (grass, shrub, surface bole) but never directly into their elevated canopy (the
        ladder, emergent during spread). A fraction of the air share (convective_fraction) is convected
        to nearby fuel to ignite it (Phase 5.5c) -- the spread driver, since radiation preheats the fuel
        ahead but is too weak to ignite it alone; skewed by a von Mises angular kernel whose bias blends
        the wind (lee bias) with the upslope terrain gradient (faster spread uphill, Phase 5.5d). All
        rates use pre-step temperatures.
        """
        R, cp_air = pc.UNIVERSAL_GAS_CONSTANT, pc.CP_AIR
        dt, d_mix = self.dt, self.combustion_mixing_depth
        B, E_a, cp_fuel = self.arrhenius_pre, self.activation_energy, self.fuel_specific_heat
        a_s, h = self.fuel_specific_surface, self.surface_mass_transfer
        hhv, s = self.heat_of_combustion, self.stoich_oxygen
        present = fuel > self.fuel_burnt_threshold
        air0 = air_temperatures.select(-3, 0)   # surface-level (..., H, W) view, for (H, W) shape refs

        # Surface mass flux psi [kg/(m^2_surface*s)]: kinetic chemistry in series with O2 film
        # diffusion. The kinetic flux underflows to ~0 at ambient T (cold fuel inert); the diffusive
        # flux caps it once flame-hot, so the rate is physically bounded with no ad-hoc residence cap.
        o2_factor = (oxygen.unsqueeze(-3) / pc.O2_DENSITY_REF).clamp(min=0.0)   # (N, H, W) broadcast
        psi_kin = B * torch.exp(-E_a / (R * fuel_temperatures.clamp(min=1.0)))  # (N, H, W) kinetic
        psi_dif = h * o2_factor                                                 # O2 film transport
        psi = psi_kin * psi_dif / (psi_kin + psi_dif).clamp(min=1e-30)          # series resistances

        # Areal consumption = reactive surface (a_s*fuel) burning at psi, integrated exactly over
        # the tick: kappa = a_s*psi [1/s], Dfuel = fuel*(1 - exp(-kappa*dt)) is bounded by the fuel
        # present (no clamp needed). tau ~ 1/kappa is the emergent flaming residence time.
        kappa = a_s * psi                                                       # (N, H, W) [1/s]
        fuel_demand = torch.where(present, fuel * -torch.expm1(-kappa * dt), torch.zeros_like(fuel))

        # Oxygen that burn would need, as a near-surface density draw (areal demand s*Dfuel over
        # the d_mix-deep combustion layer). Both reactants are required, so oxygen limits fuel as
        # well as the reverse: if the draw exceeds the O2 present, scale the whole reaction down
        # so fuel and oxygen are consumed in stoichiometric step (a fire starved of air burns less
        # fuel and releases less heat). This keeps O2 a genuine limiter, not just a rate throttle.
        o2_demand = (s * fuel_demand).sum(dim=-3) / d_mix          # (H, W)
        o2_limit  = torch.where(
            o2_demand > 0.0, (oxygen / o2_demand).clamp(max=1.0), torch.ones_like(o2_demand)
        )  # (H, W) in [0, 1]
        fuel_consumed   = fuel_demand * o2_limit.unsqueeze(-3)     # (N, H, W)
        oxygen_consumed = o2_demand * o2_limit                     # (H, W) == min(o2_demand, oxygen)

        fuel   = fuel - fuel_consumed
        oxygen = oxygen - oxygen_consumed

        # Heat released per type [J/m^2], split between air and fuel.
        burn_heat_per_type = hhv * fuel_consumed              # (N, H, W)
        f = self.burn_heat_fuel_fraction

        # Fuel self-heat: each type warms by its share f*HHV*Dfuel over its thermal mass fuel*c_p.
        # This sets the flame temperature (bounded by the Phase 3 T^4 sink); convective ignition
        # below is gated off above flame_gate so it never drives the flame, only ignites cold fuel.
        C_fuel  = cp_fuel * fuel
        safe_C  = torch.where(present, C_fuel, torch.ones_like(C_fuel))
        dT_self = torch.where(present, burn_heat_per_type * f / safe_C, torch.zeros_like(C_fuel))
        fuel_temperatures = (fuel_temperatures + dT_self).clamp(min=0.0)

        # Air share (1-f) rises in a shallow buoyant plume of depth d_plume (Phase 5.5b), not the
        # full ~1 km column, using each level's near-surface density rho = p_ref/(R_d*T_a) (= m/eta at
        # reference pressure): C_plume = rho*d_plume*c_p_air, ~30x less mass than the column, so a
        # burn warms the air strongly (a sub-grid plume scale, mirroring oxygen's d_mix). Hotter air
        # is lighter -> smaller capacity -> slightly more responsive, bounded by the T^4 sink. Stage 2:
        # each fuel's air-share is routed to ITS air level (index_add over fuel_level_index), so canopy
        # combustion warms the canopy air and surface combustion the surface air. The plume depth stays
        # plume_mixing_depth for every level (Option A) -- the surface level's plume mass therefore still
        # matches step_dynamics' lean buoyancy slab (m_plume), so fire-wind is unchanged.
        rho_level = pc.P_REF / (pc.GAS_CONSTANT_DRY_AIR * air_temperatures.clamp(min=1.0))  # (L, H, W)
        C_plume   = rho_level * self.plume_mixing_depth * cp_air                            # (L, H, W)
        air_share_type = burn_heat_per_type * (1.0 - f)      # (N, H, W) [J/m^2] each type's air share
        air_share = torch.zeros_like(air_temperatures)       # (L, H, W)
        air_share.index_add_(-3, self._fuel_level_index, air_share_type)   # sum each level's shares

        # Crown-initiation preheat (Phase 4): the trunk base basks in the surface fire. A fraction of
        # the surface-level combustion air-share conducts straight into the co-located surface bole
        # instead of warming the surface air (energy-conserving: it is removed from that level's air
        # share). It is decoupled from convective_fraction -- which is calibrated low for realistic ROS
        # and so cannot itself ignite the heavy, coarse bole -- so the crown transition is gated by fire
        # DURATION: a brief light grass fire deposits too little to cross the bole's ignition, while a
        # sustained or heavier surface fire accumulates enough to torch the trunk and start the climb.
        # Only into a present, sub-flaming bole (a flaming bole is combustion-controlled). No-op without
        # a split bole (single-fuel test worlds).
        si = self._bole_surface_idx
        if si is not None and self.bole_preheat_fraction > 0.0:
            lvl_s = self.fuel_levels[si]                              # the surface bole's air level (0)
            surf_share = air_share.select(-3, lvl_s)                 # (H, W) surface air-share
            bole_gate = (fuel.select(-3, si) > self.fuel_burnt_threshold) & \
                        (fuel_temperatures.select(-3, si) < self.flame_gate_temperature)
            C_bole = (cp_fuel[si] * fuel.select(-3, si)).clamp(min=1e-9)   # (H, W) bole thermal mass
            deposited = torch.where(bole_gate, self.bole_preheat_fraction * surf_share,
                                    torch.zeros_like(surf_share))     # (H, W) energy into the bole
            fuel_temperatures = fuel_temperatures.clone()
            fuel_temperatures[..., si, :, :] = (
                fuel_temperatures[..., si, :, :] + deposited / C_bole
            ).clamp(min=0.0)
            air_share[..., lvl_s, :, :] = air_share[..., lvl_s, :, :] - deposited   # conserve energy

        # Convective ignition (Phase 5.5c): radiation preheats the fuel ahead but can't ignite a
        # neighbour at a realistic flame temperature, so a fraction of the fresh air-share heat is
        # convected to nearby fuel to ignite it -- the propagation driver. It is sourced from this tick's
        # combustion (not the advected air field, so wind cannot sweep the source away) and skewed
        # downwind/upslope by the convective deposit kernel. It is deposited only into sub-flaming
        # fuel (below flame_gate -- flaming fuel is combustion-controlled), with a thermal-mass floor
        # so near-burnt cells do not superheat, and clamped so one tick cannot drive fuel past the
        # gate; whatever the fuel does not absorb stays in the plume air, so energy is conserved.
        if x_wind_vel is None:
            x_wind_vel = torch.zeros_like(air0)
        if y_wind_vel is None:
            y_wind_vel = torch.zeros_like(air0)
        if terrain is None:
            terrain = torch.zeros_like(air0)
        cf = self.convective_fraction

        # Bias vector for the convective deposit (the directional skew). Blend the wind (lee bias,
        # leg c) with the upslope terrain gradient (Rothermel slope effect, leg d). The wind term is
        # scaled by speed, not just direction (Phase 6, Option 2): b_wind = convective_wind_bias *
        # (wind / u_ref), so a faster wind grows |b| and throws the ignition heat harder downwind --
        # wind *speed* drives the rate of spread (paired with the channeling gain above, a gap both
        # speeds the wind and focuses the head). grad(z) is the slope (= tan of the slope angle)
        # pointing uphill, scaled by its own magnitude so the slope term has magnitude tan^2(slope) --
        # the front skews uphill ever harder on steeper ground. The deposit kernel (see
        # _convective_deposit) reads |b| as the forward concentration, so wind and slope aligning
        # focuses the deposit into a tighter, faster head; |b| is capped only as a numerical guard.
        u_ref = self.convective_wind_ref
        sgx, sgy = self._periodic_grad(terrain, self.cell_size_m)   # grad(z) = tan(slope), uphill +
        smag = torch.sqrt(sgx ** 2 + sgy ** 2)                      # tan(slope)
        bx = self.convective_wind_bias * (x_wind_vel / u_ref) + self.convective_slope_bias * sgx * smag
        by = self.convective_wind_bias * (y_wind_vel / u_ref) + self.convective_slope_bias * sgy * smag
        # Per-level convective deposit: spread each level's plume heat by the kernel (a spatial
        # convolution, so run once per fuel-bearing level), then let each fuel receive only from ITS
        # level. This is the ladder DURING spread -- a grass front's surface-level plume ignites
        # neighbours' surface fuels but deposits nothing into their canopy level.
        delivered = torch.zeros_like(air_temperatures)       # (L, H, W) [J/m^2]
        for lvl in self._levels_with_fuel:
            delivered.select(-3, lvl).copy_(
                self._convective_deposit(cf * air_share.select(-3, lvl), bx, by).clamp(min=0.0)
            )
        delivered_f = delivered.index_select(-3, self._fuel_level_index)  # (N,H,W) each fuel's level deposit

        C_floor    = (cp_fuel * 10.0 * self.fuel_burnt_threshold).expand_as(C_fuel)  # ~burnt-mass floor
        C_dep      = torch.maximum(C_fuel, C_floor)
        # Split the convected ignition heat between the cell's fuel types by reactive surface area
        # (a_s*fuel), not bulk mass: the flame heat is intercepted at the fuel surface, so fine
        # high-SAV fuel (grass) takes a share set by its exposed area rather than its weight. The split
        # is normalised WITHIN each air level (Stage 2), so the level-L plume heat is shared only among
        # level-L fuels -- one tree's canopy sharing a grass cell no longer draws the grass's surface
        # plume, and the ladder holds cell by cell.
        surface    = a_s * fuel                                          # (N, H, W)
        surf_level = torch.zeros_like(air_temperatures)                  # (L, H, W)
        surf_level.index_add_(-3, self._fuel_level_index, surface)
        surf_denom = surf_level.index_select(-3, self._fuel_level_index).clamp(min=1e-30)  # (N,H,W)
        surf_frac  = surface / surf_denom
        gate       = present & (fuel_temperatures < self.flame_gate_temperature)
        dT_conv    = torch.where(gate, delivered_f * surf_frac / C_dep, torch.zeros_like(C_fuel))
        capped     = torch.minimum(fuel_temperatures + dT_conv,
                                   torch.full_like(fuel_temperatures, self.flame_gate_temperature))
        dT_conv    = (capped - fuel_temperatures).clamp(min=0.0)
        # Energy each level's fuel took up, scattered back to that level (fuels sharing a level sum).
        absorbed   = torch.zeros_like(air_temperatures)      # (L, H, W) [J/m^2]
        absorbed.index_add_(-3, self._fuel_level_index, C_fuel * dT_conv)
        fuel_temperatures = (fuel_temperatures + dT_conv).clamp(min=0.0)

        # Per-level air bookkeeping (conserved, off-grid edge loss aside): each level keeps the
        # non-convected share (1-cf)*air_share; each receiver's leftover convected heat its fuel did
        # not take up (delivered - absorbed) warms that level's plume air.
        air_temperatures = (
            air_temperatures + ((1.0 - cf) * air_share + (delivered - absorbed)) / C_plume
        ).clamp(min=0.0)

        return air_temperatures, fuel_temperatures, fuel, oxygen.clamp(min=0.0)

    def apply_radiant_heat(
        self,
        fuel_temperatures: torch.Tensor,  # per-type fuel temperature T_f [K]   (..., N, H, W)
        fuel: torch.Tensor,               # per-type biomass [kg/m^2]           (..., N, H, W)
        temp_eq: torch.Tensor,            # rest temperature profile [K]        (..., H, W)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Spread a flame's radiant heat to its neighbours (the fire-propagation mechanism).

        This does NOT emit or cool: the fuel's grey-body longwave eps*sigma*T_f^4 is already
        radiated (and removed from the fuel) by apply_radiation, which owns the self-limiting T^4
        sink. Here we only *redistribute* the super-ambient part of that emission -- the heat a
        flame throws off above the resting glow, eps*sigma*(T_f^4 - temp_eq^4) -- onto nearby fuel
        by the inverse-square kernel. apply_radiation already accounted this energy as leaving to
        space, so depositing the absorbed neighbour share back is energy-consistent (the rest --
        sky escape + off-grid -- truly leaves); subtracting temp_eq^4 keeps a quiescent world from
        radiating its own equilibrium glow, so only flame heat above ambient propagates. Rank-
        agnostic (single world or (..., N, H, W)).
        """
        eps, sigma, dt = self.emissivity, pc.STEFAN_BOLTZMANN, self.dt
        present = fuel > self.fuel_burnt_threshold

        # Super-ambient flame emission, area-weighted exactly as apply_radiation weights the
        # longwave loss, so we can never redistribute more energy than the emitter actually shed.
        _, a_fuel = self._cover_fractions(fuel)
        excess = (
            a_fuel * eps * sigma
            * (fuel_temperatures ** 4 - temp_eq.unsqueeze(-3) ** 4).clamp(min=0.0)
            * dt
        )  # (..., N, H, W) [J/m^2]
        excess = torch.where(present, excess, torch.zeros_like(excess))

        # A fraction escapes upward to space; the rest is spread to neighbours by the
        # inverse-square kernel (the 'same'-mode convolution also drops energy off the grid edges).
        to_neighbours = (1.0 - self.sky_escape_fraction) * excess.sum(dim=-3)   # (..., H, W)
        radiant_flux = self._radiant_convolve(to_neighbours).clamp(min=0.0)     # (..., H, W) absorbed

        # Deposit the absorbed flux into each present type proportional to its mass fraction,
        # divided by thermal mass C_fuel = fuel*c_p,fuel to get the temperature rise. Unlike the
        # convective ignition deposit (update_fire), this redistribution is NOT surface-weighted and
        # has no gate/cap: surface-weighting would funnel the flux into the high-SAV, low-mass fuel
        # (grass) sharing a dense cell and let its T^4 emission run away. Mass-weighting keeps the
        # redistribution bounded; the convective deposit alone carries the surface-aware ignition.
        C_fuel = self.fuel_specific_heat * fuel                  # (..., N, H, W) [J/(m^2*K)]
        safe_C = torch.where(present, C_fuel, torch.ones_like(C_fuel))
        total_fuel = fuel.sum(dim=-3)                            # (..., H, W)
        fuel_frac = fuel / total_fuel.clamp(min=1e-12).unsqueeze(-3)
        dT_fuel = torch.where(
            present,
            radiant_flux.unsqueeze(-3) * fuel_frac / safe_C,
            torch.zeros_like(C_fuel),
        )

        return fuel_temperatures + dT_fuel, radiant_flux

    # ---------------------------------------------------------------------------
    # Full per-tick step + observation
    # ---------------------------------------------------------------------------

    def step_fields(
        self, s: SimState, *, advance_oxygen: bool = True, advance_fire: bool = True
    ) -> SimState:
        """Advance every field one tick (the pure physics of one env step).

        This is the single source of truth for the per-tick sequence — both
        FirecrackerEnv.step (single world) and BatchedRollout (B worlds) call it, so they
        can never drift apart. It carries no rendering/EMA bookkeeping. Because every op is
        rank-agnostic, the same call advances a 2-D world or a batched (B, H, W) stack.

        `advance_oxygen` / `advance_fire` default to the full step. They let a caller skip the
        oxygen transport and combustion stages -- the two costly stages that do not feed back into
        mass, the wind, or the air temperature. The quiescent map spin-up (gen_maps.spin_up_maps)
        turns both off: it only needs the developed mass/wind and wind-smeared air, and a rest-state
        world has no fire, so those stages are pure wasted work there. Keeping them as flags on this
        one method (rather than a hand-rolled subset in the generator) means the spin-up reuses the
        exact runtime sequence and automatically inherits any future stage.

        The state is mutated in place and returned for convenience.
        """
        # Shallow-water momentum core: advance the prognostic wind, then advect mass and
        # energy (E = m*c_p*T) by it. This refreshes s.x/y_wind_vel too — the wind is now
        # prognostic, not a diagnostic of the pressure field.
        self.step_dynamics(s)

        # Diagnostic terrain-channeling surface wind (Phase 6, Option 2): the fire and oxygen read
        # the prognostic wind sped up through gaps by the static channeling gain (speedup only --
        # direction is preserved, since the core already deflects the flow). The prognostic field is
        # left untouched. Falls back to the prognostic wind when channeling is off or no gain was
        # precomputed (e.g. a hand-built state), so the single-layer path is unchanged.
        u_fire, v_fire = s.x_wind_vel, s.y_wind_vel
        if self.channeling_enabled and s.channel_gain is not None:
            u_fire = s.x_wind_vel * s.channel_gain
            v_fire = s.y_wind_vel * s.channel_gain
        s.x_wind_fire, s.y_wind_fire = u_fire, v_fire

        # Fuel<->air coupling is per-level: exchange_fuel_air_heat (Stage 1) and update_fire's air-share
        # + convective deposit (Stage 2) both route heat to each fuel's own level. apply_radiation stays
        # surface-level (the ground sensible flux warms level 0 only). The inter-level convection then
        # carries the near-surface heat up, and the bole-segment conduction climbs fire up the trunk.
        # temp_eq is a level stack; its surface level anchors the surface stages, its top level the vent.
        temp_eq0 = _surface_level(s.temp_eq)

        # Surface radiative energy balance: sun warms the ground/fuel skin, longwave cools it,
        # and the ground sheds sensible heat to the air (which is transparent to radiation).
        if self.radiation_enabled:
            air0, s.fuel_temperatures, s.ground_temperature = self.apply_radiation(
                _surface_level(s.air_temperatures), s.ground_temperature,
                s.fuel_temperatures, s.fuel, s.mass, temp_eq0
            )
            s.air_temperatures = _with_surface(s.air_temperatures, air0)

        # Conductive fuel<->air heat: each fuel couples to its own air level (surface fuels to the
        # surface level, canopy fuels to the canopy level) against that level's thin-slab capacity.
        s.air_temperatures, s.fuel_temperatures = self.exchange_fuel_air_heat(
            s.air_temperatures, s.fuel_temperatures, s.fuel
        )

        # Oxygen transport (advection + diffusion + replenishment). Passive -- it never feeds back
        # into mass/wind/air -- so a spin-up that only bakes those can skip it (advance_oxygen).
        if advance_oxygen:
            s.oxygen = self.diffuse_and_advect_oxygen(s.oxygen, u_fire, v_fire)
            # Fresh-air replenishment toward the elevation oxygen profile (combustion below
            # still draws this down, so a vigorous fire can outpace it locally).
            if self.relaxation_enabled:
                s.oxygen = self.relax_to_equilibrium(s.oxygen, s.oxygen_eq, self.oxygen_rate)

        if advance_fire and self.fire_enabled:
            s.air_temperatures, s.fuel_temperatures, s.fuel, s.oxygen = self.update_fire(
                s.air_temperatures, s.fuel_temperatures, s.fuel, s.oxygen, s.mass,
                u_fire, v_fire, s.terrain,
            )
            if self.radiant_heat_enabled:
                s.fuel_temperatures, s.radiant_flux = self.apply_radiant_heat(
                    s.fuel_temperatures, s.fuel, temp_eq0
                )
            # Bole-segment conduction: the trunk climbs fire from the surface bole up to the canopy
            # bole (fast up, slow down) -- the ladder's dedicated conduit. No-op without a split bole.
            s.fuel_temperatures = self.conduct_bole_segments(s.fuel_temperatures, s.fuel)

        # Inter-level convection: carry heat *upward* through the air stack (surface -> canopy ->
        # above), gated on instability. This is what gives the surface/canopy slabs a sink -- the heat
        # they take from fuel and combustion cascades up to the above-canopy level, where the vent
        # sheds it. No-op on a quiescent world and for a single-level stack.
        s.air_temperatures = self.convect_air_levels(s.air_temperatures)

        # Vent the plume heat out of the top (above-canopy) level -- the single air-temperature sink:
        # without it the radiatively-transparent air has no vertical escape and a large fire's plume
        # heat accumulates without bound. Gated by its own toggle (independent of fire), so the closed
        # conservation core can switch it off; one-directional, so on a quiescent world it is a no-op.
        if self.venting_enabled:
            s.air_temperatures = self.vent_plume_heat(s.air_temperatures, s.temp_eq)

        return s

    @staticmethod
    def build_observation(
        air_temperatures: torch.Tensor,   # (..., H, W) near-surface air temperature [K]
        fuel_temperatures: torch.Tensor,  # (..., N_fuel, H, W)
        fuel: torch.Tensor,               # (..., N_fuel, H, W)
        terrain: torch.Tensor,            # (..., H, W) elevation [m]; observed as its slope, not height
        wind_x: torch.Tensor,             # (..., H, W) near-surface fire wind u [m/s]
        wind_y: torch.Tensor,             # (..., H, W) near-surface fire wind v [m/s]
        ignition_thresholds: torch.Tensor,  # (N_fuel, 1, 1) per-type ignition temperature [K]
        cell_size_m: float = pc.DEFAULT_CELL_SIZE_M,  # grid spacing dx=dy [m], for the slope gradient
    ) -> torch.Tensor:
        """Stack the observation channels into a normalized (..., C, H, W) observation.

        The channels are (see obs_channel_names): the near-surface air temperature, then one
        fuel-temperature channel per fuel type (in the fuel stack's order, no longer reduced to the
        hottest with amax), then one vegetation-mass channel per fuel type (same order), then the two
        signed terrain-slope components (slope_x, slope_y = grad(terrain), uphill +), the two
        near-surface wind components -- the same velocity the fire reads for spread (prognostic wind x
        channeling gain) -- and finally a binary "ignited" mask (1 where any fuel type at the cell is
        at/above its own ignition threshold, i.e. the cell is burning). N_fuel is read from the fuel
        stack, so C = 6 + 2*N_fuel. With a leading batch axis the result is the B x C x N x N tensor
        the world model trains on.

        Terrain enters as its slope, not its height: it is the slope that skews fire spread (Rothermel
        slope effect), and a conv stack struggles to synthesize that spatial derivative from a raw
        height field, so grad(terrain) = tan(slope) is supplied directly (periodic centered
        difference over cell_size_m, matching the spread physics' own _periodic_grad). The vegetation
        biome is not an observation channel -- the per-type fuel-mass channels already carry the fuel
        the model needs.

        The raw SI fields span very different magnitudes (K vs kg/m^2 vs m/s vs slope), so each
        channel is mapped to roughly [0, 1] by the documented affine window in
        physics_constants.OBS_NORM (value - offset) / scale; every per-type fuel channel shares
        the "fuel" window. Wind and slope are signed about 0; the high side is not clamped, so flames,
        strong gusts, and steep ground ride a little above 1.
        """
        # Burning mask: any fuel type at/above its own ignition threshold (the canonical "burning"
        # test, shared with rendering / the cells_burning stat). ignition_thresholds is (N_fuel,1,1)
        # so it broadcasts against the (..., N_fuel, H, W) temperature stack; reduce over the types.
        ignited = (fuel_temperatures >= ignition_thresholds).any(dim=-3).to(fuel_temperatures.dtype)
        # Signed terrain slope grad(z) = tan(slope), uphill +; the same periodic centered difference
        # the spread physics reads (see update_fire), so the observation exposes exactly that signal.
        slope_x, slope_y = Simulation._periodic_grad(terrain, cell_size_m)
        channels = [
            air_temperatures.select(-3, 0),     # air_temperature: surface-level air (canopy/above
                                                # levels are added to the obs once they carry signal)
            *fuel_temperatures.unbind(dim=-3),  # fuel_temperature_<type>: per-type temperature
            *fuel.unbind(dim=-3),               # fuel_<type>: per-type vegetation mass
            slope_x,                            # slope_x: terrain slope d/dx (signed, uphill +)
            slope_y,                            # slope_y: terrain slope d/dy (signed, uphill +)
            wind_x,                             # wind_x: near-surface fire wind u
            wind_y,                             # wind_y: near-surface fire wind v
            ignited,                            # ignited: any fuel type above its ignition threshold
        ]
        obs = torch.stack(channels, dim=-3)
        n_fuel = fuel.shape[-3]
        windows = (
            [pc.OBS_NORM[c] for c in OBS_CHANNELS_PRE]
            + [pc.OBS_NORM["fuel_temperature"]] * n_fuel
            + [pc.OBS_NORM["fuel"]] * n_fuel
            + [pc.OBS_NORM[c] for c in OBS_CHANNELS_POST]
            + [pc.OBS_NORM[OBS_CHANNEL_IGNITED]]
        )
        offsets = torch.tensor([w[0] for w in windows], dtype=obs.dtype, device=obs.device)
        scales  = torch.tensor([w[1] for w in windows], dtype=obs.dtype, device=obs.device)
        return (obs - offsets[:, None, None]) / scales[:, None, None]
