# Physical Units Conversion — Design Doc

Status: **proposed** · Target: convert Firecracker from a nondimensional, per-tick
model to a fully physical SI simulation.

## Goal & decisions

Convert every state variable, coefficient, and equation to SI units (Kelvin, kg,
metres, seconds, joules, pascals). Three decisions frame the work:

1. **Representation:** state tensors hold **true SI values** (not scaled internals).
2. **Fidelity:** **full physical rewrite** — replace heuristics with real laws, not
   just attach units to the existing forms.
3. **Wind:** a **real prognostic momentum equation**, not the `-∇p` diagnostic.

### Resolved parameters

These were pinned during design review and are assumed throughout:

| Choice | Decision |
|---|---|
| terrain `elev_max` | **500 m** over the 2.56 km domain |
| ambient wind | prevailing baseline **5–15 m/s** per map (random bearing) |
| peak wind | **~30 m/s** calibration target (terrain-channeled gusts; sets `g′`, `C_d`, CFL) |
| solar forcing | **constant daytime net flux**, sized so radiative equilibrium = `T₀` |
| Coriolis | **omitted** (negligible at 2.5 km scale) |
| combustion law | **Arrhenius, O₂-limited** (smooth ignition, no hard threshold) |
| oxygen field | **partial density [kg/m³]** (ambient ≈ 0.28 kg/m³) |

### Consequence: wind, pressure, and convection become one solver

Today, convective transport (`transport_mass_energy`, mass/energy diffusing down the
Φ-potential) and wind (`compute_wind_from_pressure`, a `-∇p` diagnostic) are two
separate heuristics. The decisions above collapse them into a single **single-layer
shallow-water atmosphere** with a prognostic velocity field. The current Φ-diffusion
is the over-damped limit of exactly this system, so it is a clean replacement.

## The current model (baseline)

Everything in `step_fields` is **per-tick** and **per-cell**; there is no `dx` or `dt`
in the physics:

- `steps_per_second=10` → `ms_per_step=100` is used *only* for real-time display
  throttling in `main.py`.
- `_laplace`, `_diffuse` (`sigma` in *cells*), and `_advect_field` (velocity in
  *cells/tick*, CFL "≤ 1 cell") all assume `dx = 1 cell`.
- Temperatures are O(1) (`air≈0.5`, `ignition≈1.5`); mass has mean 1; oxygen is a
  `[0,1]` saturation fraction; energy is `E = m·T` with an implicit `c_p = 1`.

Conversion is therefore **redimensionalization plus law replacement**: introduce real
length/time/temperature/mass scales, then swap heuristic forms for physical ones.

## Fundamental scales & constants

| Symbol | Meaning | Value |
|---|---|---|
| `dx`, `dy` | cell size | 10 m (cell area `A = 100 m²`) |
| `dt` | timestep | `1 / steps_per_second = 0.1 s` |
| domain | grid extent | 256 × 256 cells = 2.56 km square |
| `g` | gravity | 9.81 m/s² |
| `R_d` | specific gas constant, dry air | 287.05 J/(kg·K) |
| `c_p,air` | specific heat, air (const. p) | 1005 J/(kg·K) |
| `σ` | Stefan–Boltzmann | 5.670e-8 W/(m²·K⁴) |
| `T₀` | sea-level reference temp (30 °C, fire weather) | 303.15 K |
| `p₀` | sea-level pressure | 101325 Pa |
| `ρ₀` | sea-level air density at `T₀` | 1.165 kg/m³ |
| `Γ` | environmental lapse rate | 6.5e-3 K/m |
| `H_p` | pressure scale height (`R_d·T₀/g`) | ≈ 8870 m |

Combustion constants (per fuel type, config-driven; representative values):

| Quantity | Symbol | Grass | Tree/wood |
|---|---|---|---|
| heat of combustion | `HHV` [J/kg] | ~1.6e7 | ~1.8e7 |
| stoichiometric O₂ demand | `s` [kg O₂/kg fuel] | ~1.4 | ~1.4 |
| specific heat | `c_p,fuel` [J/(kg·K)] | ~1800 | ~2300 |
| ignition (pyrolysis) onset | `T_ign` [K] | ~550 | ~600 |
| Arrhenius pre-exponential | `A_pre` [1/s] | tune | tune |
| Arrhenius activation energy | `E_a` [J/mol] | tune | tune |

Constants live in a new `src/physics_constants.py`; per-fuel and tunable values stay
in `cfg/default.toml` under documented, unit-annotated keys.

## State variables (SI)

| Field | Was | Becomes | Unit |
|---|---|---|---|
| `mass` | column mass, mean 1 | shallow boundary-layer areal mass `m` (level lid) | kg/m² |
| `air_temperatures` | ~0.5 | `T` | K |
| `ground_temperature` | — | surface (soil/litter) skin `T_g` (Phase 3) | K |
| `(x,y)_wind_vel` | cells/tick, diagnostic | prognostic velocity `(u,v)` | m/s |
| `oxygen` | `[0,1]` fraction | O₂ partial density | kg/m³ |
| `fuel` (per type) | arbitrary | biomass areal density | kg/m² |
| `fuel_temperatures` | ~0.5 | `T_fuel` | K |
| `terrain` | `[0,1]` | elevation `z` | m |
| `radiant_flux` | diagnostic | absorbed radiant flux | W/m² |

Derived: surface pressure `p_s = m·g` [Pa]; column heat capacity `C_air = m·c_p,air`
[J/(m²·K)].

**Precision:** `Simulation.dtype → torch.float64`. SI magnitudes span ~1e-1…1e9, which
erodes float32 and the tight conservation tolerances. This roughly doubles tensor
memory; the `rollout.py` pool note (~3.2 GB at float32) must be recomputed (~6.4 GB),
so revisit `buffer_dtype` / pool sizing.

## Governing equations

### Shallow boundary-layer atmosphere over terrain (replaces transport + wind)

The dynamical layer is a **shallow atmospheric boundary layer** (depth ~ `layer_depth_ref`,
comparable to terrain), **not** the full column — this is what makes terrain a real
obstacle. Its free-surface height, with a thermally expanding layer (ideal gas: warmer
layers are taller):

```
thickness  η(m,T) = m · R_d / p_ref · T      [m]   (m is the boundary-layer areal mass)
s(x,y)     = z(x,y) + η(m,T)                  [m]   (free-surface height)
```

The map's `mass` is initialised to a **level lid** (`s` flat → `η = h_ref + (elev_max − z)`),
so the layer is thinner over high terrain. Because `η ~ O(1 km)` (not ~8.6 km), a hill is a
large constriction, and continuity makes the wind speed up where the layer thins.

Prognostic momentum, forced by the free-surface gradient, with weak interior surface
friction and eddy viscosity (Coriolis omitted at this scale):

```
∂u/∂t + (u·∇)u = -g′ ∂s/∂x - C_d·u + ν ∇²u
∂v/∂t + (u·∇)v = -g′ ∂s/∂y - C_d·u + ν ∇²v
```

Conservative continuity, energy and momentum advection (same velocity carries all):

```
∂m/∂t          + ∇·(m u)              = 0
∂(m c_p T)/∂t  + ∇·(m c_p T u)        = Q_rad + Q_burn + Q_cond     [W/m²]
∂(m u)/∂t      + ∇·(m u ⊗ u)          = forcing
```

**Open domain via a boundary sponge.** Advection is computed periodically, but within
`sponge_width` cells of the edges the wind is relaxed toward the per-map synoptic `u_amb`
and the mass/temperature toward their level-lid rest values (exact-exponential, so stable
at any strength). The upwind sponge injects the synoptic wind; the downwind sponge absorbs
outflow and damps wrap-around — so the wind **enters one edge, crosses the terrain, and
exits**. The interior feels only weak friction, so terrain shapes the flow. (Closed,
conserving behaviour is recovered with `sponge_strength = drag_coeff = 0`, used by the
conservation tests.)

**Observed orographic behaviour** (verified): windward blocking (slower), and
lee/downslope **acceleration** above ambient — the foehn / downslope-wind pattern that
drives real wildfires. Strength scales with how shallow the layer is (`layer_depth_ref`):
shallower → stronger channeling and gap winds. **Katabatic** drainage emerges later from
the `η`-buoyancy term once Phase 3 builds cold-slope temperature gradients.

**CFL:** substepped on `(|u| + √(g′η))·dt/dx`; with `g′ ≈ 0.2`, `η ≈ 1 km`, the gravity-wave
speed is ~15 m/s (transcritical with the wind → strong terrain response).

### Radiation (replaces `relax_to_equilibrium` for temperature)

Sunlight is absorbed **at the surface**, not by the air — clear air is nearly transparent to
shortwave radiation. So the solar flux heats the **ground/litter skin** and the **fuel**, each
of which emits longwave `εσT⁴`; the air column is warmed only by **sensible exchange** with the
ground. The two surface bodies share the incoming beam by a **vegetation cover fraction**
`c = 1 − exp(−fuel_total/κ)` (Beer–Lambert canopy interception), with area shares
`a_ground = 1−c` and `a_fuel,i = c·fuel_i/fuel_total`. Both the absorbed solar and the emitted
longwave are area-weighted (this keeps every body's balance vanishing at the rest profile):

```
S_net   = ε σ temp_eq⁴                                          [W/m²]  (per-cell anchor)
ground:  C_g dT_g/dt = a_g·S_net − a_g·εσT_g⁴ − k_ga·(T_g − T_a)
air:     m c_p dT_a/dt = k_ga·(T_g − T_a)                       (transparent to radiation)
fuel i:  C_fi dT_fi/dt = a_fi·S_net − a_fi·εσT_fi⁴              (C_fi = fuel_i·c_p,fuel)
```

`S_net` is fixed **per-cell** to the map's rest profile (`S_net = εσ temp_eq⁴`), so radiative
equilibrium sits exactly at `temp_eq` (`= T₀ − Γz`, i.e. `T₀` over flat ground, ≈ 455 W/m²): a
quiescent world holds station and matches the boundary sponge's `temp_eq` target. **Heat capacity
sets the response speed**: fine fuel (`C ≈ 10³ J/m²/K`) heats and dries in the sun within minutes,
the ground skin (`C_g ≈ 10⁵`) follows over hours, and the deep air column (`m c_p ≈ 10⁶`) lags
over days — so fuels lead the air, as they do in reality. `temperature_rate_low/high` are
**retired**. The pre-SI fuel↔air conduction (`exchange_fuel_air_heat`) is unchanged here (fixed in
Phase 4). Oxygen replenishment (`oxygen_rate`) is the only remaining Newtonian relaxation.

### Combustion (replaces `update_fire`)

Smooth Arrhenius reaction rate, limited by both fuel and oxygen (molar `R = 8.314`):

```
k(T_f)  = A_pre · exp(-E_a / (R · T_f))              [1/s]   Arrhenius rate
Δfuel   = fuel · k · (O₂ / O₂_ref) · dt              [kg/m²] capped at the fuel present
```

No hard ignition threshold — the Arrhenius exponential makes the rate vanishingly small
at ambient temperature on its own. The oxygen dependence is the *fractional* availability
`O₂/O₂_ref` (`O₂_ref` = ambient sea-level O₂ density ≈ 0.27 kg/m³), which keeps `A_pre` a
clean 1/s and smoothly extinguishes the fire as oxygen is drawn down. `A_pre` and `E_a` are
tuned so cold fuel is inert and ignition onset lands near the pyrolysis temperatures.

The oxygen field is the **near-surface combustion-layer density** [kg/m³], and `d_mix`
(~30 m) is that layer's depth. The areal burn draws `s·Δfuel` from a `d_mix`-deep slab, so a
vigorous fire can locally **deplete its own oxygen**; resupply is modelled separately —
**advection** feeds fresh air horizontally (the wind, Phase 2) and the **oxygen relaxation**
mixes it down vertically. This is the **ventilation-limited** regime: sparse fuel breathes
freely (fuel-limited), dense fuel can clamp O₂ to zero and lean on wind/mixing to keep burning.
`d_mix` is a modelling scale (like the wind core's `g′`/`ν`) that sets *when* oxygen bites — not
the ~1 km boundary layer, which would make O₂ effectively unlimited and the field pointless.

```
ΔO₂    = s · Δfuel / d_mix           [kg/m³]
Q_burn = HHV · Δfuel                 [J/m²], split between air and fuel by burn_heat_fuel_fraction
```

Both reactants are required, so the limiting is **two-way**: if the oxygen draw would exceed the
O₂ present, the whole reaction is scaled down (fuel, oxygen, and heat together) so they are
consumed in stoichiometric step — a fire starved of air burns less fuel, not the same fuel with
the books forced to balance.

The air share warms the column over `m·c_p,air`. Fuel↔air conduction (`exchange_fuel_air_heat`)
keeps its exact two-body exponential form, now with real heat capacities `C_air = m·c_p,air`,
`C_fuel = fuel·c_p,fuel`, a conductance `k` in W/(m²·K), and the decay exponent carrying `·dt`.

### Fire radiant transfer (replaces `apply_radiant_heat`)

Real grey-body emission with inverse-square neighbour redistribution (kernel radius in
metres) and a sky-escape fraction:

```
emission = ε σ T_fuel⁴ · dt          [J/m²], capped at the cell's thermal energy
```

The `T⁴` sink self-limits flame temperature, now with the actual `σ`.

## Initial-state generation (`gen_maps.py`)

| Field | New SI construction |
|---|---|
| terrain `z` | normalised Perlin × `elev_max` (e.g. 0–500 m) |
| temperature | `T(z) = T₀ − Γ·z` |
| mass | shallow boundary-layer areal mass on a level lid: `η = h_ref + (elev_max − z)`, `m = η·p₀/(R_d·T)` (`column_mass_profile`, the full barometric column, is kept only for reference) |
| oxygen | ambient O₂ density scaled with pressure/altitude |
| fuel | biomass kg/m² (grass ~0.5–2, forest ~5–20), elevation-thinned |
| ambient wind | per-map prevailing `u_amb = (speed·cosθ, speed·sinθ)` [m/s], random bearing θ and speed drawn from a config range; the synoptic target the momentum drag relaxes toward (Phase 2) |

**All 1016 maps in `maps/` and the 7 fixtures become invalid and must be
regenerated.** Stamp a `units_version` field into each `.npz`; `validate_against_config`
rejects mismatched/legacy maps with a clear "regenerate" message.

## Rendering, observations, config

- **Rendering** (`firecracker_env.py`): every `[0,1]`-assuming scale must be rebased to
  Kelvin/SI ranges — `_AIR_TEMP_DISPLAY_THRESHOLD`, the ignition-relative fire colors in
  `build_fire_surface`, pressure/column-height normalisation, etc.
- **Observations:** `OBS_CHANNELS` now span physical ranges; document a per-channel
  normalisation (mean/scale) for the world model and apply it in `build_observation` or
  at the trainer boundary.
- **Config:** rewrite `cfg/default.toml` with physical values and a unit annotation on
  every key; add a `[units]` section (`cell_size_m`, reference state, etc.).
- **`main.py`:** annotate the info printouts with units.

## Tests

- Rewrite `tests/scenarios.py` builders in SI (no more `air=0.5, mass=1.0,
  ignition=1.5`).
- Conservation tests split into **closed** (no radiation/fire → exact mass & energy
  conservation, now `Σ m·c_p·T`) and **open** (budget closure against accounted
  radiative/combustion fluxes).
- Re-baseline equilibrium (hydrostatic rest state stationary) and stability/CFL tests
  for `float64` and the new gravity-wave speed.

## Phased delivery

Each phase is independently testable; we do not change everything at once.

0. **Scaffolding** ✅ *done* (commit `b21863b`) — `src/physics_constants.py` (constants of
   nature + ISA-style reference atmosphere, `T_REF=303.15 K` fire-weather, `H_p=8870 m`
   derived from it). `Simulation.dtype → torch.float64` (SI spans ~1e-1..1e9; erodes
   float32 + tight tolerances). `cell_size_m` (dx=10 m) + `dt` (=1/steps_per_second=0.1 s)
   read onto `Simulation` from a new `[units]` config block. `units_version` stamped into
   every `.npz`; `validate_against_config` rejects mismatches with a regenerate hint.
   *Note:* float64 ~doubles tensor memory — revisit `rollout.py` pool sizing.
1. **SI state + I/O** ✅ *done* (commit `b21863b`) — `gen_maps.py` emits SI fields:
   terrain m (`relief·elev_max`), air `T = T_REF − Γ·z`, oxygen = `O2_frac·p/(R_d·T)`,
   fuel kg/m² (grass continuous, trees discrete counts). Mass was the exact barometric
   column then (superseded by Phase 2's boundary-layer mass). All rendering in
   `firecracker_env.py` rescaled to Kelvin/SI windows (`build_color_surface(field,scale,
   lo,hi)`, oxygen/terrain normalised by max, fire ramp `FIRE_COLOR_TEMP_SPAN_K`).
   `pressure_profile` uses the exact lapse-rate barometric formula (`p=p₀(1−Γz/T_REF)^
   (g/(R_d·Γ))`, isothermal `exp(−z/H_p)` only as Γ→0 fallback) — runtime-free (baked at
   gen). All 1024 maps regenerated; suite green.
2. **Shallow boundary-layer wind core** ✅ *done* (commit `4d0b2ab`) — `Simulation.
   step_dynamics` (replaces the old `transport_mass_energy` + diagnostic wind, both
   deleted). Per substep: forcing `u += dt(−g′∇s − C_d·u + ν∇²u)`, `s = terrain + η`,
   `η = m·R_d·T/p_ref`, periodic centered grad/Laplacian; then conservative upwind
   `_advect_periodic` of `m`, `E=m·cp·T`, `m·u`, `m·v`; recover `u,v=mom/m`, `T=E/(cp·m)`.
   CFL substeps on `(|u|+√(g′η))·dt/dx`. **Open domain**: after the loop, an exact-exp
   sponge (`_sponge_rate`, quadratic taper over `sponge_width` edge cells, cached per
   shape) relaxes `u,v→u_amb`, `m→mass_eq`, `T→temp_eq` — injects ambient upwind, absorbs
   outflow downwind. `mass` = shallow boundary-layer areal mass on a **level lid**
   (`gen_maps.boundary_layer_mass`: `η=h_ref+(elev_max−z)`, `m=η·p₀/(R_d·T)`; ~1.2e3
   kg/m², thinner over peaks). `SimState` gained `u_amb_x/y` + `mass_eq`; per-map ambient
   wind (random bearing, 5–15 m/s) baked into maps; env/rollout init wind = ambient,
   `mass_eq` = loaded mass. `units_version → 3`, maps regenerated. `[momentum]` config:
   `layer_depth_ref=1000`, `reduced_gravity=0.2`, `drag_coeff=0.0005`, `viscosity=50`,
   `sponge_width=8`, `sponge_strength=10`, `cfl_target=0.5`. Dead no-flux/blur/diagnostic-
   wind helpers pruned; `main.py` printout updated. **Verified**: windward blocking +
   lee/downslope (foehn) acceleration, flat-world holds ambient, steady-until-perturbed,
   stable under strong wind/steep terrain, closed-core (sponge off) conserves mass/energy/O₂.
   *Caveat:* default `layer_depth_ref=1000 m` is the recommended value — **lowering it makes
   valleys *slower*, not faster** (level-lid continuity: fast over thin-layer peaks, slow in
   deep valleys); see Known limitations.
3. **Radiation** ✅ *done* — `Simulation.apply_radiation` replaces the temperature
   `relax_to_equilibrium` with a **surface energy balance (canopy-split model)**: sunlight is
   absorbed at the surface (air is shortwave-transparent) and split between the **ground skin**
   and the **fuel** by a cover fraction `c = 1 − exp(−fuel_total/κ)`; each absorbs `a·S_net` and
   emits `a·εσT⁴` (area-weighted), the air is warmed only by sensible exchange with the ground
   (`k_ga·(T_g − T_a)`). `S_net` is fixed **per-cell** to `ε·σ·temp_eq⁴`, so radiative equilibrium
   = the rest profile `temp_eq` (= `T_REF` over flat ground, ≈ 455 W/m²) and matches the sponge
   target. **Heat capacity sets the speed**: fine fuel (`C ≈ 10³`) leads in minutes, ground skin
   (`C_g ≈ 10⁵`) over hours, deep air (`m·cp ≈ 10⁶`) over days — so fuels heat/dry ahead of the
   air, the key fire-weather effect. New `ground_temperature` field (inits to `temp_eq`, derived —
   no map regen) + per-type SI `specific_heat` (grass 1800, tree 2300). New `[radiation]` knobs
   (`emissivity=0.95`, `ground_heat_capacity=1e5`, `ground_air_exchange=10`, `cover_fuel_scale=1`);
   **retired** `temperature_rate_low/high`, `temperature_relax_rate`, the `temp_relax_rate` field.
   The pre-SI fuel↔air conduction (`exchange_fuel_air_heat`) and pre-SI `heat_capacity` are left for
   Phase 4. `relax_to_equilibrium` now drives only oxygen. Forward Euler is stable (area-weighting
   ties absorbed flux to mass → gentle steps). *Tests* (`tests/test_radiation.py`): equilibrium
   fixed point, fuel-leads-ground-leads-air, cover shading, bare-ground limit, analytic ground
   step, 455 W/m² calibration, quiescent world holds station. *Seam:* the fuel longwave `εσT_f⁴`
   here and Phase 5's flame emission both use `εσT_f⁴` — Phase 5 must **redistribute** that
   emission to neighbours, not add a second one (no double-count today: `apply_radiant_heat` is
   still pre-SI). *Emergent (future):* cold slopes → smaller `η` → `−g′∇s` → katabatic drainage.
4. **Combustion + SI oxygen** ✅ *done* — `update_fire` rewritten to a smooth Arrhenius,
   oxygen-limited rate `k = A·exp(−E_a/(R·T_fuel))` [1/s] (molar `R`), throttled by the local
   oxygen fraction `O₂/O₂_ref` — no hard ignition threshold (cold fuel is inert because the
   exponential underflows at ambient `T`). Per tick `Δfuel = (fuel·k·(O₂/O₂_ref)·dt)` capped at
   the fuel present; `Q_burn = HHV·Δfuel` [J/m²] (HHV 1.6e7 grass / 1.8e7 tree) split air/fuel by
   `burn_heat_fuel_fraction`, the air share over `m·cp_air` (was a raw add with **no** heat
   capacity). **Oxygen coupling:** O₂ is a partial density [kg/m³] but the burn is areal [kg/m²];
   they reconcile through a shallow **combustion mixing depth** `d_mix` (config, ~30 m — the near-
   surface air the fire entrains, *not* the ~1 km boundary layer), so `ΔO₂ = s·Δfuel/d_mix`
   (s≈1.4). Limiting is **two-way** — if the O₂ draw exceeds what's present, the whole reaction
   (fuel, oxygen, heat) scales down in stoichiometric step, so a fire starved of air burns less
   fuel. Using `d_mix` rather than the full layer keeps oxygen a real local limiter, not just a
   rate throttle. **`exchange_fuel_air_heat` fixed:** `C_air = m·cp_air` (was missing `·cp_air`),
   `C_fuel = fuel·cp_fuel`, decay exponent now carries `·dt`. **Consolidated** every fuel
   heat-capacity use onto the SI `specific_heat` key and **dropped** the pre-SI `heat_capacity`,
   `consumption_rate`, `burn_heat_scale`; `[fuel_types]` gained SI `heat_of_combustion`,
   `stoich_oxygen`, `arrhenius_pre`, `activation_energy`; `[fire]` dropped
   `oxygen_consumption_rate`/`oxygen_extinction_threshold`, gained `combustion_mixing_depth`.
   New constants `UNIVERSAL_GAS_CONSTANT`, `O2_DENSITY_REF`. No on-disk change (oxygen already
   kg/m³, advected since Phase 2) — **no `units_version` bump, no map regen**. *Tests*
   (`tests/test_combustion.py`): cold-fuel inert, ignition + heat release, oxygen throttle/
   extinction, stoichiometric O₂ budget, HHV split, Arrhenius monotonic in `T`;
   `test_heat_exchange` rebaselined to `cp_air`/`cp_fuel`/`dt`; `test_radiant_heat` rebaselined
   for the `specific_heat` repoint. *Known:* at flame temps `k·dt ≫ 1`, so a hot cell burns its
   fuel out in one tick (explicit-Euler saturation, capped at the fuel present — finite, but
   sharp); revisit with sub-tick consumption in Phase 7. **Fire spread still awaits Phase 5**
   (`apply_radiant_heat` is the propagation mechanism and is still pre-SI; burnt cells have
   `C_fuel=0` and emit nothing).
5. **Fire radiant transfer** ✅ *done* — `apply_radiant_heat` rewritten to a grey-body
   **redistribution** (the *preheating* leg of spread, not an ignition driver — radiation warms
   fuel toward pyrolysis but rarely ignites it alone, as in real fires). It does **not** emit or
   cool: the fuel's longwave `εσT_f⁴` is already radiated (and removed) by `apply_radiation`
   (Phase 3), which owns the self-limiting `T⁴` sink. This stage redistributes only the
   **super-ambient** part `a_fuel·εσ(T_f⁴ − temp_eq⁴)·dt` [J/m²] onto nearby fuel by the
   inverse-square `_radiant_convolve`, so a flame's heat warms its neighbours. Subtracting
   `temp_eq⁴` keeps a quiescent world from radiating its own equilibrium glow; the `a_fuel`
   area-weighting (shared with Phase 3 via the new `_cover_fractions` helper) caps the
   redistributed energy at what the emitter actually shed, so it can never amplify. Phase 3
   already booked that emission as leaving to space, so depositing the absorbed neighbour share
   back is energy-consistent (sky escape + off-grid loss truly leave). **Kernel radius in metres**
   (`kernel_radius_m=100`, → cells by `cell_size_m`; the inverse-square weight is ~1% by 10 cells,
   so 100 m is the sweet spot). `[radiant_heat]` dropped the pre-SI `emission_scale`/
   `emission_exponent` and the ignition-threshold gate; reuses `[radiation].emissivity` and the
   real `σ`. No on-disk change — **no `units_version` bump, no map regen**. *Tests*
   (`tests/test_radiant_heat.py`): quiescent world does not spread, sub-ambient fuel is inert, a
   lone hot cell warms neighbours without self-heating, no deposit into fuel-free cells, delivered
   energy bounded by emission. **Fire does not yet *spread*** — a single cell's radiative
   preheating is (correctly) too weak to ignite a neighbour alone, and the Phase 4 one-tick
   burnout leaves an igniting cell with no fuel to emit. Sustained flaming + convective ignition
   is **Phase 5.5**.
5.5. **Sustained burn + plume convection** *(the "fire spreads" deliverable — surfaced by
   Phase 5)* — Phase 5 confirmed radiation alone can't propagate fire, by design: it preheats.
   Ignition in real fires comes from convection (hot plume gases) and direct flame contact, and
   our model can't currently sustain a flame at all — fast burn (`A=1e8`) consumes all fuel in one
   tick (nothing left to emit), slow burn self-extinguishes below the pyrolysis knee. Energy is
   ample (one grass cell releases ~30× a neighbour's ignition energy); the blockers are
   *sustainability* and *delivery*. Three legs:
   - **(a) Sustained flaming (linchpin):** rate-limit combustion so a cell stays hot *and* fuelled
     over a residence time (~seconds), not one tick — sub-tick/exponential consumption — and retain
     enough combustion heat in the flame to hold it above the knee (revisit `burn_heat_fuel_fraction`
     or model a flame temperature). Also fixes the Phase 4 explicit-Euler burnout (was deferred to
     Phase 7).
   - **(b) Shallow plume layer:** the combustion air-share heat currently dumps into the full
     boundary-layer column (`m·c_p ≈ 1.2e6`) → only ~16 K per burn, a negligible convective signal.
     Route it through a shallow near-surface plume depth (like oxygen's `d_mix`) so the air carries
     a strong thermal signal that wind advects downwind. Couples to the existing `η`-buoyancy →
     stronger indraft.
   - **(c)** Radiation (Phase 5) + convection (b) then jointly preheat and ignite the fuel ahead;
     calibrate emission/ignition and demonstrate a propagating **front** (and lee/downwind bias
     under ambient wind). Add a spread test over full `step_fields` ticks.
   - **(d) Upslope spread bias (Rothermel slope effect):** fire spreads faster uphill, because the
     flame and plume tilt toward the upslope fuel, shortening the flame-to-fuel distance and
     intensifying preheating. Much of this should **emerge** from (b): a buoyant plume over a slope
     flows uphill through the existing free-surface forcing (`−g′∇s`, `s = terrain + η`; combustion
     heat raises `η` → upslope indraft), so the convective preheat already biases uphill. Add the
     **radiative** half explicitly by tilting the `_radiant_convolve` kernel toward the upslope
     (and downwind) direction — a slope/wind-skewed kernel instead of the isotropic inverse-square
     — so radiant preheating also favors the fuel ahead/above. Calibrate the spread-rate increase
     against Rothermel's slope factor (rate of spread rising ~`tan²(slope)`), and add a test that
     an ignition on a ramp advances faster upslope than downslope.
   - *Optional:* a **fuel moisture** field — radiation's biggest real preheating job is drying;
     moisture would gate ignition and make the radiation→convection handoff faithful.
6. **Observations / rendering / config / main** — `OBS_CHANNELS` now span physical ranges;
   add a documented per-channel normalisation (mean/scale) for the world model (in
   `build_observation` or trainer boundary). Revisit rendering color windows for real flame
   temps. Full `cfg/default.toml` pass (units on every key, drop "pre-SI" markers). Update
   `main.py` info printouts.
7. **Test rebaseline + numerical hardening** — consolidate SI tests; split conservation into
   **closed** (exact) vs **open** (inflow/outflow budget). Regenerate `fixtures/` + port
   `tools/visualize.py` (legacy nondimensional, would fail `units_version`). Re-verify CFL/
   stability at 256² with all subsystems on; check float64 throughput (~20 steps/s CPU at
   256²; device-agnostic GPU path untested).

## Implementation conventions (for any continuation)

- **SI everywhere**, `Simulation.dtype = torch.float64`. **Rank-agnostic** ops (trailing
  `…,H,W`) so single-world and batched `(B,H,W)` share `step_fields`.
- **`units_version`** (`map_loader.py`, currently **3**): bump on any on-disk semantics
  change, update the comment, **regenerate all 1024 maps** (`python src/gen_maps.py
  --count 1024 --out maps --seed 0`, ~2 min); `validate_against_config` rejects mismatches.
- **No pre-SI physics code remains** — every stage of `step_fields` is now SI (Phases 0–5).
  A single fire burns physically and radiantly **preheats** its neighbours, but does **not yet
  spread** (ignition needs sustained flaming + convection — Phase 5.5). (`relax_to_equilibrium`
  is retained but now only drives oxygen replenishment; temperature is the physical balance of
  Phase 3.)
- **Closed-core conservation tests**: pass `momentum={"sponge_strength":0,"drag_coeff":0}`
  to recover exact periodic conservation (the open sponge otherwise adds/removes air).
- **Fire→wind coupling already wired**: forcing `−g′∇s`, `s=terrain+η`, `η=m·R_d·T/p_ref`;
  heating `T` raises `η` → surface indraft. Keep `T` in `η`.
- **Tests build maps in temp dirs** via `scenarios.py`; none read `maps/`. SI scenario
  builders: `si_flat`, `si_hill`, `si_ridge`; `build_map(..., ambient_wind=(x,y))`;
  `make_config(..., momentum={...})`.

## Calibration left (numerical, done against tests in each phase)

- **Phase 2** *(done; revisit if needed)*: `g′`/`C_d`/`layer_depth_ref` set the foehn/
  channeling strength; defaults give prevailing 5–15 m/s with lee gusts ~20 m/s.
- **Phase 3** *(done)*: per-cell `S_net = εσ·temp_eq⁴` lands radiative equilibrium on the
  rest profile (`≈ 455 W/m²` at `T_REF`); `ε=0.95`.
- **Phase 4** *(done)*: Arrhenius defaults (grass `A=1e8`, `E_a=1e5`; tree `A=5e7`,
  `E_a=1.1e5`) make cold fuel inert at `T_REF` and the rate climb steeply through the cfg
  pyrolysis temps; per-fuel `HHV`/`s`/`cp_fuel` pinned; `d_mix=30 m`. **Revisit in Phase 5.5**:
  `A=1e8` gives sub-ms residence (one-tick burnout) — the burn rate must be limited for sustained
  flaming and spread.
- **Phase 5** *(done)*: `kernel_radius_m=100` (≈10 cells; inverse-square weight ~1% by there),
  `sky_escape_fraction=0.2`, reusing `ε=0.95` from `[radiation]`. Tuned as a *preheating* leg, not
  an ignition driver — spread calibration is Phase 5.5.

## Known limitations / deferred

- **Valley/gap (Venturi) winds NOT captured**: the level-lid shallow layer gives fast
  ridges / **slow valleys** + strong foehn lee winds; lowering `layer_depth_ref` makes
  valleys *slower*, not faster. True fast-valley/gap winds need blocking-terrain (wetting–
  drying) physics — optional future enhancement, not in the phase plan.
- **Lee separation/turbulence** regime not modelled (hydraulic/foehn regime instead).
- `column_mass_profile` (full barometric column) kept in `gen_maps` for reference but unused
  by dynamics; `equilibrium_mass` kept only for the legacy `fixtures/`/visualizer scenarios.
