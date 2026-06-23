# Physical Units Conversion — Design Doc

Status: **done** — phases 0–8 complete: Firecracker is converted from a nondimensional, per-tick
model to a fully physical SI simulation (scaffolding through terrain-channeling wind + wind-driven
spread, observation/config/render polish, and test rebaseline + numerical hardening).

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
shallower → stronger channeling and lee/foehn winds (but valleys stay *slow* — fast valley/gap
winds are not captured by this core; see Known limitations). **Katabatic** drainage emerges later from
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

The **fuel share** `f·Q_burn` (`f = burn_heat_fuel_fraction`) warms the fuel over its own thermal
mass and **sets the flame temperature** (bounded by the `T⁴` sink). The **air share** `(1−f)·Q_burn`
does not warm the deep column: it rises in a shallow buoyant **plume** of depth `d_plume`
(`C_plume = ρ_local·d_plume·c_p,air`, `ρ_local = p_ref/(R_d·T_a)`; Phase 5.5b), so a burn raises a
strong, advectable air signal that lifts `η` → indraft. A fraction `convective_fraction` of that
air share is then **convected onto nearby fuel to ignite it** — the front's propagation driver
(Phase 5.5c). Radiation (`apply_radiant_heat`) preheats the fuel ahead but is too weak to ignite it
at realistic flame temperatures, so convection (hot plume gas / flame contact) carries the ignition
and the spread is convection-dominant, as it is for grass fires. That convective deposit is sourced
from the fresh burn heat (wind cannot sweep it away), gated to
sub-flaming fuel (below `flame_gate_temperature`, so flaming fuel stays combustion-controlled and the
front does not run away), and skewed by a per-cell energy-conserving **von Mises angular kernel**
(`deposit(δ) ∝ K0(δ)·exp(b·δ̂)`, normalised per source) along a bias vector `b` that blends the unit
wind (lee bias) with the upslope terrain gradient (`b = b_wind·ŵ + b_slope·∇z·|∇z|`; the slope term
scales as `tan²(slope)` for the Rothermel upslope effect, Phase 5.5d). `|b|` is the forward
concentration: the kernel is non-negative for any `|b|` (no clamp), so wind and slope aligning keeps
focusing the deposit into a tighter, faster head instead of saturating (`|b|` is capped only as a
numerical guard against `exp` overflow at the spurious terrain-wrap gradient). Whatever the fuel does
not absorb stays in the plume air, so energy is conserved (off-grid edge loss aside).

Fuel↔air conduction (`exchange_fuel_air_heat`) keeps its exact two-body exponential form, with real
heat capacities `C_air = m·c_p,air`, `C_fuel = fuel·c_p,fuel`, a conductance `k` in W/(m²·K), and the
decay exponent carrying `·dt`.

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
   sharp); resolved in Phase 5.5a (surface-area combustion). **Fire spread still awaits Phase 5**
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
   real `σ`. An `enabled` flag (`[radiant_heat].enabled`, default `true`, parsed as
   `radiant_heat_enabled`, gates the `apply_radiant_heat` call in `step_fields`) toggles this
   preheating leg independently of the Phase-3 `T⁴` sink — turning it off isolates convection-only
   spread, which `test_radiant_preheat_needs_convection_but_accelerates_it` uses to show preheat
   alone can't sustain a front but accelerates one convection carries (1 / 67 / 89 burnt cells for
   preheat-only / convection-only / both). No on-disk change — **no `units_version` bump, no map
   regen**. *Tests*
   (`tests/test_radiant_heat.py`): quiescent world does not spread, sub-ambient fuel is inert, a
   lone hot cell warms neighbours without self-heating, no deposit into fuel-free cells, delivered
   energy bounded by emission. At this stage fire did **not yet spread** — a single cell's radiative
   preheating is (correctly) too weak to ignite a neighbour alone; sustained flaming + convective
   ignition (**Phase 5.5**, now done) is what makes the front propagate.
5.5. **Sustained burn + plume convection** *(the "fire spreads" deliverable — surfaced by
   Phase 5)* — Phase 5 confirmed radiation alone can't propagate fire, by design: it preheats.
   Ignition in real fires comes from convection (hot plume gases) and direct flame contact, and
   our model can't currently sustain a flame at all — fast burn (`A=1e8`) consumes all fuel in one
   tick (nothing left to emit), slow burn self-extinguishes below the pyrolysis knee. Energy is
   ample (one grass cell releases ~30× a neighbour's ignition energy); the blockers are
   *sustainability* and *delivery*. Four legs:
   - **(a) Sustained flaming via surface-area combustion (linchpin)** ✅ *done* (`fd2d1c2`) — the
     one-tick burnout is an
     artefact of treating combustion as a *bulk* Arrhenius rate of the fuel's mean temperature. Real
     fuel reacts only at its *surface*, so the burn is surface-area-controlled — which is also why
     grass (high surface-area-to-volume) flashes while a log (low SAV) smolders: same material and
     `HHV`, only the geometry differs. Model it as a heterogeneous surface flux. Per fuel type add a
     surface-area-to-volume ratio `σ [1/m]` (cured grass ~12000, a 10 cm log ~40) and a particle
     density `ρ_p` (~500 kg/m³), giving a specific reactive surface `a_s = σ/ρ_p [m²/kg]` (grass ~24,
     log ~0.08 — a ~300× span). The surface mass flux `ψ [kg/(m²·s)]` puts chemistry and oxygen
     transport in series: `ψ_kin = B·exp(−E_a/(R·T_f))` (kinetic — the ignition gate, cold fuel
     inert), `ψ_dif = h·(O₂/O₂_ref)` (O₂ film diffusion to the surface), `ψ = (1/ψ_kin + 1/ψ_dif)⁻¹`.
     Areal consumption `Δfuel = fuel·(1 − exp(−a_s·ψ·dt))` (exact-exponential, bounded by the fuel
     present — retires the `clamp(max=fuel)` hack). Cold → kinetic-limited (inert); flame-hot →
     transport-limited at `ψ_dif`, the physical speed cap ("the outside burns before the inside can
     heat up"). The flaming residence time then *emerges*, `τ ≈ 1/(a_s·ψ)`: grass ~seconds, a log
     ~minutes. The existing two-way stoichiometric O₂ *inventory* limiter stays as the whole-cell
     "ran out of air" backstop (a coarser scale than the per-particle film transport). A steady
     surface burn over `τ` releases `HHV·Δfuel` gradually, so the flame holds above the pyrolysis
     knee far more readily than a one-tick dump (co-calibrate with `burn_heat_fuel_fraction`). No new
     state field; reinterprets `[fuel_types].arrhenius_pre` as the surface pre-exponential `B`, adds
     `σ`/`ρ_p` per fuel and `[fire].surface_mass_transfer` (`h`). Also fixes the Phase 4
     explicit-Euler burnout (was deferred to Phase 8). *Optional later:* a shrinking-core state so
     `a_s` evolves as particles burn down, and wind-enhanced `h` (Sherwood ∝ Re); `σ` also feeds
     Rothermel in (d). The two-state flame-temperature model is the fallback if `B`/`h`/`σ`
     calibration proves stubborn. *Result:* in the full loop a lit grass cell holds ~1000–2400 K
     (well above the 573 K knee) and burns down over ~10 s, then self-extinguishes — finite, no
     runaway (the Phase 3 `T⁴` sink bounds it). The ~2400 K peak is hot for grass (real ~1100–1500 K)
     but stable, so flame-temperature calibration moves to (c) rather than being tuned in isolation.
   - **(b) Shallow plume layer** ✅ *done* — the combustion air-share heat was dumped into the full
     boundary-layer column (`m·c_p ≈ 1.2e6`) → only ~16 K per burn, a negligible convective signal.
     It is now injected into a shallow near-surface **plume** of depth `plume_mixing_depth` (`d_plume
     ≈ 30 m`, mirroring oxygen's `d_mix`), using the **local** near-surface density `ρ = p_ref/(R_d·
     T_a)` (= `m/η` at reference pressure): `C_plume = ρ·d_plume·c_p,air ≈ 3.5e4`, so the same burn
     warms the air **~34× more** — a strong, advectable thermal signal that lifts `η` → indraft.
     Being local, hotter air is lighter → smaller capacity → slightly more responsive (a mild
     feedback, bounded by the Phase 3 `T⁴` sink). This is a deliberate **sub-grid plume scale**: a
     single-layer model conserving column energy is stuck at ~16 K, so concentrating the air share
     in a shallow slab is the only way to express plume buoyancy (symmetric to `d_mix` concentrating
     the O₂ draw). New `[fire].plume_mixing_depth`; no new state field, no `units_version` bump.
     *Tests:* the HHV split now checks the air rise against `C_plume`, plus a plume-concentration
     test (air ΔT ≫ the full-column rise, finite). The indraft + downwind-warming + propagating-
     **front** demonstration is in leg (c)'s `test_spread` (over multi-tick `step_fields`).
   - **(c) Convective front propagation** ✅ *done* — calibration confirmed radiation alone **cannot**
     drive spread: at a realistic flame temperature its `εσT⁴` preheat lifts a neighbour only ~30–90 K
     (far short of the ~573 K pyrolysis knee), and it ignites a neighbour only by running the flame
     away to ~4000 K — so radiation stays the *preheat* leg, as designed. Ignition (and thus spread)
     is driven by **convection**: `update_fire` now convects a fraction (`convective_fraction`, 0.12)
     of the plume's air-share heat onto nearby fuel to ignite it (hot plume gas / flame contact). The
     spread is convection-dominant, realistic for grass; radiation contributes preheat ahead of the
     front but cannot ignite alone. It is sourced from *this
     tick's* combustion heat (not the advected air field, so wind can't sweep the source away),
     deposited only into **sub-flaming** fuel (below `flame_gate_temperature`, 700 K — flaming fuel is
     combustion-controlled, which bounds the front and stops the runaway), over a short
     `convective_radius_m` (10 m) reach, and **skewed downwind** by a per-cell, energy-conserving
     **von Mises angular kernel** (`deposit(δ) ∝ K0(δ)·exp(b·δ̂)`, bias `b = convective_wind_bias·ŵ`)
     — giving the lee bias (the same kernel carries the upslope bias in leg d). The flame temperature
     is kept realistic by lowering `burn_heat_fuel_fraction` to **0.05**
     and diluting the plume to `plume_mixing_depth` **100 m**. *Result* (one-shot hot cell, then
     released): a self-sustaining front at a realistic ~1540 K flame (no runaway) spreading ~2–5 m/s,
     symmetric with no wind and biased downwind under ambient wind. No new state field, no
     `units_version` bump. *Tests* (`tests/test_spread.py`, full `step_fields` ticks): one-cell
     self-propagation (symmetric), downwind bias, physical flame band, and no spontaneous ignition.
   - **(d) Upslope spread bias (Rothermel slope effect)** ✅ *done* — fire spreads faster uphill,
     because the flame and plume tilt toward the upslope fuel, shortening the flame-to-fuel distance.
     Part of this already **emerges** from (b) (a buoyant plume over a slope flows uphill via the
     free-surface forcing `−g′∇s`), but the level-lid rest state cancels the terrain gradient there,
     so the explicit driver is the **leg (c) convective kernel**: `_convective_deposit`'s bias vector
     is now a blend `b = convective_wind_bias·ŵ + convective_slope_bias·∇z·|∇z|`. The slope term is
     `∇z` (= `tan(slope)`, uphill) scaled by its own magnitude, so it points uphill with magnitude
     `tan²(slope)` — the front skews uphill ever harder on steeper ground, tracking Rothermel's `~tan²`
     slope factor. **The kernel was upgraded from the linear dipole to a von Mises angular form**
     (`deposit(δ) ∝ K0(δ)·exp(b·δ̂)`): the old dipole stayed non-negative only for `|b| ≤ 1` and so had
     to clamp the bias, which *saturated* the skew and killed the wind/slope synergy; the angular
     kernel is non-negative for any `|b|`, reading `|b|` as the forward concentration, so aligning wind
     and slope keeps focusing the deposit into a tighter, faster head. It is still **directional**
     (redistributes the convective heat, does not boost the total burn), so it captures the
     upslope/downslope asymmetry and the alignment synergy, not an absolute rate-of-spread gain.
     `update_fire` gained a `terrain` argument; `_radiant_convolve` is left isotropic (convection is
     the spread driver, so biasing it suffices). No new state field, no `units_version` bump.
     *Result* (ramp, no wind): a front that advances clearly faster upslope than downslope (head >
     flanks > backing) and stays symmetric across the slope; aligning a wind with the upslope drives a
     tighter, faster head still. *Tests* (`tests/test_spread.py`): upslope-biased spread on a ~31° ramp,
     and wind-aligned-with-slope focusing a faster head with pinched flanks (the non-saturation property).
   - *Optional:* a **fuel moisture** field — radiation's biggest real preheating job is drying;
     moisture would gate ignition and make the radiation→convection handoff faithful.
6. **Terrain-channeling surface wind + wind-driven spread (Phase 6, Option 2)** ✅ *done* — the
   prognostic core is terrain-responsive (windward blocking, lee/foehn acceleration) but its
   level-lid layer *deepens* in valleys, so it makes gaps **slow** and cannot produce the gap/Venturi
   speedup (the strict xfail `test_narrow_channel_does_not_speed_up_wind`, kept). Rather than a second
   prognostic layer (Option 1 — built, shelved on `phase6-two-layer` as research-grade; see Known
   limitations), the gap speedup is imposed as a **diagnostic rule**: `compute_channel_gain`
   precomputes a static per-map gain by scanning the **cross-stream relief** (the line perpendicular
   to the fixed synoptic wind, sampled along the true bearing by bilinear `grid_sample` so any wind
   angle works) out to `radius_m` each way, grading each step's openness by relief rather than a hard
   wall threshold: `openness = clamp(1 − rise/height_scale_m, 0, 1)` (rise = terrain above the cell),
   shadowed outward by a running min (a wall hides everything beyond it) and summed into an open width
   per side. The two sides combine as `W_eff = max(W_plus, W_minus)` — the *more-open* side limits the
   boost — and `gain = clamp(radius/W_eff, 1, gain_max)` (flat → 1, a tight gap → up to `gain_max`),
   lightly Gaussian-smoothed. The `max` combiner is the key: a true valley is confined **both** ways
   so `W_eff` is small and the gain is large, while a hillside, mountain base or one side of a saddle
   stays open on one side → `W_eff = radius` → gain ≈ 1 (a bare gradient would wrongly accelerate
   those). Both wall *height* (via the grading) and *distance* (a closer wall caps the width sooner)
   feed the gain continuously. `step_fields` multiplies the wind the **fire and oxygen** read by this
   gain (speedup only — direction preserved, since the core already deflects), leaving the
   **prognostic field untouched** (so the xfail stays honest). New `SimState.channel_gain
   /x_wind_fire/y_wind_fire` (optional — hand-built states are unaffected), `[channeling]` config
   (`enabled` **on by default**, `radius_m=300`, `height_scale_m=100`, `gain_max=3`, `smooth_sigma=1`).
   **Wind speed now drives rate-of-spread** (Phase 6's second half): the convective deposit's wind
   bias was the *unit* wind (direction only); it is now `convective_wind_bias·(wind/convective_wind_ref)`
   (`u_ref=10 m/s`), so a faster wind grows the forward concentration and throws the ignition heat
   harder downwind. Paired with the channeling gain, a gap both speeds the wind and focuses the head.
   *Result:* head reach grows ~3→6→9→11 cells as wind goes 0→5→10→15 m/s with the flanks pinching in;
   the channeled surface wind exceeds ambient in a gap. No new state on disk, **no `units_version`
   bump, no map regen**. *Tests:* `test_equilibrium.py` gains channel-speedup + flat-gain-unity
   tests (xfail kept); `test_spread.py` gains a wind-speed-drives-spread test and the preheat-synergy
   thresholds were re-tuned to the speed-scaled deposit (~1/25/37 burnt for preheat/convection/both).
   *Deferred:* wind-scaled convective *reach* (spotting) and the prognostic gap jet (Option 1).
7. **Observations / rendering / config / main** ✅ *done* — the world-model observation is now
   **normalised**: `OBS_CHANNELS` span physical ranges, so `Simulation.build_observation` maps each
   channel to roughly `[0, 1]` by a documented affine window `(value − offset)/scale` from
   `physics_constants.OBS_NORM` (`fuel_temperature` `(T_REF, 2000)`, `fuel` `(0, FUEL_REF=20)`,
   `terrain` `(0, ELEV_MAX=500)`); the high side is unclamped so flames ride a little above 1.
   Rank-agnostic, so the batched rollout pool stores normalised samples. The gym env's
   `observation_space` was made honest (raw air-temp `Box` in Kelvin, distinct from the world-model
   path; the env obs rework waits for a real agent). Rendering: `FIRE_COLOR_TEMP_SPAN_K` widened
   (600 → 1600 K) so the flame ramp resolves the real ~573–2400 K band. Config: rewrote the stale
   "pre-SI" header, dropped dead knobs (`[simulation].blur_sigma`, `[wind].smooth_sigma`/
   `temporal_smoothing`), added unit annotations, and relabelled `[convection]` as **legacy**
   (it fed only `gen_maps`' retained reference mass profile; removed in Phase 8). `main.py`
   printouts gained units and a `[channeling]` section. No `units_version` bump, no map regen.
   *Tests:* `tests/test_observation.py` (per-channel normalisation, hottest-type reduction, batched
   == single); `test_batched.py` obs assertion rebaselined to the normalised channels.
8. **Test rebaseline + numerical hardening** ✅ *done* — conservation split into **closed** vs
   **open**: `tests/test_conservation.py` keeps the closed (sponge/friction off) exact mass/energy/
   oxygen tests and adds open-domain throughflow tests (a warm anomaly is flushed out by the sponge
   while the closed core conserves it; a blob released upwind drifts downwind), with
   `test_radiation.py::test_quiescent_world_holds_station` as the open baseline. The pre-SI scenario
   builders in `tests/scenarios.py` were rewritten to SI (`uniform`/`hot_blob`/`corner_blob`/
   `oxygen_saturation`/`mass_gradient` over the flat rest state; `equilibrium`/`off_equilibrium` as
   SI level-lid rest / off-rest states), and `tools/visualize.py`'s `write_fixtures_toml` emits an SI
   config — `fixtures/` regenerated (`units_version=3`, reload cleanly). The legacy `[convection]`
   block and its `gen_maps` reference-mass methods (`equilibrium_mass`, `column_mass_profile`) were
   deleted. New `tests/test_stability.py::test_full_model_stable_at_production_grid` steps a burning
   256² hill under a strong wind with **every subsystem on** (wind + radiation + combustion + radiant
   transfer + channeling) and stays finite/positive/bounded — the configuration the radiation
   broadcast fix unblocked. Throughput measured by `tools/bench.py`: **~15 steps/s CPU (float64) at
   256² with all subsystems on** (~65 ms/step). The device-agnostic GPU path is written but
   **unverified** (no CUDA on the development machine). No `units_version` bump, no map regen.
   *Suite:* 66 passed, 1 xfailed (the gap-wind prognostic xfail, kept on purpose).

## Implementation conventions (for any continuation)

- **SI everywhere**, `Simulation.dtype = torch.float64`. **Rank-agnostic** ops (trailing
  `…,H,W`) so single-world and batched `(B,H,W)` share `step_fields`.
- **`units_version`** (`map_loader.py`, currently **3**): bump on any on-disk semantics
  change, update the comment, **regenerate all 1024 maps** (`python src/gen_maps.py
  --count 1024 --out maps --seed 0`, ~2 min); `validate_against_config` rejects mismatches.
- **No pre-SI physics code remains** — every stage of `step_fields` is now SI (Phases 0–5).
  A fire burns physically, radiantly **preheats** its neighbours, and now **spreads** as a
  self-sustaining convective front that also spreads faster uphill (Phase 5.5a–d).
  (`relax_to_equilibrium` is retained but now only drives oxygen replenishment; temperature is the
  physical balance of Phase 3.)
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
  pyrolysis temps; per-fuel `HHV`/`s`/`cp_fuel` pinned; `d_mix=30 m`. **Recast in Phase 5.5**:
  `A=1e8` as a *bulk* rate gives sub-ms residence (one-tick burnout); 5.5 makes combustion a
  surface-area-controlled flux (SAV `σ` + O₂ film transport `h`), reinterpreting `arrhenius_pre` as
  the surface pre-exponential `B`, so the burn rate is physically limited for sustained flaming.
- **Phase 5** *(done)*: `kernel_radius_m=100` (≈10 cells; inverse-square weight ~1% by there),
  `sky_escape_fraction=0.2`, reusing `ε=0.95` from `[radiation]`. Tuned as a *preheating* leg, not
  an ignition driver — spread calibration is Phase 5.5.
- **Phase 5.5c** *(done)*: convective spread tuned to a realistic self-sustaining front —
  `burn_heat_fuel_fraction=0.05` and `plume_mixing_depth=100 m` hold the flame at ~1540 K;
  `convective_fraction=0.12`, `flame_gate_temperature=700 K`, `convective_radius_m=10`,
  `convective_wind_bias=1.0` give ~2–5 m/s spread with a clean lee bias. Radiation was confirmed too
  weak to ignite at realistic flame temps, so convection is the spread driver (radiation stays preheat).
- **Phase 5.5d** *(done)*: the convective deposit kernel is a **von Mises angular** form (replacing the
  linear dipole, whose unit-magnitude clamp saturated the skew and killed the wind/slope synergy). It is
  non-negative for any bias magnitude, so `|b|` is a true forward concentration. `convective_slope_bias`
  was retuned to **2.5** for the uncapped kernel (was 4.0 under the clamped dipole): on a ~31° ramp the
  front shows a clear elliptical shape (head > flanks > backing) and a wind aligned with the upslope
  focuses a tighter, faster head. `convective_wind_bias=1.0` is unchanged (wind-only is identical to the
  old dipole at `|b|=1`, so leg c stays calibrated). `convective_concentration_max=12` caps `|b|` only as
  a numerical `exp`-overflow guard at the spurious terrain-wrap gradient. Still directional, so it
  captures the upslope/downslope asymmetry and the alignment synergy, not an absolute ROS gain.
- **Phase 6 (Option 2)** *(done)*: `convective_wind_ref=10 m/s` with `convective_wind_bias=1.0` gives a
  clean wind-driven story (head reach ~3→6→9→11 cells over 0→15 m/s, flanks pinching) without the
  thin-finger runaway that `convective_wind_bias≈2` produces at high wind. Channeling
  `radius_m=300`, `height_scale_m=100`, `gain_max=3`, `smooth_sigma=1` give a gap gain ~3 on the
  `si_channel` walls; on real maps the gain saturates at 3 in tight valleys with a mean ~1.1, and
  low ground (valleys) reads a higher mean gain than ridges while cross-wind ridges stay exactly 1.
  Tune `height_scale_m` (the rise that fully closes a channel; smaller → more bumps boost) and
  `gain_max` if real maps over- or under-channel. Going past the ~R11 head
  ceiling at extreme winds needs wind-scaled reach (spotting), deferred.

## Known limitations / deferred

- **Valley/gap (Venturi) winds NOT captured by the prognostic core**: the level-lid shallow
  layer gives fast ridges / **slow valleys** + strong foehn lee winds; lowering `layer_depth_ref`
  makes valleys *slower*, not faster. True prognostic fast-valley/gap winds need blocking-terrain
  (wetting–drying) physics. A prognostic **two-layer surface-wind core** was built for this and
  **shelved as research-grade**: in this sponge-bounded shallow water the gap flow self-limits
  below ambient (the windward dam bleeds out the open boundary; the layer stays subcritical so a
  constriction deepens rather than speeds up). It lives on branch `phase6-two-layer`, write-up in
  `docs/phase6b_investigation.md`. The strict xfail `test_narrow_channel_does_not_speed_up_wind`
  (a wind-aligned channel asserting the real-Venturi speedup) stays pinned to the **prognostic**
  field and keeps failing on purpose.
  **Resolved for the fire (Phase 6, Option 2 — done):** the fire and oxygen instead read a cheap
  **diagnostic terrain-channeling near-surface wind** — the continuity speedup imposed on the
  single-layer wind, `gain = clamp(R/max(W_plus, W_minus), 1, gain_max)` from the graded cross-stream
  relief (`openness = clamp(1 − rise/height_scale_m, 0, 1)`, shadowed and summed per side, the
  more-open side limiting the boost so only two-sided valleys channel; `compute_channel_gain`,
  `[channeling]` block, on by default) — applied speedup-only with the
  prognostic field untouched, so the xfail above stays valid. It is paired with a **wind-magnitude
  term in the convective deposit** (`b_wind = convective_wind_bias·wind/convective_wind_ref`, was
  the unit wind) so wind *speed* now drives rate-of-spread, not just direction. See the Phase 6
  entry (item 6 under Phased delivery) and `compute_channel_gain` in `src/simulation.py`. (The
  *prognostic* gap jet — Option 1 — remains deferred on `phase6-two-layer`.)
- **Lee separation/turbulence** regime not modelled (hydraulic/foehn regime instead).
- **GPU path unverified** — the code is device-agnostic (`Simulation.device`), but only the CPU
  (float64) path has been exercised; the development machine has no CUDA.
