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

Smooth Arrhenius reaction rate, limited by both fuel and oxygen:

```
ṙ = A_pre · exp(-E_a / (R T_fuel)) · fuel · [O₂]      [kg/(m²·s)]
```

No hard ignition threshold — the Arrhenius exponential makes the rate vanishingly small
at ambient temperature on its own (a tiny numerical floor avoids denormals only).
`A_pre` and `E_a` are tuned in Phase 4 so cold fuel is inert and ignition onset lands
near the pyrolysis temperatures in the constants table. Stoichiometric and energetic
coupling per tick `Δfuel = ṙ·dt`:

```
ΔO₂   = s · Δfuel                    (capped at available O₂)
Q_burn = HHV · Δfuel / dt            [W/m²], split between air and fuel by burn_heat_fuel_fraction
```

Fuel↔air conduction (`exchange_fuel_air_heat`) keeps its exact two-body exponential
form but with real heat capacities: `C_air = m·c_p,air` (currently missing the
`·c_p,air`), `C_fuel = fuel·c_p,fuel`, and a conductance `k` in W/(m²·K).

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
4. **Combustion + SI oxygen** — rewrite `update_fire`; fix `exchange_fuel_air_heat`.
   Arrhenius O₂-limited rate `ṙ = A·exp(−E_a/(R·T_fuel))·fuel·[O₂]` [kg/m²/s] (no hard
   threshold; tiny numerical floor). Per tick `Δfuel = ṙ·dt`: `ΔO₂ = s·Δfuel` (s≈1.4,
   capped at available O₂); `Q_burn = HHV·Δfuel` (HHV≈1.6–1.8e7 J/kg) split air/fuel by
   `burn_heat_fuel_fraction`. **Fix:** `exchange_fuel_air_heat` `C_air` must be
   `air_mass·cp_air` (currently missing `·cp`); `C_fuel = fuel·cp_fuel` — **consolidate onto the
   SI `specific_heat` key added in Phase 3** (grass 1800, tree 2300) and drop the pre-SI
   `heat_capacity`. Oxygen is already kg/m³ advected
   by the wind (Phase 2); couple consumption + ambient mixing. Tune `A,E_a` so cold fuel is
   inert, ignition ≈ pyrolysis temps (grass 573 K, tree 600 K, already in cfg). Rewrite
   `[fire]`/`[fuel_types]` combustion constants to SI. Fire heats `T` → `η` → indraft wind
   (keep `T` in `η`). *Tests:* O₂/fuel stoichiometric budget; HHV split; no ambient burn;
   fire-driven indraft.
5. **Fire radiant transfer** — rewrite `apply_radiant_heat`: emit `E = ε·σ·T_fuel⁴·dt`
   [J/m²] capped at the cell's thermal energy; `sky_escape_fraction` to space; rest spread
   by `_radiant_convolve` (inverse-square) with **kernel radius in metres** (`radius_cells =
   radius_m/dx`). `[radiant_heat]` → real `ε,σ` (drop pre-SI `emission_scale=1e-6`),
   exponent 4 fixed. Re-baseline `tests/test_radiant_heat.py` magnitudes for SI `T`.
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
- **Pre-SI code still live** (marked): `update_fire`, `apply_radiant_heat`,
  `exchange_fuel_air_heat`, and the cfg combustion constants — replaced in Phases 4–5. So
  stepping with fire on is not yet fully physical. (`relax_to_equilibrium` is retained but
  now only drives oxygen replenishment; temperature is the physical balance of Phase 3.)
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
- **Phase 4**: tune Arrhenius `A`, `E_a` so cold fuel is inert and ignition ≈ the cfg
  pyrolysis temps; pin per-fuel `HHV`, `s`, `cp_fuel`.

## Known limitations / deferred

- **Valley/gap (Venturi) winds NOT captured**: the level-lid shallow layer gives fast
  ridges / **slow valleys** + strong foehn lee winds; lowering `layer_depth_ref` makes
  valleys *slower*, not faster. True fast-valley/gap winds need blocking-terrain (wetting–
  drying) physics — optional future enhancement, not in the phase plan.
- **Lee separation/turbulence** regime not modelled (hydraulic/foehn regime instead).
- `column_mass_profile` (full barometric column) kept in `gen_maps` for reference but unused
  by dynamics; `equilibrium_mass` kept only for the legacy `fixtures/`/visualizer scenarios.
