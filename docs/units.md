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
| target max wind | **~20 m/s** (calibrate `g′`, `C_d` toward this) |
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
| `mass` | column mass, mean 1 | column mass per area `m` | kg/m² |
| `air_temperatures` | ~0.5 | `T` | K |
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

### Single-layer shallow-water atmosphere (replaces transport + wind)

Column-top geopotential height, with a thermally expanding layer (ideal gas: warmer
columns are taller):

```
thickness  η(m,T) = m · R_d · T / p_ref      [m]      (vertically integrated layer)
H(x,y)     = z(x,y) + η(m,T)                  [m]      (column-top height)
```

Prognostic momentum (pressure-gradient force via reduced gravity g′ on the free
surface `H`, plus surface drag and eddy viscosity; Coriolis omitted at this scale).
`g′` and `C_d` are calibrated in Phase 2 so a representative `∇H` drives steady winds
up to ~20 m/s (drag balance `u ≈ (g′/C_d)|∇H|`):

```
∂u/∂t + (u·∇)u = -g′ ∂H/∂x - C_d·u + ν ∇²u
∂v/∂t + (u·∇)v = -g′ ∂H/∂y - C_d·v + ν ∇²v
```

Conservative mass continuity and energy advection (same velocity carries both):

```
∂m/∂t          + ∇·(m u)              = 0
∂(m c_p T)/∂t  + ∇·(m c_p T u)        = Q_rad + Q_burn + Q_cond     [W/m²]
```

Discretisation: finite-volume, conservative face fluxes (reuse the telescoping
upwind/centered structure already in `_advect_upwind` / `_phi_diffuse_step`), no-flux
walls. **CFL:** advective `|u|dt/dx` is comfortable (~0.2 at 20 m/s), but the
gravity-wave speed `c = √(g′H)` sets the real limit — substep on `c` as the current
code substeps on displacement. Reduced gravity `g′` for a thin layer keeps `c` modest;
**Phase 2 must verify and add substepping.**

### Radiation (replaces `relax_to_equilibrium` for temperature)

Surface energy balance per column:

```
Q_rad = S_net - ε σ T⁴                      [W/m²]
dT/dt = Q_rad / (m c_p,air)
```

`S_net` is the constant net absorbed daytime flux. Size it to the reference state so
radiative equilibrium sits at `T₀`: `S_net = ε σ T₀⁴ ≈ 0.95·5.67e-8·303.15⁴ ≈
455 W/m²`. This makes the equilibrium-temperature test exactly `T_eq = (S_net/εσ)^¼ =
T₀`.

Altitude-dependent cooling now emerges naturally: a thinner (high-altitude) column has
smaller `m c_p` heat capacity, so it heats/cools faster. `temperature_rate_low/high`
are **retired**. Oxygen replenishment (`oxygen_rate`) is replaced by mixing toward the
ambient O₂ profile (or by the advective exchange itself).

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
| mass | hydrostatic `m(z) = p(z)/g`, with the exact barometric `p(z) = p₀·(1−Γz/T₀)^(g/(R_d·Γ))` (isothermal `exp(−z/H_p)` only as the Γ→0 fallback) |
| oxygen | ambient O₂ density scaled with pressure/altitude |
| fuel | biomass kg/m² (grass ~0.5–2, forest ~5–20), elevation-thinned |
| `mass` rest state | re-derive `equilibrium_mass` as the true hydrostatic balance |

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

0. **Scaffolding** ✅ *done* — `physics_constants.py`, `Simulation.dtype→float64`,
   `cell_size_m`/`dt` on `Simulation`, `[units]` config, `units_version` (=1) stamped in
   maps + rejected by `validate_against_config`.
1. **SI state + I/O (dynamics off)** ✅ *done* — `gen_maps.py` emits SI fields
   (terrain m, T K, mass kg/m², O₂ kg/m³, fuel kg/m²); `maps/` regenerated; all rendering
   rescaled to SI. Existing test suite still green. *Deferred:* `fixtures/` regeneration
   and `OBS_CHANNELS` normalisation (Phase 6), and the dynamics constants under
   `[convection]`/`[relaxation]`/`[fire]`/`[radiant_heat]`/`[wind]`/`[oxygen]` (Phases 2–5)
   — so stepping the SI maps is not yet physically meaningful.
2. **Shallow-water core** — momentum + continuity + energy solver replacing
   `transport_mass_energy` and the wind diagnostic; CFL substepping. *Riskiest phase
   (new prognostic velocity); most test coverage.*
3. **Radiation** — surface energy balance replacing temperature relaxation.
4. **Combustion** — Arrhenius + stoichiometry + `HHV`; oxygen as a real field; real
   `c_p` in fuel/air conduction.
5. **Fire radiant transfer** — `ε σ T⁴` + inverse-square redistribution.
6. **Observations / rendering / config docs** — normalisation, color scales, config
   rewrite, `main.py`.
7. **Test rebaseline + numerical hardening** — SI scenarios, closed/open conservation,
   CFL.

## Calibration left for implementation phases

The design parameters are pinned (see *Resolved parameters*). What remains is numerical
calibration done against tests inside the relevant phase, not open design questions:

- **Phase 2:** tune reduced gravity `g′` and drag `C_d` so a representative `∇H` yields
  steady winds up to ~20 m/s, and confirm the gravity-wave CFL `√(g′H)·dt/dx < 1`
  (add substepping if not).
- **Phase 3:** confirm `S_net ≈ 370 W/m²` lands radiative equilibrium at `T₀`; pick
  emissivity `ε` (≈0.95) and albedo consistent with that net flux.
- **Phase 4:** tune Arrhenius `A_pre`, `E_a` so cold fuel is inert and ignition onset
  matches the table's pyrolysis temperatures; pin per-fuel `HHV`, `s`, `c_p,fuel`.
- **Phase 1:** choose biomass densities (grass ~0.5–2 kg/m², forest ~5–20 kg/m²). The
  hydrostatic initial mass uses the exact lapse-rate barometric profile (`H_p` is only
  the Γ→0 isothermal fallback).
</content>
</invoke>
