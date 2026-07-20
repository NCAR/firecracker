# Vertical Air Levels — Design Doc

Status: **Phase 1 done + committed; Phase 2 Step A done + committed; combined Phase 2+3 (per-level
thermodynamics) is the current in-progress work.** Physics decisions resolved through the design
review below. Phase 0 validated the multi-level dynamics in isolation (`tools/vertical_levels/
phase0_dynamics_spike.py`): the advection scheme is stable, the old fire-wind amplifier is ~15×
(not 89×) and is an energy-over-count, and the ~15× weaker energy-correct fire-wind is **accepted**
(no `reduced_gravity` retune).

## ⏸ RESUMPTION STATE (context handoff)

**Git:** clean at commit `17c4aa6` (base `f3c69db`). History: `9244cf6` = Phase 1 (air level axis
L=1, lean buoyancy, `tree_bole` split N=5, canopy SAV 5000→10000); `17c4aa6` = Phase 2 Step A
(materialized L=3 env-persisted air state, behavior-preserving).

**Next task — the user chose (a): combine Phase 2 + Phase 3** into one per-level-thermodynamics change
(a gentler temperature-only Step B was rejected because the thin canopy slab has no heat sink until
the Phase 3 transfer/vent, so Step B alone can't be verified). Implement in two verifiable stages:

- **Stage 1 — heat budget** (do first): (i) rewrite `exchange_fuel_air_heat` to couple each fuel to
  *its level's* air using thin-slab thermal mass `ρ·depth·cp` (surface 5.83, canopy 11.65 kg/m² — vs
  today's full-column ~1747, a ~300× change); (ii) new inter-level **upward** convection stage
  (surface→canopy→above, rate ∝ `(T_lower−T_upper)₊`, exact-exponential like `exchange_fuel_air_heat`
  at `:~825`); (iii) **energy-anomaly vent on the above-canopy (top) level only** (repurpose
  `vent_plume_heat`); (iv) **align `rollout.py` and `gen_maps.py` constructors to L=3** via
  `lift_air_levels` — REQUIRED, because `exchange` will `index_select` level 1 and an L=1 air tensor
  will crash. Then run the suite and rebaseline (expect shifts in `test_heat_exchange`, `test_spread`,
  `test_combustion`, `test_equilibrium`) — surface each rebaseline to the user, don't silently adjust.
- **Stage 2 — ladder**: per-level combustion **air-share** routing in `update_fire` (each fuel's
  air-share → its level); **per-level convective ignition deposit** (a burning cell's level-L plume
  ignites only neighbours' level-L fuels); **bole-segment vertical conduction** (fast-up/slow-down
  two-body relaxation between `tree_bole_surface` and `tree_bole_canopy`). Verify with
  `tools/vertical_levels/forest_ladder_test.py` (grass must not ignite the canopy without the bole).

**Key facts / decisions to preserve:**
- **Buoyancy stays LEAN** (sub-area 2 form: surface-anomaly × `m_plume = ρ·plume_mixing_depth ≈ 116.5`).
  Per-level *masses* remain deferred; `SimState.mass` stays a single total `(H,W)`. Per-level air
  *temperatures* drive fuel/ladder behavior, NOT the wind. Don't reopen the buoyancy in 2+3.
- `Simulation` already has: `air_level_depths=(5,10)`, `num_air_levels=3`, `fuel_levels`
  (`= [1 if "canopy" in name else 0 for name in fuel_type_names]` → grass/shrub/tree_bole_surface→0,
  tree_canopy/tree_bole_canopy→1), and module fn `lift_air_levels(field, n)`. Helpers
  `_surface_level`/`_with_surface` do the axis −3 boundary conversion.
- **Planned precompute** (add in `__init__` right after `fuel_levels`): `_fuel_level_index` `(N,)` long
  tensor; `_air_level_capacity` `(L,1,1) = ρ_ref·depth·cp` (above-canopy depth ≈ a config value ~100 m,
  its exact value only affects the vent's diagnostic T since no fuel couples there). Add an
  `[air_levels]` config block (depths, `transfer_rate`, `vent_rate`) or Simulation defaults.
- `exchange` rewrite sketch: `air_f = air.index_select(-3, lvl)`; `C_air_f =
  _air_level_capacity.index_select(0, lvl)`; two-body exact-exp relaxation; scatter the air
  back-reaction with `air_gain.index_add_(-3, lvl, -(C_fuel*dT_fuel))` then `air += air_gain /
  _air_level_capacity`. Remove the `air_mass`/`s.mass` argument. Update the `step_fields` call to pass
  the full `(L,H,W)` air (drop the `_surface_level`/`_with_surface` wrap for exchange).
- **Heat-budget sink logic:** surface level (0) already has a sink (the `step_dynamics` edge sponge);
  canopy/above levels get their sink ONLY from the new inter-level transfer + top vent — that's why
  Stage 1 must include them or the canopy air runs away during fire.
- `apply_radiation` can stay surface-level for now (ground→surface air); its per-level split is minor.

**Open follow-ups (unchanged):** `maps/*.npz` still 4-type (app viewer broken until regenerated —
deferred to cluster); canopy SAV 10000 is committed but contested (the ladder will properly gate
crowning, which is the justification); `tools/vertical_levels/` holds the preserved Phase-0/ROS/regime/
ladder harnesses (were scratchpad-only; copied into the repo so they survive — Phase 4 re-measures
with `ros_test.py` / `forest_regime_test.py` / `forest_ladder_test.py`).

**Phase 1 progress (production code, all physics tests green):**
- *Sub-area 1 — data model (L = 1):* `air_temperatures`/`temp_eq` carry a level axis (−3); every stage
  reads/writes the surface level via `_surface_level`/`_with_surface`; `build_observation` unbinds the
  axis. Behaviour-preserving refactor across `SimState`, `step_fields`/`step_dynamics`, all 6
  constructors, env, and tests.
- *Sub-area 2 — multi-level buoyancy (lean):* `step_dynamics` only. A super-ambient surface anomaly is
  buoyant over just a `ρ·plume_mixing_depth` slab (matching how `update_fire` heats the air), so
  fire-wind is ~15× weaker (measured **~13×** in the real sim) while the orographic wind and
  mass/energy conservation are untouched. **Deferred to Phase 3** (where the canopy level first
  differs from the surface): the full per-level *state* — `SimState.mass`/air as real `(L,H,W)`,
  fully-advected slabs, env `(L,H,W)` storage, and map regeneration.
- *Sub-area 3 — `tree_bole` split (N → 5):* `cfg/default.toml` splits `tree_bole` into
  `tree_bole_surface` (0–5 m, 1.43 kg/tree) and `tree_bole_canopy` (5–15 m, 2.87 kg/tree) — same
  coarse-wood properties for now; per-level air coupling and the fast-up/slow-down segment conduction
  come in the vertical-levels phases. The `tree`-prefix logic in `gen_maps` generalizes (verified:
  bakes `(5,H,W)` maps); `rendering` updated for both segments; obs channels derive to `C = 16`. All
  physics tests green. **Follow-up:** regenerate the committed `maps/*.npz` (still 4-type) — deferred,
  likely alongside the cluster retraining.

## Goal

Give the air temperature a **vertical dimension** so that fire spread respects fuel height. Today
there is one sheet of near-surface air, and every fuel type — grass, shrub, bole, canopy — exchanges
heat with that same sheet. That is why a **ground grass fire ignites the elevated canopy directly**,
with no trunk to carry the flame up. Measured, this is a hard violation: in a bed of grass + canopy
with **no bole at all**, a surface ignition still burns 99.7% of the canopy; and in a full stand the
canopy crosses ignition at a downstream cell where the bole never does.

Splitting the air into stacked levels makes the **ladder-fuel** rule emergent geometry rather than a
special-case gate: grass heats only the surface air, the canopy only ignites once the *canopy-level*
air is hot, and the only things that heat the canopy level are (a) the **trunk**, whose fire climbs
fast from the surface bole segment into the canopy bole segment, and (b) buoyant heat convected up
from the surface. It also lets a
fire **drive its own wind** through the existing shallow-water buoyancy, and gives an intensity-gated
crown transition for free (the bole is hard to catch, so the crown only engages once the surface fire
is intense enough to light the trunk).

## Resolved decisions

Pinned during the design review; assumed throughout unless revisited.

| Choice | Decision |
|---|---|
| Number of levels | **3**: surface, canopy, above-canopy |
| Level geometry | surface **0–5 m**, canopy **5–15 m**, above-canopy **15 m → top** |
| Fuel↔level membership | grass, shrub, `tree_bole_surface` → surface; `tree_canopy`, `tree_bole_canopy` → canopy (**every fuel maps to exactly one level**) |
| Tree bole | **split into two stacked fuel types** — `tree_bole_surface` (0–5 m) + `tree_bole_canopy` (5–15 m); mass split ⅓/⅔ by height (tunable `fuel_per_tree`), same coarse-wood SAV/ignition |
| Bole vertical conduction | exact-exponential two-body relaxation between the two segments, rate by sign of ΔT: **`k_up ≫ k_down`** (fire climbs the trunk, doesn't descend) |
| Wind | **single shared** `(u, v)` for all levels (one momentum field) |
| Buoyancy driver | total column height `η_total = Σ η_i`, `η_i = m_i·R_d·T_i / P_REF` |
| Fire drives wind? | **Yes** — the above-canopy level is a real advected, warming level, not a pure sink |
| Inter-level heat transfer | **heat-only**, upward, rate ∝ (T_lower − T_upper) via **exact-exponential relaxation** (self-limiting, unconditionally stable); no vertical mass flux (v1) |
| Plume / thin-level sink | air-share injects into the **surface** level; **fast proportional upward transfer** carries it up before it accumulates; the above-canopy vent is the only sink |
| Mass / energy advection | **fully advect each level's mass + energy** (conservative — total `Σ mᵢ·cp·Tᵢ` preserved); **mass floor + thin-slab advection substep** guard `T=E/(m·cp)` against blow-up near divergence |
| Venting | **energy-anomaly** vent on the above-canopy level only |
| Ignition/spread deposit | **per level** (a burning cell's level-L plume ignites neighbours' level-L fuels) |
| ML obs channels | **expose all 3 levels** as air channels; retrain. With `N = 5` fuel types, `C = 8 + 2N = 18` |
| Map migration | **regenerate** the map set with per-level `air`/`temp_eq` `(3, H, W)`; units-version bump |
| Vertical mass transfer | **deferred** (see Q2); revisit only if the fire-wind sign is wrong |

## The model in plain English

Three stacked sheets of air per cell:

- **Level 0 — surface (0–5 m):** the air among the grass, shrubs, and tree trunks.
- **Level 1 — canopy (5–15 m):** the air up in the crowns.
- **Level 2 — above-canopy (15 m → top):** the deep remainder of the column. Holds essentially all
  the column mass, anchors the baseline surface pressure and wind, receives rising fire heat, drives
  buoyancy through its thickness, and is the only level that vents to space. It does **not** warm any
  fuel.

Each fuel couples only to the level at its height (one-to-one). Grass, shrub, and the **surface bole**
(0–5 m) trade heat with level 0; the **canopy foliage** and the **canopy bole** (5–15 m) trade heat
with level 1. The **trunk is split into two stacked fuel segments** — surface bole and canopy bole —
with anisotropic vertical conduction between them: fire climbs the trunk fast (surface → canopy) and
descends slowly, so once the trunk *base* catches, fire runs *up* it into the crown, but a crown fire
does not readily run back *down*. Heat also moves **upward** between air levels by convection whenever
a lower level is hotter than the one above it (warm air rises; the reverse does not happen through this
term). So a grass fire warms level 0, and that heat only reaches the canopy's air by rising or by the
bole carrying fire up. The canopy cannot catch until its own level is hot — the ladder.

The wind stays a single shared field. Heating any level increases that level's hydrostatic thickness
`η_i`, which increases the total column height `Σ η_i`; the existing shallow-water solver then spills
mass into neighbouring columns exactly as it does today. That is the entire fire-drives-wind
mechanism — no new momentum machinery.

## Key quantitative facts

Constants: `P_REF = 101325 Pa`, `R_d = 287.05 J/(kg·K)`, `cp_air = 1005 J/(kg·K)`,
`ρ_ref = 1.165 kg/m³`, `g = 9.81 m/s²`.

**Mass split.** The model's air `mass` is **not** the full atmospheric column — it is a shallow-water
**boundary-layer mass** (`layer_depth_ref = 1000 m`, `gen_maps.boundary_layer_mass`), ≈ **1747 kg/m²**
on flat terrain (a ~1500 m layer), *not* `P_REF/g ≈ 10329`. The thin slabs are `ρ_ref·depth`:

| level | depth | mass | share of the ~1747 kg/m² layer |
|---|---|---|---|
| surface | 5 m | 5.83 kg/m² | 0.33% |
| canopy | 10 m | 11.65 kg/m² | 0.67% |
| above-canopy | rest (~1490 m) | ~1730 kg/m² | 99.0% |

**Buoyancy per unit heat is level-independent.** Since `η_i = m_i·R_d·T_i/P_REF` and injecting heat
`Q` into a level gives `ΔT_i = Q/(m_i·cp)`, the thickness bump is

```
Δη_i = m_i·R_d·ΔT_i / P_REF = R_d·Q / (cp·P_REF)          (the m_i cancels)
```

= `2.82e-6 m` per `J/m²`, whatever level absorbs it. Consequences:

- **Total fire-driven column rise = `R_d/(cp·P_REF) × (total super-ambient heat retained in the
  column)`.** Redistributing heat *between* levels is buoyancy-neutral for the sum. So wind strength
  depends on the **total retained heat and the vent rate**, not on the number of levels. Three levels
  is plenty for the wind; levels matter only for *where fuel is warmed* (the ladder) and for *routing
  heat to the top vent*.
- **The current code over-counts fire-wind ~15× (an energy-conservation violation).** Today
  combustion heat sets `ΔT` for a shallow 100 m plume slab, but `step_dynamics` then applies that
  `ΔT` to the **full ~1747 kg/m² boundary-layer mass** when forming `η` — putting ~15× more thermal
  energy into the buoyancy than the fire released. Measured amplifier = `mass/(ρ·d_plume)
  = 1747/116.5 = 15.0×` (Phase 0 spike measured **14.9×**; the earlier "89×" used the wrong
  full-column mass). The faithful multi-level model conserves energy and is therefore *correctly*
  ~15× weaker.
- **Decision (Phase 0): accept the weaker, energy-correct fire-wind.** `reduced_gravity` cannot
  restore it — the fire-wind is strongly *sublinear* in `g'` (a 100× increase buys only ~2.5×,
  because larger `g'` radiates the warm anomaly away as faster gravity waves), and `g'` is a *global*
  knob shared with the orographic/synoptic wind calibrated to ~30 m/s. Since total buoyancy from heat
  `Q` is `R_d·Q/(cp·P_REF)` regardless of level distribution, no level-routing recovers it either.
  Weaker fire-wind is also consistent with fixing the measured ~3–4× too-fast ROS. Revisit only if
  Phase 4 behavior shows fires need to drive more wind (then via an explicit, physical fire-buoyancy
  term, never by inflating `g'`).

## Detailed design (per `step_fields` stage)

Reference: `src/simulation.py:1033` (`step_fields`) drives the per-tick sequence. Air becomes a
leading-axis stack `air_temperatures : (L=3, H, W)`; `temp_eq` likewise gains a level axis (rest
profile at each level's mid-height via the existing lapse rate).

1. **`step_dynamics` (`:597`) — wind + transport.**
   - Buoyancy forcing reads `η_total = Σ_i m_i·R_d·T_i / P_REF` in place of the single `m·R_d·T`.
   - **Fully advect each level's mass and energy** (`Eᵢ = mᵢ·cp·Tᵢ`) conservatively by the shared
     wind, exactly as the current single level is transported; momentum advected once against the
     total mass. This conserves `Σ mᵢ·cp·Tᵢ` to round-off (so `test_closed_core_conserves` stays at
     1e-6) and carries each level's temperature with the flow (downwind plume tilt).
   - **Thin-slab guard:** the surface (5.83) and canopy (11.65) slabs are tiny, so where the wind
     diverges around a hot cell their mass can approach zero and `T = E/(m·cp)` blow up. Guard with a
     **mass floor** (a divide-by-zero safety valve that only triggers near strong divergence — never
     in the closed conservation test, so exact conservation there is unaffected) plus an **advection
     substep** for the thin slabs so the floor rarely fires.
   - CFL/substepping unchanged in practice: the wave speed `√(g'·η_total)` is dominated by the deep
     level, so the substep count is ~as today.

2. **`apply_radiation` (`:699`) — surface energy balance.** Ground sensible flux warms **level 0**
   only. Each fuel exchanges longwave/solar with its single level (one-to-one gather).

3. **`exchange_fuel_air_heat` (`:783`) — conductive fuel↔air.** Per level: grass/shrub/surface-bole
   against level 0; canopy foliage/canopy-bole against level 1. Air thermal mass per level is
   `ρ·depth·cp`.

4. **`update_fire` (`:824`) — combustion + spread.**
   - Each fuel's combustion **air-share heat** is injected into its level's air (grass/shrub/
     surface-bole → 0; canopy/canopy-bole → 1). The plume thermal-mass slab (`C_plume`) per level is
     the level depth, not the current fixed 100 m.
   - The **convective ignition deposit runs per level.** A burning cell's level-L plume heat ignites
     only neighbours' level-L fuels. This is the mechanism that enforces the ladder *during spread*,
     not just at a single cell: a grass front deposits into neighbours' surface fuels (grass, shrub,
     surface-bole) but never directly into their
     canopy.

5. **Inter-level air convection (new stage).** Upward-only, heat-only transfer between adjacent air
   levels, gated on instability: flux `∝ (T_lower − T_upper)₊`. Energy-conserving (what leaves a level
   enters the one above). No vertical mass flux in v1 (Q2). Order: surface → canopy → above-canopy.

6. **Bole-segment vertical conduction (new stage).** Between the two bole *fuel* types
   `tree_bole_surface` and `tree_bole_canopy` (co-located, same trunk). Exact-exponential two-body
   relaxation (as `exchange_fuel_air_heat:814`), with a **direction-dependent rate**: `k_up` when the
   surface segment is hotter (fire climbing), `k_down ≪ k_up` when the canopy segment is hotter (fire
   reluctant to descend). Energy-conserving between the two segments' thermal masses `fuel·cp`. This
   is the dedicated fast upward ladder — distinct from, and tuned jointly with, the air convection in
   stage 5 (both carry heat up; guard against over-eager crowning).

7. **`vent_plume_heat` (`:740`) — the only sink.** Becomes an **energy-anomaly** vent on the
   **above-canopy level only**: shed a fraction of `m·cp·(T − temp_eq)` per tick at rate
   `λ ≈ updraft/depth`. Replaces (does not stack with) the old single-level temperature-excess vent.

## Design-review analysis (Q2 / Q3 / Q4)

### Q2 — Why heat-only, and why mass transfer is deferred

Heat-only inter-level transfer is a single conservative scalar exchange. Adding vertical **mass**
transfer drags mass, its carried energy, and (physically) its carried momentum along with it, and:

- **Stiffness** — the thin slabs hold only ~6 and ~12 kg/m², so any meaningful vertical flux is a
  large *relative* change per tick, demanding a small dt or implicit solve.
- **It couples into the pressure solver** — moving mass vertically changes each level's `η` *inside*
  the CFL-substepped wind loop, so it cannot be a clean operator-split post-step the way venting is.
- **It fights the shared-wind assumption** — real updrafts loft low-level momentum upward, but we
  have one velocity for all levels, so there is nowhere consistent to put lofted momentum.
- **More silent failure modes** for mass/energy/momentum conservation across a stiff thin layer.

The one real payoff is a **sign** issue. In shallow water a heated column is a geopotential *high*
(`η` rises) and `−g'·∇η` pushes air **outward** — so heat-only fire-wind is divergent **outflow**, a
fire cell blowing air away from itself, not the surface **indraft** real fires generate. The only way
to flip that with this machinery is to let the plume **eject mass out the top**, lowering the column's
surface pressure toward a surface **low** → convergence. That is the genuine benefit of mass
transfer. But with a single shared column-averaged wind, true indraft (inflow low + outflow aloft,
opposite signs by height) is unreachable regardless — mass ejection would only flip the *net* column
response from divergent to convergent, crudely.

**Decision:** ship heat-only. Hold a **targeted top-of-column mass-ejection** term (the vent removes a
little *mass*, not just energy, from the above-canopy level) in reserve as the lever if the fire-wind
sign proves wrong or important once we can observe it. Full bidirectional inter-level mass exchange is
not worth the cost at 10 m resolution.

### Q3 — Implications of advecting all levels

Advecting per-level `m_i` and `E_i` by the shared wind costs ~2× the transport work in
`step_dynamics` (3 masses + 3 energies vs. 1 + 1; momentum unchanged) and adds no CFL substeps. But
because one wind moves all levels *identically*, the only thing per-level advection must preserve is
the vertical **temperature** structure — advect a lumped energy instead and every gust would smear the
profile. Per-level **mass** advection buys little and introduces the **thin-slab depletion** risk:
around a hot, spilling fire cell the wind diverges, a 6 kg/m² slab can lose most of its mass in a
tick, and `T = E/(m·cp)` then blows up.

**Revised in review round 2 → fully advect.** The pinned-slab idea was reconsidered: since
conservative advection preserves `Σ mᵢ·cp·Tᵢ` no matter how mass redistributes, the closed
conservation test holds either way, and fully advecting every level's mass+energy is simpler and more
uniform than maintaining two transport schemes. The thin-slab depletion risk is handled with a **mass
floor** (divide-by-zero guard, active only near the strong divergence around fires) plus a **thin-slab
advection substep**, not by pinning. See the resolved decisions below.

### Q4 — Why energy-anomaly venting (not temperature-excess)

The vent is the **only sink** for fire heat (the air is radiatively transparent), so its form decides
both **stability** (unbounded accumulation warns in the current code) and **fire-wind magnitude**
(buoyancy ∝ total retained heat, and the vent sets retention; residence `1/λ` is the plume's
dynamical memory).

A rate on `(T − temp_eq)` is mass-dependent in a physically arbitrary way: dump heat `Q` into the deep
above-canopy level (~10⁴ kg/m²) and `ΔT = Q/(10⁴·cp)` is tiny, so the vent removes almost nothing →
heat piles up → slow buoyancy runaway; dump the same `Q` into a thin slab and it vents hard. Same
heat, opposite behavior, purely from the level's mass.

An **energy-anomaly** vent sheds a fraction of the heat *content* `m·cp·(T − temp_eq)` per tick, so
the sink is mass-independent: combustion adds `Q_in`, the vent removes `λ·Q_retained`, steady state
`Q_retained = Q_in/λ` → bounded, tunable buoyancy wherever the heat sits. It is also how "heat that
rises is lost to fuel but drives advection first" becomes concrete: the above-canopy level *holds* the
energy (driving `η`/wind) for `1/λ`, then releases it to space. `λ` is the physically meaningful knob.

## Conflicts / downstream impact with current code

- **Observation channels & trained models.** `in_channels = len(obs_channel_names(...))`
  (`strided_autoencoder.py:209`) and `air_temperature` is a single `OBS_CHANNELS_PRE` entry
  (`simulation.py:36`). Exposing 3 levels changes `C = 6 + 2N → 8 + 2N`. Acceptable — the models are
  being retrained on the cluster. Needs `OBS_NORM` windows for the two new levels and updated
  `build_observation` (`:1108`).
- **Map format & generator.** Maps store one `air_temperatures` and one `temp_eq` `(H, W)`
  (`map_loader.py:85`, v4). **Decision: regenerate** with per-level `air`/`temp_eq` `(3, H, W)` rather
  than upgrade at load — `gen_maps.air_temperature_profile` (`:306`) supplies the per-level rest
  profiles (lapse rate at each mid-height), `spin_up_maps` develops all three (thin slabs near-trivial
  at rest), and `validate_against_config` (`map_loader.py:264`) + `units_version` move to the new
  shape. The cached `maps/*.npz` set is re-baked.
- **Venting/conservation tests.** `test_venting`, `test_conservation`, `test_equilibrium` must be
  re-derived: the vent moves to the above-canopy level as an energy-anomaly sink, and the new
  inter-level transfer must be shown energy-conserving.
- **Surface reads / rendering.** `_spawn_fire_patch` (`firecracker_env.py:869`) must ignite the
  **surface** level (a ground ignition). Rendering/stats (`:742`, `:977`) pick a level to display
  (surface) or add a canopy view. `column_height` (`:775`) is display-only.
- **Fire-wind.** Removing the ~15× energy-over-count leaves fire-wind ~15× weaker; **accepted as-is**
  (Phase 0 decision). `reduced_gravity` stays at its baseline-wind calibration and is *not* used to
  restore fire-wind (it can't — sublinear and global).

## Open questions / deferred

- **Fire-wind sign** — confirm, once observable, whether heat-only outflow is acceptable or whether we
  need the reserved top-of-column mass-ejection term for indraft.
- **`λ` and `reduced_gravity` calibration** — set against realistic fire-wind and the ROS targets
  (forest surface ≈ 10% of wind, crown ≈ shrub, grass fastest).
- **Bole depth split** — ⅓/⅔ is a placeholder; may tune once the crown transition is observable.

## Resolved bugs & fixes (review round 2)

Concrete issues found reading the code, each with the chosen fix.

| # | Problem | Chosen fix |
|---|---|---|
| 1 | Thin combustion levels have no sink → runaway | Fast **proportional** upward transfer (rate ∝ ΔT); self-limiting cascade to the top vent |
| 2 | Air-share into a 5 m slab = ~900 K/tick direct add | Inject into the surface level but relax with the **exact-exponential** inter-level transfer (`exchange_fuel_air_heat:814` pattern) — unbounded rate is safe; κ sets steady surface-air temp (tuning) |
| 3 | Closed-core energy conservation (1e-6) | **Fully advect** mass+energy per level → `Σ mᵢ·cp·Tᵢ` conserved exactly; floor never triggers in the closed test |
| 4 | Two advection schemes | Avoided — one conservative scheme for all levels |
| 5 | Bole couples to two levels (not a broadcast) | **Eliminated** — the bole is split into two fuel types (`tree_bole_surface`, `tree_bole_canopy`), each on one level, so fuel→level is one-to-one; the new piece is the anisotropic bole-segment conduction term (below) |
| 6 | Rank collision — `air.unsqueeze(-3)` now aliases L↔N | Audit `:797/:802/:820/:911`; replace with explicit level-indexed ops |
| 7 | `_spawn_fire_patch` indexes level axis | `[..., patch]` targeting the **surface** level |
| 8 | Map validation rejects `(L,H,W)` | **Regenerate** maps with per-level fields; units bump; update `validate_against_config` |
| 9 | Env display/stats assume `(H,W)` | Show surface level; `column_height` uses `η_total`; confirm `_crop` passes a leading axis |
| 10 | Obs channel count | Expose all 3 levels (`C = 8 + 2N`, `N = 5` → 18); add `OBS_NORM` windows; unbind level axis in `build_observation` |
| 11 | Thin-slab depletion blow-up | Mass floor (divide guard) + thin-slab advection substep |
| 12 | spin-up / CFL | Develop all 3 levels; wave speed uses `η_total` (deep-dominated) |

## Phased implementation plan (de-risked)

0. **Dynamics spike (no fuel coupling). — DONE** (`scratchpad/phase0_dynamics_spike.py`). Built a
   3-level `step_dynamics` (Σ `η_i` buoyancy, fully-advected levels, mass floor + substep) from the
   real primitives and compared fire-wind to the single-layer core. **Results:** (a) multi-level
   advection is **stable** — finite, thin-slab mass held at 5.82 kg/m², floor never stressed, for
   `g'` 0.2→20; (b) the amplifier is **14.9×** (not 89× — the wrong-mass error), an energy-over-count;
   (c) `reduced_gravity` **cannot** restore fire-wind (sublinear + global). **Decision: accept the
   ~15× weaker, energy-correct fire-wind** and leave `g'` at its baseline calibration. (Bugs 3, 4, 6,
   11, 12 exercised; advection scheme validated.)
1. **Data model & maps.** Thread the level axis through `SimState`, `map_loader` (regenerate + units
   bump), `gen_maps` per-level rest profiles + spin-up, env plumbing (bugs 7, 9),
   `build_observation`/`obs_channel_names`/`OBS_NORM` (bug 10). **Split `tree_bole` → `tree_bole_surface`
   + `tree_bole_canopy`** in `cfg/default.toml` (⅓/⅔ mass, shared tree count); `N` becomes 5.
2. **Fuel↔air per level.** Split `apply_radiation` and `exchange_fuel_air_heat` by level; with the bole
   now two single-level fuels, this is a clean one-to-one gather (bug 5 gone).
3. **Fire per level + transfers + vent.** Per-level convective ignition deposit; surface-level air-share
   injection with **exact-exponential proportional upward air transfer** (bugs 1, 2); **anisotropic
   bole-segment conduction** (`k_up ≫ k_down`); above-canopy energy-anomaly vent; re-derive
   conservation/venting tests. Verify the ladder on `forest_ladder_test` (grass cannot ignite the
   canopy without the surface bole catching and the fire climbing the trunk).
4. **Re-tune & re-measure.** Recalibrate `reduced_gravity`, air-share, transfer rate κ, vent rate `λ`;
   re-run ROS / regime / ladder harnesses; confirm intensity-gated crowning and realistic fuel
   ordering.
