# The Simulation Engine

This is an in-depth tour of Firecracker's physics engine — what state it holds, the fixed
sequence of steps it runs every tick, and how each subsystem works. It describes the code in
[`src/simulation.py`](../src/simulation.py); the deepest reference for the physical derivations
(and the reasoning behind the constants) is [`units.md`](units.md).

The whole model runs in **true SI units**: temperature in kelvin, mass in kg/m², wind in m/s,
oxygen in kg/m³, terrain in meters, energy in joules. Every field is a PyTorch tensor on one
device (a GPU when available, otherwise the CPU — the exact same code runs on both), stored in
double precision because the state spans a huge range of magnitudes and the conservation checks
are tight.

---

## The grid and the clock

The world is a square grid of cells. Two numbers set the physical scale:

- **`cell_size_m`** — how many meters wide each cell is (10 m by default, so a 256×256 grid is a
  2.56 km patch).
- **`seconds_per_tick`** — how much simulated time one tick advances (1 second by default).

`seconds_per_tick` is deliberately separate from how fast the window redraws. The display is
throttled to `steps_per_second` ticks per wall-clock second (5 by default), but each of those
ticks can advance any amount of physics time. At the defaults, the simulation runs at 5× real
time. The internal solvers sub-step themselves for stability (see the CFL note below), so a larger
time step stays stable — it just integrates the combustion a bit more coarsely.

### The padded grid and open boundaries

The physics actually runs on a grid that is *larger* than what you see. A ring of `sponge_width`
cells (8 by default) is added on every side. In that ring — the "sponge" — the wind and air are
gently relaxed toward the prevailing background conditions, which makes the domain behave like an
open patch of a bigger world: wind flows in one edge and out the other without piling up or
bouncing back. The observation, the rendered image, and the interactive controls all use only the
inner (sponge-free) region. Maps are generated at the padded size to match.

---

## The state

All per-tick fields are bundled in a `SimState` dataclass. The main ones:

| Field | Meaning | Units |
|-------|---------|-------|
| `mass` | Boundary-layer air mass per area (this doubles as surface pressure) | kg/m² |
| `air_temperatures` | Air temperature, split into vertical **levels** (see below) | K |
| `ground_temperature` | Surface (soil/litter) skin temperature | K |
| `fuel` | Vegetation mass, **one layer per fuel type** | kg/m² |
| `fuel_temperatures` | Temperature of each fuel layer | K |
| `oxygen` | Near-surface oxygen partial density | kg/m³ |
| `terrain` | Elevation | m |
| `x_wind_vel`, `y_wind_vel` | Prognostic wind velocity | m/s |
| `x_wind_fire`, `y_wind_fire` | The near-surface wind the fire and oxygen actually read (wind × terrain-channeling gain) | m/s |
| `channel_gain` | Static per-map terrain speed-up factor (≥ 1) | — |
| `u_amb_x`, `u_amb_y` | The per-map prevailing ("synoptic") wind the sponge relaxes toward | m/s |
| `radiant_flux` | Last tick's absorbed radiant heat (a diagnostic for the view) | W/m² |

Three more fields — `temp_eq`, `oxygen_eq`, `mass_eq` — are the constant "rest state" targets each
field relaxes toward (the radiative-equilibrium temperature, the fresh-air oxygen profile, and the
level-lid rest mass). They are fixed for an episode but travel with the state so that stepping is a
pure function of it.

### Fuel types

A cell does not hold one lump of "fuel"; it holds a **stack of separate fuel layers**, each with
its own physical constants. The default config defines five:

- **`grass`** — fine, fast-burning surface fuel.
- **`shrub`** — woodier, slightly slower, energy-dense chaparral-type scrub.
- **`tree_canopy`** — fine foliage/twigs that carry a crown fire.
- **`tree_bole_surface`** — the lower tree trunk (0–5 m), coarse slow-burning wood.
- **`tree_bole_canopy`** — the upper trunk and big limbs (5–15 m).

Each type has a specific heat, an ignition threshold, a heat of combustion, an oxygen demand, an
Arrhenius reaction rate, a surface-area-to-volume ratio (fine grass has a huge one and burns out in
seconds; a coarse log has a tiny one and smoulders for minutes), and so on — all in
[`cfg/default.toml`](../cfg/default.toml) under `[fuel_types.*]`. You can add or remove fuel types
just by editing that config; the engine sizes all its per-type arrays from it.

### Vertical air levels

The near-surface air is not a single sheet either — it is split into stacked slabs so that a
surface fire and a crown fire live in different air:

- **Level 0 — surface** (0–5 m): grass, shrub, and the lower tree trunk couple here.
- **Level 1 — canopy** (5–15 m): tree foliage and the upper trunk couple here.
- **Level 2 — above-canopy** (the deep remainder): carries no fuel; it is where plume heat finally
  vents to the sky.

Each fuel layer trades heat only with the thin slab at its own height. Heat then moves *upward*
between levels, and the trunk carries fire up from level 0 to level 1 (the "ladder"). This is what
lets a ground fire climb into the crown, and lets an intense fire shed its heat out the top instead
of overheating without bound.

---

## One tick, step by step

Every tick calls `Simulation.step_fields`, which runs the following sequence. This one method is
the single source of truth — both the interactive environment (one world) and the batched data
collector (many worlds at once) call it, so they can never drift apart. The operations are written
to work on a bare 2-D world or a batched stack of worlds interchangeably.

### 1. Wind, mass, and air advection (`step_dynamics`)

The wind is **prognostic** — it is a real, evolving velocity field, not a shortcut computed from
the pressure. The model is a shallow layer of atmosphere sitting on the terrain. The height of the
top of that layer is `terrain + eta`, where `eta` is the layer thickness derived from the air mass
and temperature. The wind accelerates down the slope of that top surface. Because the modeled
layer is shallow (roughly as deep as the terrain is tall), the air gets squeezed as it flows over
high ground, which — by conservation of mass — speeds it up over crests and channels it through
gaps.

Each tick this stage:

1. Computes the forcing (the downhill push of the free surface), applies weak surface friction, and
   smooths the wind with eddy viscosity.
2. **Conservatively advects** the air mass, the momentum, and *every air level's heat energy* along
   that wind. Advecting each level's energy is what makes the plume tilt downwind — the canopy and
   above-canopy heat drift with the flow just like the surface does.
3. Relaxes the outer sponge ring toward the prevailing wind and rest mass (the open boundary).

For stability the whole stage is **CFL sub-stepped**: it works out the worst-case distance any
signal could travel in one tick (the wind speed plus the gravity-wave speed) and, if that exceeds a
target fraction of a cell, splits the tick into several smaller solver steps.

### 2. The terrain-channeling wind (the wind the fire feels)

The prognostic wind above is terrain-responsive, but its shallow-layer physics actually makes
valleys *slow* (the layer deepens in low ground), which is the opposite of the gap speed-up a real
Venturi valley produces. Rather than add a second, more expensive airflow layer, the model applies
a **static per-map gain**: for each cell it scans the terrain across the wind direction, measures
how tightly the channel is walled in on both sides, and computes a speed-up factor (capped, and
larger for taller and closer walls). A true valley confined on both sides gets a big boost; a
hillside open on one side stays near 1.

This gain (`channel_gain`) is precomputed once per map. The fire and the oxygen read the prognostic
wind **multiplied by this gain** (`x_wind_fire` / `y_wind_fire`) — direction unchanged, speed
boosted through gaps. That is also the wind you see in the Wind view.

### 3. Surface radiation (`apply_radiation`)

Sunlight is absorbed at the *surface*, not by the air (air is transparent to sunlight). The
incoming solar energy is split between bare ground and the fuel/canopy by a vegetation "cover
fraction" (denser vegetation intercepts more). Each surface — ground and each fuel layer — absorbs
its share of sunlight and radiates heat back out as `emissivity × σ × T⁴` (the Stefan–Boltzmann
law). The air is warmed only indirectly, by conduction from the ground.

The incoming solar flux is pinned per cell so that the radiative equilibrium sits exactly at that
cell's baked **rest temperature** (`temp_eq`) — the elevation-and-sunlight profile the map generator
computed, warmer on sun-facing slopes and cooler in shadow. How fast each surface responds is set by
its thermal mass: fine fuel heats and dries quickly, the ground skin follows, and the deep air
column lags far behind. This T⁴ emission is also the self-limiting brake on flame temperature —
the hotter a flame gets, the harder it radiates its heat away.

### 4. Fuel ↔ air conduction (`exchange_fuel_air_heat`)

Each fuel layer conducts heat with the thin air slab at its own height. This uses an exact
two-body relaxation: over one tick the temperature gap between the fuel and its air decays toward
their shared (mass-weighted) equilibrium by an exact exponential, so it can never overshoot no
matter the time step. Energy is conserved — whatever the fuel gains, its air level loses.

### 5. Oxygen transport (`diffuse_and_advect_oxygen`, `relax_to_equilibrium`)

Oxygen drifts with the fire wind (the same conservative advection as the air mass), then mixes by a
small Gaussian diffusion, then is relaxed toward its fresh-air rest profile (mixing down from
aloft). A vigorous fire can still locally outpace this replenishment and choke itself. *(This whole
stage is skipped when a caller asks for it — the map spin-up does, since a quiescent map has no
fire.)*

### 6. Combustion and fire spread (`update_fire`)

This is the core of the fire model. For every fuel layer at every cell:

- **The burn rate** comes from real chemistry in series with oxygen supply. A kinetic (Arrhenius)
  rate `B·exp(-Eₐ/RT)` governs cold fuel — it is vanishingly small at ambient temperature, which is
  the (soft) ignition gate: cold fuel is simply inert. Once the fuel is flame-hot, the rate is
  instead capped by how fast oxygen can diffuse to the fuel surface. Combining the two as
  resistances-in-series means the slower one always wins. The reactive **surface area** (from each
  fuel's surface-area-to-volume ratio) scales the whole thing, so fine grass burns out in seconds
  and a coarse log over minutes — all emerging from one geometric property.
- **Oxygen limits fuel too.** The oxygen a burn would need is drawn from a shallow near-surface
  slab. If that demand exceeds the oxygen present, the whole reaction scales down so fuel and oxygen
  are consumed in step — a fire starved of air burns less fuel and releases less heat.
- **The released heat is split**: a small fraction stays in the fuel (this sets the flame
  temperature), the rest goes into a shallow, buoyant plume slab of air above the cell. Injecting
  the heat into a *shallow* slab (not the whole kilometre-deep column) is what makes a burn raise a
  strong, drift-able temperature signal that in turn lifts the air and drives the indraft.

**Fire spread** is convection-dominant (realistic for grass): radiation alone is too weak to ignite
a neighbor at realistic flame temperatures, so ignition is carried by hot plume gas. A fraction of
each flame's plume heat (`convective_fraction`) is deposited onto nearby fuel, skewed **downwind and
uphill** by an angular kernel whose bias blends the wind (stronger wind throws the heat harder
downwind) with the upslope terrain gradient (fires run faster up steeper ground). A separate small
**isotropic near-field** term spreads a little heat in every direction, which is what lets a small
point ignition close its perimeter and establish rather than blowing all its heat downwind and
starving. Deposited heat only ignites *sub-flaming* fuel — already-flaming fuel is combustion-
controlled — which keeps the front from running away.

This whole stage is routed **per air level**: a grass fire's surface plume ignites neighbors'
surface fuels but never reaches directly into their canopy. A separate **crown-preheat** term feeds
part of the surface fire's heat straight into the trunk base, so that only a *sustained* surface
fire (not a quick flash of light grass) can eventually torch the trunk and start climbing.

### 7. Flame radiation to neighbors (`apply_radiant_heat`)

The surface-radiation stage already emitted (and cooled) each flame's gray-body heat. This stage
*redistributes* the part of that emission above the ambient glow onto nearby fuel by an
inverse-square kernel, so flames preheat their neighbors. A fraction escapes upward to the sky.
On its own this preheat cannot sustain a fire, but combined with the convective ignition above it
speeds the spread.

### 8. The ladder — trunk conduction (`conduct_bole_segments`)

The tree trunk carries fire between its two stacked segments by conduction, with a
**direction-dependent rate**: fast when the lower segment is hotter (fire climbs the trunk into the
crown) and slow when the upper segment is hotter (a crown fire does not readily run back down). This
is the dedicated conduit that turns a surface fire into a crown fire.

### 9. Upward inter-level convection (`convect_air_levels`)

Heat cascades *upward* through the air stack (surface → canopy → above-canopy), but only where the
lower level is hotter than the one above it (warm air rises; the reverse does not mix down). It is
energy-conserving and self-limiting. This is what gives the surface and canopy slabs a way to shed
the heat they take from the fire.

### 10. Venting to the sky (`vent_plume_heat`)

The single ultimate heat sink. The air is otherwise transparent to radiation and has no vertical
escape, so without this a big fire's plume heat would accumulate without bound and conduct back into
the fuel — a runaway. The top (above-canopy) level's *super-ambient* heat is relaxed toward the rest
profile, modeling a buoyant plume carrying heat up and out while fresh air entrains. It only ever
*cools* air that is hotter than ambient, so a quiescent world is untouched.

---

## The observation

For the machine-learning side, `build_observation` packs the state into a normalized multi-channel
image (the same thing the environment returns as its Gymnasium observation). The channels, in order:

1. Surface air temperature.
2. One temperature channel per fuel type.
3. One vegetation-mass channel per fuel type.
4. The two components of the terrain **slope** (not the height — it is the slope that skews fire
   spread, and it is supplied directly because a convolutional network struggles to synthesize a
   derivative from a raw height field).
5. The two near-surface wind components (the fire wind).
6. A binary "ignited" mask (1 where any fuel layer is at or above its ignition threshold).

So the channel count is `6 + 2 × (number of fuel types)` — 16 for the default five fuels. Each
channel is mapped to roughly the `[0, 1]` range by a fixed, physically-motivated affine window (in
`physics_constants.OBS_NORM`); the high side is intentionally left unclamped so flames and strong
gusts ride a little above 1 rather than saturating.

---

## Turning subsystems off

Most stages have an on/off toggle in the config (`[radiation]`, `[relaxation]`, `[venting]`,
`[fire]`, `[radiant_heat]`, `[channeling]`, `[air_levels]`). Turning them off is useful for
isolating behavior — for example, disabling venting and fire recovers a closed, exactly-conserving
transport core, which the conservation tests rely on.
