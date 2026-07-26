"""
Crown fire vs. the fuel ladder -- what happens when the fire starts *aloft* in the canopy instead
of on the ground.

The vegetation ladder is deliberately asymmetric: fire climbs a tree trunk far faster than it
descends it (bole_conduction_up >> bole_conduction_down), and inter-level air convection only ever
carries heat *upward* (warm air rises). So a fire lit in the crown should race canopy-to-canopy
through the connected foliage but barely reach the fine surface fuels below, whereas the same fire
lit on the ground climbs the ladder and involves everything.

These tests build a uniform five-fuel world (grass, shrub, tree_canopy, tree_bole_surface,
tree_bole_canopy) -- the full ladder -- ignite a small patch either in the canopy or on the ground,
run the whole per-tick loop, and pin: a canopy ignition spreads crown-to-crown well beyond the seed
patch; that same canopy ignition leaves the grass/shrub/surface-trunk essentially unburnt; and the
ground-ignition control instead consumes every layer (so the world plainly *can* carry fire between
layers -- the crown fire's restraint is the ladder asymmetry, not an inert map).
"""

import numpy as np
import torch

import physics_constants as pc
from conftest import make_config
from simulation import Simulation, SimState, lift_air_levels
from gen_maps import MapGenerator

GRID = 64
TICKS = 150
SEED = 2                       # ignition patch half-width -> a (2*SEED+1)^2 = 25-cell square
SEED_CELLS = (2 * SEED + 1) ** 2

# The full vertical fuel stack: surface fines (grass, shrub), the crown foliage (tree_canopy), and
# the trunk split into its surface and canopy segments. "canopy" in the name -> crown air level (1);
# "tree_bole_surface"/"tree_bole_canopy" wire up the bole-segment conduction ladder.
FUELS = ("grass", "shrub", "tree_canopy", "tree_bole_surface", "tree_bole_canopy")
I_GRASS, I_SHRUB, I_CANOPY, I_BOLE_SURF, I_BOLE_CANOPY = range(5)


def _sim() -> Simulation:
    # Fire + radiation + the inter-level air convection ladder; relaxation keeps the ambient steady.
    return Simulation(make_config(GRID, FUELS, fire=True, radiation=True,
                                  relaxation=True, air_convection=True))


def _state(sim: Simulation, ignite: str) -> SimState:
    """A flat, wind-free SI world with all five fuels present everywhere, ignited either in the
    canopy (`ignite="canopy"` -> a hot tree_canopy patch) or on the ground (`ignite="ground"` -> a
    hot grass+shrub patch). The patch is a one-shot temperature seed, then released to the physics."""
    gen = MapGenerator(make_config(GRID, FUELS))
    terrain = np.zeros((GRID, GRID), dtype=np.float64)
    air = gen.air_temperature_profile(terrain)
    mass = gen.boundary_layer_mass(terrain)
    oxygen = gen.oxygen_profile(terrain, air)
    t = lambda a: torch.as_tensor(a, dtype=sim.dtype, device=sim.device)
    air_t, mass_t, oxy_t, terr_t = t(air), t(mass), t(oxygen), t(terrain)

    fuel = torch.ones((5, GRID, GRID), dtype=sim.dtype, device=sim.device)               # every layer present
    fuel_t = torch.full((5, GRID, GRID), pc.T_REF, dtype=sim.dtype, device=sim.device)   # all at ambient
    c = GRID // 2
    patch = (slice(c - SEED, c + SEED + 1), slice(c - SEED, c + SEED + 1))
    if ignite == "canopy":
        fuel_t[I_CANOPY][patch] = 1200.0            # light the crown only
    elif ignite == "ground":
        fuel_t[I_GRASS][patch] = 1200.0             # light the fine surface fuels only
        fuel_t[I_SHRUB][patch] = 1200.0
    else:                                           # pragma: no cover - guard for typos
        raise ValueError(ignite)

    z = torch.zeros_like(mass_t)
    ax = torch.zeros((1, 1), dtype=sim.dtype, device=sim.device)
    return SimState(
        mass=mass_t, air_temperatures=lift_air_levels(air_t, sim.num_air_levels),
        ground_temperature=air_t.clone(),
        fuel_temperatures=fuel_t, fuel=fuel, oxygen=oxy_t, terrain=terr_t,
        temp_eq=lift_air_levels(air_t.clone(), sim.num_air_levels),
        oxygen_eq=oxy_t.clone(), mass_eq=mass_t.clone(),
        x_wind_vel=z.clone(), y_wind_vel=z.clone(), u_amb_x=ax, u_amb_y=ax,
        radiant_flux=z.clone(),
    )


def _run(ignite: str, ticks: int = TICKS) -> tuple[torch.Tensor, SimState, Simulation]:
    """Advance a freshly-ignited world; return (initial fuel, final state, sim)."""
    sim = _sim()
    s = _state(sim, ignite)
    fuel0 = s.fuel.clone()
    for _ in range(ticks):
        sim.step_fields(s)
    return fuel0, s, sim


def _burnt(fuel0: torch.Tensor, s: SimState) -> np.ndarray:
    """Per-fuel count of cells that lost >half their fuel (i.e. actually burned)."""
    consumed = (fuel0 - s.fuel).cpu().numpy()          # (5, H, W) fuel lost
    return (consumed > 0.5).reshape(5, -1).sum(axis=1)


def _canopy_reach(fuel0: torch.Tensor, s: SimState) -> tuple[int, int, int, int]:
    """Burnt-canopy reach (left, right, up, down) in cells from the centre, along the centre lines."""
    burnt = ((fuel0 - s.fuel)[I_CANOPY].cpu().numpy() > 0.5)
    c = GRID // 2
    cols = np.where(burnt[c])[0]
    rows = np.where(burnt[:, c])[0]
    left = c - cols.min() if len(cols) else 0
    right = cols.max() - c if len(cols) else 0
    up = c - rows.min() if len(rows) else 0
    down = rows.max() - c if len(rows) else 0
    return int(left), int(right), int(up), int(down)


def test_canopy_fire_spreads_crown_to_crown():
    """A patch lit in the crown ignites a self-sustaining canopy fire that spreads well beyond the
    seed, symmetrically with no wind, and carries the co-located upper trunk with it."""
    fuel0, s, _ = _run("canopy")
    burnt = _burnt(fuel0, s)

    # Spread: many more crowns burn than were seeded, reaching several cells in every direction.
    assert burnt[I_CANOPY] > 2 * SEED_CELLS
    left, right, up, down = _canopy_reach(fuel0, s)
    assert min(left, right, up, down) >= 3                       # propagated past the seed patch
    assert max(left, right, up, down) - min(left, right, up, down) <= 2   # symmetric with no wind
    # The upper (canopy) trunk segment shares the crown air level, so it burns along with the foliage.
    assert burnt[I_BOLE_CANOPY] > 2 * SEED_CELLS
    assert torch.isfinite(s.fuel_temperatures).all()


def test_canopy_fire_spares_the_surface_fuels():
    """The crown fire does not climb *down* the ladder: with fire descending a trunk far slower than
    it climbs (and air convection only rising), the grass, shrub and surface trunk stay unburnt --
    unlike a ground ignition, which involves every layer. The contrast rules out an inert map."""
    fuel0_c, s_c, _ = _run("canopy")
    burnt_c = _burnt(fuel0_c, s_c)

    # A canopy start leaves the fine surface fuels (and the surface trunk) untouched.
    assert burnt_c[I_GRASS] == 0
    assert burnt_c[I_SHRUB] == 0
    assert burnt_c[I_BOLE_SURF] == 0

    # A ground start on the same world climbs the ladder and consumes every layer -- proof the world
    # can carry fire between layers, so the crown fire's restraint above is the ladder asymmetry.
    fuel0_g, s_g, _ = _run("ground")
    burnt_g = _burnt(fuel0_g, s_g)
    assert burnt_g[I_GRASS] > 2 * SEED_CELLS
    assert burnt_g[I_SHRUB] > 2 * SEED_CELLS
    assert burnt_g[I_CANOPY] > 2 * SEED_CELLS                    # fire reached the crown from below
    assert burnt_g[I_BOLE_SURF] > 2 * SEED_CELLS
