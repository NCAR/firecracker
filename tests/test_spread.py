"""
Fire spread across full step_fields ticks (Phase 5.5c) -- the "fire propagates" deliverable.

With sustained flaming (leg a) and the shallow plume (leg b) in place, update_fire convects a
fraction of the plume's air-share heat onto nearby fuel (convective_fraction) to ignite it -- the
mechanism that moves the front. Radiation preheats the fuel ahead but is too weak to ignite it at a
realistic flame temperature, so spread is convection-dominant (realistic for grass). These tests
drive the whole per-tick loop and pin:
a one-shot hot cell self-propagates into a spreading front (symmetric with no wind), the spread is
biased downwind under an ambient wind (lee bias), the flame temperature stays in a physical band
(no runaway), and a world with no ignition source never spontaneously lights.
"""

import numpy as np
import torch

import physics_constants as pc
from conftest import make_config
from simulation import Simulation, SimState
from gen_maps import MapGenerator

GRID = 64
TICKS = 150


def _state(sim: Simulation, ambient=(0.0, 0.0), ignite=True) -> SimState:
    """A flat SI world, fuel everywhere, optionally one hot ignition cell at the centre."""
    gen = MapGenerator(make_config(GRID))
    terrain = np.zeros((GRID, GRID), dtype=np.float64)
    air = gen.air_temperature_profile(terrain)
    mass = gen.boundary_layer_mass(terrain)
    oxygen = gen.oxygen_profile(terrain, air)
    t = lambda a: torch.as_tensor(a, dtype=sim.dtype, device=sim.device)
    air_t, mass_t, oxy_t, terr_t = t(air), t(mass), t(oxygen), t(terrain)

    fuel = torch.full((1, GRID, GRID), 1.0, dtype=sim.dtype, device=sim.device)   # uniform grass
    fuel_t = torch.full((1, GRID, GRID), pc.T_REF, dtype=sim.dtype, device=sim.device)
    if ignite:
        fuel_t[0, GRID // 2, GRID // 2] = 1200.0   # one-shot hot cell, then released

    ax = torch.tensor(float(ambient[0]), dtype=sim.dtype, device=sim.device).view(1, 1)
    ay = torch.tensor(float(ambient[1]), dtype=sim.dtype, device=sim.device).view(1, 1)
    return SimState(
        mass=mass_t, air_temperatures=air_t, ground_temperature=air_t.clone(),
        fuel_temperatures=fuel_t, fuel=fuel, oxygen=oxy_t, terrain=terr_t,
        temp_eq=air_t.clone(), oxygen_eq=oxy_t.clone(), mass_eq=mass_t.clone(),
        x_wind_vel=torch.zeros_like(mass_t) + ax, y_wind_vel=torch.zeros_like(mass_t) + ay,
        u_amb_x=ax, u_amb_y=ay, radiant_flux=torch.zeros_like(mass_t),
    )


def _sim() -> Simulation:
    return Simulation(make_config(GRID, fire=True, radiation=True, relaxation=True))


def _run(sim: Simulation, s: SimState, ticks: int = TICKS) -> float:
    """Advance the world; return the hottest front fuel temperature while the front is still finite."""
    front_temp = 0.0
    for _ in range(ticks):
        sim.step_fields(s)
        if float((s.fuel[0] < 0.5).sum()) < 0.6 * GRID * GRID:   # before the domain is fully involved
            front_temp = max(front_temp, float(s.fuel_temperatures.max()))
    return front_temp


def _extents(s: SimState) -> tuple[int, int, int, int]:
    """Burnt-region reach (left, right, up, down) in cells from the centre, along the centre lines."""
    burnt = (s.fuel[0].cpu().numpy() < 0.5)
    c = GRID // 2
    cols = np.where(burnt[c])[0]
    rows = np.where(burnt[:, c])[0]
    left = c - cols.min() if len(cols) else 0
    right = cols.max() - c if len(cols) else 0
    up = c - rows.min() if len(rows) else 0
    down = rows.max() - c if len(rows) else 0
    return int(left), int(right), int(up), int(down)


def test_fire_propagates_from_one_cell():
    """A single hot cell, then released, ignites a self-sustaining front that spreads outward.

    No wind, so the spread is symmetric: the burnt region reaches several cells in every direction
    (the fire crosses cells radiation alone could never ignite), driven by convective ignition.
    """
    sim = _sim()
    s = _state(sim)
    _run(sim, s)

    left, right, up, down = _extents(s)
    assert min(left, right, up, down) >= 3            # propagated well beyond the ignition cell
    assert int((s.fuel[0].cpu().numpy() < 0.5).sum()) > 30   # a real burnt area, not one cell
    assert max(left, right, up, down) - min(left, right, up, down) <= 2   # symmetric with no wind
    assert torch.isfinite(s.fuel_temperatures).all()


def test_spread_is_biased_downwind():
    """Under an ambient wind the front reaches further downwind than upwind (the lee bias)."""
    sim = _sim()
    s = _state(sim, ambient=(8.0, 0.0))   # wind toward +x (downwind = right)
    _run(sim, s)

    left, right, up, down = _extents(s)
    assert right > left + 3       # clearly further downwind than upwind
    assert right > down           # and biased along-wind versus cross-wind
    assert torch.isfinite(s.fuel_temperatures).all()


def test_flame_temperature_is_physical():
    """The propagating front holds a realistic flame temperature -- bounded, no runaway."""
    sim = _sim()
    s = _state(sim)
    front_temp = _run(sim, s)

    assert 1000.0 < front_temp < 2100.0     # grass-flame band, not the pre-5.5c ~2400 K / runaway
    assert torch.isfinite(s.fuel_temperatures).all()
    assert torch.isfinite(s.air_temperatures).all()


def test_no_spontaneous_ignition():
    """With no ignition source a fuelled world never lights -- spread needs a fire to start."""
    sim = _sim()
    s = _state(sim, ignite=False)
    _run(sim, s)

    np.testing.assert_allclose(s.fuel[0].cpu().numpy(), 1.0, atol=1e-6)   # no fuel consumed
    assert float(s.fuel_temperatures.max()) < 400.0                       # nothing approached pyrolysis
