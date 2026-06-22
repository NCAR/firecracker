"""
Fire spread across full step_fields ticks (Phase 5.5c/d) -- the "fire propagates" deliverable.

With sustained flaming (leg a) and the shallow plume (leg b) in place, update_fire convects a
fraction of the plume's air-share heat onto nearby fuel (convective_fraction) to ignite it -- the
mechanism that moves the front. Radiation preheats the fuel ahead but is too weak to ignite it at a
realistic flame temperature, so spread is convection-dominant (realistic for grass). The convective
deposit is skewed by a von Mises angular kernel whose bias blends the wind (lee bias) with the
upslope terrain gradient (Rothermel slope effect). These tests drive the whole per-tick loop and
pin: a one-shot hot cell self-propagates into a spreading front (symmetric with no wind), the spread
is biased downwind under an ambient wind (lee bias) and upslope on a ramp (slope effect), wind
aligned with the slope speeds/focuses the head while wind opposing it slows the head (the kernel
does not saturate), the flame stays in a physical temperature band (no runaway), and an unlit world
never spontaneously ignites.
"""

import numpy as np
import torch

import physics_constants as pc
from conftest import make_config
from simulation import Simulation, SimState
from gen_maps import MapGenerator

GRID = 64
TICKS = 150


def _state(sim: Simulation, ambient=(0.0, 0.0), ignite=True, terrain=None) -> SimState:
    """An SI world (flat unless `terrain` given), fuel everywhere, optionally one hot ignition cell."""
    gen = MapGenerator(make_config(GRID))
    terrain = np.zeros((GRID, GRID), dtype=np.float64) if terrain is None else np.asarray(terrain, dtype=np.float64)
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


def test_spread_is_biased_upslope():
    """On a slope (no wind) the front reaches further uphill than downhill (Rothermel slope effect).

    Terrain ramps uphill toward +x, so the convective-ignition kernel skews uphill (right) and the
    fire spreads further upslope than downslope, while staying roughly symmetric across the slope.
    """
    sim = _sim()
    # Ramp rising toward +x: ~380 m over the grid -> slope tan ~0.6 (~31 deg, uphill = +x = right).
    ramp = np.tile(np.linspace(0.0, 380.0, GRID, dtype=np.float64), (GRID, 1))
    s = _state(sim, terrain=ramp)
    _run(sim, s)

    left, right, up, down = _extents(s)
    assert right > left + 3       # clearly further upslope (right) than downslope (left)
    assert right > up             # and biased along-slope versus cross-slope
    assert abs(up - down) <= 2    # roughly symmetric across the slope (no cross-slope bias)
    assert torch.isfinite(s.fuel_temperatures).all()


def test_wind_slope_alignment_changes_spread():
    """Wind aligned with the upslope speeds the head fire; wind opposing it slows the head.

    The von Mises deposit reads the combined wind+slope bias as a forward concentration (it does not
    saturate), so the head-fire reach responds monotonically to alignment: a wind blowing up the
    slope drives the longest, tightest head (flanks pinched), the slope alone is intermediate, and a
    wind blowing down the slope cancels much of the slope bias and shortens the upslope head -- the
    wind/slope synergy real fires show (and what the old unit-capped dipole could not express).
    """
    ramp = np.tile(np.linspace(0.0, 380.0, GRID, dtype=np.float64), (GRID, 1))   # uphill toward +x

    def head(ambient):
        sim = _sim()
        s = _state(sim, ambient=ambient, terrain=ramp)
        _run(sim, s)
        _, right, up, down = _extents(s)
        return right, up, down, s

    right_slope, up_s, down_s, _   = head((0.0, 0.0))     # slope only, no wind
    right_align, up_a, down_a, s_a = head((8.0, 0.0))     # wind up the slope (aligned, +x)
    right_opp,   _,    _,    _      = head((-8.0, 0.0))    # wind down the slope (opposed)

    assert right_align > right_slope > right_opp      # head reach rises with wind/slope alignment
    assert (up_a + down_a) < (up_s + down_s)          # aligned head is tighter (flanks pinched)
    assert torch.isfinite(s_a.fuel_temperatures).all()


def test_wind_speed_drives_spread_rate():
    """A stronger wind drives the head fire markedly further downwind (Phase 6, Option 2).

    The convective deposit's wind bias scales with wind *speed* (convective_wind_bias*(wind/u_ref)),
    not just direction, so a faster wind grows the forward concentration and throws the ignition heat
    harder downwind: the head reaches further and the flanks pinch in. Flat and windless terrain
    isolates the effect (no slope), comparing a light wind against a strong one along +x.
    """
    def reach(speed):
        sim = _sim()
        s = _state(sim, ambient=(speed, 0.0))
        _run(sim, s)
        return _extents(s)   # (left, right, up, down)

    _, right_weak, up_weak, down_weak       = reach(3.0)
    _, right_strong, up_strong, down_strong = reach(12.0)

    assert right_strong >= right_weak + 3                       # the strong-wind head runs much further
    assert (up_strong + down_strong) < (up_weak + down_weak)    # and the front pinches into a tighter head


def test_radiant_preheat_needs_convection_but_accelerates_it():
    """Radiant preheat alone can't sustain a front, but it speeds one the convection carries.

    Three runs, flat and windless from one hot cell, with the Phase-3 surface balance on
    throughout (radiation=True) -- only the propagation legs differ. The radiant_heat stage
    redistributes a flame's super-ambient emission onto neighbours (preheat); convection deposits
    plume heat onto nearby fuel (the ignition driver). Preheat on its own (no convection) cannot
    raise a neighbour to ignition, so the fire dies at the source; convection alone spreads; the two
    together burn markedly more than convection alone -- preheat brings fuel closer to ignition so
    convection lights it sooner (~1 / 25 / 37 burnt cells for preheat-only / convection-only / both).
    (radiation_enabled is left on in every run: it owns the eps*sigma*T^4
    flame-cooling sink, so toggling it would conflate cooling with preheat -- only radiant_heat.enabled
    isolates the preheat leg.)
    """
    def burnt(convective_fraction=None, radiant_heat=True):
        sim = _sim()
        if convective_fraction is not None:
            sim.convective_fraction = convective_fraction
        sim.radiant_heat_enabled = radiant_heat
        s = _state(sim)
        _run(sim, s)
        return int((s.fuel[0].cpu().numpy() < 0.5).sum())

    burnt_preheat = burnt(convective_fraction=0.0)   # preheat only, no convective ignition
    burnt_conv    = burnt(radiant_heat=False)         # convection only, no flame preheat
    burnt_both    = burnt()                            # both legs (the default)

    assert burnt_preheat <= 3                # preheat alone cannot sustain a front (it dies)
    assert burnt_conv > 20                   # convection alone spreads a real fire
    assert burnt_both > burnt_conv * 1.15    # adding preheat clearly accelerates the spread


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
