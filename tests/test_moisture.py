"""
Fire-suppression moisture (the water-drop action) -- Simulation.apply_moisture + the env wiring.

Water dropped on a cell (SimState.moisture, deposited by the env's action) is a heat sink: each
tick it absorbs the super-ambient heat of the co-located fuel and surface air and boils off at the
latent heat of vaporization, pinning wet fuel at the boiling point so the cell cannot ignite until
the water is gone. A baseline (sun + wind) term also dries a drop out with no fire present. These
tests pin: a single call cools a hot wet cell to the boiling point and removes the matching latent
heat as evaporated water; a wet cell sitting in a fire never ignites while an identical dry one
burns; the water eventually boils off and the once-wet cell is then free to burn; ambient drying
clears a drop on a cold world (faster in wind); and a completed action deposits water on its cell.
"""

import numpy as np
import pytest
import torch

import physics_constants as pc
from conftest import make_config, build_map
from simulation import Simulation, SimState, lift_air_levels
from gen_maps import MapGenerator

GRID = 48
T_DRY = pc.WATER_BOILING_POINT


def _sim(**kw) -> Simulation:
    return Simulation(make_config(GRID, fire=True, radiation=True, relaxation=True, **kw))


def _state(sim: Simulation, *, ignite_center=False, moisture=None) -> SimState:
    """A flat SI grass world; optionally a hot center cell and/or a moisture field [kg/m^2]."""
    gen = MapGenerator(make_config(GRID))
    terrain = np.zeros((GRID, GRID), dtype=np.float64)
    air = gen.air_temperature_profile(terrain)
    mass = gen.boundary_layer_mass(terrain)
    oxygen = gen.oxygen_profile(terrain, air)
    t = lambda a: torch.as_tensor(a, dtype=sim.dtype, device=sim.device)
    air_t, mass_t, oxy_t, terr_t = t(air), t(mass), t(oxygen), t(terrain)

    fuel = torch.full((1, GRID, GRID), 1.0, dtype=sim.dtype, device=sim.device)
    fuel_t = torch.full((1, GRID, GRID), pc.T_REF, dtype=sim.dtype, device=sim.device)
    if ignite_center:
        fuel_t[0, GRID // 2, GRID // 2] = 1200.0

    moist_t = torch.zeros_like(mass_t) if moisture is None else t(moisture)
    z = torch.zeros_like(mass_t)
    return SimState(
        mass=mass_t, air_temperatures=lift_air_levels(air_t, sim.num_air_levels),
        ground_temperature=air_t.clone(),
        fuel_temperatures=fuel_t, fuel=fuel, oxygen=oxy_t, terrain=terr_t,
        temp_eq=lift_air_levels(air_t.clone(), sim.num_air_levels),
        oxygen_eq=oxy_t.clone(), mass_eq=mass_t.clone(),
        x_wind_vel=z.clone(), y_wind_vel=z.clone(),
        u_amb_x=torch.zeros((1, 1), dtype=sim.dtype), u_amb_y=torch.zeros((1, 1), dtype=sim.dtype),
        radiant_flux=z.clone(), moisture=moist_t,
    )


# --- apply_moisture in isolation (deterministic, one call) -------------------------------------

def test_water_cools_hot_cell_to_boiling_and_evaporates_matching_mass():
    """One call: an abundant-water cell whose fuel/air are far above boiling is cooled to the
    boiling point, and the water boiled off equals exactly the absorbed heat / latent heat."""
    sim = _sim()
    sim.moisture_base_dry_rate = 0.0    # isolate the thermal (latent-heat) term from ambient drying
    sim.moisture_wind_dry_coeff = 0.0
    fuel = torch.ones((1, 1, 1), dtype=sim.dtype)
    fuel_t = torch.full((1, 1, 1), 1500.0, dtype=sim.dtype)
    air = torch.full((1, 1, 1), 1500.0, dtype=sim.dtype)     # (L=1, H=1, W=1)
    m0 = 100.0                                               # kg/m^2 -- more than enough to absorb it
    moist = torch.full((1, 1), m0, dtype=sim.dtype)
    speed = torch.zeros((1, 1), dtype=sim.dtype)

    # Heat the water should absorb = fuel + air super-ambient heat above the boiling point.
    C_fuel = sim.fuel_specific_heat[0] * 1.0 * (1500.0 - T_DRY)
    rho = pc.P_REF / (pc.GAS_CONSTANT_DRY_AIR * 1500.0)
    C_air = rho * sim.plume_mixing_depth * pc.CP_AIR * (1500.0 - T_DRY)
    expected_evap = float(C_fuel + C_air) / pc.WATER_LATENT_HEAT

    m1, ft1, at1 = sim.apply_moisture(moist, fuel_t, air, fuel, speed)

    assert float(ft1[0, 0, 0]) == pytest.approx(T_DRY)      # fuel pinned at boiling
    assert float(at1[0, 0, 0]) == pytest.approx(T_DRY)      # surface air cooled to boiling
    boiled = m0 - float(m1[0, 0])
    assert boiled == pytest.approx(expected_evap, rel=1e-6)  # energy -> latent heat balances


def test_wet_fuel_cannot_exceed_boiling_point():
    """While water remains, fuel is gated at the boiling point even if it starts white-hot."""
    sim = _sim()
    fuel = torch.ones((1, 1, 1), dtype=sim.dtype)
    fuel_t = torch.full((1, 1, 1), 2000.0, dtype=sim.dtype)
    air = torch.full((1, 1, 1), pc.T_REF, dtype=sim.dtype)
    moist = torch.full((1, 1), 50.0, dtype=sim.dtype)
    m1, ft1, _ = sim.apply_moisture(moist, fuel_t, air, fuel, torch.zeros((1, 1), dtype=sim.dtype))
    assert float(m1[0, 0]) > 0.0                            # still wet
    assert float(ft1[0, 0, 0]) <= T_DRY + 1e-6             # held at boiling


def test_dry_fuel_is_untouched():
    """No water -> the gate/heat-sink is a no-op: a hot dry cell keeps its temperature and no
    phantom water appears."""
    sim = _sim()
    fuel = torch.ones((1, 1, 1), dtype=sim.dtype)
    fuel_t = torch.full((1, 1, 1), 1500.0, dtype=sim.dtype)
    air = torch.full((1, 1, 1), 1500.0, dtype=sim.dtype)
    moist = torch.zeros((1, 1), dtype=sim.dtype)
    m1, ft1, at1 = sim.apply_moisture(moist, fuel_t, air, fuel, torch.zeros((1, 1), dtype=sim.dtype))
    assert float(m1[0, 0]) == 0.0
    assert float(ft1[0, 0, 0]) == pytest.approx(1500.0)
    assert float(at1[0, 0, 0]) == pytest.approx(1500.0)


def test_ambient_drying_clears_a_drop_and_wind_speeds_it():
    """With no fire, a drop still dries out (sun) and faster in wind."""
    sim = _sim()
    fuel = torch.ones((1, 1, 1), dtype=sim.dtype)
    fuel_t = torch.full((1, 1, 1), pc.T_REF, dtype=sim.dtype)   # cold -> no thermal boil-off
    air = torch.full((1, 1, 1), pc.T_REF, dtype=sim.dtype)

    def dry(speed, ticks=200):
        m = torch.full((1, 1), 5.0, dtype=sim.dtype)
        sp = torch.full((1, 1), float(speed), dtype=sim.dtype)
        for _ in range(ticks):
            m, fuel_t2, _ = sim.apply_moisture(m, fuel_t.clone(), air, fuel, sp)
        return float(m[0, 0])

    calm, windy = dry(0.0), dry(10.0)
    assert calm < 5.0                    # sun alone removes some water
    assert windy < calm                  # wind carries it away faster


# --- full step_fields integration ---------------------------------------------------------------

def _fire_around_target(sim: Simulation, moisture=None) -> SimState:
    """A world with a hot ring of fire surrounding the center target cell (the target itself is not
    lit -- fire has to reach it), optionally with a moisture field."""
    c = GRID // 2
    s = _state(sim, moisture=moisture)
    ring = np.zeros((GRID, GRID), dtype=bool)
    ring[c - 2:c + 3, c - 2:c + 3] = True
    ring[c, c] = False                                     # everything but the target
    s.fuel_temperatures[0][torch.as_tensor(ring)] = 1200.0
    return s


def test_wet_cell_reached_by_fire_does_not_ignite_while_an_identical_dry_cell_burns():
    """A heavily-wet target surrounded by fire stays unburned and near boiling, whereas the same
    target with no water ignites and loses fuel -- the water is a genuine retardant, not a flag."""
    c = GRID // 2

    sim = _sim()
    dry = _fire_around_target(sim)
    for _ in range(40):
        sim.step_fields(dry)
    assert float(dry.fuel[0, c, c]) < 0.9                  # dry target caught and burned

    sim2 = _sim()
    heavy = np.zeros((GRID, GRID)); heavy[c, c] = 80.0
    wet = _fire_around_target(sim2, moisture=heavy)
    for _ in range(40):
        sim2.step_fields(wet)
    assert float(wet.moisture[c, c]) > 0.0                 # water not yet exhausted
    assert float(wet.fuel[0, c, c]) > 0.999                # fuel intact -- never ignited
    assert float(wet.fuel_temperatures[0, c, c]) <= T_DRY + 1.0   # pinned at boiling


def test_water_boils_off_then_the_cell_can_burn():
    """A light drop on a cell surrounded by sustained fire boils away, and the once-wet cell then
    ignites and burns -- the suppression is temporary, gated on water remaining."""
    c = GRID // 2
    sim = _sim()
    s = _state(sim)
    # A hot 5x5 block around the target keeps feeding it heat after its own water is gone.
    s.fuel_temperatures[0, c - 2:c + 3, c - 2:c + 3] = 1200.0
    light = np.zeros((GRID, GRID)); light[c, c] = 1.0      # a light drop on the center only
    s.moisture = torch.as_tensor(light, dtype=sim.dtype)

    burned_at = None
    for tick in range(120):
        sim.step_fields(s)
        if burned_at is None and float(s.fuel[0, c, c]) < 0.5:
            burned_at = tick
    assert float(s.moisture[c, c]) == 0.0                  # water fully boiled off
    assert burned_at is not None                           # the once-wet cell did eventually burn


# --- env action wiring --------------------------------------------------------------------------

def test_completed_action_near_fire_drops_water(make_env):
    """A two-step action completed over a burning action-cell deposits water on that cell's block,
    raising its moisture; a fire-free selection stays a no-op."""
    from scenarios import _si_state
    terrain = np.zeros((GRID, GRID), dtype=np.float32)
    air, mass, oxygen = _si_state(terrain)
    fuel = np.ones((1, GRID, GRID), dtype=np.float32)          # grass everywhere so a seed can burn
    config = make_config(GRID, fire=True, radiation=True)
    # A realistic pyrolysis threshold so ambient fuel is NOT flagged as burning (the test default of
    # 1.5 K would mark every cell on fire, defeating the near-fire action gate).
    config["fuel_types"]["grass"]["ignition_threshold"] = 573.0
    map_data = build_map(GRID, terrain=terrain, air=air, mass=mass, oxygen=oxygen, fuel=fuel)
    env = make_env(config, map_data)
    env.reset(seed=0)
    assert env.action_grid_size == GRID                    # 1 sim cell per action-cell on this grid
    r = c = GRID // 2
    env._spawn_fire_patch(r, c)                            # a burning patch of radius ~5 at the centre

    # The deposit itself lays down exactly water_drop_amount over the action-cell's block.
    p = env._pad
    m_before = env._moisture.clone()
    env._apply_water_drop(r, c)
    assert float(env._moisture[p + r, p + c] - m_before[p + r, p + c]) == pytest.approx(
        env._water_drop_amount)
    env._moisture = m_before.clone()                       # undo before the action-driven checks

    # A two-step action completed over a cell just outside the fire but adjacent to it goes through
    # and drops water there (a cool cell, so it isn't instantly boiled off within the tick).
    tr, tc = r, c + 6                                      # outside the disk, neighbour of a burning cell
    before = float(env._moisture[p + tr, p + tc])
    env.step(tr)                                           # row step
    env.step(tc)                                           # column step completes the selection
    assert env._selected_action_cell == (tr, tc)
    assert float(env._moisture[p + tr, p + tc]) > before   # water landed and persists near (not on) fire

    # A selection far from any fire is a no-op -- the action doesn't go through, so no drop path runs
    # (moisture still evolves under the physics each tick, so we check the selection, not the field).
    fr = fc = 1
    assert not env._action_cell_near_fire(fr, fc)
    env.step(fr)
    env.step(fc)
    assert env._selected_action_cell is None


def test_water_drop_footprint_is_a_circle_around_the_square(make_env):
    """With a coarse action grid (c > 1 sim cells per action-cell), a drop wets a disk that
    circumscribes the action-cell's square: every square cell is wet, the disk bulges a little beyond
    the square edges, and cells outside the circumscribing circle stay dry."""
    G = 32
    config = make_config(G)
    config["action"] = {"grid_size": 4}                # 8x8 sim cells per action-cell
    env = make_env(config, build_map(G))
    env.reset(seed=0)
    c = env._action_cell_cells
    assert c == 8
    p = env._pad
    arow = acol = 1                                    # a middle action-cell, room to bulge each way
    r0, c0 = p + arow * c, p + acol * c
    env._apply_water_drop(arow, acol)
    m = env._moisture

    assert (m[r0:r0 + c, c0:c0 + c] > 0).all()          # the whole square block is wet
    assert float(m[r0 - 1, c0 + 3]) > 0                 # one row above the square (near centre) -> inside disk
    assert float(m[r0 - 4, c0 + 3]) == 0               # farther out, beyond the circle -> dry
    assert float(m[r0 - 3, c0 - 3]) == 0               # a diagonal corner well outside the circle -> dry
