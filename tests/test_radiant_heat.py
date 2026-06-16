"""
Operator-level tests for apply_radiant_heat.

Radiation must be conservative: a burning cell loses exactly the energy it emits,
a fraction escapes to space, and the rest is redistributed to neighbours. The
historical bug deposited flux into neighbours without ever cooling the emitter, so
the T^exponent term acted as a free energy *source* and a hot fire diverged to inf
(then NaN, crashing wind/render). These tests pin radiation down as a strict sink
and assert a hot fire stays finite.
"""

import numpy as np

from conftest import make_config
from simulation import Simulation


def _sim(scale: float = 1e-6, exponent: float = 4.0) -> Simulation:
    config = make_config(fuel_type_names=("grass",))
    config["radiant_heat"] = {
        "kernel_radius": 5,
        "emission_scale": scale,
        "emission_exponent": exponent,
        "sky_escape_fraction": 0.2,
    }
    return Simulation(config)


def _fuel_energy(sim, fuel_temps, fuel) -> float:
    c_fuel = sim.heat_capacities[:, None, None] * fuel
    return float((c_fuel * fuel_temps).sum())


def test_radiant_heat_is_a_net_sink():
    """Total fuel thermal energy must not increase: radiation only redistributes and loses."""
    sim = _sim()
    shape = (16, 16)
    fuel = np.full((1, *shape), 1.0, dtype=np.float32)
    fuel_temps = np.full((1, *shape), 10.0, dtype=np.float32)  # all burning, hot
    oxygen = np.ones(shape, dtype=np.float32)

    e0 = _fuel_energy(sim, fuel_temps, fuel)
    new_ft, _flux = sim.apply_radiant_heat(fuel_temps, fuel, oxygen)
    e1 = _fuel_energy(sim, new_ft, fuel)

    assert e1 <= e0 + 1e-3          # never a source
    assert e1 < e0                  # with sky escape + boundary loss, strictly cooling


def test_radiant_heat_does_not_diverge():
    """A uniformly hot fire must stay finite and bounded under repeated radiation."""
    sim = _sim()
    shape = (16, 16)
    fuel = np.full((1, *shape), 1.0, dtype=np.float32)
    fuel_temps = np.full((1, *shape), 50.0, dtype=np.float32)
    oxygen = np.ones(shape, dtype=np.float32)

    for _ in range(200):
        fuel_temps, _ = sim.apply_radiant_heat(fuel_temps, fuel, oxygen)

    assert np.isfinite(fuel_temps).all()
    assert fuel_temps.max() <= 50.0 + 1e-3   # pure radiation can only cool, never heat above start
    assert fuel_temps.min() >= 0.0           # emitter cooling never drives temperature negative


def test_radiant_heat_redistributes_to_neighbours():
    """A single hot cell must cool while its neighbours warm (no self-heating)."""
    sim = _sim(scale=1e-4)               # strong enough to move temperatures visibly in one tick
    shape = (16, 16)
    fuel = np.full((1, *shape), 1.0, dtype=np.float32)
    fuel_temps = np.zeros((1, *shape), dtype=np.float32)
    fuel_temps[0, 8, 8] = 20.0           # one hot, burning cell
    oxygen = np.ones(shape, dtype=np.float32)

    new_ft, _ = sim.apply_radiant_heat(fuel_temps, fuel, oxygen)

    assert new_ft[0, 8, 8] < 20.0        # the emitter cooled
    assert new_ft[0, 8, 9] > 0.0         # a neighbour was warmed
    assert np.isfinite(new_ft).all()
