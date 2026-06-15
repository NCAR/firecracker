"""
Operator-level tests for exchange_fuel_air_heat.

These call the Simulation method directly (no map/env) to pin down the exact
two-body relaxation claims: energy conservation, relaxation toward the
mass-weighted equilibrium without overshoot, and the massless-ash limit.
"""

import numpy as np

from conftest import make_config
from simulation import Simulation


def _sim() -> Simulation:
    # grass gets default heat_capacity = 1.0.
    return Simulation(make_config(fuel_type_names=("grass",)))


def _thermal_energy(sim, air, fuel_temps, fuel, mass) -> float:
    c_fuel = sim.heat_capacities[:, None, None] * fuel
    return float((mass * air).sum() + (c_fuel * fuel_temps).sum())


def test_heat_exchange_conserves_energy():
    sim = _sim()
    shape = (8, 8)
    air = np.full(shape, 1.0, dtype=np.float32)
    fuel = np.full((1, *shape), 0.5, dtype=np.float32)
    fuel_temps = np.zeros((1, *shape), dtype=np.float32)
    mass = np.ones(shape, dtype=np.float32)

    e0 = _thermal_energy(sim, air, fuel_temps, fuel, mass)
    new_air, new_ft = sim.exchange_fuel_air_heat(air, fuel_temps, fuel, mass)
    e1 = _thermal_energy(sim, new_air, new_ft, fuel, mass)

    np.testing.assert_allclose(e1, e0, rtol=1e-5)


def test_heat_exchange_relaxes_without_overshoot():
    sim = _sim()
    shape = (8, 8)
    air = np.full(shape, 1.0, dtype=np.float32)         # hot air
    fuel = np.full((1, *shape), 0.5, dtype=np.float32)
    fuel_temps = np.zeros((1, *shape), dtype=np.float32)  # cold fuel
    mass = np.ones(shape, dtype=np.float32)

    # Mass-weighted equilibrium the pair relaxes toward.
    c_air, c_fuel = mass, sim.heat_capacities[0] * fuel[0]
    t_eq = (c_air * air + c_fuel * fuel_temps[0]) / (c_air + c_fuel)

    new_air, new_ft = sim.exchange_fuel_air_heat(air, fuel_temps, fuel, mass)

    # Air cools toward T_eq but does not cross it; fuel warms toward T_eq but does not cross it.
    assert np.all(new_air <= air + 1e-6)
    assert np.all(new_air >= t_eq - 1e-6)
    assert np.all(new_ft[0] >= fuel_temps[0] - 1e-6)
    assert np.all(new_ft[0] <= t_eq + 1e-6)


def test_massless_ash_is_stable():
    """Fuel at ~zero mass must not produce NaN/Inf and must leave the air unchanged."""
    sim = _sim()
    shape = (8, 8)
    air = np.full(shape, 1.0, dtype=np.float32)
    fuel = np.zeros((1, *shape), dtype=np.float32)        # massless ash
    fuel_temps = np.full((1, *shape), 5.0, dtype=np.float32)
    mass = np.ones(shape, dtype=np.float32)

    new_air, new_ft = sim.exchange_fuel_air_heat(air, fuel_temps, fuel, mass)

    assert np.isfinite(new_air).all() and np.isfinite(new_ft).all()
    np.testing.assert_allclose(new_air, air, atol=1e-6)
