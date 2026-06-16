"""
Operator-level tests for exchange_fuel_air_heat.

These call the Simulation method directly (no map/env) to pin down the exact
two-body relaxation claims: energy conservation, relaxation toward the
mass-weighted equilibrium without overshoot, and the massless-ash limit.
"""

import numpy as np
import torch

from conftest import make_config
from simulation import Simulation


def _sim() -> Simulation:
    # grass gets default heat_capacity = 1.0.
    return Simulation(make_config(fuel_type_names=("grass",)))


def _full(sim, shape, value) -> torch.Tensor:
    return torch.full(shape, float(value), dtype=sim.dtype, device=sim.device)


def _thermal_energy(sim, air, fuel_temps, fuel, mass) -> float:
    c_fuel = sim.heat_capacities * fuel   # heat_capacities is already (N, 1, 1)
    return float((mass * air).sum() + (c_fuel * fuel_temps).sum())


def test_heat_exchange_conserves_energy():
    sim = _sim()
    shape = (8, 8)
    air = _full(sim, shape, 1.0)
    fuel = _full(sim, (1, *shape), 0.5)
    fuel_temps = torch.zeros((1, *shape), dtype=sim.dtype, device=sim.device)
    mass = _full(sim, shape, 1.0)

    e0 = _thermal_energy(sim, air, fuel_temps, fuel, mass)
    new_air, new_ft = sim.exchange_fuel_air_heat(air, fuel_temps, fuel, mass)
    e1 = _thermal_energy(sim, new_air, new_ft, fuel, mass)

    np.testing.assert_allclose(e1, e0, rtol=1e-5)


def test_heat_exchange_relaxes_without_overshoot():
    sim = _sim()
    shape = (8, 8)
    air = _full(sim, shape, 1.0)                                    # hot air
    fuel = _full(sim, (1, *shape), 0.5)
    fuel_temps = torch.zeros((1, *shape), dtype=sim.dtype, device=sim.device)  # cold fuel
    mass = _full(sim, shape, 1.0)

    # Mass-weighted equilibrium the pair relaxes toward.
    c_air, c_fuel = mass, sim.heat_capacities[0] * fuel[0]
    t_eq = (c_air * air + c_fuel * fuel_temps[0]) / (c_air + c_fuel)

    new_air, new_ft = sim.exchange_fuel_air_heat(air, fuel_temps, fuel, mass)

    # Air cools toward T_eq but does not cross it; fuel warms toward T_eq but does not cross it.
    assert torch.all(new_air <= air + 1e-6)
    assert torch.all(new_air >= t_eq - 1e-6)
    assert torch.all(new_ft[0] >= fuel_temps[0] - 1e-6)
    assert torch.all(new_ft[0] <= t_eq + 1e-6)


def test_massless_ash_is_stable():
    """Fuel at ~zero mass must not produce NaN/Inf and must leave the air unchanged."""
    sim = _sim()
    shape = (8, 8)
    air = _full(sim, shape, 1.0)
    fuel = torch.zeros((1, *shape), dtype=sim.dtype, device=sim.device)  # massless ash
    fuel_temps = _full(sim, (1, *shape), 5.0)
    mass = _full(sim, shape, 1.0)

    new_air, new_ft = sim.exchange_fuel_air_heat(air, fuel_temps, fuel, mass)

    assert torch.isfinite(new_air).all() and torch.isfinite(new_ft).all()
    np.testing.assert_allclose(new_air.cpu().numpy(), air.cpu().numpy(), atol=1e-6)
