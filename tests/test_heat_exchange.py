"""
Operator-level tests for exchange_fuel_air_heat.

These call the Simulation method directly (no map/env) to pin down the exact
two-body relaxation claims: energy conservation, relaxation toward the
mass-weighted equilibrium without overshoot, and the massless-ash limit. Each
fuel now couples to its own air level (fuel_levels); the air side's heat
capacity is that level's thin-slab capacity (_air_level_capacity), not the full
column mass, so a single grass type exchanges only with the surface level.
"""

import numpy as np
import torch

import physics_constants as pc
from conftest import make_config
from simulation import Simulation


def _sim() -> Simulation:
    # grass gets the default specific_heat = 2000 J/(kg*K) and couples to the surface level (0).
    return Simulation(make_config(fuel_type_names=("grass",)))


def _full(sim, shape, value) -> torch.Tensor:
    return torch.full(shape, float(value), dtype=sim.dtype, device=sim.device)


def _air_stack(sim, shape, value) -> torch.Tensor:
    """A (L, H, W) air stack with every level at `value`."""
    return _full(sim, (sim.num_air_levels, *shape), value)


def _thermal_energy(sim, air, fuel_temps, fuel) -> float:
    # SI heat capacities: each air level's thin-slab capacity C = rho*depth*c_p (_air_level_capacity),
    # summed over levels, plus the fuel fuel*c_p_fuel.
    c_fuel = sim.fuel_specific_heat * fuel   # (N, 1, 1) broadcast over (N, H, W)
    air_energy = (sim._air_level_capacity * air).sum()   # (L,1,1) * (L,H,W)
    return float(air_energy + (c_fuel * fuel_temps).sum())


def test_heat_exchange_conserves_energy():
    sim = _sim()
    shape = (8, 8)
    air = _air_stack(sim, shape, 1.0)
    fuel = _full(sim, (1, *shape), 0.5)
    fuel_temps = torch.zeros((1, *shape), dtype=sim.dtype, device=sim.device)

    e0 = _thermal_energy(sim, air, fuel_temps, fuel)
    new_air, new_ft = sim.exchange_fuel_air_heat(air, fuel_temps, fuel)
    e1 = _thermal_energy(sim, new_air, new_ft, fuel)

    np.testing.assert_allclose(e1, e0, rtol=1e-5)


def test_heat_exchange_relaxes_without_overshoot():
    sim = _sim()
    shape = (8, 8)
    air = _air_stack(sim, shape, 1.0)                              # hot air (all levels)
    fuel = _full(sim, (1, *shape), 0.5)
    fuel_temps = torch.zeros((1, *shape), dtype=sim.dtype, device=sim.device)  # cold fuel

    # Mass-weighted equilibrium the surface pair relaxes toward: grass couples to level 0, whose
    # capacity is the surface slab capacity _air_level_capacity[0], not the column mass.
    c_air = sim._air_level_capacity[0]
    c_fuel = sim.fuel_specific_heat[0] * fuel[0]
    t_eq = (c_air * air[0] + c_fuel * fuel_temps[0]) / (c_air + c_fuel)

    new_air, new_ft = sim.exchange_fuel_air_heat(air, fuel_temps, fuel)

    # Surface air cools toward T_eq but does not cross it; fuel warms toward T_eq but does not cross it.
    assert torch.all(new_air[0] <= air[0] + 1e-6)
    assert torch.all(new_air[0] >= t_eq - 1e-6)
    assert torch.all(new_ft[0] >= fuel_temps[0] - 1e-6)
    assert torch.all(new_ft[0] <= t_eq + 1e-6)
    # The canopy/above levels have no surface fuel, so they are untouched.
    assert torch.allclose(new_air[1:], air[1:])


def test_massless_ash_is_stable():
    """Fuel at ~zero mass must not produce NaN/Inf and must leave the air unchanged."""
    sim = _sim()
    shape = (8, 8)
    air = _air_stack(sim, shape, 1.0)
    fuel = torch.zeros((1, *shape), dtype=sim.dtype, device=sim.device)  # massless ash
    fuel_temps = _full(sim, (1, *shape), 5.0)

    new_air, new_ft = sim.exchange_fuel_air_heat(air, fuel_temps, fuel)

    assert torch.isfinite(new_air).all() and torch.isfinite(new_ft).all()
    np.testing.assert_allclose(new_air.cpu().numpy(), air.cpu().numpy(), atol=1e-6)
