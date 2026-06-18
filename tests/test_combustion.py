"""
Smooth Arrhenius, oxygen-limited combustion (Phase 4).

update_fire is a continuous reaction rate k = A*exp(-E_a/(R*T_fuel)) [1/s], throttled by the
local oxygen fraction, with no hard ignition threshold: cold fuel is inert only because the
exponential underflows at ambient temperature. Burned mass releases its heat of combustion HHV,
split between fuel and air by burn_heat_fuel_fraction; oxygen [kg/m^3] couples to the areal burn
[kg/m^2] through the shallow combustion mixing depth d_mix. These tests pin the inert-cold limit,
ignition under heat, the oxygen throttle, the stoichiometric O2 budget, the HHV energy split, and
the Arrhenius monotonicity in temperature. Rates are checked per step in regimes chosen so the
fuel cap and the oxygen cap do not bind (those limits are exercised separately).
"""

import numpy as np
import torch

import physics_constants as pc
from conftest import make_config
from simulation import Simulation


def _sim() -> Simulation:
    # grass falls to the _ft_tensor combustion defaults (A=1e8, E_a=1e5, HHV=1.6e7,
    # s=1.4, cp=2000); d_mix defaults to 30 m and dt to 0.1 s.
    return Simulation(make_config(fuel_type_names=("grass",)))


def _f(sim, value, shape=(8, 8)) -> torch.Tensor:
    return torch.full(shape, float(value), dtype=sim.dtype, device=sim.device)


def _burn(sim, *, temp, oxygen, fuel=1.0, mass=1500.0):
    """Run one update_fire from a uniform state; return (air, fuel_t, fuel, oxygen) tensors."""
    shape = (8, 8)
    air = _f(sim, pc.T_REF)
    fuel_t = _f(sim, temp).unsqueeze(0)
    fuel_a = _f(sim, fuel).unsqueeze(0)
    oxy = _f(sim, oxygen)
    m = _f(sim, mass)
    return sim.update_fire(air, fuel_t, fuel_a, oxy, m)


def test_cold_fuel_is_inert():
    """At ambient temperature the Arrhenius rate underflows, so fuel barely burns over a long run."""
    sim = _sim()
    air = _f(sim, pc.T_REF)
    fuel_t = _f(sim, pc.T_REF).unsqueeze(0)
    fuel = _f(sim, 1.0).unsqueeze(0)
    oxy = _f(sim, pc.O2_DENSITY_REF)
    mass = _f(sim, 1500.0)

    for _ in range(500):
        air, fuel_t, fuel, oxy = sim.update_fire(air, fuel_t, fuel, oxy, mass)

    np.testing.assert_allclose(fuel.cpu().numpy(), 1.0, atol=1e-6)   # essentially no consumption
    assert torch.isfinite(fuel_t).all()


def test_hot_fuel_ignites_and_releases_heat():
    """Fuel well above pyrolysis burns: fuel drops, oxygen is drawn down, and heat is released."""
    sim = _sim()
    _, fuel_t, fuel, oxy = _burn(sim, temp=700.0, oxygen=pc.O2_DENSITY_REF)

    assert float(fuel.mean()) < 1.0                       # fuel consumed
    assert float(oxy.mean()) < pc.O2_DENSITY_REF          # oxygen consumed
    assert float(fuel_t.mean()) > 700.0                   # combustion heated the fuel


def test_oxygen_throttles_and_extinguishes():
    """With no oxygen the rate vanishes; half the oxygen burns about half the fuel."""
    sim = _sim()

    # No oxygen -> no reaction at all.
    air0, ft0, fuel0, oxy0 = _burn(sim, temp=700.0, oxygen=0.0)
    np.testing.assert_allclose(fuel0.cpu().numpy(), 1.0, atol=1e-12)
    np.testing.assert_allclose(air0.cpu().numpy(), pc.T_REF, atol=1e-12)
    np.testing.assert_allclose(oxy0.cpu().numpy(), 0.0, atol=1e-12)

    # Rate is linear in the oxygen fraction (in the unsaturated regime): half O2 -> half burn.
    _, _, fuel_full, _ = _burn(sim, temp=600.0, oxygen=pc.O2_DENSITY_REF)
    _, _, fuel_half, _ = _burn(sim, temp=600.0, oxygen=0.5 * pc.O2_DENSITY_REF)
    d_full = 1.0 - float(fuel_full.mean())
    d_half = 1.0 - float(fuel_half.mean())
    np.testing.assert_allclose(d_half, 0.5 * d_full, rtol=1e-6)


def test_oxygen_limited_burn_is_clamped():
    """When oxygen runs short it limits fuel too: the reaction scales to the O2 available."""
    sim = _sim()
    o2_start = 0.01                                  # far less than a full hot burn would demand
    _, _, fuel, oxy = _burn(sim, temp=1000.0, oxygen=o2_start, fuel=1.0)

    fuel_burned = 1.0 - float(fuel.mean())
    s, d_mix = float(sim.stoich_oxygen[0]), sim.combustion_mixing_depth

    assert fuel_burned < 1.0                         # NOT a full burn -- oxygen capped it
    np.testing.assert_allclose(float(oxy.mean()), 0.0, atol=1e-9)   # all oxygen consumed
    # Fuel actually burned matches the oxygen that was available (stoichiometric, not over-drawn).
    np.testing.assert_allclose(s * fuel_burned / d_mix, o2_start, rtol=1e-6)


def test_stoichiometric_oxygen_budget():
    """Oxygen drawn (density * mixing depth) equals s * fuel burned, when neither cap binds."""
    sim = _sim()
    _, _, fuel, oxy = _burn(sim, temp=640.0, oxygen=pc.O2_DENSITY_REF)

    fuel_burned = 1.0 - float(fuel.mean())
    o2_drawn_areal = (pc.O2_DENSITY_REF - float(oxy.mean())) * sim.combustion_mixing_depth
    s = float(sim.stoich_oxygen[0])
    np.testing.assert_allclose(o2_drawn_areal, s * fuel_burned, rtol=1e-6)
    assert fuel_burned > 0.0


def test_hhv_energy_split():
    """Released heat HHV*Dfuel is split between air and fuel by burn_heat_fuel_fraction."""
    sim = _sim()
    mass = 1500.0
    air, fuel_t, fuel, _ = _burn(sim, temp=640.0, oxygen=pc.O2_DENSITY_REF, mass=mass)

    fuel_burned = 1.0 - float(fuel.mean())
    hhv = float(sim.heat_of_combustion[0])
    cp_fuel = float(sim.fuel_specific_heat[0])
    f = sim.burn_heat_fuel_fraction

    # Air share: m*c_p_air*dT_air = (1-f)*HHV*Dfuel.
    air_energy = mass * pc.CP_AIR * (float(air.mean()) - pc.T_REF)
    np.testing.assert_allclose(air_energy, (1.0 - f) * hhv * fuel_burned, rtol=1e-6)

    # Fuel share: c_p_fuel*fuel_new*dT_fuel = f*HHV*Dfuel.
    fuel_energy = cp_fuel * float(fuel.mean()) * (float(fuel_t.mean()) - 640.0)
    np.testing.assert_allclose(fuel_energy, f * hhv * fuel_burned, rtol=1e-6)


def test_arrhenius_monotonic_in_temperature():
    """Hotter fuel burns faster (within the unsaturated regime)."""
    sim = _sim()
    burned = []
    for temp in (560.0, 600.0, 640.0):
        _, _, fuel, _ = _burn(sim, temp=temp, oxygen=pc.O2_DENSITY_REF)
        burned.append(1.0 - float(fuel.mean()))

    assert burned[0] < burned[1] < burned[2]
    assert burned[0] > 0.0
