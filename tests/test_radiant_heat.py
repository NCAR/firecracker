"""
Flame radiant transfer (Phase 5) -- the fire-spread mechanism.

apply_radiant_heat does NOT emit or cool: the fuel's grey-body longwave eps*sigma*T_f^4 is
already radiated (and removed) by apply_radiation, which owns the self-limiting T^4 sink. This
stage only *redistributes* the super-ambient part eps*sigma*(T_f^4 - temp_eq^4)*dt onto nearby
fuel by the inverse-square kernel, so a flame ignites its neighbours. These tests pin that down:
a quiescent world at temp_eq does not spread, a lone hot cell warms its neighbours while its own
temperature is untouched, fuel below ambient is inert, flux is never deposited into fuel-free
cells, and the energy delivered to neighbours never exceeds the super-ambient emission (it is a
redistribution, not an amplifier). Fire actually propagating across full step_fields ticks is
covered by test_spread.
"""

import numpy as np
import torch

import physics_constants as pc
from conftest import make_config
from simulation import Simulation


def _sim(kernel_radius_m: float = 100.0, sky_escape: float = 0.2) -> Simulation:
    config = make_config(fuel_type_names=("grass",))
    config["radiant_heat"] = {
        "kernel_radius_m": kernel_radius_m,
        "sky_escape_fraction": sky_escape,
    }
    return Simulation(config)


def _full(sim, shape, value) -> torch.Tensor:
    return torch.full(shape, float(value), dtype=sim.dtype, device=sim.device)


def _emitted_excess(sim, fuel_temps, fuel, temp_eq) -> float:
    """Total super-ambient emission [J/m^2] the redistribution draws from (a_fuel-weighted)."""
    _, a_fuel = sim._cover_fractions(fuel)
    excess = (
        a_fuel * sim.emissivity * pc.STEFAN_BOLTZMANN
        * (fuel_temps ** 4 - temp_eq.unsqueeze(-3) ** 4).clamp(min=0.0) * sim.dt
    )
    return float(excess.sum())


def _deposited_energy(sim, fuel, dT) -> float:
    """Thermal energy [J/m^2] actually delivered to neighbours = sum(C_fuel * dT)."""
    return float((sim.fuel_specific_heat * fuel * dT).sum())


def test_equilibrium_does_not_spread():
    """All fuel at the rest profile -> zero super-ambient excess -> nothing moves."""
    sim = _sim()
    shape = (16, 16)
    fuel = _full(sim, (1, *shape), 1.0)
    temp_eq = _full(sim, shape, pc.T_REF)
    fuel_temps = temp_eq.clone().unsqueeze(0)

    new_ft, flux = sim.apply_radiant_heat(fuel_temps, fuel, temp_eq)

    np.testing.assert_allclose(new_ft.cpu().numpy(), fuel_temps.cpu().numpy(), atol=1e-9)
    np.testing.assert_allclose(flux.cpu().numpy(), 0.0, atol=1e-12)


def test_below_equilibrium_is_inert():
    """Fuel colder than ambient emits no flame radiation (the excess is clamped at zero)."""
    sim = _sim()
    shape = (16, 16)
    fuel = _full(sim, (1, *shape), 1.0)
    temp_eq = _full(sim, shape, pc.T_REF)
    fuel_temps = _full(sim, (1, *shape), pc.T_REF - 30.0)   # below the rest profile

    new_ft, flux = sim.apply_radiant_heat(fuel_temps, fuel, temp_eq)

    np.testing.assert_allclose(new_ft.cpu().numpy(), fuel_temps.cpu().numpy(), atol=1e-9)
    np.testing.assert_allclose(flux.cpu().numpy(), 0.0, atol=1e-12)


def test_redistributes_to_neighbours_without_self_heating():
    """A lone hot cell warms its neighbours; its own temperature is untouched (no self-absorption)."""
    sim = _sim(kernel_radius_m=30.0)   # 3-cell reach so the deposit is concentrated and visible
    shape = (16, 16)
    fuel = _full(sim, (1, *shape), 1.0)
    temp_eq = _full(sim, shape, pc.T_REF)
    fuel_temps = temp_eq.clone().unsqueeze(0)
    fuel_temps[0, 8, 8] = 1200.0       # one hot cell, the rest at ambient

    new_ft, flux = sim.apply_radiant_heat(fuel_temps, fuel, temp_eq)

    assert float(new_ft[0, 8, 8]) == 1200.0        # emitter untouched: kernel centre is zero and
                                                    # its neighbours (at temp_eq) emit nothing back
    assert float(new_ft[0, 8, 9]) > pc.T_REF       # an adjacent neighbour was warmed
    assert float(flux[8, 9]) > 0.0
    assert torch.isfinite(new_ft).all()


def test_no_deposit_into_fuel_free_cells():
    """Absorbed flux only heats cells that actually hold fuel; bare cells stay put."""
    sim = _sim(kernel_radius_m=30.0)
    shape = (16, 16)
    fuel = _full(sim, (1, *shape), 1.0)
    fuel[0, 8, 9] = 0.0                # a bare cell right next to the emitter
    temp_eq = _full(sim, shape, pc.T_REF)
    fuel_temps = temp_eq.clone().unsqueeze(0)
    fuel_temps[0, 8, 8] = 1200.0

    new_ft, _ = sim.apply_radiant_heat(fuel_temps, fuel, temp_eq)

    np.testing.assert_allclose(float(new_ft[0, 8, 9]), pc.T_REF, atol=1e-12)  # no fuel -> no heating
    assert float(new_ft[0, 7, 8]) > pc.T_REF                                  # a fuelled neighbour still warms


def test_delivered_energy_bounded_by_emission():
    """Neighbours receive no more than the super-ambient emission, less sky escape and edge loss."""
    sim = _sim()
    shape = (24, 24)
    fuel = _full(sim, (1, *shape), 1.0)
    temp_eq = _full(sim, shape, pc.T_REF)
    fuel_temps = temp_eq.clone().unsqueeze(0)
    fuel_temps[0, 12, 12] = 1400.0

    new_ft, _ = sim.apply_radiant_heat(fuel_temps, fuel, temp_eq)
    dT = new_ft - fuel_temps

    emitted = _emitted_excess(sim, fuel_temps, fuel, temp_eq)
    delivered = _deposited_energy(sim, fuel, dT)

    assert delivered > 0.0                                   # energy did reach neighbours
    assert delivered <= (1.0 - sim.sky_escape_fraction) * emitted + 1e-6   # never amplifies
    assert delivered < emitted                               # sky escape (+ edge loss) is a real loss
