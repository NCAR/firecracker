"""
Surface radiative energy balance (Phase 3, canopy-split model).

Sunlight is absorbed at the surface -- the air is transparent to shortwave -- and split
between bare ground and fuel by a vegetation cover fraction; each emits longwave by its area
share, and the air is warmed only by sensible exchange with the ground. The net flux S_net is
pinned per-cell to eps*sigma*temp_eq^4, so radiative equilibrium sits at temp_eq (= T_REF over
flat ground). These tests pin the fixed point, the heat-capacity ordering (fuel leads, ground
follows, air lags), the cover partition, the bare-ground limit, the analytic ground step, the
455 W/m^2 calibration, and the wiring through step_fields. Real radiative timescales are
~hours--days, so behaviour is checked per step / over short runs, not by stepping to equilibrium.
"""

import numpy as np
import torch

import physics_constants as pc
from conftest import to_numpy
from simulation import Simulation
from scenarios import make_config, si_flat


def _sim() -> Simulation:
    return Simulation(make_config(radiation=True))


def _f(sim: Simulation, value: float, shape=(8, 8)) -> torch.Tensor:
    return torch.full(shape, float(value), dtype=sim.dtype, device=sim.device)


def test_equilibrium_is_a_fixed_point():
    """With every body at temp_eq the net flux is zero, so nothing moves."""
    sim = _sim()
    temp_eq = _f(sim, pc.T_REF)
    mass = _f(sim, 1500.0)
    fuel = _f(sim, 1.0).unsqueeze(0)            # (1, H, W) grass present
    fuel_t = temp_eq.clone().unsqueeze(0)

    air, fuel_t_new, ground = sim.apply_radiation(
        temp_eq.clone(), temp_eq.clone(), fuel_t, fuel, mass, temp_eq
    )

    np.testing.assert_allclose(to_numpy(air), to_numpy(temp_eq), atol=1e-9)
    np.testing.assert_allclose(to_numpy(ground), to_numpy(temp_eq), atol=1e-9)
    np.testing.assert_allclose(to_numpy(fuel_t_new[0]), to_numpy(temp_eq), atol=1e-9)


def test_fuel_leads_ground_leads_air():
    """Cold-started bodies warm under the sun in heat-capacity order: fuel >> ground >> air."""
    sim = _sim()
    temp_eq = _f(sim, pc.T_REF)
    mass = _f(sim, 1500.0)
    fuel = _f(sim, 1.0).unsqueeze(0)

    start = pc.T_REF - 30.0      # all bodies below the radiative-equilibrium target
    air, ground = _f(sim, start), _f(sim, start)
    fuel_t = _f(sim, start).unsqueeze(0)

    for _ in range(300):
        air, fuel_t, ground = sim.apply_radiation(air, ground, fuel_t, fuel, mass, temp_eq)

    d_fuel = float(fuel_t[0].mean()) - start
    d_ground = float(ground.mean()) - start
    d_air = float(air.mean()) - start

    assert d_fuel > d_ground > d_air >= 0.0    # fine fuel fastest, deep air column slowest
    assert d_fuel > 1e-2                       # fuel response is actually meaningful
    # All stay below the target they relax toward (no overshoot past temp_eq).
    assert float(fuel_t.max()) <= pc.T_REF + 1e-9


def test_more_fuel_shades_the_ground():
    """Higher fuel load -> higher canopy cover -> less sun on the ground -> slower ground warming."""
    sim = _sim()
    temp_eq = _f(sim, pc.T_REF)
    mass = _f(sim, 1500.0)
    start = pc.T_REF - 30.0

    def ground_gain(fuel_load: float) -> float:
        fuel = _f(sim, fuel_load).unsqueeze(0)
        _, _, ground = sim.apply_radiation(
            _f(sim, start), _f(sim, start), _f(sim, start).unsqueeze(0), fuel, mass, temp_eq
        )
        return float(ground.mean()) - start

    sparse = ground_gain(0.1)   # mostly bare ground
    dense = ground_gain(3.0)    # near-full canopy
    assert sparse > dense > 0.0


def test_bare_cell_ground_absorbs_full_solar():
    """A fuel-free cell: cover -> 0, so the ground takes the full S_net and warms; fuel term is inert."""
    sim = _sim()
    temp_eq = _f(sim, pc.T_REF)
    mass = _f(sim, 1500.0)
    fuel = _f(sim, 0.0).unsqueeze(0)            # no vegetation

    start = pc.T_REF - 30.0
    air, fuel_t_new, ground = sim.apply_radiation(
        _f(sim, start), _f(sim, start), _f(sim, start).unsqueeze(0), fuel, mass, temp_eq
    )

    assert float(ground.mean()) > start         # ground heated by the full solar flux
    np.testing.assert_allclose(to_numpy(fuel_t_new[0]), start, atol=1e-9)  # no fuel -> no fuel heating
    assert np.isfinite(to_numpy(ground)).all()


def test_ground_step_matches_analytic():
    """One ground step equals T_g + dt*(a_g*S_net - a_g*eps*sigma*T_g^4 - k_ga*(T_g - T_a))/C_g."""
    sim = _sim()
    temp_eq = _f(sim, pc.T_REF)
    mass = _f(sim, 1500.0)
    fuel = _f(sim, 0.7).unsqueeze(0)
    air = _f(sim, pc.T_REF)
    ground = _f(sim, pc.T_REF + 40.0)
    fuel_t = _f(sim, pc.T_REF).unsqueeze(0)

    _, _, ground_new = sim.apply_radiation(air, ground, fuel_t, fuel, mass, temp_eq)

    eps, sigma = sim.emissivity, pc.STEFAN_BOLTZMANN
    cover = 1.0 - np.exp(-0.7 / sim.cover_fuel_scale)
    a_g = 1.0 - cover
    s_net = eps * sigma * pc.T_REF ** 4
    q_g = a_g * s_net - a_g * eps * sigma * float(ground[0, 0]) ** 4 - sim.ground_air_exchange * (
        float(ground[0, 0]) - pc.T_REF
    )
    expected = float(ground[0, 0]) + sim.dt * q_g / sim.ground_heat_capacity
    np.testing.assert_allclose(float(ground_new[0, 0]), expected, rtol=1e-12)


def test_calibration_net_flux_at_reference():
    """S_net at the reference temperature is the documented ~455 W/m^2 daytime flux."""
    sim = _sim()
    s_net = sim.emissivity * pc.STEFAN_BOLTZMANN * pc.T_REF ** 4
    assert abs(s_net - 455.0) < 5.0


def test_quiescent_world_holds_station(make_env):
    """A flat, windless world at its rest profile stays put through step_fields with radiation on."""
    _, m = si_flat(grid=16, ambient=(0.0, 0.0))
    env = make_env(make_config(16, radiation=True), m)

    before = to_numpy(env._air_temperatures).copy()
    for _ in range(100):
        env.step(0)

    after = to_numpy(env._air_temperatures)
    assert np.isfinite(after).all()
    np.testing.assert_allclose(after, before, atol=1e-2)
