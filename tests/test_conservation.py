"""
Conservation of the advection core (Phase 2).

The full model uses open (sponge) boundaries, which deliberately add/remove air at the
edges, so domain totals are NOT conserved in normal operation. To test the conservative
advection scheme in isolation we close the domain: with the boundary sponge and surface
friction off, the solver is pure periodic shallow-water transport, which conserves mass,
energy (sum m*c_p*T) and oxygen to round-off.
"""

import numpy as np

import physics_constants as pc
from conftest import DEFAULT_GRID, make_config, to_numpy
from scenarios import si_hill

CP = pc.CP_AIR
CLOSED = {"sponge_strength": 0.0, "drag_coeff": 0.0}   # no edge sponge, no friction -> closed


def _totals(env):
    m = env._mass
    return (float(m.sum()), float((m * CP * env._air_temperatures).sum()), float(env._oxygen.sum()))


def test_closed_core_conserves(make_env):
    """Closed (periodic) core: mass, energy and oxygen are conserved as the wind stirs them."""
    _, m = si_hill(ambient=(12.0, 4.0))
    env = make_env(make_config(DEFAULT_GRID, momentum=CLOSED), m)

    m0, e0, o0 = _totals(env)
    for _ in range(50):
        env.step(0)
    m1, e1, o1 = _totals(env)

    assert np.isfinite(to_numpy(env._air_temperatures)).all()
    np.testing.assert_allclose(m1, m0, rtol=1e-9)
    np.testing.assert_allclose(e1, e0, rtol=1e-6)
    np.testing.assert_allclose(o1, o0, rtol=1e-9)


def test_closed_core_keeps_mass_positive(make_env):
    """The conservative upwind transport never drives the layer mass negative."""
    _, m = si_hill(ambient=(15.0, 6.0), peak_m=500.0)
    env = make_env(make_config(DEFAULT_GRID, momentum=CLOSED), m)
    for _ in range(50):
        env.step(0)
    assert float(env._mass.min()) > 0.0
