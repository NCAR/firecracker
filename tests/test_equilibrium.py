"""
The convective rest state.

equilibrium_mass() claims to return the mass field that makes the transport
potential Phi uniform — i.e. a stationary state. These tests verify that a map
initialized at equilibrium does not drift, while an off-equilibrium one does.
The same scenarios are rendered by tools/visualize.py.
"""

import numpy as np

from scenarios import equilibrium, off_equilibrium


def test_equilibrium_mass_is_stationary(make_env):
    env = make_env(*equilibrium())
    before = env._mass.copy()
    env.step(0)

    # grad(Phi) ~ 0 at rest, so the mass field barely moves (only float-level drift).
    assert np.abs(env._mass - before).max() < 1e-4


def test_off_equilibrium_mass_drifts(make_env):
    """Uniform mass over sloped terrain is not a rest state, so it must move."""
    env = make_env(*off_equilibrium())
    before = env._mass.copy()
    env.step(0)

    assert np.abs(env._mass - before).max() > 1e-3
    # Mass is only moved around, never created or destroyed.
    np.testing.assert_allclose(env._mass.sum(), before.sum(), rtol=1e-4)
