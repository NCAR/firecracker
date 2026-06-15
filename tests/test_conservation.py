"""
Conservation invariants of the full step() pipeline.

All scenarios here run with fire and relaxation disabled and zero fuel, so the
only active physics is convective transport, diffusion, and oxygen advection —
the parts that claim *exact* conservation under no-flux walls. The same
scenarios are rendered by tools/visualize.py.
"""

import numpy as np

from conftest import total_air_energy, total_mass, total_oxygen
from scenarios import corner_blob, hot_blob, oxygen_saturation, uniform


def test_uniform_field_is_stationary(make_env):
    """With no gradients anywhere, nothing should move: the step is a no-op."""
    env = make_env(*uniform())

    before = (env._mass.copy(), env._air_temperatures.copy(), env._oxygen.copy())
    for _ in range(5):
        env.step(0)

    np.testing.assert_allclose(env._mass, before[0], atol=1e-6)
    np.testing.assert_allclose(env._air_temperatures, before[1], atol=1e-6)
    np.testing.assert_allclose(env._oxygen, before[2], atol=1e-6)


def test_transport_conserves_mass_and_energy(make_env):
    """A hot blob convects and diffuses, but total mass and energy are preserved."""
    env = make_env(*hot_blob())

    m0, e0 = total_mass(env), total_air_energy(env)
    for _ in range(25):
        env.step(0)

    assert np.isfinite(env._air_temperatures).all()
    np.testing.assert_allclose(total_mass(env), m0, rtol=1e-4)
    np.testing.assert_allclose(total_air_energy(env), e0, rtol=1e-4)


def test_no_flux_walls_lose_nothing(make_env):
    """A hot blob jammed into a corner must not leak through the boundary.

    Conservation of the total *is* the no-flux test: if the walls leaked, the
    totals would drop.
    """
    env = make_env(*corner_blob())

    m0, e0 = total_mass(env), total_air_energy(env)
    for _ in range(40):
        env.step(0)

    np.testing.assert_allclose(total_mass(env), m0, rtol=1e-4)
    np.testing.assert_allclose(total_air_energy(env), e0, rtol=1e-4)


def test_oxygen_advection_conserves_and_saturates(make_env):
    """Oxygen advecting into already-full cells stays conserved and capped at 1.0."""
    env = make_env(*oxygen_saturation())

    o0 = total_oxygen(env)
    for _ in range(20):
        env.step(0)

    assert env._oxygen.max() <= 1.0 + 1e-6
    np.testing.assert_allclose(total_oxygen(env), o0, rtol=1e-4)
