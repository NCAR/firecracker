"""
Stability / correctness of the relaxational transport.

Conservation (test_conservation.py) is necessary but not sufficient: a scheme can
conserve mass exactly while sloshing it into a checkerboard. These tests assert
the dynamics actually *behave* — gradients equalize smoothly toward uniform, no
odd/even stripes grow, and oxygen keeps moving with the wind instead of freezing
at a piled steady state. They would have caught the checkerboard / oxygen-freeze
bugs that conservation alone missed.
"""

import numpy as np

from conftest import to_numpy
from scenarios import mass_gradient, oxygen_saturation


def test_mass_gradient_stays_monotone(make_env):
    """A monotone pressure ramp must relax while staying monotone (no stripes).

    The checkerboard instability shows up as adjacent high/low cells, i.e. a break
    in monotonicity. Requiring the columns to stay non-increasing left-to-right is a
    direct, interpretable "no checkerboard" check.
    """
    env = make_env(*mass_gradient())
    for _ in range(300):
        env.step(0)

    # Every row must remain non-increasing across columns (small float tolerance).
    assert np.all(np.diff(to_numpy(env._mass), axis=1) <= 1e-5)


def test_mass_gradient_equalizes(make_env):
    """The ramp must measurably flatten toward uniform over time."""
    env = make_env(*mass_gradient())
    var0 = float(env._mass.var())
    for _ in range(400):
        env.step(0)
    var1 = float(env._mass.var())

    assert var1 < 0.1 * var0                       # substantially equalized
    np.testing.assert_allclose(float(env._mass.sum()), env._mass.numel(), rtol=1e-4)  # mean-1 preserved


def test_high_rate_transport_is_stable(make_env):
    """Bumping the transport rate must speed up relaxation, not blow it up.

    With the wrong (advective) CFL the diffusion went unstable above rate ~0.25 and
    *concentrated* mass instead of equalizing it. Run well past that, at rate 0.5.
    """
    config, m = mass_gradient()
    config["convection"] = {"pressure_transport_rate": 0.5, "buoyancy_transport_rate": 0.5}
    env = make_env(config, m)
    var0 = float(env._mass.var())
    for _ in range(200):
        env.step(0)

    assert np.isfinite(to_numpy(env._mass)).all()
    assert float(env._mass.min()) > 0.0          # no cell evacuated by anti-diffusion
    assert float(env._mass.var()) < 0.05 * var0  # equalized, not concentrated


def test_oxygen_does_not_freeze(make_env):
    """Oxygen must keep moving (the bug snapped it to a piled steady state in 1 step)."""
    env = make_env(*oxygen_saturation())
    for _ in range(5):
        env.step(0)
    early = to_numpy(env._oxygen)
    for _ in range(45):
        env.step(0)
    later = to_numpy(env._oxygen)

    # It should have changed appreciably between step 5 and step 50.
    assert np.abs(later - early).max() > 1e-3


def test_oxygen_step_diffuses(make_env):
    """The sharp 0.4/1.0 oxygen boundary must smooth out toward uniform, staying capped."""
    env = make_env(*oxygen_saturation())
    o0 = float(env._oxygen.sum())
    var0 = float(env._oxygen.var())
    for _ in range(200):
        env.step(0)

    assert float(env._oxygen.var()) < 0.3 * var0   # boundary has blurred toward uniform
    assert float(env._oxygen.max()) <= 1.0 + 1e-6  # still capped
    np.testing.assert_allclose(float(env._oxygen.sum()), o0, rtol=1e-4)  # conserved
