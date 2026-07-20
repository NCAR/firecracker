"""
Conservation of the dynamical core (Phase 2), split into closed vs open domains.

The solver runs on an *open* domain: a boundary sponge relaxes the edge belt toward the synoptic
free-stream, deliberately adding air upwind and removing it downwind, so domain totals are NOT
conserved in normal operation. Two regimes are pinned here:

  * CLOSED -- with the sponge and surface friction off the solver is pure periodic shallow-water
    transport, which conserves mass, energy (sum m*c_p*T) and oxygen to round-off.
  * OPEN -- with the sponge live the domain is a throughflow: a perturbation is carried downwind
    and flushed out (the thermal anomaly decays toward zero), where the closed core would conserve
    it. (A quiescent world at the rest profile instead holds station -- see
    tests/test_radiation.py::test_quiescent_world_holds_station, the open-domain baseline.)
"""

import numpy as np

import physics_constants as pc
from conftest import DEFAULT_GRID, make_config, to_numpy
from scenarios import si_flat, si_hill, disc

CP = pc.CP_AIR
CLOSED = {"sponge_strength": 0.0, "drag_coeff": 0.0}   # no edge sponge, no friction -> closed


def _totals(env):
    m = env._mass
    return (float(m.sum()), float((m * CP * env._air_temperatures.select(-3, 0)).sum()), float(env._oxygen.sum()))


# ---------------------------------------------------------------------------
# Closed domain: exact conservation
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Open domain: throughflow budget (sponge inflow/outflow)
# ---------------------------------------------------------------------------

def _hot_blob_flat(grid, ambient, col):
    """Flat SI rest state with a warm air blob (an anomaly above the T_REF rest profile)."""
    _, m = si_flat(grid=grid, ambient=ambient)
    blob = disc(grid, grid // 2, col, 3, value=pc.T_REF + 80.0, base=0.0)
    m.air_temperatures = np.where(blob > 0.0, blob, m.air_temperatures).astype(np.float32)
    return m


def _thermal_anomaly(env):
    """Heat held above the flat rest profile: sum(m * c_p * (T - T_REF)) [J/m^2 * cells]."""
    return float((env._mass * CP * (env._air_temperatures.select(-3, 0) - pc.T_REF)).sum())


def test_open_domain_flushes_anomaly_while_closed_conserves_it(make_env):
    """The open sponge carries a warm anomaly out of the domain; the closed core conserves it.

    Same flat world and warm blob stepped two ways: with the live boundary sponge the interior
    heat is advected to the edges and removed (the thermal anomaly decays toward zero), whereas the
    closed periodic core conserves it exactly -- the defining difference between the two regimes.
    """
    grid, ambient = 24, (10.0, 0.0)
    open_env   = make_env(make_config(grid), _hot_blob_flat(grid, ambient, grid // 2))
    closed_env = make_env(make_config(grid, momentum=CLOSED), _hot_blob_flat(grid, ambient, grid // 2))

    a0 = _thermal_anomaly(open_env)
    for _ in range(400):
        open_env.step(0)
        closed_env.step(0)

    assert np.isfinite(to_numpy(open_env._air_temperatures)).all()
    # Closed core conserves the anomaly; the open sponge flushes most of it out.
    np.testing.assert_allclose(_thermal_anomaly(closed_env), a0, rtol=1e-4)
    assert abs(_thermal_anomaly(open_env)) < 0.5 * abs(a0)


def test_open_domain_carries_anomaly_downwind(make_env):
    """A warm blob released upwind drifts downwind (+x) under a steady synoptic wind."""
    grid, ambient = 32, (12.0, 0.0)
    env = make_env(make_config(grid), _hot_blob_flat(grid, ambient, grid // 4))

    cols = np.arange(grid)

    def centroid_col(env):
        # Centroid over the observed interior (crop the padded sponge ring), matching `cols`.
        mass = to_numpy(env._crop(env._mass))
        air = to_numpy(env._crop(env._air_temperatures.select(-3, 0)))
        w = np.clip(mass * (air - pc.T_REF), 0.0, None)
        return float((w.sum(axis=0) * cols).sum() / w.sum())

    x0 = centroid_col(env)
    for _ in range(40):
        env.step(0)
    x1 = centroid_col(env)

    assert np.isfinite(to_numpy(env._air_temperatures)).all()
    assert x1 > x0 + 1.0   # the anomaly is advected toward the downwind edge
