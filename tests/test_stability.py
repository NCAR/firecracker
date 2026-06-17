"""
Stability of the shallow boundary-layer wind solver (Phase 2).

The open-domain solver pushes strong winds over steep terrain with an explicit scheme,
so these tests assert it stays finite, keeps the layer mass positive, and does not grow
a grid-scale (checkerboard) oscillation -- including when the layer is made shallower
(stronger terrain constriction) or the wind is strong.
"""

import numpy as np

from conftest import to_numpy
from scenarios import make_config, build_map, si_hill, si_ridge
from scenarios import _si_state, gaussian_hill


def test_strong_wind_steep_terrain_is_stable(make_env):
    """A fast synoptic wind over a tall hill stays finite with positive layer mass."""
    env = make_env(*si_hill(ambient=(28.0, 10.0), peak_m=500.0))
    for _ in range(400):
        env.step(0)

    assert np.isfinite(to_numpy(env._x_wind_vel)).all()
    assert np.isfinite(to_numpy(env._air_temperatures)).all()
    assert float(env._mass.min()) > 0.0


def test_no_grid_scale_checkerboard(make_env):
    """The wind field stays smooth: no growing odd/even (Nyquist) oscillation."""
    env = make_env(*si_ridge(ambient=(25.0, 0.0), peak_m=450.0))
    for _ in range(400):
        env.step(0)

    u, v = to_numpy(env._x_wind_vel), to_numpy(env._y_wind_vel)
    sp = np.sqrt(u ** 2 + v ** 2)
    lap = (np.roll(sp, 1, 0) + np.roll(sp, -1, 0) + np.roll(sp, 1, 1) + np.roll(sp, -1, 1) - 4 * sp)
    assert np.abs(lap).max() < 0.5 * float(sp.std()) + 1.0


def test_shallow_layer_is_stable(make_env):
    """A shallower layer (stronger terrain constriction, faster channeling) stays stable."""
    grid = 32
    terrain = gaussian_hill(grid, 400.0)
    air, mass, oxygen = _si_state(terrain)
    config = make_config(grid, momentum={"layer_depth_ref": 600.0})
    env = make_env(config, build_map(grid, terrain=terrain, air=air, mass=mass,
                                     oxygen=oxygen, ambient_wind=(20.0, 0.0)))
    for _ in range(300):
        env.step(0)

    assert np.isfinite(to_numpy(env._mass)).all()
    assert float(env._mass.min()) > 0.0
