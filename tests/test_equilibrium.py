"""
Terrain-shaped ambient wind (Phase 2 shallow boundary-layer core).

The synoptic wind enters at the upwind sponge and is reshaped by terrain as it crosses
the domain. Over flat ground it holds the ambient value; over a ridge it shows the
orographic signature -- windward blocking (slow) and lee/downslope acceleration (fast),
the foehn / downslope-wind pattern. Once settled it is steady until perturbed.
"""

import numpy as np
import torch

from conftest import to_numpy
from scenarios import si_flat, si_ridge


def _speed(env) -> np.ndarray:
    return to_numpy(torch.sqrt(env._x_wind_vel ** 2 + env._y_wind_vel ** 2))


def test_flat_world_holds_ambient_wind(make_env):
    """Over flat terrain the interior wind stays uniform at the synoptic ambient."""
    ambient = (12.0, -5.0)
    env = make_env(*si_flat(grid=48, ambient=ambient))
    for _ in range(150):
        env.step(0)

    speed = np.hypot(*ambient)
    interior = _speed(env)[12:-12, 12:-12]   # away from the sponge belt
    # Near the ambient (a touch below, from weak surface friction) and essentially uniform.
    np.testing.assert_allclose(interior.mean(), speed, rtol=2e-2)
    assert interior.std() < 0.1


def test_ridge_blocks_windward_and_accelerates_lee(make_env):
    """A ridge across the wind: slower on the windward slope, faster on the lee slope."""
    env = make_env(*si_ridge(grid=48, ambient=(15.0, 0.0), peak_m=400.0))
    for _ in range(800):
        env.step(0)

    sp = _speed(env)
    row, c = sp.shape[0] // 2, sp.shape[1] // 2
    windward = sp[row, c - 8]    # upwind slope
    lee = sp[row, c + 8]         # downwind slope
    assert lee > windward + 1.0          # clear orographic asymmetry
    assert lee > 15.0                    # lee/downslope acceleration above ambient


def test_wind_is_steady_until_perturbed(make_env):
    """Once spun up, the wind field is (near-)constant step-to-step."""
    env = make_env(*si_ridge(grid=48, ambient=(15.0, 0.0), peak_m=400.0))
    for _ in range(800):
        env.step(0)

    before = _speed(env)
    for _ in range(20):
        env.step(0)
    assert np.abs(_speed(env) - before).max() < 1e-2
