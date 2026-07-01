"""
Terrain-shaped ambient wind (Phase 2 shallow boundary-layer core).

The synoptic wind enters at the upwind sponge and is reshaped by terrain as it crosses
the domain. Over flat ground it holds the ambient value; over a ridge it shows the
orographic signature -- windward blocking (slow) and lee/downslope acceleration (fast),
the foehn / downslope-wind pattern. Once settled it is steady until perturbed.
"""

import numpy as np
import pytest
import torch

from conftest import to_numpy
from scenarios import make_config, si_channel, si_flat, si_ridge


def _speed(env) -> np.ndarray:
    # Observed interior only (drop the padded sponge ring), so the wind checks see the same field
    # the env exposes -- not the boundary belt the open-domain sponge holds at the free stream.
    return to_numpy(env._crop(torch.sqrt(env._x_wind_vel ** 2 + env._y_wind_vel ** 2)))


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


@pytest.mark.xfail(strict=True, reason=(
    "Gap/Venturi winds are not captured: the level-lid shallow-water core gives a deep (slow) "
    "layer in the low channel, so wind funnelled through it does not speed up above ambient. "
    "True fast-channel winds need blocking-terrain (wetting-drying) physics -- see "
    "docs/units.md Known limitations. strict=True so this flips to a failure if the model "
    "ever gains gap physics."
))
def test_narrow_channel_does_not_speed_up_wind(make_env):
    """A narrow channel aligned with the wind should accelerate it (real Venturi) -- but doesn't.

    Two high walls flank a low channel running along the x-wind. A real atmosphere funnels the
    wind through the gap and speeds it up; the level-lid core makes the low channel a deep, slow
    layer instead, so the channel wind stays at/below ambient. We assert the real-Venturi
    expectation (channel faster than ambient), which fails -- documenting the limitation.
    """
    ambient = (15.0, 0.0)
    env = make_env(*si_channel(grid=48, ambient=ambient, peak_m=800.0))
    for _ in range(800):
        env.step(0)

    sp = _speed(env)
    c = sp.shape[0] // 2
    channel = sp[c, 12:-12].mean()      # along the channel centreline, away from the sponge
    assert channel > np.hypot(*ambient)  # real Venturi speedup -- the model fails this


def _fire_speed(env) -> np.ndarray:
    """Magnitude of the diagnostic near-surface wind the fire/oxygen read (channeling applied)."""
    return to_numpy(torch.sqrt(env._x_wind_fire ** 2 + env._y_wind_fire ** 2))


def test_channeling_speeds_up_surface_wind(make_env):
    """With channeling on, the wind the fire reads speeds up through the gap (the imposed Venturi).

    The prognostic core leaves the channel slow (see the xfail above), but the diagnostic channeling
    gain multiplies the near-surface wind by the cross-stream confinement, so the wind the fire/oxygen
    feel along the channel exceeds the synoptic ambient -- the gap speedup, imposed as a rule.
    """
    ambient = (15.0, 0.0)
    _, m = si_channel(grid=48, ambient=ambient, peak_m=800.0)
    env = make_env(make_config(48, channeling={"enabled": True}), m)
    for _ in range(800):
        env.step(0)

    fire = _fire_speed(env)
    c = fire.shape[0] // 2
    channel = fire[c, 12:-12].mean()       # along the channel centreline, away from the sponge
    assert channel > np.hypot(*ambient)    # the channeled surface wind exceeds ambient


def test_channeling_is_unity_over_flat_ground(make_env):
    """Over flat terrain the gain is 1, so the wind the fire reads equals the prognostic wind."""
    ambient = (12.0, -5.0)
    env = make_env(make_config(48, channeling={"enabled": True}), si_flat(grid=48, ambient=ambient)[1])
    for _ in range(150):
        env.step(0)

    # No relief -> no confinement -> gain == 1 everywhere, so the fire wind is the prognostic wind.
    np.testing.assert_allclose(to_numpy(env._x_wind_fire), to_numpy(env._x_wind_vel), rtol=1e-12)
    np.testing.assert_allclose(to_numpy(env._y_wind_fire), to_numpy(env._y_wind_vel), rtol=1e-12)


def test_wind_is_steady_until_perturbed(make_env):
    """Once spun up, the wind field is (near-)constant step-to-step."""
    env = make_env(*si_ridge(grid=48, ambient=(15.0, 0.0), peak_m=400.0))
    # The observed interior carries no sponge damping at its edges (the ring holds it), so the
    # ridge flow settles on the domain's own (longer) timescale; 800 ticks leaves it mid-transient.
    for _ in range(1600):
        env.step(0)

    before = _speed(env)
    for _ in range(20):
        env.step(0)
    assert np.abs(_speed(env) - before).max() < 1e-2
