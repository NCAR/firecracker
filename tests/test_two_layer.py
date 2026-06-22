"""
Two-layer surface-wind core (Phase 6, leg a).

With [momentum].two_layer on, the single boundary-layer slab becomes the UPPER layer and a
thin surface layer rides underneath it. Leg (a) adds only the structure and dynamics (no
terrain blocking, interfacial coupling, or fire rewiring yet), so these tests pin the
foundations: the rest-state mass splits exactly into the two layers, both layers hold the
synoptic ambient over flat ground and stay steady, the closed (periodic) core conserves each
layer's mass and energy independently, a ridge world stays finite, and the split is wired the
same way through the batched rollout collector. The single-layer path being unchanged is
covered by the rest of the suite running green with two_layer off (the default).
"""

import numpy as np
import torch

import physics_constants as pc
from conftest import make_config, to_numpy
from map_loader import save_map
from rollout import BatchedRollout
from scenarios import build_map, si_flat, si_ridge

CP = pc.CP_AIR


def _two_layer_config(grid: int, **momentum) -> dict:
    """make_config with the two-layer core enabled (plus any momentum overrides)."""
    return make_config(grid, momentum={"two_layer": True, **momentum})


def _upper_speed(env) -> np.ndarray:
    return to_numpy(torch.sqrt(env._x_wind_vel ** 2 + env._y_wind_vel ** 2))


def _surface_speed(env) -> np.ndarray:
    return to_numpy(torch.sqrt(env._x_wind_surface ** 2 + env._y_wind_surface ** 2))


def test_mass_split_is_exact_and_positive(make_env):
    """The surface + upper masses sum back to the loaded level-lid mass, both strictly positive."""
    _, m = si_ridge(grid=48, ambient=(0.0, 0.0), peak_m=400.0)
    env = make_env(_two_layer_config(48), m)

    total = to_numpy(env._mass + env._mass_surface)
    np.testing.assert_allclose(total, m.mass, rtol=1e-6)
    assert float(env._mass.min()) > 0.0
    assert float(env._mass_surface.min()) > 0.0
    # The surface layer is thinner over the ridge crest than in the flanking low ground
    # (level interface), so its mass is not uniform.
    assert float(env._mass_surface.max()) > float(env._mass_surface.min())


def test_flat_world_holds_ambient_in_both_layers(make_env):
    """Over flat terrain both layers settle at the synoptic ambient and stay uniform."""
    ambient = (12.0, -5.0)
    _, m = si_flat(grid=48, ambient=ambient)
    env = make_env(_two_layer_config(48), m)
    for _ in range(400):
        env.step(0)

    speed = np.hypot(*ambient)
    belt = (slice(12, -12), slice(12, -12))   # interior, away from the sponge
    upper = _upper_speed(env)[belt]
    surface = _surface_speed(env)[belt]
    # A touch below ambient from the (stronger, on the surface layer) interior friction.
    assert speed * 0.9 < upper.mean() <= speed * 1.01
    assert speed * 0.9 < surface.mean() <= speed * 1.01
    assert upper.std() < 0.1 and surface.std() < 0.1


def test_both_layers_steady_until_perturbed(make_env):
    """Once spun up, both layers are (near-)constant step-to-step."""
    _, m = si_ridge(grid=48, ambient=(15.0, 0.0), peak_m=400.0)
    env = make_env(_two_layer_config(48), m)
    for _ in range(800):
        env.step(0)

    before_u, before_s = _upper_speed(env), _surface_speed(env)
    for _ in range(20):
        env.step(0)
    assert np.abs(_upper_speed(env) - before_u).max() < 1e-2
    assert np.abs(_surface_speed(env) - before_s).max() < 1e-2


def test_closed_core_conserves_each_layer(make_env):
    """Closed (periodic) core: each layer independently conserves its mass and energy.

    With the sponge and both layers' drag off, the solver is pure periodic shallow-water
    transport per layer, so each layer's total mass and energy (sum m*c_p*T) hold to round-off.
    """
    closed = {"sponge_strength": 0.0, "drag_coeff": 0.0, "surface_drag_coeff": 0.0}
    _, m = si_ridge(grid=32, ambient=(8.0, 3.0), peak_m=300.0)
    env = make_env(_two_layer_config(32, **closed), m)

    def totals():
        mu, ms = env._mass, env._mass_surface
        tu, ts = env._air_temperatures, env._air_temperatures_surface
        return (
            float(mu.sum()), float(ms.sum()),
            float((mu * CP * tu).sum()), float((ms * CP * ts).sum()),
        )

    mu0, ms0, eu0, es0 = totals()
    for _ in range(200):
        env.step(0)
    mu1, ms1, eu1, es1 = totals()

    assert np.isfinite(to_numpy(env._air_temperatures_surface)).all()
    np.testing.assert_allclose(mu1, mu0, rtol=1e-9)
    np.testing.assert_allclose(ms1, ms0, rtol=1e-9)
    np.testing.assert_allclose(eu1, eu0, rtol=1e-6)
    np.testing.assert_allclose(es1, es0, rtol=1e-6)


def test_ridge_world_stays_finite(make_env):
    """A two-layer ridge world runs many steps with every field finite (CFL/stability)."""
    _, m = si_ridge(grid=48, ambient=(15.0, 0.0), peak_m=400.0)
    env = make_env(_two_layer_config(48), m)
    for _ in range(800):
        env.step(0)

    for field in (
        env._mass, env._mass_surface,
        env._x_wind_vel, env._y_wind_vel, env._x_wind_surface, env._y_wind_surface,
        env._air_temperatures, env._air_temperatures_surface,
    ):
        assert torch.isfinite(field).all()


def test_rollout_splits_surface_layer(tmp_path):
    """The batched rollout collector applies the same split and steps two-layer worlds."""
    grid = 16
    config = _two_layer_config(grid)
    terrain = np.tile(np.linspace(0.0, 300.0, grid, dtype=np.float32), (grid, 1))
    save_map(tmp_path / "m.npz", build_map(grid, terrain=terrain, ambient_wind=(6.0, 0.0)))

    roll = BatchedRollout(
        config, maps_dir=str(tmp_path), grid_size=grid,
        num_envs=3, steps=8, stride=4, buffer_device="cpu", seed=0,
    )
    s = roll._state
    assert s.mass_surface is not None
    # Upper + surface reconstruct the loaded mass for every world, both layers positive.
    loaded_mass = build_map(grid, terrain=terrain).mass
    total = to_numpy(s.mass + s.mass_surface)[0]
    np.testing.assert_allclose(total, loaded_mass, rtol=1e-6)
    assert float(s.mass_surface.min()) > 0.0

    pool = roll.collect()
    assert torch.isfinite(pool).all()
