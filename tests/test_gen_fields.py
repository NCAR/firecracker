"""
Tests for the array-agnostic field builders in gen_maps.MapGenerator.

The derived-field math (temperature, pressure, mass, oxygen, vegetation thinning) runs
on either a NumPy array (the CPU path and the test API) or a torch tensor (the batched
GPU build path). These pin down both halves of that contract:
  * NumPy in -> NumPy out, float32, unchanged from the original CPU behaviour, and
  * torch in -> torch out, same values (so the GPU build reuses the same physics).
"""

import sys
from pathlib import Path

import numpy as np
import torch

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import load_config  # noqa: E402
from gen_maps import MapGenerator  # noqa: E402

GRID = 24


def _gen() -> MapGenerator:
    return MapGenerator(load_config())


def _elevation_np() -> np.ndarray:
    # A smooth ramp of plausible elevations [m], spanning sea level to near the peak.
    z = np.linspace(0.0, 450.0, GRID * GRID, dtype=np.float32).reshape(GRID, GRID)
    return z


def test_numpy_in_numpy_out_float32():
    gen = _gen()
    z = _elevation_np()
    air = gen.air_temperature_profile(z)
    for field in (air, gen.pressure_profile(z), gen.boundary_layer_mass(z),
                  gen.oxygen_profile(z, air),
                  gen.normalize_grid(z)):
        assert isinstance(field, np.ndarray)
        assert field.dtype == np.float32


def test_torch_in_torch_out_matches_numpy():
    gen = _gen()
    z_np = _elevation_np()
    z_t = torch.as_tensor(z_np, dtype=torch.float64)

    air_np = gen.air_temperature_profile(z_np)
    air_t = gen.air_temperature_profile(z_t)
    assert isinstance(air_t, torch.Tensor)

    pairs = [
        (air_np, air_t),
        (gen.pressure_profile(z_np), gen.pressure_profile(z_t)),
        (gen.boundary_layer_mass(z_np), gen.boundary_layer_mass(z_t)),
        (gen.oxygen_profile(z_np, air_np), gen.oxygen_profile(z_t, air_t)),
        (gen.normalize_grid(z_np), gen.normalize_grid(z_t)),
    ]
    for arr_np, arr_t in pairs:
        assert isinstance(arr_t, torch.Tensor)
        # float32 numpy vs float64 torch: compare with a float32-scale tolerance.
        assert np.allclose(arr_np, arr_t.numpy(), rtol=1e-5, atol=1e-4)


def test_zero_lapse_rate_uses_isothermal_pressure():
    # The gamma <= 0 branch uses exp(); exercise it on both backends.
    gen = _gen()
    gen.temperature_lapse_rate = 0.0
    z_np = _elevation_np()
    z_t = torch.as_tensor(z_np, dtype=torch.float64)
    p_np = gen.pressure_profile(z_np)
    p_t = gen.pressure_profile(z_t)
    assert np.allclose(p_np, p_t.numpy(), rtol=1e-5, atol=1e-2)


def test_batched_normalize_is_per_map():
    # A batched (B, H, W) tensor normalises each map over its own extent: every map's
    # min -> 0 and max -> 1 independently, regardless of the other maps' ranges.
    gen = _gen()
    batch = torch.stack([
        torch.linspace(0.0, 1.0, GRID * GRID, dtype=torch.float64).reshape(GRID, GRID),
        torch.linspace(-5.0, 50.0, GRID * GRID, dtype=torch.float64).reshape(GRID, GRID),
    ])
    out = gen.normalize_grid(batch)
    assert out.shape == batch.shape
    per_map_min = out.amin(dim=(-2, -1))
    per_map_max = out.amax(dim=(-2, -1))
    assert torch.allclose(per_map_min, torch.zeros(2, dtype=out.dtype), atol=1e-9)
    assert torch.allclose(per_map_max, torch.ones(2, dtype=out.dtype), atol=1e-9)


def test_batched_sample_perlin_matches_per_map():
    # sample_perlin_grid with a list of bases stacks the per-base fields, so a future
    # batched build matches looping generate_map.
    gen = _gen()
    bases = [1, 7, 19]
    batch = gen.sample_perlin_grid(GRID, bases, scale=24.0)
    assert batch.shape == (3, GRID, GRID)
    stacked = torch.stack([gen.sample_perlin_grid(GRID, b, scale=24.0) for b in bases])
    assert torch.allclose(batch, stacked)


def _ns_ridge() -> np.ndarray:
    # A north-south-running cosine ridge: elevation varies only with the row index (which runs
    # south), so the slopes face due south on one flank and due north on the other.
    i = np.arange(GRID)[:, None]
    return (200.0 * np.cos(2.0 * np.pi * i / GRID)).repeat(GRID, axis=1).astype(np.float32)


def test_sunlight_range_and_diffuse_floor():
    # The exposure field is a dimensionless [0, 1] fraction, floored by the diffuse skylight.
    gen = _gen()
    sun = gen.sunlight_exposure(_ns_ridge())
    assert isinstance(sun, np.ndarray) and sun.dtype == np.float32
    assert sun.shape == (GRID, GRID)
    assert sun.min() >= gen.sun_diffuse_fraction - 1e-6
    assert sun.max() <= 1.0 + 1e-6


def test_sunlight_south_face_brighter_than_north():
    # Northern-hemisphere physics: the south-facing flank of the ridge catches far more sun than
    # the north-facing flank (which falls back to the diffuse floor).
    gen = _gen()
    sun = gen.sunlight_exposure(_ns_ridge())
    south_face = sun[GRID // 4, GRID // 2]      # dz/di < 0 -> surface faces south
    north_face = sun[3 * GRID // 4, GRID // 2]  # dz/di > 0 -> surface faces north
    assert south_face > north_face


def test_sunlight_torch_matches_numpy():
    # NumPy and torch backends produce the same field (the GPU build reuses the same geometry).
    gen = _gen()
    z = _ns_ridge()
    sun_np = gen.sunlight_exposure(z)
    sun_t = gen.sunlight_exposure(torch.as_tensor(z, dtype=torch.float64))
    assert isinstance(sun_t, torch.Tensor)
    assert np.allclose(sun_np, sun_t.numpy(), rtol=1e-5, atol=1e-4)


def test_surface_temperature_flat_ground_is_ambient():
    # A flat, open map gets the flat-exposure everywhere, so the baked surface temperature is the
    # unmodulated elevation profile -- sunlight only perturbs the surface around that flat baseline.
    gen = _gen()
    z = np.zeros((GRID, GRID), dtype=np.float32)
    surf = gen.surface_temperature(z, gen.sunlight_exposure(z))
    assert np.allclose(surf, gen.air_temperature_profile(z), atol=1e-3)


def test_surface_temperature_sunny_warmer_than_shaded():
    # The sun-facing flank of the ridge bakes warmer than the shaded north flank.
    gen = _gen()
    z = _ns_ridge()
    surf = gen.surface_temperature(z, gen.sunlight_exposure(z))
    assert surf[GRID // 4, GRID // 2] > surf[3 * GRID // 4, GRID // 2]


def test_surface_temperature_moderation_bounds():
    # moderation = 1 removes the solar swing (pure ambient profile); moderation = 0 is the full
    # (vacuum) radiative response -- shaded cells well below ambient, sunlit cells above.
    gen = _gen()
    z = _ns_ridge()
    sun = gen.sunlight_exposure(z)
    ambient = gen.air_temperature_profile(z)
    gen.sun_temp_moderation = 1.0
    assert np.allclose(gen.surface_temperature(z, sun), ambient, atol=1e-4)
    gen.sun_temp_moderation = 0.0
    full = gen.surface_temperature(z, sun)
    assert full.min() < ambient.min() - 1.0
    assert full.max() > ambient.max() + 0.5


def test_surface_temperature_torch_matches_numpy():
    gen = _gen()
    z = _ns_ridge()
    surf_np = gen.surface_temperature(z, gen.sunlight_exposure(z))
    z_t = torch.as_tensor(z, dtype=torch.float64)
    surf_t = gen.surface_temperature(z_t, gen.sunlight_exposure(z_t))
    assert isinstance(surf_t, torch.Tensor)
    assert np.allclose(surf_np, surf_t.numpy(), rtol=1e-5, atol=1e-3)


def test_biomes_partition_the_grid():
    # Every cell lands in exactly one biome (the three masks tile the grid without overlap),
    # on both backends.
    gen = _gen()
    z = _elevation_np()
    surf = gen.surface_temperature(z, gen.sunlight_exposure(z))
    woodland, grass, shrub = gen.classify_biomes(z, surf)
    exactly_one = woodland.astype(int) + grass.astype(int) + shrub.astype(int)
    assert np.array_equal(exactly_one, np.ones_like(exactly_one))

    z_t = torch.as_tensor(z, dtype=torch.float64)
    surf_t = gen.surface_temperature(z_t, gen.sunlight_exposure(z_t))
    w_t, g_t, s_t = gen.classify_biomes(z_t, surf_t)
    assert torch.equal(w_t, torch.as_tensor(woodland))
    assert torch.equal(g_t, torch.as_tensor(grass))
    assert torch.equal(s_t, torch.as_tensor(shrub))


def test_biome_labels_follow_temperature_and_elevation():
    # Woodland is the cool cells (surface T <= woodland_temp_max); among the warm cells, low ground
    # is grassland and higher ground shrubland.
    gen = _gen()
    z = _elevation_np()
    surf = gen.surface_temperature(z, gen.sunlight_exposure(z))
    woodland, grass, shrub = gen.classify_biomes(z, surf)
    assert np.all(surf[woodland] <= gen.woodland_temp_max)
    assert np.all(surf[grass] > gen.woodland_temp_max)
    assert np.all(z[grass] <= gen.grass_elev_max)
    assert np.all(z[shrub] > gen.grass_elev_max)


def test_biome_weights_partition_softly_and_track_classification():
    # The soft weights sum to 1 everywhere, blend smoothly (interior values strictly between 0 and 1
    # near the borders), and collapse onto the hard classification as the softness -> 0. Fields that
    # straddle both thresholds (temperature around woodland_temp_max, elevation around grass_elev_max)
    # so every biome and its borders are exercised.
    gen = _gen()
    surf = np.linspace(294.0, 302.0, GRID * GRID, dtype=np.float32).reshape(GRID, GRID)
    z = np.linspace(150.0, 350.0, GRID * GRID, dtype=np.float32).reshape(GRID, GRID)
    w_wood, w_grass, w_shrub = gen.biome_weights(z, surf)
    total = w_wood + w_grass + w_shrub
    assert np.allclose(total, 1.0, atol=1e-5)
    assert np.all(w_wood >= 0) and np.all(w_grass >= 0) and np.all(w_shrub >= 0)
    # A transition band exists: some cell is a genuine blend, not a hard 0/1 label.
    assert np.any((w_wood > 0.05) & (w_wood < 0.95))

    # Shrinking the softness collapses the ramps toward steps: the winning (argmax) biome then
    # agrees with the hard classification everywhere but the thin residual transition band.
    gen.woodland_temp_softness = 0.2
    gen.grass_elev_softness = 5.0
    soft_label = np.argmax(np.stack(gen.biome_weights(z, surf)), axis=0)
    woodland, grass, shrub = gen.classify_biomes(z, surf)
    hard_label = np.where(woodland, 0, np.where(grass, 1, 2))
    assert (soft_label == hard_label).mean() > 0.95

    # torch backend matches numpy.
    gen2 = _gen()
    wt = gen2.biome_weights(torch.as_tensor(z, dtype=torch.float64),
                            torch.as_tensor(surf, dtype=torch.float64))
    ref = gen2.biome_weights(z, surf)
    for a_t, a_np in zip(wt, ref):
        assert isinstance(a_t, torch.Tensor)
        assert np.allclose(a_t.numpy(), a_np, rtol=1e-5, atol=1e-4)


def test_biome_tree_density_ranks_woodland_over_shrub_over_grass():
    # Woodland is the densest tree biome, grassland the sparsest, shrubland in between; the density
    # is one flat value per biome (no slope dependence), on both backends.
    gen = _gen()
    assert gen.tree_density_woodland > gen.tree_density_shrubland > gen.tree_density_grassland

    ones = np.ones((GRID, GRID), dtype=bool)
    zeros = np.zeros((GRID, GRID), dtype=bool)
    wood = gen.biome_tree_density(ones, zeros, zeros)
    grass = gen.biome_tree_density(zeros, ones, zeros)
    shrub = gen.biome_tree_density(zeros, zeros, ones)
    assert np.allclose(wood, gen.tree_density_woodland)
    assert np.allclose(grass, gen.tree_density_grassland)
    assert np.allclose(shrub, gen.tree_density_shrubland)

    shrub_t = gen.biome_tree_density(
        torch.as_tensor(zeros), torch.as_tensor(zeros), torch.as_tensor(ones))
    assert isinstance(shrub_t, torch.Tensor)
    assert np.allclose(shrub, shrub_t.numpy())


def test_apply_noise_floor_lifts_trough_to_floor():
    # The [0,1] noise is remapped onto [floor, 1]: its min rises to the floor, its max stays at 1,
    # and the ordering/texture is preserved. Grass carries a nonzero floor in the default config.
    gen = _gen()
    gi = gen.fuel_type_names.index("grass")
    floor = float(gen.fuel_noise_floors[gi])
    assert floor > 0.0
    noise = np.linspace(0.0, 1.0, GRID * GRID, dtype=np.float32).reshape(GRID, GRID)
    lifted = gen.apply_noise_floor(noise, gi)
    assert np.isclose(lifted.min(), floor, atol=1e-6)
    assert np.isclose(lifted.max(), 1.0, atol=1e-6)
    # Monotone remap: same ordering as the input (no texture inversion).
    assert np.all(np.diff(lifted.ravel()) >= -1e-6)

    # A zero-floor fuel is returned unchanged; torch in -> torch out matches.
    lifted_t = gen.apply_noise_floor(torch.as_tensor(noise, dtype=torch.float64), gi)
    assert isinstance(lifted_t, torch.Tensor)
    assert np.allclose(lifted, lifted_t.numpy(), atol=1e-6)


def test_biome_grass_multiplier_woodland_between_zero_and_grassland():
    # Grass runs full-strength on the grassland, a reduced fraction under the woodland, and none on
    # the shrubland (which is left as the implicit zero).
    gen = _gen()
    ones = np.ones((GRID, GRID), dtype=bool)
    zeros = np.zeros((GRID, GRID), dtype=bool)
    assert np.allclose(gen.biome_grass_multiplier(ones, zeros), 1.0)          # all grassland
    assert np.allclose(gen.biome_grass_multiplier(zeros, ones),               # all woodland
                       gen.woodland_grass_fraction)
    assert 0.0 < gen.woodland_grass_fraction < 1.0


def test_device_property_resolves():
    # The lazy device property resolves to a real torch device without needing a GPU.
    dev = MapGenerator(load_config()).device
    assert isinstance(dev, torch.device)
    assert dev.type in ("cpu", "cuda")
