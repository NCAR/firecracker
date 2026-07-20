"""
gen_maps.py

Synthetic map producer for the Firecracker simulation.

Owns the procedural initial-state generation that used to live inside
Simulation/reset(): octave Perlin terrain, elevation-driven temperature/oxygen
profiles, the convective rest-state pressure, the grass/tree fuel layers, and a
static per-cell solar-exposure field (south-facing slopes catch more sun, shadowed
valleys less).
`MapGenerator` reads what it needs straight from the config dict and emits a
`MapData`, the same contract a future real-data importer would target.

The whole rest state is built as torch tensors on `MapGenerator.device` (CUDA when
available, else CPU -- set [environment].device or pass --device to force one), using
the vectorized torch Perlin kernel in perlin.py instead of a per-cell scalar loop. So
both the noise and the field math run on the GPU. A whole chunk of maps is built in one
batched pass (`generate_maps`): the Perlin kernel samples every field for the batch at
once (one base per map) and the field math broadcasts over the leading map dim, so the
maps are built together rather than one at a time. Map generation is not required to be
deterministic (only the maps, once written, must stay stable), so every per-map random
draw (noise bases, wind, tree counts) comes from a single shared numpy rng. Fields move
back to numpy at the MapData boundary, then the chunk is spun up on the same device.

After the rest state is built, `spin_up_maps` steps the full field physics to its
developed state and bakes the developed mass, air temperature and prognostic wind
back into each map, so a loaded map already shows the terrain-shaped flow (windward
blocking, lee/foehn acceleration) and the wind-smeared air temperature instead of
relaxing into them at runtime. (Stepping the full physics -- not the wind alone --
is what lets the air settle into its developed, advected pattern rather than washing
out.) This is the one place generation touches the physics engine; the emitted
`MapData` is still a plain baked state.

Run as a script to bake a batch of maps to disk:

    python src/gen_maps.py --count 8 --out maps --seed 0

Generation is embarrassingly parallel, so a large run can be sharded across processes / GPUs /
nodes without ever producing duplicate maps (disjoint output index blocks + an independent random
stream per shard). Fan across the local GPUs in one command:

    python src/gen_maps.py --count 1000 --seed 0 --gpus cuda:0,cuda:1,cuda:2,cuda:3

or launch the shards yourself (one process per rank, e.g. under MPI/PBS across nodes), pinning a
GPU per rank and passing the same --count/--seed/--shards with a distinct --shard-id:

    python src/gen_maps.py --count 1000 --seed 0 --shards $NRANKS --shard-id $RANK --device cuda
"""

import argparse

import numpy as np
import torch

import physics_constants as pc
from config import load_config, boundary_pad, physics_grid_size
from map_loader import MapData, save_map
from perlin import perlin_noise

DEFAULT_GRID_SIZE = 256


# ---------------------------------------------------------------------------
# Array-library helpers
#
# The field builders (temperature, pressure, mass, oxygen, vegetation thinning) are
# pure elementwise math, so they run unchanged on either a NumPy array (the CPU path
# and the test API) or a torch tensor (the batched GPU build path). These tiny helpers
# pick the right module and storage cast without importing torch at module load -- torch
# is only touched when a tensor is actually passed in, keeping rest-state generation
# import-light (the engine dependency stays confined to the spin-up).
# ---------------------------------------------------------------------------

def _is_torch(arr) -> bool:
    return type(arr).__module__.split(".", 1)[0] == "torch"


def _xp(arr):
    """The array module backing `arr`: torch for a tensor, numpy otherwise."""
    if _is_torch(arr):
        import torch
        return torch
    return np


def _store(arr):
    """Cast a built field to its storage dtype.

    NumPy fields are stored float32 (the MapData/on-disk convention); torch tensors are
    left in the engine's compute dtype (float64) until the MapData boundary, so the GPU
    build keeps full precision through to the spin-up.
    """
    return arr if _is_torch(arr) else arr.astype(np.float32)


def _resolve_device(name: str | None):
    """Pick the generation device (mirrors simulation._resolve_device): an explicit
    config value wins, else CUDA when present. Imports torch lazily."""
    import torch
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MapGenerator:
    """Procedural initial-state generator driven entirely by the config dict."""

    def __init__(self, cfg: dict | None = None):
        # Target device for the batched GPU build path (slice 3). Stored as the raw
        # config name and resolved lazily via the `device` property, so constructing a
        # generator (e.g. in the tests) never imports torch.
        self._device_name = (cfg or {}).get("environment", {}).get("device")

        # Cell size [m]: the physical grid spacing, used by the sunlight/shadow geometry.
        self.cell_size_m = float((cfg or {}).get("units", {}).get("cell_size_m", 10.0))
        self._device = None

        noise_cfg = (cfg or {}).get("noise", {})
        self.noise_scale       = float(noise_cfg.get("scale",      64.0))
        self.noise_octaves     = int(noise_cfg.get("octaves",       4))
        self.noise_persistence = float(noise_cfg.get("persistence", 0.5))
        self.noise_lacunarity  = float(noise_cfg.get("lacunarity",  2.0))

        terrain = (cfg or {}).get("terrain", {})
        self.terrain_scale       = float(terrain.get("scale",      96.0))
        self.terrain_octaves     = int(terrain.get("octaves",       6))
        self.terrain_persistence = float(terrain.get("persistence", 0.5))
        self.terrain_lacunarity  = float(terrain.get("lacunarity",  2.0))
        # Exponent applied to the normalised [0,1] height map: >1 biases toward flat low-elevation
        # terrain with sharper, less frequent peaks; 1 leaves the raw Perlin field unchanged.
        self.terrain_exponent    = float(terrain.get("exponent",     1.0))
        # SI elevation: the normalised [0,1] height map is scaled to metres by elev_max.
        self.elev_max               = float(terrain.get("elev_max",               500.0))
        # Environmental lapse rate [K/m]: air temperature falls T_REF - rate*elevation.
        self.temperature_lapse_rate = float(terrain.get("temperature_lapse_rate", pc.LAPSE_RATE))

        # Per-map synoptic wind: speed [m/s] drawn uniformly from this range, random bearing.
        wind = (cfg or {}).get("wind", {})
        self.ambient_speed_min = float(wind.get("ambient_speed_min",  5.0))
        self.ambient_speed_max = float(wind.get("ambient_speed_max", 15.0))

        # Free-stream depth of the modelled boundary layer [m]; sets the level-lid rest state.
        momentum = (cfg or {}).get("momentum", {})
        self.layer_depth_ref = float(momentum.get("layer_depth_ref", 1000.0))
        # Wind spin-up: how many solver ticks to develop the orographic flow before baking it
        # into the map. The flow is unsteady (lee waves never freeze), so this is "developed",
        # not a fixed point: ~800 ticks is where the blocking/foehn/channeling structure is
        # fully formed (see docs/units.md). 0 skips the spin-up (maps stay in the level-lid rest
        # state with uniform wind, the pre-v4 behaviour).
        self.spinup_steps = int(momentum.get("spinup_steps", 800))

        # Static solar exposure field (sunlight_exposure): the sun is sampled along the clear-sky
        # daily arc (hour angle) over a few seasons (declination) at this latitude; each sample's
        # beam is gated by a horizon ray-march so deep valleys lose their direct sun.
        sun = (cfg or {}).get("sunlight", {})
        self.latitude_deg         = float(sun.get("latitude_deg",        40.0))
        self.axial_tilt_deg       = float(sun.get("axial_tilt_deg",      23.44))
        self.sun_num_seasons      = int(sun.get("num_seasons",            1))
        self.sun_num_hours        = int(sun.get("num_hours",             12))
        self.sun_hour_half_angle  = float(sun.get("hour_half_angle_deg", 90.0))
        self.sun_shadows          = bool(sun.get("shadows",             True))
        self.sun_max_shadow_steps = int(sun.get("max_shadow_steps",       0))   # 0 = grid size
        self.sun_diffuse_fraction = float(sun.get("diffuse_fraction",    0.15))
        # How strongly the atmosphere damps the bare radiative temperature swing the sun drives
        # (surface_temperature): 0 = pure (vacuum) radiative response, 1 = no solar effect (uniform
        # elevation profile). Earth's air insulation + lateral mixing keep shade only "a bit cool".
        self.sun_temp_moderation  = float(sun.get("temp_moderation",     0.7))

        # Non-overlapping vegetation biomes. Each cell is labelled exactly one biome from its
        # elevation and baked surface temperature (see classify_biomes) for the discrete biome map,
        # but the vegetation reads soft membership weights that blend across the thresholds (see
        # biome_weights), so the fuel transitions smoothly at biome borders instead of stepping.
        biomes = (cfg or {}).get("biomes", {})
        self.woodland_temp_max = float(biomes.get("woodland_temp_max", 298.0))  # K: cool cells -> woodland
        self.grass_elev_max    = float(biomes.get("grass_elev_max",    250.0))  # m: warm+low -> grassland
        # Border softness: the width of the smooth transition band the vegetation blends over at each
        # biome threshold (temperature for the woodland edge, elevation for the grassland/shrubland
        # edge). Larger = wider, gentler blend; -> 0 recovers a hard edge. See biome_weights.
        self.woodland_temp_softness = float(biomes.get("woodland_temp_softness", 1.5))   # K
        self.grass_elev_softness    = float(biomes.get("grass_elev_softness",    35.0))  # m
        # Fraction of the grassland's grass loading that also grows in the (cooler, tree-dense)
        # woodland: >0 so woodland floors keep some grass, <1 so it stays below the open grassland.
        self.woodland_grass_fraction = float(biomes.get("woodland_grass_fraction", 0.3))
        # Per-biome exponential-mean tree count (drives sample_tree_counts): one value per biome,
        # scaled per cell only by the tree noise -- dense woodland canopy, a sparse grassland
        # sprinkle, and a modest shrubland cover in between.
        self.tree_density_woodland    = float(biomes.get("tree_density_woodland",    8.0))
        self.tree_density_grassland   = float(biomes.get("tree_density_grassland",   0.6))
        self.tree_density_shrubland   = float(biomes.get("tree_density_shrubland",   1.2))

        fuel_types_cfg = (cfg or {}).get("fuel_types", {})
        self.fuel_type_names: list[str] = list(fuel_types_cfg.keys())
        self.num_fuel_types: int = len(self.fuel_type_names)

        def _ft(name: str, key: str, default: float) -> float:
            return float(fuel_types_cfg.get(name, {}).get(key, default))

        self.fuel_noise_scales = np.array([_ft(n, "noise_scale",      32.0) for n in self.fuel_type_names], dtype=np.float32)
        self.spawn_densities   = np.array([_ft(n, "spawn_density",     0.5)  for n in self.fuel_type_names], dtype=np.float32)
        # Coverage floor for a continuous fuel's [0,1] noise: the noise is remapped to [floor, 1],
        # so even low-noise patches keep a base loading (floor of the peak) instead of thinning to
        # bare ground. Raises overall grass/shrub cover so the map reads less barren; 0 = raw noise.
        self.fuel_noise_floors = np.array([_ft(n, "noise_floor",      0.0)  for n in self.fuel_type_names], dtype=np.float32)

        # Tree components: any fuel type whose name starts with "tree" (e.g. tree_canopy, tree_bole)
        # is part of the same physical trees, so they share one per-cell count and spatial
        # distribution. fuel_per_tree is per type (the biomass each tree contributes to that
        # component); the count/distribution params (noise/spawn) are read from the first
        # tree component.
        self.fuel_per_tree = np.array(
            [_ft(n, "fuel_per_tree", 0.0) for n in self.fuel_type_names], dtype=np.float32
        )
        self.tree_indices = [i for i, n in enumerate(self.fuel_type_names) if n.startswith("tree")]
        first_tree = self.fuel_type_names[self.tree_indices[0]] if self.tree_indices else "tree"
        self.max_trees_per_cell = int(_ft(first_tree, "max_trees_per_cell", 3))

    @property
    def device(self):
        """The resolved torch device for the GPU build path (CUDA when available,
        else CPU; an explicit config name wins). Resolved and cached on first use."""
        if self._device is None:
            self._device = _resolve_device(self._device_name)
        return self._device

    # -----------------------------------------------------------------------
    # Field builders
    # -----------------------------------------------------------------------

    def sample_perlin_grid(
        self,
        size: int,
        base,
        scale: float | None = None,
        octaves: int | None = None,
        persistence: float | None = None,
        lacunarity: float | None = None,
    ) -> torch.Tensor:
        """Raw octave Perlin field(s) on self.device via the vectorized torch kernel.

        `base` is an int (-> (size, size)) or a sequence of ints (-> (B, size, size),
        one field per base), so a whole chunk of maps can be sampled in one call.
        """
        scale       = self.noise_scale       if scale is None else scale
        octaves     = self.noise_octaves     if octaves is None else octaves
        persistence = self.noise_persistence if persistence is None else persistence
        lacunarity  = self.noise_lacunarity  if lacunarity is None else lacunarity
        return perlin_noise(
            size, size,
            scale=scale, octaves=octaves, persistence=persistence, lacunarity=lacunarity,
            base=base, device=self.device, dtype=torch.float64,
        )

    @staticmethod
    def normalize_grid(grid):
        """Min/max-normalise to [0, 1]. For a batched (B, H, W) tensor each map is
        normalised independently (over its own H, W); a flat map maps to all zeros."""
        if _is_torch(grid) and grid.ndim == 3:
            lo = grid.amin(dim=(-2, -1), keepdim=True)
            hi = grid.amax(dim=(-2, -1), keepdim=True)
            span = hi - lo
            return _store(torch.where(span > 0, (grid - lo) / span, torch.zeros_like(grid)))
        lo = grid.min()
        hi = grid.max()
        if hi == lo:
            return _store(_xp(grid).zeros_like(grid))
        return _store((grid - lo) / (hi - lo))

    def create_grid(
        self,
        size: int,
        scale: float | None = None,
        base=None,
        octaves: int | None = None,
        persistence: float | None = None,
        lacunarity: float | None = None,
        rng: np.random.Generator | None = None,
    ) -> torch.Tensor:
        """Normalised [0, 1] Perlin field on self.device. A None base is drawn from rng."""
        if base is None:
            base = int((rng or np.random.default_rng()).integers(0, 256))
        raw = self.sample_perlin_grid(
            size, base, scale=scale,
            octaves=octaves, persistence=persistence, lacunarity=lacunarity,
        )
        return self.normalize_grid(raw)

    def create_terrain(
        self, size: int, base=None, rng: np.random.Generator | None = None
    ) -> torch.Tensor:
        """Normalised [0, 1] elevation height map from octave Perlin noise.

        Raised to terrain_exponent (default 1, no-op): >1 biases toward flat
        low-elevation terrain with sharper, less frequent high-elevation peaks.
        """
        height = self.create_grid(
            size,
            scale=self.terrain_scale,
            base=base,
            octaves=self.terrain_octaves,
            persistence=self.terrain_persistence,
            lacunarity=self.terrain_lacunarity,
            rng=rng,
        )
        return height ** self.terrain_exponent

    def air_temperature_profile(self, elevation_m):
        """Air temperature [K] from the environmental lapse rate: T = T_REF - Gamma*z."""
        return _store(pc.T_REF - self.temperature_lapse_rate * elevation_m)

    def pressure_profile(self, elevation_m):
        """Hydrostatic surface pressure [Pa].

        Exact barometric formula for the constant lapse rate Gamma, integrating
        dp/dz = -p*g/(R_d*T(z)) with T(z) = T_REF - Gamma*z:
            p(z) = p_REF * (1 - Gamma*z/T_REF) ** (g / (R_d*Gamma))
        This is consistent with air_temperature_profile and stays accurate to high
        elevations (no isothermal approximation). It reduces to the isothermal
        p_REF*exp(-z/H_p) as Gamma -> 0, which is the fallback used when the lapse
        rate is zero.
        """
        gamma = self.temperature_lapse_rate
        if gamma <= 0.0:
            return _store(pc.P_REF * _xp(elevation_m).exp(-elevation_m / pc.PRESSURE_SCALE_HEIGHT))
        exponent = pc.GRAVITY / (pc.GAS_CONSTANT_DRY_AIR * gamma)
        base = 1.0 - gamma * elevation_m / pc.T_REF   # = T(z)/T_REF, positive for z < T_REF/Gamma
        return _store(pc.P_REF * base ** exponent)

    def boundary_layer_mass(self, elevation_m):
        """Level-lid boundary-layer mass per area [kg/m^2] -- the shallow-water rest state.

        The modelled layer is a shallow near-surface layer, not the whole column. At rest its
        free surface s = terrain + h is level, so the layer is thinner over high terrain:
            h(z) = h_ref + (elev_max - z)        (depth, >= h_ref)
        The areal mass that gives that hydrostatic thickness (eta = m*R_d*T/p_ref = h) is
            m = h * p_ref / (R_d * T(z)).
        Making the layer shallow (h ~ O(km), comparable to terrain) is what lets continuity
        speed the wind up over crests and channel it through gaps.
        """
        lid = self.elev_max + self.layer_depth_ref            # level free-surface height [m]
        h = lid - elevation_m                                  # layer depth [m]
        T = self.air_temperature_profile(elevation_m)
        return _store(h * pc.P_REF / (pc.GAS_CONSTANT_DRY_AIR * T))

    def oxygen_profile(self, elevation_m, air_temperatures):
        """Ambient O2 partial density [kg/m^3] = O2 mass fraction * air density p/(R_d*T)."""
        air_density = self.pressure_profile(elevation_m) / (pc.GAS_CONSTANT_DRY_AIR * air_temperatures)
        return _store(pc.O2_MASS_FRACTION * air_density)

    def classify_biomes(self, elevation_m, surface_temperature):
        """Partition the grid into the three non-overlapping vegetation biomes.

        Returns (woodland, grassland, shrubland) boolean masks that tile the grid (every cell is
        in exactly one):
          * woodland  -- cool cells (surface_temperature <= woodland_temp_max), any slope. Cool
                         ground is high and/or shaded, so this captures the tops and the cold
                         hollows regardless of steepness;
          * grassland -- the remaining warm cells on low ground (elevation <= grass_elev_max);
          * shrubland -- the remaining warm cells above that elevation (higher than the grassland
                         but, being warm rather than cool, not high enough to have turned to woodland).
        Pure elementwise comparisons/logical ops, so it runs on a NumPy array or a torch tensor;
        the masks stay boolean (not cast to the float storage dtype)."""
        woodland = surface_temperature <= self.woodland_temp_max
        warm = ~woodland
        low = elevation_m <= self.grass_elev_max
        grassland = warm & low
        shrubland = warm & ~low
        return woodland, grassland, shrubland

    def biome_weights(self, elevation_m, surface_temperature):
        """Soft per-cell biome membership (woodland, grassland, shrubland), each in [0, 1] and
        summing to 1 -- the smooth counterpart of classify_biomes used to place vegetation.

        The hard thresholds are replaced by logistic ramps so the biomes blend over a transition
        band instead of stepping at a border: the woodland share rises smoothly as the surface
        temperature drops through woodland_temp_max (band width woodland_temp_softness), and the
        remaining (warm) share is split between grassland and shrubland by a smooth ramp on elevation
        through grass_elev_max (band width grass_elev_softness). Feeding these weights (instead of the
        boolean masks) to biome_grass_multiplier / biome_tree_density -- both linear in their
        arguments -- makes the grass, shrub and tree densities cross-fade across biome borders. As
        the softness knobs -> 0 the ramps become steps and this reduces to classify_biomes. Array-
        agnostic (NumPy or torch)."""
        xp = _xp(surface_temperature)
        w_wood = 1.0 / (1.0 + xp.exp((surface_temperature - self.woodland_temp_max) / self.woodland_temp_softness))
        share_grass = 1.0 / (1.0 + xp.exp((elevation_m - self.grass_elev_max) / self.grass_elev_softness))
        warm = 1.0 - w_wood
        return _store(w_wood), _store(warm * share_grass), _store(warm * (1.0 - share_grass))

    def apply_noise_floor(self, noise, fuel_index: int):
        """Lift a fuel's [0,1] noise onto [floor, 1] so low-noise patches keep a base loading.

        floor = fuel_noise_floors[fuel_index]; the remap is floor + (1-floor)*noise, so the noise
        texture is preserved but its trough sits at `floor` of the peak instead of zero -- filling
        the bare gaps that read as barren. floor = 0 is a no-op. Array-agnostic (NumPy or torch)."""
        f = float(self.fuel_noise_floors[fuel_index])
        return _store(f + (1.0 - f) * noise) if f > 0.0 else _store(noise)

    def biome_grass_multiplier(self, grassland, woodland):
        """Per-cell scale on the grass loading by biome, in [0, 1].

        Grass grows at full strength on the open grassland and at woodland_grass_fraction of that
        under the cooler, tree-dense woodland canopy (less grass than the grassland, but not none);
        the shrubland gets no grass. The masks are the boolean biome labels from classify_biomes.
        Array-agnostic (NumPy or torch)."""
        return _store(grassland + woodland * self.woodland_grass_fraction)

    def biome_tree_density(self, woodland, grassland, shrubland):
        """Per-cell exponential-mean tree count (the sample_tree_counts density), set by biome.

        Trees grow in every biome, at one mean per biome (scaled per cell only by the tree noise):
        a dense woodland canopy, a sparse grassland sprinkle, and a modest shrubland cover in
        between. The masks are the boolean biome labels from classify_biomes; combining them selects
        one density per cell. Array-agnostic (NumPy or torch)."""
        density = (woodland * self.tree_density_woodland
                   + grassland * self.tree_density_grassland
                   + shrubland * self.tree_density_shrubland)
        return _store(density)

    def sample_tree_counts(self, density: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
        """Stochastic per-cell tree counts on density.device.

        Each cell's count is floor(Exp(mean=density)) capped at max_trees_per_cell, so
        cells with higher density average more trees (mean 0 -> always 0). The unit-mean
        exponential draws come from the per-map numpy rng (so the count stays reproducible
        and device-independent) and are scaled on-device by the density: Exp(mean=d) = d * E.
        """
        e = rng.standard_exponential(size=tuple(density.shape))
        e = torch.as_tensor(e, dtype=density.dtype, device=density.device)
        samples = density.clamp(min=0.0) * e
        return _store(torch.floor(samples).clamp(max=float(self.max_trees_per_cell)))

    def _sun_directions(self):
        """Unit sun vectors (East, North, Up) sampled over the clear-sky daily arc.

        The sun is sampled at the configured latitude over `num_hours` hour angles (across
        +/- hour_half_angle each side of solar noon) and `num_seasons` declinations (equinox
        only when 1, else spread over +/- axial_tilt). Standard solar-position geometry gives,
        for latitude phi, declination delta and hour angle h:
            Up    = sin(phi) sin(delta) + cos(phi) cos(delta) cos(h)
            East  = -cos(delta) sin(h)
            North =  sin(delta) cos(phi) - cos(delta) sin(phi) cos(h)
        so at solar noon the sun sits due south (North < 0) at elevation 90-phi -- which is why
        south-facing slopes catch it. Samples below the horizon (Up <= 0) are dropped. Returns
        three numpy arrays (E, N, U), one entry per above-horizon sample (possibly empty)."""
        phi = np.deg2rad(self.latitude_deg)
        if self.sun_num_seasons <= 1:
            decls = np.array([0.0])                          # equinox
        else:
            decls = np.deg2rad(np.linspace(-self.axial_tilt_deg, self.axial_tilt_deg,
                                           self.sun_num_seasons))
        hours = np.deg2rad(np.linspace(-self.sun_hour_half_angle, self.sun_hour_half_angle,
                                       max(1, self.sun_num_hours)))
        E, N, U = [], [], []
        for d in decls:
            for h in hours:
                up = np.sin(phi) * np.sin(d) + np.cos(phi) * np.cos(d) * np.cos(h)
                if up <= 0.0:                                # sun below the horizon
                    continue
                E.append(-np.cos(d) * np.sin(h))
                N.append(np.sin(d) * np.cos(phi) - np.cos(d) * np.sin(phi) * np.cos(h))
                U.append(up)
        return np.array(E), np.array(N), np.array(U)

    def sunlight_exposure(self, elevation_m):
        """Average relative solar exposure per cell, in [0, 1] (dimensionless).

        A static field: how much sunlight a cell receives on average over the day/year,
        relative to a flat, fully-exposed surface. It folds in the two dominant terrain effects:
          * slope/aspect -- the Lambertian cos(incidence) between the surface normal and the sun,
            so (northern hemisphere) south-facing slopes catch the sun and north-facing ones are
            dim; and
          * cast shadow -- a horizon ray-march toward each sun position, so deep valleys and
            pole-facing hollows shaded by the surrounding terrain lose their direct beam.
        The sun is sampled along the clear-sky daily arc over a few seasons (see _sun_directions);
        each sample is weighted by sin(elevation) as a beam-strength proxy and gated by the
        cast-shadow test. A small diffuse_fraction of isotropic skylight is added as a floor, so a
        fully shaded cell stays dim rather than black. Drives the baked surface temperature
        (surface_temperature): sunnier cells settle warmer, shaded cells cooler.

        Built on torch (grid_sample for the shadow march); a NumPy elevation is accepted and
        returns NumPy float32, a torch tensor returns a torch tensor in the compute dtype.

        Accepts either a single (H, W) elevation or a batched (B, H, W) stack of maps: a 2-D input
        returns a 2-D field, a batched input returns (B, H, W) with every map evaluated together in
        one pass (the whole shadow ray-march runs across the batch on the GPU). The sun-sample
        geometry is terrain-independent, so it is computed once and shared over the batch.
        """
        import torch
        as_torch = _is_torch(elevation_m)
        t = elevation_m if as_torch else torch.as_tensor(elevation_m, dtype=torch.float64)
        squeeze = t.ndim == 2                     # remember to drop the batch dim on a single-map call
        if squeeze:
            t = t[None]
        device, dtype = t.device, t.dtype
        B, H, W = t.shape
        dz = float(self.cell_size_m)
        diffuse = self.sun_diffuse_fraction

        def _finish(out):
            out = out[0] if squeeze else out
            return _store(out) if as_torch else out.cpu().numpy().astype(np.float32)

        E, N, U = self._sun_directions()
        if E.size == 0:                          # polar night: only the diffuse floor reaches the ground
            return _finish(torch.full((B, H, W), diffuse, dtype=dtype, device=device))

        Es = torch.as_tensor(E, dtype=dtype, device=device)            # (S,)
        Ns = torch.as_tensor(N, dtype=dtype, device=device)
        Us = torch.as_tensor(U, dtype=dtype, device=device)
        S = Es.shape[0]

        # Upward surface normal from each map's terrain gradient. Grid convention: row i runs south,
        # column j runs east, so the north gradient is -d/di. n = (-dz/dE, -dz/dN, 1)/|.|.
        gy, gx = torch.gradient(t, spacing=dz, dim=(1, 2))             # (B, H, W) each: gy=dz/di, gx=dz/dj
        inv_norm = torch.rsqrt(gx * gx + gy * gy + 1.0)
        nE, nN, nU = -gx * inv_norm, gy * inv_norm, inv_norm           # (B, H, W) each

        # Lambertian incidence for every (map, sun sample) pair (clamped at the terminator).
        cos_inc = (nE[:, None] * Es[None, :, None, None]
                   + nN[:, None] * Ns[None, :, None, None]
                   + nU[:, None] * Us[None, :, None, None]).clamp(min=0.0)   # (B, S, H, W)

        # Cast shadow: march each cell toward the sun's horizontal direction and test whether the
        # terrain ever rises above the line of sight (height + horizontal_distance * tan(elevation)).
        # The ray geometry is map-independent, so the sample coordinates are shared across the batch
        # and only the sampled terrain differs between maps.
        visible = torch.ones((B, S, H, W), dtype=dtype, device=device)
        if self.sun_shadows:
            hmag = torch.sqrt(Es * Es + Ns * Ns).clamp(min=1e-6)       # horizontal speed (S,)
            tan_elev = Us / hmag                                       # rise per unit horizontal run
            di = (-Ns / hmag)                                         # row step per cell (north = -i)
            dj = (Es / hmag)                                          # col step per cell
            ii = torch.arange(H, device=device, dtype=dtype)
            jj = torch.arange(W, device=device, dtype=dtype)
            gi, gj = torch.meshgrid(ii, jj, indexing="ij")            # (H, W) base coords
            terr = t[:, None, None].expand(B, S, 1, H, W).reshape(B * S, 1, H, W)  # grid_sample input
            shadowed = torch.zeros((B, S, H, W), dtype=torch.bool, device=device)
            zmin, zmax = float(t.min()), float(t.max())
            tan_min = float(tan_elev.min())
            steps = self.sun_max_shadow_steps or max(H, W)
            for k in range(1, steps + 1):
                # Once the lowest line of sight clears the highest terrain, nothing can shadow.
                if tan_min > 0.0 and zmin + k * dz * tan_min > zmax:
                    break
                r = gi[None] + di[:, None, None] * k                  # (S, H, W) sample row
                c = gj[None] + dj[:, None, None] * k                  # sample col
                gxn = c / (W - 1) * 2.0 - 1.0                          # normalised grid_sample coords
                gyn = r / (H - 1) * 2.0 - 1.0
                grid = torch.stack((gxn, gyn), dim=-1)                 # (S, H, W, 2)
                grid = grid[None].expand(B, S, H, W, 2).reshape(B * S, H, W, 2)
                sampled = torch.nn.functional.grid_sample(
                    terr, grid, mode="bilinear", padding_mode="zeros", align_corners=True
                ).reshape(B, S, H, W)                                  # terrain along the ray, per map
                los = t[:, None] + (k * dz) * tan_elev[None, :, None, None]   # line-of-sight height
                in_bounds = (r >= 0) & (r <= H - 1) & (c >= 0) & (c <= W - 1)  # (S, H, W)
                shadowed |= in_bounds[None] & (sampled > los)
            visible = (~shadowed).to(dtype)

        # Beam-weighted (sin elevation) average of the unshadowed cosine incidence, plus a uniform
        # diffuse skylight floor so fully shaded ground is dim, not black.
        direct = (Us[None, :, None, None] * cos_inc * visible).sum(1) / Us.sum()   # (B, H, W)
        out = (diffuse + (1.0 - diffuse) * direct).clamp(0.0, 1.0)
        return _finish(out)

    def flat_exposure(self) -> float:
        """The sunlight_exposure value a flat, fully-open cell receives (no slope, no shadow).

        A flat cell's surface normal points straight up, so its incidence cosine is sin(elevation)
        for every sun sample; the beam-weighted average is then Sum(U^2)/Sum(U), plus the diffuse
        floor. This is the reference exposure that reads as "ordinary" ground: surface_temperature
        leaves a cell at this exposure at the unmodulated ambient profile, so flat ground stays at
        T_REF. Pure function of the sun-sampling geometry (independent of terrain)."""
        _, _, U = self._sun_directions()
        if U.size == 0:
            return self.sun_diffuse_fraction
        direct_flat = float((U * U).sum() / U.sum())
        return self.sun_diffuse_fraction + (1.0 - self.sun_diffuse_fraction) * direct_flat

    def surface_temperature(self, elevation_m, sunlight):
        """Baked per-cell surface (rest) temperature [K] from elevation and solar exposure.

        The surface a cell settles at is its elevation ambient profile T_REF - Gamma*z, warmed or
        cooled by how much sun it gets. The bare radiative response (absorbed solar ~ exposure,
        emission ~ sigma*T^4) would give T ~ ambient*(exposure/flat)^(1/4) -- realistic in a vacuum,
        but on Earth the atmosphere insulates and warm surrounding air mixes laterally, so shaded
        ground is only "a bit cool", not lunar-cold. We model that by blending the bare radiative
        temperature back toward the ambient air by `temp_moderation` in [0, 1]: 0 = pure radiative
        swing, 1 = no solar effect (the old uniform profile). A flat, open cell (exposure = flat)
        stays exactly at the ambient profile, so flat ground keeps the calibrated T_REF.

        This is the surface skin temperature the radiation balance equilibrates to and the air is
        initialised from; sunnier slopes run warmer, shaded valleys cooler.
        """
        ambient = self.air_temperature_profile(elevation_m)            # T_REF - Gamma*z
        rel = sunlight / self.flat_exposure()                         # exposure relative to flat ground
        radiative = ambient * rel ** 0.25                             # bare (vacuum) radiative response
        m = self.sun_temp_moderation
        return _store(ambient + (1.0 - m) * (radiative - ambient))


def generate_maps(
    gen: MapGenerator,
    grid_size: int,
    count: int,
    rng: np.random.Generator,
    seeds: list[int | None] | None = None,
) -> list[MapData]:
    """Build `count` fully-baked, unperturbed initial states in one batched GPU pass.

    The whole chunk's rest states are built as (count, ...) torch tensors on gen.device (CUDA
    when available): the octave Perlin kernel samples every terrain/fuel field for the batch at
    once (one random base per map) and all the field math -- temperature, pressure, mass, oxygen,
    sunlight/shadows, biomes, tree counts -- broadcasts over the leading map dim, so a chunk of
    maps is built together instead of one at a time. Map generation is not required to be
    deterministic (only the maps, once written, must stay stable), so every per-map random draw
    (noise bases, tree counts, wind) comes straight from a single shared numpy rng. Fields move
    back to numpy only at the MapData boundary, where the batch is sliced into one MapData per map.

    Fire is intentionally not baked here: ignition (random spawn or click) is a
    runtime concern applied by the environment, so maps describe the world at rest.

    All fields are SI: terrain [m], air_temperatures [K], mass [kg/m^2], oxygen
    [kg/m^3], fuel [kg/m^2]. `sunlight` is a static dimensionless [0,1] solar-exposure field that
    drives the baked surface (rest) temperature: sunnier cells warmer, shaded cells cooler.
    """
    B = int(count)
    device = gen.device
    if seeds is None:
        seeds = [None] * B
    names = gen.fuel_type_names

    # One random Perlin base per map for each noise field. Order/independence between fields does
    # not matter (generation need not be reproducible), so every base is drawn from the shared rng.
    def bases():
        return list(rng.integers(0, 256, size=B))

    # Normalised [0,1] relief is scaled to metres for the elevation (one field per map).
    relief = gen.create_terrain(grid_size, base=bases())
    terrain = relief * gen.elev_max

    # Static solar exposure [0,1]: south-facing slopes catch the sun, shadowed valleys lose it.
    sunlight = gen.sunlight_exposure(terrain)

    # Surface (rest) temperature: the elevation profile warmed/cooled by how much sun each cell
    # gets (sunny slopes warmer, shaded valleys cooler), atmosphere-moderated. This is the radiative
    # equilibrium the surface settles at and the temperature the air is initialised to; the runtime
    # ground<->air radiation coupling then carries it into the air. Oxygen follows the air density.
    surface_temperature = gen.surface_temperature(terrain, sunlight)
    air_temperatures = surface_temperature
    oxygen = gen.oxygen_profile(terrain, air_temperatures)

    N = gen.num_fuel_types
    fuel = torch.zeros((B, N, grid_size, grid_size), dtype=torch.float64, device=device)

    # Vegetation biomes: each cell has a soft membership in woodland / grassland / shrubland (from
    # its elevation and baked surface temperature) that blends smoothly across the biome borders, so
    # the fuel cross-fades there instead of stepping. Within a biome the density comes purely from a
    # Perlin noise map (elevation/slope/temperature no longer shape it): grass grows on the grassland
    # (and, more thinly, under the woodland), shrub on the shrubland, and trees in every biome at a
    # per-biome mean -- dense woodland canopy, a sparse grassland sprinkle, and a modest shrubland
    # cover. The weights are linear, so biome_grass_multiplier/biome_tree_density blend the densities.
    w_wood, w_grass, w_shrub = gen.biome_weights(terrain, surface_temperature)

    if "grass" in names:
        # Grass: continuous Perlin density [kg/m^2], full-strength on the grassland and thinned to
        # woodland_grass_fraction under the woodland canopy (none on the shrubland).
        gi = names.index("grass")
        grass_noise = gen.apply_noise_floor(
            gen.create_grid(grid_size, scale=float(gen.fuel_noise_scales[gi]), base=bases()), gi)
        grass_biome = gen.biome_grass_multiplier(w_grass, w_wood)
        fuel[:, gi] = grass_noise * float(gen.spawn_densities[gi]) * grass_biome

    if "shrub" in names:
        # Shrub: continuous Perlin density [kg/m^2] (its own noise field, so it is spatially
        # independent of the grass), confined to the shrubland.
        si = names.index("shrub")
        shrub_noise = gen.apply_noise_floor(
            gen.create_grid(grid_size, scale=float(gen.fuel_noise_scales[si]), base=bases()), si)
        fuel[:, si] = shrub_noise * float(gen.spawn_densities[si]) * w_shrub

    if gen.tree_indices:
        # Trees: one stochastic per-cell count (exponential mean = biome-set density * noise, capped)
        # drives every co-located tree component (canopy + bole), each scaled by its own fuel_per_tree
        # -- so the canopy and bole of the same trees always share a cell. The count's noise scale is
        # taken from the first tree component; its density is set per biome (biome_tree_density).
        t0 = gen.tree_indices[0]
        tree_noise = gen.create_grid(grid_size, scale=float(gen.fuel_noise_scales[t0]), base=bases())
        density = tree_noise * gen.biome_tree_density(w_wood, w_grass, w_shrub)
        tree_counts = gen.sample_tree_counts(density, rng)             # (B, H, W)
        for ti in gen.tree_indices:
            fuel[:, ti] = tree_counts * float(gen.fuel_per_tree[ti])

    # Each fuel type starts at ambient air temperature.
    fuel_temperatures = air_temperatures.unsqueeze(1).expand(B, N, grid_size, grid_size)

    # Shallow boundary-layer mass at its level-lid rest state (thinner over high terrain).
    mass = gen.boundary_layer_mass(terrain)

    # Per-map synoptic (prevailing) wind: random bearing, speed from the configured range.
    speed = rng.uniform(gen.ambient_speed_min, gen.ambient_speed_max, size=B)
    bearing = rng.uniform(0.0, 2.0 * np.pi, size=B)
    ambient_wind_x = speed * np.cos(bearing)
    ambient_wind_y = speed * np.sin(bearing)

    def to_np(t: torch.Tensor) -> np.ndarray:
        return t.detach().contiguous().to("cpu").numpy().astype(np.float32)

    terrain_np = to_np(terrain)
    sunlight_np = to_np(sunlight)
    air_np = to_np(air_temperatures)
    mass_np = to_np(mass)
    oxygen_np = to_np(oxygen)
    fuel_np = to_np(fuel)
    fuel_temp_np = to_np(fuel_temperatures)

    maps: list[MapData] = []
    for i in range(B):
        air_i = air_np[i]
        oxygen_i = oxygen_np[i]
        maps.append(MapData(
            terrain=terrain_np[i],
            sunlight=sunlight_np[i],
            air_temperatures=air_i,
            mass=mass_np[i],
            oxygen=oxygen_i,
            fuel=fuel_np[i],
            fuel_temperatures=fuel_temp_np[i],
            fuel_type_names=list(gen.fuel_type_names),
            grid_size=grid_size,
            temp_eq=air_i.copy(),
            oxygen_eq=oxygen_i.copy(),
            seed=seeds[i],
            source="gen_maps",
            ambient_wind_x=float(ambient_wind_x[i]),
            ambient_wind_y=float(ambient_wind_y[i]),
        ))
    return maps


def generate_map(
    gen: MapGenerator,
    grid_size: int,
    rng: np.random.Generator,
    seed: int | None = None,
) -> MapData:
    """Build one fully-baked, unperturbed initial state (a single-map `generate_maps`).

    Retained for the single-map callers (tools/bench.py, tests); the map pipeline builds a whole
    chunk at once via generate_maps.
    """
    return generate_maps(gen, grid_size, 1, rng, seeds=[seed])[0]


def spin_up_maps(maps: list[MapData], cfg: dict | None, steps: int, chunk: int = 16) -> None:
    """Develop the orographic wind for each map and bake it back in (in place).

    Each map is built in its level-lid rest state (uniform synoptic wind, flat free surface, sharp
    sunlight air temperature). That is not the developed world a real terrain produces, and the
    solver only relaxes into it over hundreds of ticks. We run that relaxation here, once, at
    generation time: stack the rest states into a batch and step the FULL field physics `steps`
    times (the engine is rank-agnostic, so the batch runs far faster than one map at a time), then
    write the developed mass, air temperature and prognostic wind back into each MapData. The
    level-lid rest mass is preserved as `mass_eq` (the open-boundary sponge target) and the sunlight
    rest temperature as `temp_eq`, so the runtime sponge still injects the correct free stream.

    The air temperature is developed too, not reset to the sharp rest profile: the wind advects it
    while the surface radiation balance and fuel/air coupling hold it at its sunlight range, so it
    settles into the wind-smeared pattern it actually equilibrates to -- which is why we step
    step_fields here, not step_dynamics (wind alone has no thermal source and would wash the air out
    toward uniform). The map is quiescent (fuel far below ignition), so combustion stays inert. A
    loaded map then starts already in this developed flow instead of visibly smearing over the first
    frames.

    The flow is genuinely unsteady (lee/gravity waves do not settle to a frozen field), so the
    baked state is a developed snapshot, not a fixed point: it loads already terrain-shaped and
    keeps evolving naturally.

    Spin-up is the dominant cost of map generation (~19x the Perlin rest-state build), all in the
    batched solver. On CPU its per-map throughput peaks around a batch of ~16 (wider batches lose
    to memory bandwidth), so `chunk` defaults there rather than to the pipeline's save chunk.
    """
    if steps <= 0 or not maps:
        return
    # Simulation is imported here (not at module load) so the engine dependency stays confined
    # to the spin-up; the rest-state build only needs the lightweight torch Perlin kernel.
    from simulation import Simulation, SimState

    sim = Simulation(cfg)

    def t(arr):
        return torch.as_tensor(arr, dtype=sim.dtype, device=sim.device)

    for start in range(0, len(maps), chunk):
        batch = maps[start:start + chunk]
        mass = torch.stack([t(m.mass) for m in batch])              # (B, H, W) level-lid rest
        air  = torch.stack([t(m.air_temperatures) for m in batch])
        terrain = torch.stack([t(m.terrain) for m in batch])
        amb_x = t([m.ambient_wind_x for m in batch]).view(-1, 1, 1)
        amb_y = t([m.ambient_wind_y for m in batch]).view(-1, 1, 1)
        s = SimState(
            mass=mass,
            air_temperatures=air.unsqueeze(-3),        # air level axis (-3); L = 1 in Phase 1
            ground_temperature=air.clone(),
            fuel_temperatures=torch.stack([t(m.fuel_temperatures) for m in batch]),
            fuel=torch.stack([t(m.fuel) for m in batch]),
            oxygen=torch.stack([t(m.oxygen) for m in batch]),
            terrain=terrain,
            temp_eq=air.clone().unsqueeze(-3),                     # rest temperature (sponge target)
            oxygen_eq=torch.stack([t(m.oxygen) for m in batch]),
            mass_eq=mass.clone(),                                   # level-lid rest mass (sponge target)
            x_wind_vel=torch.zeros_like(mass) + amb_x,
            y_wind_vel=torch.zeros_like(mass) + amb_y,
            u_amb_x=amb_x,
            u_amb_y=amb_y,
            radiant_flux=torch.zeros_like(mass),
            channel_gain=sim.compute_channel_gain(terrain, amb_x, amb_y),
        )
        # Step the field physics (not just the wind): the wind develops the orographic flow, and the
        # surface radiation balance + fuel/air coupling hold the air at its sunlight range while the
        # wind advects it -- so the air settles into its developed, wind-smeared state rather than the
        # sharp rest profile (or, under wind-only stepping, washing out toward uniform). We reuse the
        # runtime step_fields (one source of truth) but turn off the two stages a quiescent map does
        # not need: oxygen transport (passive -- never affects mass/wind/air) and combustion (no fire
        # at rest). Both are ~30% / ~43% of a tick, so this is the bulk of the spin-up cost, and the
        # baked mass/air/wind are identical to the full step (verified to round-off).
        for _ in range(steps):
            sim.step_fields(s, advance_oxygen=False, advance_fire=False)

        dev_mass = s.mass.cpu().numpy()
        dev_air  = s.air_temperatures.select(-3, 0).cpu().numpy()   # surface level -> (B, H, W)
        dev_u    = s.x_wind_vel.cpu().numpy()
        dev_v    = s.y_wind_vel.cpu().numpy()
        for i, m in enumerate(batch):
            m.mass_eq = m.mass.copy()                  # the level-lid rest state -> sponge target
            m.mass = dev_mass[i].astype(np.float32)    # developed initial state
            # Developed (wind-smeared) air temperature: the equilibrium the full physics settles
            # into, so a loaded map starts already in its developed flow instead of the sharp
            # sunlight rest profile that visibly smears over the first frames.
            m.air_temperatures = dev_air[i].astype(np.float32)
            m.x_wind_vel = dev_u[i].astype(np.float32)
            m.y_wind_vel = dev_v[i].astype(np.float32)
        print(f"  spun up maps {start + 1}-{start + len(batch)} / {len(maps)} ({steps} ticks)")


def _run_shard(
    config: dict,
    grid_size: int,
    observed: int,
    pad: int,
    *,
    count: int,
    start_index: int,
    out: str,
    prefix: str,
    chunk: int,
    shard_id: int,
    num_shards: int,
    device: str | None,
    root_seq: np.random.SeedSequence,
) -> None:
    """Generate one shard of a (possibly multi-process) run: the maps whose global index falls in
    this shard's contiguous block of the `count` total.

    Sharding is what makes generation safe to fan out across processes / GPUs / nodes without
    duplicate maps. Two guarantees keep the shards disjoint:
      * output index range -- shard s owns the contiguous global block [s*count//N, (s+1)*count//N),
        written at `start_index + global_index`, so no two shards ever target the same file; and
      * random stream -- each shard draws from its own SeedSequence child (root_seq.spawn(N)[s]),
        which numpy guarantees is statistically independent, so no two shards draw the same maps
        even though generation is non-deterministic. Per-map provenance seeds are spawned from the
        same shard stream, so every written map still records a distinct seed.

    A single-shard run (num_shards == 1) is exactly the old whole-run behaviour.
    """
    if device is not None:
        config.setdefault("environment", {})["device"] = device
    gen = MapGenerator(config)
    chunk = max(1, chunk)

    lo = shard_id * count // num_shards
    hi = (shard_id + 1) * count // num_shards
    n_shard = hi - lo
    tag = "" if num_shards == 1 else f"[shard {shard_id}/{num_shards}] "

    shard_seq = root_seq.spawn(num_shards)[shard_id]
    rng = np.random.default_rng(shard_seq)
    map_seeds = [int(s.generate_state(1)[0]) for s in shard_seq.spawn(n_shard)]

    if n_shard == 0:
        print(f"{tag}no maps for this shard.", flush=True)
        return

    print(f"{tag}Generating {n_shard} map(s) [global {start_index + lo}..{start_index + hi - 1}] at "
          f"{grid_size}x{grid_size} (observed {observed}x{observed} + {pad}-cell sponge padding) "
          f"into '{out}/' (device={gen.device}, chunk={chunk}, spinup={gen.spinup_steps} ticks) ...",
          flush=True)

    # Process the maps in chunks: build each rest state on the device (the torch Perlin kernel and
    # field math run on the GPU when present), spin up the whole chunk in one batched solver pass,
    # then save it before moving on. So progress is durable (each chunk is on disk before the next
    # starts) and memory stays bounded (one chunk resident at a time).
    done = 0
    for start in range(0, n_shard, chunk):
        n = min(start + chunk, n_shard) - start
        maps = generate_maps(gen, grid_size, n, rng, seeds=map_seeds[start:start + n])
        if gen.spinup_steps > 0:
            spin_up_maps(maps, config, gen.spinup_steps)
        for off, m in enumerate(maps):
            gidx = start_index + lo + start + off
            save_map(f"{out}/{prefix}_{gidx:04d}.npz", m)
        done += len(maps)
        print(f"  {tag}[{done}/{n_shard}] generated + spun up + saved", flush=True)
    print(f"{tag}Done ({n_shard} maps).", flush=True)


def _shard_worker(rank, config, meta, gpus, root_seq) -> None:
    """torch.multiprocessing entry point: run shard `rank` pinned to gpus[rank] (see main --gpus)."""
    _run_shard(
        config, meta["grid_size"], meta["observed"], meta["pad"],
        count=meta["count"], start_index=meta["start_index"], out=meta["out"],
        prefix=meta["prefix"], chunk=meta["chunk"],
        shard_id=rank, num_shards=len(gpus), device=gpus[rank], root_seq=root_seq,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Firecracker maps")
    parser.add_argument("--count", type=int, default=1, help="number of maps to generate")
    parser.add_argument("--out", default="maps", help="output directory (default: maps)")
    parser.add_argument("--config", metavar="PATH", help="path to a TOML config file")
    parser.add_argument("--seed", type=int, default=None, help="master seed for reproducibility")
    parser.add_argument("--prefix", default="map", help="output filename prefix")
    parser.add_argument(
        "--start-index", type=int, default=0,
        help="first output file index (e.g. 8 to append after map_0007 without overwriting)",
    )
    parser.add_argument(
        "--device", default=None,
        help="generation device: 'cuda', 'cpu', or omit to auto-select (cuda when available). "
             "Overrides [environment].device from the config.",
    )
    parser.add_argument(
        "--chunk", type=int, default=32,
        help="maps per generate->spin-up->save chunk (bounds memory; saves progress durably)",
    )
    parser.add_argument(
        "--shards", type=int, default=1,
        help="split --count into this many disjoint shards for multi-process / multi-node "
             "generation; this process builds only its --shard-id slice (no duplicate maps across "
             "shards). Launch one process per rank with the same --count/--seed/--shards.",
    )
    parser.add_argument(
        "--shard-id", type=int, default=0,
        help="which shard (0..shards-1) this process generates (see --shards)",
    )
    parser.add_argument(
        "--gpus", default=None,
        help="comma-separated devices (e.g. 'cuda:0,cuda:1,cuda:2,cuda:3') to fan this run across on "
             "THIS node: spawns one process per device, each building a disjoint shard. Overrides "
             "--device/--shards/--shard-id.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    observed = int((config or {}).get("environment", {}).get("grid_size", DEFAULT_GRID_SIZE))
    # Maps are generated at the padded physics size (observed + 2*boundary_pad): the open-boundary
    # sponge lives in the ring, so the observed interior the env crops out is sponge-free.
    pad = boundary_pad(config)
    grid_size = physics_grid_size(config, observed)
    # Shared seed root: every shard spawns an independent child stream from this, so the shards are
    # coordinated (given --seed) yet never draw the same maps.
    root_seq = np.random.SeedSequence(args.seed)

    # Fan out across local GPUs: one process per device, each a disjoint shard. Parent must not touch
    # CUDA before spawning, so device selection is deferred into the workers (spawn start method).
    if args.gpus:
        gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
        if not gpus:
            parser.error("--gpus was empty")
        import torch.multiprocessing as mp
        meta = dict(count=args.count, start_index=args.start_index, out=args.out,
                    prefix=args.prefix, chunk=args.chunk, grid_size=grid_size,
                    observed=observed, pad=pad)
        print(f"Fanning {args.count} map(s) across {len(gpus)} device(s): {', '.join(gpus)}",
              flush=True)
        mp.spawn(_shard_worker, args=(config, meta, gpus, root_seq), nprocs=len(gpus), join=True)
        print("All shards done.", flush=True)
        return

    if not (0 <= args.shard_id < args.shards):
        parser.error(f"--shard-id must be in [0, {args.shards}) (got {args.shard_id})")

    _run_shard(
        config, grid_size, observed, pad,
        count=args.count, start_index=args.start_index, out=args.out, prefix=args.prefix,
        chunk=args.chunk, shard_id=args.shard_id, num_shards=args.shards,
        device=args.device, root_seq=root_seq,
    )


if __name__ == "__main__":
    main()
