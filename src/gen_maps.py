"""
gen_maps.py

Synthetic map producer for the Firecracker simulation.

Owns the procedural initial-state generation that used to live inside
Simulation/reset(): octave Perlin terrain, elevation-driven temperature/oxygen
profiles, the convective rest-state pressure, and the grass/tree fuel layers.
`MapGenerator` reads what it needs straight from the config dict and emits a
`MapData`, the same contract a future real-data importer would target.

The whole rest state is built as torch tensors on `MapGenerator.device` (CUDA when
available, else CPU -- set [environment].device or pass --device to force one), using
the vectorized torch Perlin kernel in perlin.py instead of a per-cell scalar loop. So
both the noise and the field math run on the GPU; only the per-map random scalars
(noise bases, wind, tree draws) are drawn from the numpy rng, which keeps output
reproducible and identical across devices. Fields move back to numpy at the MapData
boundary, then the chunk is spun up on the same device.

After the rest state is built, `spin_up_maps` runs the shallow-water wind solver
to its developed orographic state and bakes the result (mass, air temperature, and
the prognostic wind) back into each map, so a loaded map already shows the terrain-
shaped flow (windward blocking, lee/foehn acceleration) instead of relaxing into it
at runtime. This is the one place generation touches the physics engine; the emitted
`MapData` is still a plain baked state.

Run as a script to bake a batch of maps to disk:

    python src/gen_maps.py --count 8 --out maps --seed 0
"""

import argparse

import numpy as np
import torch

import physics_constants as pc
from config import load_config
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

        # Cell size [m] sets the grid spacing used to turn the elevation field into a
        # physical slope (rise/run) for the slope-based vegetation preference.
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
        # Half-saturation slope (rise/run) for the slope-based vegetation preference: cells at this
        # slope sit at the midpoint between the flat-loving and steep-loving extremes.
        self.slope_ref              = float(terrain.get("slope_ref",              0.5))
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

        fuel_types_cfg = (cfg or {}).get("fuel_types", {})
        self.fuel_type_names: list[str] = list(fuel_types_cfg.keys())
        self.num_fuel_types: int = len(self.fuel_type_names)

        def _ft(name: str, key: str, default: float) -> float:
            return float(fuel_types_cfg.get(name, {}).get(key, default))

        self.fuel_noise_scales = np.array([_ft(n, "noise_scale",      32.0) for n in self.fuel_type_names], dtype=np.float32)
        self.spawn_densities   = np.array([_ft(n, "spawn_density",     0.5)  for n in self.fuel_type_names], dtype=np.float32)

        # Slope-based vegetation preference: each fuel either prefers flat ground (grass) or steep
        # ground (trees), with slope_falloff sharpening that bias. slope_preference is "flat" or
        # "steep" (default flat); slope_falloff is the exponent on the normalised slope.
        self.slope_falloffs    = np.array([_ft(n, "slope_falloff",    1.0)  for n in self.fuel_type_names], dtype=np.float32)
        self.slope_prefer_steep = np.array(
            [str(fuel_types_cfg.get(n, {}).get("slope_preference", "flat")).lower() == "steep"
             for n in self.fuel_type_names]
        )

        # Tree components: any fuel type whose name starts with "tree" (e.g. tree_canopy, tree_bole)
        # is part of the same physical trees, so they share one per-cell count and spatial
        # distribution. fuel_per_tree is per type (the biomass each tree contributes to that
        # component); the count/distribution params (noise/spawn/slope) are read from the first
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

    def terrain_slope(self, elevation_m):
        """Per-cell terrain slope magnitude (rise/run, dimensionless) of the elevation field [m].

        Central differences along both grid axes with cell_size_m spacing (one-sided at the
        edges), then |grad z|. Drives the slope-based vegetation preference below."""
        dz = float(self.cell_size_m)
        if _is_torch(elevation_m):
            gy, gx = torch.gradient(elevation_m, spacing=dz)
            return _store((gx * gx + gy * gy).sqrt())
        gy, gx = np.gradient(elevation_m, dz)
        return _store(np.sqrt(gx * gx + gy * gy))

    def slope_vegetation_multiplier(self, slope, fuel_index: int):
        """Per-cell vegetation density scale from terrain slope, in [0, 1].

        The normalised slope s = slope/(slope + slope_ref) is 0 on flat ground and -> 1 on
        steep ground (= 0.5 at slope_ref). Flat-preferring fuels (grass) scale as (1 - s)^k and
        steep-preferring fuels (trees) as s^k, where k is the per-fuel slope_falloff -- so grass
        favours gentle ground and trees favour steeper slopes."""
        s = slope / (slope + self.slope_ref)
        k = float(self.slope_falloffs[fuel_index])
        if bool(self.slope_prefer_steep[fuel_index]):
            return _store(s ** k)
        return _store((1.0 - s) ** k)

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


def generate_map(
    gen: MapGenerator,
    grid_size: int,
    rng: np.random.Generator,
    seed: int | None = None,
) -> MapData:
    """Build one fully-baked, unperturbed initial state.

    The whole rest state is built as torch tensors on gen.device (CUDA when available),
    so the noise and field math run on the GPU; the per-map random scalars (noise bases,
    wind, tree draws) come from the numpy rng, so output is reproducible and identical
    across devices. Fields are moved back to numpy only at the MapData boundary.

    Fire is intentionally not baked here: ignition (random spawn or click) is a
    runtime concern applied by the environment, so maps describe the world at rest.

    All fields are SI: terrain [m], air_temperatures [K], mass [kg/m^2], oxygen
    [kg/m^3], fuel [kg/m^2].
    """
    # Normalised [0,1] relief is scaled to metres for the elevation; its per-cell slope
    # (rise/run) drives the vegetation preference (flat-loving grass, steep-loving trees).
    relief = gen.create_terrain(grid_size, rng=rng)
    terrain = relief * gen.elev_max
    slope = gen.terrain_slope(terrain)

    # Air temperature from the lapse rate; oxygen from the hydrostatic air density.
    air_temperatures = gen.air_temperature_profile(terrain)
    oxygen = gen.oxygen_profile(terrain, air_temperatures)

    N = gen.num_fuel_types
    fuel = torch.zeros((N, grid_size, grid_size), dtype=torch.float64, device=gen.device)

    names = gen.fuel_type_names
    if "grass" in names:
        # Grass: continuous Perlin density [kg/m^2], biased toward flatter (low-slope) ground.
        gi = names.index("grass")
        grass_noise = gen.create_grid(grid_size, scale=float(gen.fuel_noise_scales[gi]), rng=rng)
        grass_slope = gen.slope_vegetation_multiplier(slope, gi)
        fuel[gi] = grass_noise * float(gen.spawn_densities[gi]) * grass_slope

    if "shrub" in names:
        # Shrub: continuous Perlin density [kg/m^2], slope-biased like grass (its own noise field
        # so the two layers are spatially independent).
        si = names.index("shrub")
        shrub_noise = gen.create_grid(grid_size, scale=float(gen.fuel_noise_scales[si]), rng=rng)
        shrub_slope = gen.slope_vegetation_multiplier(slope, si)
        fuel[si] = shrub_noise * float(gen.spawn_densities[si]) * shrub_slope

    if gen.tree_indices:
        # Trees: one stochastic per-cell count (exponential mean = slope-biased noise density,
        # capped) drives every co-located tree component (canopy + bole), each scaled by its own
        # fuel_per_tree -- so the canopy and bole of the same trees always share a cell. The count's
        # spatial params are taken from the first tree component.
        t0 = gen.tree_indices[0]
        tree_noise = gen.create_grid(grid_size, scale=float(gen.fuel_noise_scales[t0]), rng=rng)
        tree_slope = gen.slope_vegetation_multiplier(slope, t0)
        density = tree_noise * float(gen.spawn_densities[t0]) * tree_slope
        tree_counts = gen.sample_tree_counts(density, rng)
        for ti in gen.tree_indices:
            fuel[ti] = tree_counts * float(gen.fuel_per_tree[ti])

    # Each fuel type starts at ambient air temperature.
    fuel_temperatures = air_temperatures.unsqueeze(0).expand(N, -1, -1)

    # Shallow boundary-layer mass at its level-lid rest state (thinner over high terrain).
    mass = gen.boundary_layer_mass(terrain)

    # Per-map synoptic (prevailing) wind: random bearing, speed from the configured range.
    speed = float(rng.uniform(gen.ambient_speed_min, gen.ambient_speed_max))
    bearing = float(rng.uniform(0.0, 2.0 * np.pi))
    ambient_wind_x = speed * np.cos(bearing)
    ambient_wind_y = speed * np.sin(bearing)

    def to_np(t: torch.Tensor) -> np.ndarray:
        return t.detach().contiguous().to("cpu").numpy().astype(np.float32)

    terrain_np = to_np(terrain)
    air_np = to_np(air_temperatures)
    oxygen_np = to_np(oxygen)

    return MapData(
        terrain=terrain_np,
        air_temperatures=air_np,
        mass=to_np(mass),
        oxygen=oxygen_np,
        fuel=to_np(fuel),
        fuel_temperatures=to_np(fuel_temperatures),
        fuel_type_names=list(gen.fuel_type_names),
        grid_size=grid_size,
        temp_eq=air_np.copy(),
        oxygen_eq=oxygen_np.copy(),
        seed=seed,
        source="gen_maps",
        ambient_wind_x=ambient_wind_x,
        ambient_wind_y=ambient_wind_y,
    )


def spin_up_maps(maps: list[MapData], cfg: dict | None, steps: int, chunk: int = 16) -> None:
    """Develop the orographic wind for each map and bake it back in (in place).

    Each map is built in its level-lid rest state (uniform synoptic wind, flat free surface).
    That is not the wind a real terrain produces, and the prognostic solver only relaxes into
    the terrain-shaped flow over hundreds of ticks. We run that relaxation here, once, at
    generation time: stack the rest states into a batch and step the shallow-water core `steps`
    times (the engine is rank-agnostic, so the batch runs far faster than one map at a time),
    then write the developed mass, air temperature and prognostic wind back into each MapData.
    The level-lid rest mass is preserved as `mass_eq` (the open-boundary sponge target) and the
    rest temperature as `temp_eq`, so the runtime sponge still injects the correct free stream.

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
        amb_x = t([m.ambient_wind_x for m in batch]).view(-1, 1, 1)
        amb_y = t([m.ambient_wind_y for m in batch]).view(-1, 1, 1)
        s = SimState(
            mass=mass,
            air_temperatures=air,
            ground_temperature=air.clone(),
            fuel_temperatures=torch.stack([t(m.fuel_temperatures) for m in batch]),
            fuel=torch.stack([t(m.fuel) for m in batch]),
            oxygen=torch.stack([t(m.oxygen) for m in batch]),
            terrain=torch.stack([t(m.terrain) for m in batch]),
            temp_eq=air.clone(),                                    # rest temperature (sponge target)
            oxygen_eq=torch.stack([t(m.oxygen) for m in batch]),
            mass_eq=mass.clone(),                                   # level-lid rest mass (sponge target)
            x_wind_vel=torch.zeros_like(mass) + amb_x,
            y_wind_vel=torch.zeros_like(mass) + amb_y,
            u_amb_x=amb_x,
            u_amb_y=amb_y,
            radiant_flux=torch.zeros_like(mass),
        )
        for _ in range(steps):
            sim.step_dynamics(s)

        dev_mass = s.mass.cpu().numpy()
        dev_air  = s.air_temperatures.cpu().numpy()
        dev_u    = s.x_wind_vel.cpu().numpy()
        dev_v    = s.y_wind_vel.cpu().numpy()
        for i, m in enumerate(batch):
            m.mass_eq = m.mass.copy()                  # the level-lid rest state -> sponge target
            m.mass = dev_mass[i].astype(np.float32)    # developed initial state
            m.air_temperatures = dev_air[i].astype(np.float32)
            m.x_wind_vel = dev_u[i].astype(np.float32)
            m.y_wind_vel = dev_v[i].astype(np.float32)
        print(f"  spun up maps {start + 1}-{start + len(batch)} / {len(maps)} ({steps} ticks)")


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
    args = parser.parse_args()

    config = load_config(args.config)
    if args.device is not None:
        config.setdefault("environment", {})["device"] = args.device
    grid_size = int((config or {}).get("environment", {}).get("grid_size", DEFAULT_GRID_SIZE))
    gen = MapGenerator(config)
    chunk = max(1, args.chunk)

    # Per-map seeds, spawned once so each map is reproducible regardless of chunking.
    children = np.random.SeedSequence(args.seed).spawn(args.count)
    seeds = [int(c.generate_state(1)[0]) for c in children]

    print(f"Generating {args.count} map(s) at {grid_size}x{grid_size} into '{args.out}/' "
          f"(device={gen.device}, chunk={chunk}, spinup={gen.spinup_steps} ticks) ...", flush=True)

    # Process the maps in chunks: build each rest state on the device (the torch Perlin kernel and
    # field math run on the GPU when present), spin up the whole chunk in one batched solver pass,
    # then save it before moving on. So progress is durable (each chunk is on disk before the next
    # starts) and memory stays bounded (one chunk resident at a time).
    done = 0
    for start in range(0, args.count, chunk):
        idx = range(start, min(start + chunk, args.count))
        maps = [generate_map(gen, grid_size, np.random.default_rng(children[i]), seed=seeds[i])
                for i in idx]
        if gen.spinup_steps > 0:
            spin_up_maps(maps, config, gen.spinup_steps)
        for off, m in enumerate(maps):
            save_map(f"{args.out}/{args.prefix}_{args.start_index + start + off:04d}.npz", m)
        done += len(maps)
        print(f"  [{done}/{args.count}] generated + spun up + saved", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
