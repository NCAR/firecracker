"""
gen_maps.py

Synthetic map producer for the Firecracker simulation.

Owns the procedural initial-state generation that used to live inside
Simulation/reset(): octave Perlin terrain, elevation-driven temperature/oxygen
profiles, the convective rest-state pressure, and the grass/tree fuel layers.
`MapGenerator` reads what it needs straight from the config dict and emits a
`MapData`, the same contract a future real-data importer would target.

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
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import noise

import physics_constants as pc
from config import load_config
from map_loader import MapData, save_map

DEFAULT_GRID_SIZE = 256


class MapGenerator:
    """Procedural initial-state generator driven entirely by the config dict."""

    def __init__(self, cfg: dict | None = None):
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
        self.altitude_falloffs = np.array([_ft(n, "altitude_falloff", 1.0)  for n in self.fuel_type_names], dtype=np.float32)
        self.spawn_densities   = np.array([_ft(n, "spawn_density",     0.5)  for n in self.fuel_type_names], dtype=np.float32)

        # Tree components: any fuel type whose name starts with "tree" (e.g. tree_canopy, tree_bole)
        # is part of the same physical trees, so they share one per-cell count and spatial
        # distribution. fuel_per_tree is per type (the biomass each tree contributes to that
        # component); the count/distribution params are read from the first tree component.
        self.fuel_per_tree = np.array(
            [_ft(n, "fuel_per_tree", 0.0) for n in self.fuel_type_names], dtype=np.float32
        )
        self.tree_indices = [i for i, n in enumerate(self.fuel_type_names) if n.startswith("tree")]
        first_tree = self.fuel_type_names[self.tree_indices[0]] if self.tree_indices else "tree"
        self.max_trees_per_cell = int(_ft(first_tree, "max_trees_per_cell", 3))

    # -----------------------------------------------------------------------
    # Field builders
    # -----------------------------------------------------------------------

    def sample_perlin_grid(
        self,
        size: int,
        base: int,
        scale: float | None = None,
        octaves: int | None = None,
        persistence: float | None = None,
        lacunarity: float | None = None,
    ) -> np.ndarray:
        scale       = self.noise_scale       if scale is None else scale
        octaves     = self.noise_octaves     if octaves is None else octaves
        persistence = self.noise_persistence if persistence is None else persistence
        lacunarity  = self.noise_lacunarity  if lacunarity is None else lacunarity
        grid = np.empty((size, size), dtype=np.float32)
        for row in range(size):
            for col in range(size):
                grid[row, col] = noise.pnoise2(
                    col / scale,
                    row / scale,
                    octaves=octaves,
                    persistence=persistence,
                    lacunarity=lacunarity,
                    base=base,
                )
        return grid

    @staticmethod
    def normalize_grid(grid: np.ndarray) -> np.ndarray:
        lo = grid.min()
        hi = grid.max()
        if hi == lo:
            return np.zeros_like(grid)
        return ((grid - lo) / (hi - lo)).astype(np.float32)

    def create_grid(
        self,
        size: int,
        scale: float | None = None,
        base: int | None = None,
        octaves: int | None = None,
        persistence: float | None = None,
        lacunarity: float | None = None,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        if base is None:
            base = int((rng or np.random.default_rng()).integers(0, 256))
        raw = self.sample_perlin_grid(
            size, base, scale=scale,
            octaves=octaves, persistence=persistence, lacunarity=lacunarity,
        )
        return self.normalize_grid(raw)

    def create_terrain(
        self, size: int, base: int | None = None, rng: np.random.Generator | None = None
    ) -> np.ndarray:
        """Normalised [0, 1] elevation height map from octave Perlin noise.

        The normalised map is squared to bias toward flat low-elevation terrain
        with sharper, less frequent high-elevation peaks.
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
        return (height ** 2).astype(np.float32)

    def air_temperature_profile(self, elevation_m: np.ndarray) -> np.ndarray:
        """Air temperature [K] from the environmental lapse rate: T = T_REF - Gamma*z."""
        return (pc.T_REF - self.temperature_lapse_rate * elevation_m).astype(np.float32)

    def pressure_profile(self, elevation_m: np.ndarray) -> np.ndarray:
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
            return (pc.P_REF * np.exp(-elevation_m / pc.PRESSURE_SCALE_HEIGHT)).astype(np.float32)
        exponent = pc.GRAVITY / (pc.GAS_CONSTANT_DRY_AIR * gamma)
        base = 1.0 - gamma * elevation_m / pc.T_REF   # = T(z)/T_REF, positive for z < T_REF/Gamma
        return (pc.P_REF * base ** exponent).astype(np.float32)

    def boundary_layer_mass(self, elevation_m: np.ndarray) -> np.ndarray:
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
        return (h * pc.P_REF / (pc.GAS_CONSTANT_DRY_AIR * T)).astype(np.float32)

    def oxygen_profile(self, elevation_m: np.ndarray, air_temperatures: np.ndarray) -> np.ndarray:
        """Ambient O2 partial density [kg/m^3] = O2 mass fraction * air density p/(R_d*T)."""
        air_density = self.pressure_profile(elevation_m) / (pc.GAS_CONSTANT_DRY_AIR * air_temperatures)
        return (pc.O2_MASS_FRACTION * air_density).astype(np.float32)

    def altitude_vegetation_multiplier(self, terrain: np.ndarray, fuel_index: int) -> np.ndarray:
        """Per-cell vegetation density scale from elevation: (1 - h)^falloff in [0, 1]."""
        return ((1.0 - terrain) ** float(self.altitude_falloffs[fuel_index])).astype(np.float32)

    def sample_tree_counts(self, density: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Stochastic per-cell tree counts.

        Each cell's count is floor(Exp(mean=density)) capped at max_trees_per_cell,
        so cells with higher density average more trees (mean 0 -> always 0).
        """
        samples = rng.exponential(np.maximum(density, 0.0))
        return np.minimum(np.floor(samples), self.max_trees_per_cell).astype(np.float32)


def generate_map(
    gen: MapGenerator,
    grid_size: int,
    rng: np.random.Generator,
    seed: int | None = None,
) -> MapData:
    """Build one fully-baked, unperturbed initial state.

    Fire is intentionally not baked here: ignition (random spawn or click) is a
    runtime concern applied by the environment, so maps describe the world at rest.

    All fields are SI: terrain [m], air_temperatures [K], mass [kg/m^2], oxygen
    [kg/m^3], fuel [kg/m^2].
    """
    # Normalised [0,1] relief drives both the elevation (scaled to metres) and the
    # vegetation thinning (which is a function of fractional altitude).
    relief = gen.create_terrain(grid_size, rng=rng)
    terrain = (relief * gen.elev_max).astype(np.float32)

    # Air temperature from the lapse rate; oxygen from the hydrostatic air density.
    air_temperatures = gen.air_temperature_profile(terrain)
    oxygen = gen.oxygen_profile(terrain, air_temperatures)

    N = gen.num_fuel_types
    fuel = np.zeros((N, grid_size, grid_size), dtype=np.float32)

    names = gen.fuel_type_names
    if "grass" in names:
        # Grass: continuous Perlin density [kg/m^2], thinned by altitude.
        gi = names.index("grass")
        grass_noise = gen.create_grid(grid_size, scale=float(gen.fuel_noise_scales[gi]), rng=rng)
        grass_alt = gen.altitude_vegetation_multiplier(relief, gi)
        fuel[gi] = (grass_noise * float(gen.spawn_densities[gi]) * grass_alt).astype(np.float32)

    if gen.tree_indices:
        # Trees: one stochastic per-cell count (exponential mean = altitude-thinned noise density,
        # capped) drives every co-located tree component (canopy + bole), each scaled by its own
        # fuel_per_tree -- so the canopy and bole of the same trees always share a cell. The count's
        # spatial params are taken from the first tree component.
        t0 = gen.tree_indices[0]
        tree_noise = gen.create_grid(grid_size, scale=float(gen.fuel_noise_scales[t0]), rng=rng)
        tree_alt = gen.altitude_vegetation_multiplier(relief, t0)
        density = tree_noise * float(gen.spawn_densities[t0]) * tree_alt
        tree_counts = gen.sample_tree_counts(density, rng)
        for ti in gen.tree_indices:
            fuel[ti] = (tree_counts * float(gen.fuel_per_tree[ti])).astype(np.float32)

    # Each fuel type starts at ambient air temperature.
    fuel_temperatures = np.tile(air_temperatures[np.newaxis], (N, 1, 1)).astype(np.float32)

    # Shallow boundary-layer mass at its level-lid rest state (thinner over high terrain).
    mass = gen.boundary_layer_mass(terrain)

    # Per-map synoptic (prevailing) wind: random bearing, speed from the configured range.
    speed = float(rng.uniform(gen.ambient_speed_min, gen.ambient_speed_max))
    bearing = float(rng.uniform(0.0, 2.0 * np.pi))
    ambient_wind_x = speed * np.cos(bearing)
    ambient_wind_y = speed * np.sin(bearing)

    return MapData(
        terrain=terrain,
        air_temperatures=air_temperatures,
        mass=mass,
        oxygen=oxygen,
        fuel=fuel,
        fuel_temperatures=fuel_temperatures,
        fuel_type_names=list(gen.fuel_type_names),
        grid_size=grid_size,
        temp_eq=air_temperatures.copy(),
        oxygen_eq=oxygen.copy(),
        seed=seed,
        source="gen_maps",
        ambient_wind_x=ambient_wind_x,
        ambient_wind_y=ambient_wind_y,
    )


def _generate_one(task: tuple) -> MapData:
    """Build one rest-state map. Top-level (picklable) so it can run in a worker process.

    Rest-state generation is pure-Python Perlin noise (a scalar per-cell loop), the slowest part
    of map-making and embarrassingly parallel across maps, so it is fanned out over a process pool.
    Each worker rebuilds its own MapGenerator from the config (cheap) and draws from the per-map
    SeedSequence child, so the output is identical to and independent of the serial path.
    """
    cfg, grid_size, child, map_seed = task
    gen = MapGenerator(cfg)
    rng = np.random.default_rng(child)
    return generate_map(gen, grid_size, rng, seed=map_seed)


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
    # Imported here (not at module load) so the rest-state generator stays import-light and the
    # engine dependency is confined to the spin-up.
    import torch
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
        "--jobs", type=int, default=None,
        help="parallel worker processes for rest-state generation (default: all CPUs)",
    )
    parser.add_argument(
        "--chunk", type=int, default=32,
        help="maps per generate->spin-up->save chunk (bounds memory; saves progress durably)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    grid_size = int((config or {}).get("environment", {}).get("grid_size", DEFAULT_GRID_SIZE))
    gen = MapGenerator(config)
    jobs = args.jobs if args.jobs is not None else (os.cpu_count() or 1)
    chunk = max(1, args.chunk)

    # Per-map seeds, spawned once so each map is reproducible regardless of chunking/parallelism.
    children = np.random.SeedSequence(args.seed).spawn(args.count)
    seeds = [int(c.generate_state(1)[0]) for c in children]

    print(f"Generating {args.count} map(s) at {grid_size}x{grid_size} into '{args.out}/' "
          f"({jobs} worker(s), chunk={chunk}, spinup={gen.spinup_steps} ticks) ...", flush=True)

    # Process the maps in chunks: generate the chunk's rest states in parallel, spin up the whole
    # chunk in one batched solver pass, then save it before moving on. So progress is durable (each
    # chunk is on disk before the next starts), memory stays bounded (one chunk in RAM), and the
    # parallel Perlin generation overlaps across CPUs -- the slow part of map-making.
    def run(pool) -> None:
        done = 0
        for start in range(0, args.count, chunk):
            idx = range(start, min(start + chunk, args.count))
            tasks = [(config, grid_size, children[i], seeds[i]) for i in idx]
            maps = list(pool.map(_generate_one, tasks)) if pool else [_generate_one(t) for t in tasks]
            if gen.spinup_steps > 0:
                spin_up_maps(maps, config, gen.spinup_steps)
            for off, m in enumerate(maps):
                save_map(f"{args.out}/{args.prefix}_{args.start_index + start + off:04d}.npz", m)
            done += len(maps)
            print(f"  [{done}/{args.count}] generated + spun up + saved", flush=True)

    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            run(pool)
    else:
        run(None)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
