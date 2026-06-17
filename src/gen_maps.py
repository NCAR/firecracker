"""
gen_maps.py

Synthetic map producer for the Firecracker simulation.

Owns the procedural initial-state generation that used to live inside
Simulation/reset(): octave Perlin terrain, elevation-driven temperature/oxygen
profiles, the convective rest-state pressure, and the grass/tree fuel layers.
None of this depends on the physics engine — `MapGenerator` reads what it needs
straight from the config dict and emits a `MapData`, the same contract a future
real-data importer would target.

Run as a script to bake a batch of maps to disk:

    python src/gen_maps.py --count 8 --out maps --seed 0
"""

import argparse

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

        # Convection params are dynamics knobs, but the *initial* pressure is the
        # rest state of those dynamics, so the generator reads them too.
        convection = (cfg or {}).get("convection", {})
        self.thermal_expansion    = float(convection.get("thermal_expansion",    0.5))
        self.terrain_height_scale = float(convection.get("terrain_height_scale", 0.5))
        self.buoyancy_transport_rate = float(convection.get("buoyancy_transport_rate", 0.1))
        self.pressure_transport_rate = float(convection.get("pressure_transport_rate", 0.1))

        fuel_types_cfg = (cfg or {}).get("fuel_types", {})
        self.fuel_type_names: list[str] = list(fuel_types_cfg.keys())
        self.num_fuel_types: int = len(self.fuel_type_names)

        def _ft(name: str, key: str, default: float) -> float:
            return float(fuel_types_cfg.get(name, {}).get(key, default))

        self.fuel_noise_scales = np.array([_ft(n, "noise_scale",      32.0) for n in self.fuel_type_names], dtype=np.float32)
        self.altitude_falloffs = np.array([_ft(n, "altitude_falloff", 1.0)  for n in self.fuel_type_names], dtype=np.float32)
        self.spawn_densities   = np.array([_ft(n, "spawn_density",     0.5)  for n in self.fuel_type_names], dtype=np.float32)

        tree_cfg = fuel_types_cfg.get("tree", {})
        self.fuel_per_tree      = float(tree_cfg.get("fuel_per_tree",      0.33))
        self.max_trees_per_cell = int(tree_cfg.get("max_trees_per_cell",   3))

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

    def column_mass_profile(self, elevation_m: np.ndarray) -> np.ndarray:
        """Hydrostatic column mass per area [kg/m^2]: m = p / g (weight of the air column)."""
        return (self.pressure_profile(elevation_m) / pc.GRAVITY).astype(np.float32)

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

    def equilibrium_mass(self, terrain: np.ndarray, air_temperatures: np.ndarray) -> np.ndarray:
        """Closed-form rest state (mean 1) of the convective transport: the mass that
        makes the transport potential Phi uniform.

        At rest grad(Phi)=0, i.e. Phi = c_p*m + c_b*H = C with H = gamma*terrain +
        m*(1+alpha*T). Solving per cell with k = c_p + c_b*(1 + alpha*T):
            m = (C - c_b*gamma*terrain) / k
        and C is fixed by total mass = N cells:
            C = (N + c_b*gamma*sum[terrain/k]) / sum[1/k]
        """
        c_p = self.pressure_transport_rate
        c_b = self.buoyancy_transport_rate
        gamma = self.terrain_height_scale
        k = c_p + c_b * (1.0 + self.thermal_expansion * air_temperatures)
        inv = 1.0 / k
        n_cells = float(terrain.size)
        c = (n_cells + c_b * gamma * float((terrain * inv).sum())) / float(inv.sum())
        return ((c - c_b * gamma * terrain) * inv).astype(np.float32)


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

    if N > 0:
        # Grass (type 0): continuous Perlin density [kg/m^2], thinned by altitude.
        grass_noise = gen.create_grid(grid_size, scale=float(gen.fuel_noise_scales[0]), rng=rng)
        grass_alt = gen.altitude_vegetation_multiplier(relief, 0)
        fuel[0] = (grass_noise * float(gen.spawn_densities[0]) * grass_alt).astype(np.float32)

    if N > 1:
        # Trees (type 1): per-cell counts from an exponential whose mean is the
        # altitude-thinned noise density, capped, then scaled to fuel mass [kg/m^2].
        tree_noise = gen.create_grid(grid_size, scale=float(gen.fuel_noise_scales[1]), rng=rng)
        tree_alt = gen.altitude_vegetation_multiplier(relief, 1)
        density = tree_noise * float(gen.spawn_densities[1]) * tree_alt
        tree_counts = gen.sample_tree_counts(density, rng)
        fuel[1] = (tree_counts * gen.fuel_per_tree).astype(np.float32)

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
    args = parser.parse_args()

    config = load_config(args.config)
    grid_size = int((config or {}).get("environment", {}).get("grid_size", DEFAULT_GRID_SIZE))
    gen = MapGenerator(config)

    print(f"Generating {args.count} map(s) at {grid_size}x{grid_size} into '{args.out}/' ...")
    for i, child in enumerate(np.random.SeedSequence(args.seed).spawn(args.count)):
        rng = np.random.default_rng(child)
        map_seed = int(child.generate_state(1)[0])
        m = generate_map(gen, grid_size, rng, seed=map_seed)
        path = save_map(f"{args.out}/{args.prefix}_{args.start_index + i:04d}.npz", m)
        print(f"  [{i + 1}/{args.count}] {path}  (seed={map_seed})")
    print("Done.")


if __name__ == "__main__":
    main()
