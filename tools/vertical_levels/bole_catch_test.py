"""Single-stand bole-catch test.

Builds a small uniform woodland (real cfg/default.toml physics), ignites ONLY the
canopy + grass in a central patch (the bole starts at ambient), and runs the full
step_fields loop. Reports whether the low-SAV bole catches: crosses its ignition
threshold and consumes a meaningful fraction of its mass. Runs for a sweep of
tree_canopy surface_area_to_volume values so we can see the effect of grass-parity.
"""
import sys
sys.path.insert(0, "src")
import numpy as np
import torch

from config import load_config
from simulation import Simulation, SimState, lift_air_levels
from gen_maps import MapGenerator
import physics_constants as pc

GRID = 48
TICKS = 4000            # 0.1 s/tick -> 400 s of sim time
CANOPY = 2.8           # max canopy load (4 trees)
SURFACE_BOLE = 5.72    # tree_bole_surface load (~1/3 of the old 17.2 single bole)
CANOPY_BOLE = 11.48    # tree_bole_canopy load  (~2/3) -- co-located with the ignited canopy
GRASS = 0.3            # woodland understorey grass


def run(canopy_sav: float) -> dict:
    cfg = load_config()
    cfg["fuel_types"]["tree_canopy"]["surface_area_to_volume"] = float(canopy_sav)
    sim = Simulation(cfg)
    names = sim.fuel_type_names
    gi, ci = names.index("grass"), names.index("tree_canopy")
    bsi, bci = names.index("tree_bole_surface"), names.index("tree_bole_canopy")
    # The ignited canopy is level 1, co-located with the canopy bole -- track that segment's catch.
    bi = bci
    bole_ign = float(sim.ignition_thresholds[bi])

    terrain = np.zeros((GRID, GRID), dtype=np.float64)
    gen = MapGenerator(cfg)
    air = gen.air_temperature_profile(terrain)
    mass = gen.boundary_layer_mass(terrain)
    oxygen = gen.oxygen_profile(terrain, air)
    t = lambda a: torch.as_tensor(a, dtype=sim.dtype, device=sim.device)
    air2d, mass_t, oxy_t, terr_t = t(air), t(mass), t(oxygen), t(terrain)
    air_t = lift_air_levels(air2d, sim.num_air_levels)   # (L, H, W) per-level air stack

    fuel = torch.zeros((len(names), GRID, GRID), dtype=sim.dtype, device=sim.device)
    fuel[gi] = GRASS
    fuel[ci] = CANOPY
    fuel[bsi] = SURFACE_BOLE
    fuel[bci] = CANOPY_BOLE
    ftemp = torch.full((len(names), GRID, GRID), pc.T_REF, dtype=sim.dtype, device=sim.device)

    # Ignite a central 3-cell-radius patch via the FINE fuels + air only. The bole is left
    # at ambient so it must catch on its own from the canopy's flaming heat.
    c = GRID // 2
    rr = torch.arange(GRID).view(-1, 1)
    cc = torch.arange(GRID).view(1, -1)
    patch = (rr - c) ** 2 + (cc - c) ** 2 <= 3 ** 2
    hot = float(sim.ignition_thresholds[ci]) * 2.0
    air_t[0][patch] = hot          # surface air (grass)
    air_t[1][patch] = hot          # canopy air (canopy foliage)
    ftemp[gi][patch] = hot
    ftemp[ci][patch] = hot
    # bole temperatures deliberately NOT set -- they must catch on their own from the fire

    z = torch.zeros_like(mass_t)
    ax = torch.zeros((1, 1), dtype=sim.dtype)
    s = SimState(
        mass=mass_t, air_temperatures=air_t, ground_temperature=air2d.clone(),
        fuel_temperatures=ftemp, fuel=fuel, oxygen=oxy_t, terrain=terr_t,
        temp_eq=air_t.clone(), oxygen_eq=oxy_t.clone(), mass_eq=mass_t.clone(),
        x_wind_vel=z.clone(), y_wind_vel=z.clone(), u_amb_x=ax, u_amb_y=ax.clone(),
        radiant_flux=z.clone(),
    )

    bole0 = float(fuel[bi].sum())
    canopy0 = float(fuel[ci].sum())
    peak_bole_T = 0.0
    tick_bole_ignites = None
    bole_at = {}
    checkpoints = {int(x / sim.dt) - 1 for x in (100, 200, 400)}
    for k in range(TICKS):
        s = sim.step_fields(s)
        bT = float(s.fuel_temperatures[bi].max())
        peak_bole_T = max(peak_bole_T, bT)
        if tick_bole_ignites is None and bT >= bole_ign:
            tick_bole_ignites = k
        if k in checkpoints:
            bole_at[round((k + 1) * sim.dt)] = 100.0 * (bole0 - float(s.fuel[bi].sum())) / bole0
    bole1 = float(s.fuel[bi].sum())
    canopy1 = float(s.fuel[ci].sum())
    return {
        "sav": canopy_sav,
        "a_s": canopy_sav / 500.0,
        "bole_ign_K": bole_ign,
        "peak_bole_T": peak_bole_T,
        "final_bole_T": float(s.fuel_temperatures[bi].max()),
        "bole_ignites_s": None if tick_bole_ignites is None else tick_bole_ignites * sim.dt,
        "canopy_consumed_%": 100.0 * (canopy0 - canopy1) / canopy0,
        "bole_consumed_%": 100.0 * (bole0 - bole1) / bole0,
        "bole_at": bole_at,
    }


if __name__ == "__main__":
    print(f"grid {GRID}x{GRID}, {TICKS} ticks (0.1s) = {TICKS*0.1:.0f}s sim time")
    print(f"grass SAV=12000 (a_s=24) for reference\n")
    hdr = ("canopy_SAV", "a_s", "catch@", "peakT", "finalT", "bole@100s", "@200s", "@400s")
    print("{:>10} {:>5} {:>8} {:>8} {:>8} {:>10} {:>7} {:>7}".format(*hdr))
    for sav in (5000, 10000, 12000):
        r = run(sav)
        catch = "-" if r["bole_ignites_s"] is None else f"{r['bole_ignites_s']:.1f}s"
        b = r["bole_at"]
        print("{:>10.0f} {:>5.0f} {:>8} {:>7.0f}K {:>7.0f}K {:>9.1f}% {:>6.1f}% {:>6.1f}%".format(
            r["sav"], r["a_s"], catch, r["peak_bole_T"], r["final_bole_T"],
            b.get(100, 0), b.get(200, 0), b.get(400, 0)))
