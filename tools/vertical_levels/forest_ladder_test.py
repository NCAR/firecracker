"""Ladder-fuel ordering test: can grass ignite the canopy WITHOUT the bole catching first?

Physically the canopy is elevated: a ground grass fire must climb the bole (trunk) to reach
it, so the bole should cross ignition BEFORE the canopy. Two checks:

  (1) no-bole bed (grass + canopy, bole=0): if the canopy still burns, grass is igniting the
      canopy directly -- a violation (there's no ladder to carry the fire up).
  (2) full bed (grass + canopy + bole): at a downstream probe the fire reaches naturally,
      record the tick each of bole/canopy first crosses its own ignition threshold and check
      the order (bole should be first).
"""
import sys, os
sys.path.insert(0, "src")
import numpy as np
import torch

from config import load_config
from simulation import Simulation, SimState
from gen_maps import MapGenerator
import physics_constants as pc

WX, HY = 160, 32
DT = 1.0
TICKS = 600
WIND = 5.0
GRASS_UNDER = 0.3
PROBE = 90            # downstream column reached by the spreading front


def run(canopy_load, bole_load, sav=10000):
    cfg = load_config()
    cfg.setdefault("units", {})["seconds_per_tick"] = DT
    cfg["fuel_types"]["tree_canopy"]["surface_area_to_volume"] = sav
    sim = Simulation(cfg)
    names = sim.fuel_type_names
    idx = {n: names.index(n) for n in names}
    gi, ci, bi = idx["grass"], idx["tree_canopy"], idx["tree_bole"]
    canopy_ign = float(sim.ignition_thresholds[ci])
    bole_ign = float(sim.ignition_thresholds[bi])

    terrain = np.zeros((HY, WX), dtype=np.float64)
    gen = MapGenerator(cfg)
    air = gen.air_temperature_profile(terrain)
    mass = gen.boundary_layer_mass(terrain)
    oxygen = gen.oxygen_profile(terrain, air)
    t = lambda a: torch.as_tensor(a, dtype=sim.dtype)

    fuel = torch.zeros((len(names), HY, WX), dtype=sim.dtype)
    fuel[gi] = GRASS_UNDER
    fuel[ci] = canopy_load
    fuel[bi] = bole_load
    canopy0 = float(fuel[ci].sum()) or 1.0
    bole0 = float(fuel[bi].sum()) or 1.0

    ftemp = torch.full((len(names), HY, WX), pc.T_REF, dtype=sim.dtype)
    air_t = t(air)
    air_t[:, 4:9] = float(sim.ignition_thresholds.max()) * 2.0
    ftemp[gi][:, 4:9] = float(sim.ignition_thresholds[gi]) * 2.0   # grass-only ignition

    wx = torch.full((HY, WX), WIND, dtype=sim.dtype)
    wy = torch.zeros((HY, WX), dtype=sim.dtype)
    ax = torch.tensor(WIND, dtype=sim.dtype).view(1, 1)
    ay = torch.zeros((1, 1), dtype=sim.dtype)
    s = SimState(
        mass=t(mass), air_temperatures=air_t, ground_temperature=t(air).clone(),
        fuel_temperatures=ftemp, fuel=fuel, oxygen=t(oxygen), terrain=t(terrain),
        temp_eq=t(air).clone(), oxygen_eq=t(oxygen).clone(), mass_eq=t(mass).clone(),
        x_wind_vel=wx, y_wind_vel=wy, u_amb_x=ax, u_amb_y=ay, radiant_flux=wy.clone(),
    )

    band = slice(HY // 2 - 4, HY // 2 + 4)
    t_bole = t_canopy = None
    for k in range(TICKS):
        s = sim.step_fields(s)
        cT = float(s.fuel_temperatures[ci][band, PROBE].median())
        bT = float(s.fuel_temperatures[bi][band, PROBE].median())
        if t_canopy is None and cT >= canopy_ign:
            t_canopy = k * DT
        if t_bole is None and bT >= bole_ign:
            t_bole = k * DT
    return {
        "canopy_burned_%": 100.0 * (canopy0 - float(s.fuel[ci].sum())) / canopy0 if canopy_load else 0.0,
        "bole_burned_%": 100.0 * (bole0 - float(s.fuel[bi].sum())) / bole0 if bole_load else 0.0,
        "t_bole_ign": t_bole, "t_canopy_ign": t_canopy,
    }


if __name__ == "__main__":
    print(f"grass understory {GRASS_UNDER}, grass-only ignition, wind {WIND} m/s, dt {DT}s\n")

    print("(1) NO-BOLE bed (grass + canopy, bole=0) -- can grass light the canopy with no ladder?")
    r = run(canopy_load=2.8, bole_load=0.0)
    print(f"    canopy burned: {r['canopy_burned_%']:.1f}%   "
          f"canopy ignites at probe: {r['t_canopy_ign']}")
    verdict = "VIOLATION: grass ignites canopy with no bole" if r["canopy_burned_%"] > 5 \
        else "OK: canopy does not catch without a bole"
    print(f"    -> {verdict}\n")

    print("(2) FULL bed (grass + canopy + bole) -- ordering at downstream probe:")
    r = run(canopy_load=2.8, bole_load=17.2)
    tb, tc = r["t_bole_ign"], r["t_canopy_ign"]
    print(f"    bole ignites at:   {tb} s")
    print(f"    canopy ignites at: {tc} s")
    print(f"    canopy burned {r['canopy_burned_%']:.1f}%, bole burned {r['bole_burned_%']:.1f}%")
    if tc is not None and (tb is None or tc <= tb):
        print("    -> VIOLATION: canopy catches at or before the bole (no bole-first ladder)")
    else:
        print("    -> OK: bole catches before the canopy")
