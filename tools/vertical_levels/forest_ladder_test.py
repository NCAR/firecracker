"""Ladder-fuel ordering test: can grass ignite the canopy WITHOUT the bole carrying fire up?

Physically the canopy is elevated: a ground grass fire must climb the trunk (bole) to reach it, so
the surface bole should cross ignition BEFORE the canopy. With the vertical air levels (Stage 1+2),
the ladder is emergent geometry: grass warms the surface air and deposits its convective ignition
only into surface-level fuels, so the canopy (level 1) can only catch once (a) the surface bole
ignites and the bole-segment conduction climbs fire up into the canopy bole, and (b) that canopy-level
combustion lights the crown. Two checks:

  (1) no-bole bed (grass + canopy, both bole segments = 0): if the canopy still burns, heat is
      reaching the canopy without a trunk -- a ladder violation.
  (2) full bed (grass + canopy + both bole segments): at a downstream probe the fire reaches
      naturally; record the tick each of surface-bole / canopy-bole / canopy first crosses its own
      ignition threshold and check the order (surface bole first, canopy last).
"""
import sys, os
sys.path.insert(0, "src")
import numpy as np
import torch

from config import load_config
from simulation import Simulation, SimState, lift_air_levels
from gen_maps import MapGenerator
import physics_constants as pc

WX, HY = 160, 32
DT = 1.0
TICKS = 600
WIND = 5.0
GRASS_UNDER = 0.3
PROBE = 90            # downstream column reached by the spreading front

# Bole mass splits ~1/3 surface, ~2/3 canopy (matches cfg fuel_per_tree at 4 trees/cell).
SURFACE_BOLE = 5.72
CANOPY_BOLE = 11.48


def run(canopy_load, bole_on, sav=10000, ignite_surface_bole=False):
    cfg = load_config()
    cfg.setdefault("units", {})["seconds_per_tick"] = DT
    cfg["fuel_types"]["tree_canopy"]["surface_area_to_volume"] = sav
    sim = Simulation(cfg)
    names = sim.fuel_type_names
    idx = {n: names.index(n) for n in names}
    gi = idx["grass"]
    ci = idx["tree_canopy"]
    bsi = idx["tree_bole_surface"]
    bci = idx["tree_bole_canopy"]
    canopy_ign = float(sim.ignition_thresholds[ci])
    sbole_ign = float(sim.ignition_thresholds[bsi])
    cbole_ign = float(sim.ignition_thresholds[bci])

    terrain = np.zeros((HY, WX), dtype=np.float64)
    gen = MapGenerator(cfg)
    air = gen.air_temperature_profile(terrain)
    mass = gen.boundary_layer_mass(terrain)
    oxygen = gen.oxygen_profile(terrain, air)
    t = lambda a: torch.as_tensor(a, dtype=sim.dtype)

    fuel = torch.zeros((len(names), HY, WX), dtype=sim.dtype)
    fuel[gi] = GRASS_UNDER
    fuel[ci] = canopy_load
    fuel[bsi] = SURFACE_BOLE if bole_on else 0.0
    fuel[bci] = CANOPY_BOLE if bole_on else 0.0
    canopy0 = float(fuel[ci].sum()) or 1.0
    sbole0 = float(fuel[bsi].sum()) or 1.0

    ftemp = torch.full((len(names), HY, WX), pc.T_REF, dtype=sim.dtype)
    # L=3 air stack; ignite the SURFACE level only (a ground ignition), plus a grass strip.
    air_t = lift_air_levels(t(air), sim.num_air_levels)
    air_t[0, :, 4:9] = float(sim.ignition_thresholds.max()) * 2.0     # surface-level ignition strip
    ftemp[gi][:, 4:9] = float(sim.ignition_thresholds[gi]) * 2.0      # grass ignition
    if ignite_surface_bole:                                          # also light the trunk base
        ftemp[bsi][:, 4:9] = float(sim.ignition_thresholds[bsi]) * 2.0

    wx = torch.full((HY, WX), WIND, dtype=sim.dtype)
    wy = torch.zeros((HY, WX), dtype=sim.dtype)
    ax = torch.tensor(WIND, dtype=sim.dtype).view(1, 1)
    ay = torch.zeros((1, 1), dtype=sim.dtype)
    s = SimState(
        mass=t(mass), air_temperatures=air_t, ground_temperature=t(air).clone(),
        fuel_temperatures=ftemp, fuel=fuel, oxygen=t(oxygen), terrain=t(terrain),
        temp_eq=lift_air_levels(t(air), sim.num_air_levels),
        oxygen_eq=t(oxygen).clone(), mass_eq=t(mass).clone(),
        x_wind_vel=wx, y_wind_vel=wy, u_amb_x=ax, u_amb_y=ay, radiant_flux=wy.clone(),
    )

    band = slice(HY // 2 - 4, HY // 2 + 4)
    t_sbole = t_cbole = t_canopy = None
    for k in range(TICKS):
        s = sim.step_fields(s)
        med = lambda i: float(s.fuel_temperatures[i][band, PROBE].median())
        if t_sbole is None and med(bsi) >= sbole_ign:
            t_sbole = k * DT
        if t_cbole is None and med(bci) >= cbole_ign:
            t_cbole = k * DT
        if t_canopy is None and med(ci) >= canopy_ign:
            t_canopy = k * DT
    return {
        "canopy_burned_%": 100.0 * (canopy0 - float(s.fuel[ci].sum())) / canopy0 if canopy_load else 0.0,
        "sbole_burned_%": 100.0 * (sbole0 - float(s.fuel[bsi].sum())) / sbole0 if bole_on else 0.0,
        "t_sbole_ign": t_sbole, "t_cbole_ign": t_cbole, "t_canopy_ign": t_canopy,
    }


if __name__ == "__main__":
    print(f"grass understory {GRASS_UNDER}, grass-only ignition, wind {WIND} m/s, dt {DT}s\n")

    print("(1) NO-BOLE bed (grass + canopy, no trunk) -- can heat reach the canopy with no ladder?")
    r = run(canopy_load=2.8, bole_on=False)
    print(f"    canopy burned: {r['canopy_burned_%']:.1f}%   "
          f"canopy ignites at probe: {r['t_canopy_ign']}")
    verdict = "VIOLATION: canopy catches with no bole" if r["canopy_burned_%"] > 5 \
        else "OK: canopy does not catch without a bole"
    print(f"    -> {verdict}\n")

    print("(2) FULL bed (grass + canopy + surface/canopy bole) -- ordering at downstream probe:")
    r = run(canopy_load=2.8, bole_on=True)
    ts, tcb, tc = r["t_sbole_ign"], r["t_cbole_ign"], r["t_canopy_ign"]
    print(f"    surface bole ignites at: {ts} s")
    print(f"    canopy bole ignites at:  {tcb} s")
    print(f"    canopy ignites at:       {tc} s")
    print(f"    canopy burned {r['canopy_burned_%']:.1f}%, surface bole burned {r['sbole_burned_%']:.1f}%")
    if tc is not None and (ts is None or tc <= ts):
        print("    -> VIOLATION: canopy catches at or before the surface bole (no bole-first ladder)")
    elif ts is None and tc is None:
        print("    -> NOTE: the grass fire does not ignite the coarse surface bole (Phase-4 tuning);")
        print("            check (3) confirms the trunk up-chain itself works once the base is lit.")
    else:
        print("    -> OK: surface bole catches before the canopy")

    print("\n(3) UP-CHAIN (grass + canopy + bole, surface bole lit at the source) -- does fire climb?")
    r = run(canopy_load=2.8, bole_on=True, ignite_surface_bole=True)
    tcb, tc = r["t_cbole_ign"], r["t_canopy_ign"]
    print(f"    canopy bole ignites at:  {tcb} s   canopy ignites at: {tc} s")
    print(f"    canopy burned {r['canopy_burned_%']:.1f}% (crown fire once the trunk carries fire up)")
    verdict = "OK: fire climbs the trunk and the canopy crowns" if r["canopy_burned_%"] > 20 \
        else "WEAK: the trunk up-chain did not crown the canopy"
    print(f"    -> {verdict}")
