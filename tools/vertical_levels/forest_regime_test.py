"""Forest two-regime test: surface fire vs crown fire.

Light ONLY the understory grass in a woodland bed (canopy + bole co-located above a light
grass understory) and watch whether the fire stays a slow surface fire or climbs into the
canopy and accelerates to a crown fire. Track the surface (grass) front and the crown
(canopy) front separately, report each ROS and how much canopy is consumed (did the crown
engage). Compare canopy SAV 5000 vs 10000, plus a no-canopy surface baseline.
"""
import sys, os
sys.path.insert(0, "src")
import numpy as np
import torch

from config import load_config
from simulation import Simulation, SimState
from gen_maps import MapGenerator
import physics_constants as pc

WX, HY = 220, 32
DT = 1.0
TICKS = 1000
WIND = float(os.environ.get("WIND","10.0"))
SAMPLE = 10
GRASS_UNDER = 0.3      # woodland understory grass
BOLE = 17.2


def front_ros(consumed_series, ts, dx):
    """Linear-fit ROS [m/s] over the steady window of a (t, front_m) series."""
    ts = np.array(ts)
    xs = np.array(consumed_series)
    edge_m = (WX - 15) * dx
    steady = (xs > 12 * dx) & (xs < edge_m)
    if steady.sum() < 3:
        return float("nan")
    return float(np.polyfit(ts[steady], xs[steady], 1)[0])


def run(canopy_load: float, canopy_sav: float | None, label: str) -> dict:
    cfg = load_config()
    cfg.setdefault("units", {})["seconds_per_tick"] = DT
    if canopy_sav is not None:
        cfg["fuel_types"]["tree_canopy"]["surface_area_to_volume"] = canopy_sav
    sim = Simulation(cfg)
    names = sim.fuel_type_names
    idx = {n: names.index(n) for n in names}
    dx = sim.cell_size_m
    gi, ci, bi = idx["grass"], idx["tree_canopy"], idx["tree_bole"]

    terrain = np.zeros((HY, WX), dtype=np.float64)
    gen = MapGenerator(cfg)
    air = gen.air_temperature_profile(terrain)
    mass = gen.boundary_layer_mass(terrain)
    oxygen = gen.oxygen_profile(terrain, air)
    t = lambda a: torch.as_tensor(a, dtype=sim.dtype)

    fuel = torch.zeros((len(names), HY, WX), dtype=sim.dtype)
    fuel[gi] = GRASS_UNDER
    fuel[ci] = canopy_load
    fuel[bi] = BOLE
    canopy0 = float(fuel[ci].sum()) or 1.0
    bole0 = float(fuel[bi].sum())

    ftemp = torch.full((len(names), HY, WX), pc.T_REF, dtype=sim.dtype)
    air_t = t(air)
    # SURFACE ignition: understory grass + air only. Canopy and bole start at ambient.
    air_t[:, 4:9] = float(sim.ignition_thresholds.max()) * 2.0
    ftemp[gi][:, 4:9] = float(sim.ignition_thresholds[gi]) * 2.0

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

    band = slice(HY // 2 - 6, HY // 2 + 6)
    cols = torch.arange(WX)
    ts, gfront, cfront = [], [], []

    def lead(consumed):
        lr = torch.where(consumed[band].any(1), (consumed[band] * cols).amax(1),
                         torch.zeros(band.stop - band.start, dtype=cols.dtype))
        return float(lr.float().median()) * dx

    for k in range(TICKS):
        s = sim.step_fields(s)
        if (k + 1) % SAMPLE == 0:
            ts.append((k + 1) * DT)
            gfront.append(lead(s.fuel[gi] < 0.5 * GRASS_UNDER))
            cfront.append(lead(s.fuel[ci] < 0.5 * (canopy_load or 1e9)))

    canopy_burned = 100.0 * (canopy0 - float(s.fuel[ci].sum())) / canopy0
    bole_burned = 100.0 * (bole0 - float(s.fuel[bi].sum())) / bole0
    return {
        "label": label,
        "surface_ros": front_ros(gfront, ts, dx),
        "crown_ros": front_ros(cfront, ts, dx) if canopy_load else float("nan"),
        "canopy_burned_%": canopy_burned if canopy_load else 0.0,
        "bole_burned_%": bole_burned,
    }


if __name__ == "__main__":
    print(f"woodland bed: grass understory {GRASS_UNDER} + canopy + bole {BOLE}, SURFACE ignition")
    print(f"wind {WIND} m/s, dt={DT}s, {TICKS} ticks = {TICKS*DT:.0f}s  (shrub ref ROS ~6.9 m/s)\n")
    print("{:>28} {:>13} {:>11} {:>13} {:>11}".format(
        "scenario", "surface ROS", "crown ROS", "canopy burnt", "bole burnt"))
    runs = [
        (0.0,  None, "no canopy (pure surface)"),
        (2.8,  5000, "full forest, canopy SAV 5000"),
        (2.8, 10000, "full forest, canopy SAV 10000"),
    ]
    for load, sav, lab in runs:
        r = run(load, sav, lab)
        cr = "-" if np.isnan(r["crown_ros"]) else f"{r['crown_ros']:.2f} m/s"
        print("{:>28} {:>10.2f} m/s {:>11} {:>11.1f}% {:>10.1f}%".format(
            lab, r["surface_ros"], cr, r["canopy_burned_%"], r["bole_burned_%"]))
