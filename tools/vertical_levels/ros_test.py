"""Rate-of-spread (ROS) measurement on homogeneous fuel beds, real cfg/default.toml.

Elongated domain, a planar line-ignition at the upwind (left) edge, a steady ambient
wind along +x. Track the leading edge of the *fine-fuel* consumed front (the carrier of
spread) over time and linear-fit its steady-state velocity -> ROS in m/s. Report ROS and
ROS as a fraction of the ambient wind for all-forest, all-grass, all-shrub beds.
"""
import sys
sys.path.insert(0, "src")
import numpy as np
import torch

from config import load_config
from simulation import Simulation, SimState, lift_air_levels
from gen_maps import MapGenerator
import physics_constants as pc

import os
DT = float(os.environ.get("DT", "1.0"))   # seconds_per_tick to run the physics at
WX, HY = 220, 32       # cells (2200 m x 320 m), elongated along the wind (+x)
TICKS = int(float(os.environ.get("TICKS", "1200")))
WIND = 10.0            # ambient wind [m/s] along +x
SAMPLE = 10            # record the front every SAMPLE ticks

# (name, {fuel_type: load}, fine-fuel carrier used to track the front)
SCENARIOS = [
    ("forest",    {"tree_canopy": 2.8, "tree_bole_surface": 5.72, "tree_bole_canopy": 11.48}, "tree_canopy"),
    ("grass",     {"grass": 1.0},                          "grass"),
    ("shrubland", {"shrub": 0.8},                          "shrub"),
]


def measure(fuel_loads: dict, carrier: str) -> dict:
    cfg = load_config()
    cfg.setdefault("units", {})["seconds_per_tick"] = DT
    if "CANOPY_SAV" in os.environ:
        cfg["fuel_types"]["tree_canopy"]["surface_area_to_volume"] = float(os.environ["CANOPY_SAV"])
    sim = Simulation(cfg)
    names = sim.fuel_type_names
    idx = {n: names.index(n) for n in names}
    dx, dt = sim.cell_size_m, sim.dt

    terrain = np.zeros((HY, WX), dtype=np.float64)
    gen = MapGenerator(cfg)
    air = gen.air_temperature_profile(terrain)
    mass = gen.boundary_layer_mass(terrain)
    oxygen = gen.oxygen_profile(terrain, air)
    t = lambda a: torch.as_tensor(a, dtype=sim.dtype, device=sim.device)
    air2d, mass_t, oxy_t, terr_t = t(air), t(mass), t(oxygen), t(terrain)
    air_t = lift_air_levels(air2d, sim.num_air_levels)   # (L, H, W) per-level air stack

    fuel = torch.zeros((len(names), HY, WX), dtype=sim.dtype, device=sim.device)
    for n, load in fuel_loads.items():
        fuel[idx[n]] = load
    ci = idx[carrier]
    fine0 = float(fuel_loads[carrier])

    ftemp = torch.full((len(names), HY, WX), pc.T_REF, dtype=sim.dtype, device=sim.device)
    # Planar line-ignition: columns 4..8, full height, fine fuel + air hot.
    ign_cols = slice(4, 9)
    hot = float(sim.ignition_thresholds.max()) * 2.0
    air_t[0, :, ign_cols] = hot                          # surface-level ignition strip
    ftemp[ci][:, ign_cols] = hot

    wx = torch.full((HY, WX), WIND, dtype=sim.dtype, device=sim.device)
    wy = torch.zeros((HY, WX), dtype=sim.dtype, device=sim.device)
    ax = torch.tensor(WIND, dtype=sim.dtype).view(1, 1)
    ay = torch.zeros((1, 1), dtype=sim.dtype)
    s = SimState(
        mass=mass_t, air_temperatures=air_t, ground_temperature=air2d.clone(),
        fuel_temperatures=ftemp, fuel=fuel, oxygen=oxy_t, terrain=terr_t,
        temp_eq=air_t.clone(), oxygen_eq=oxy_t.clone(), mass_eq=mass_t.clone(),
        x_wind_vel=wx, y_wind_vel=wy, u_amb_x=ax, u_amb_y=ay, radiant_flux=wy.clone(),
    )

    band = slice(HY // 2 - 6, HY // 2 + 6)   # central rows, away from top/bottom edges
    ts, xs = [], []
    for k in range(TICKS):
        s = sim.step_fields(s)
        if (k + 1) % SAMPLE == 0:
            consumed = (s.fuel[ci] < 0.5 * fine0)                 # (HY, WX) bool
            cols = torch.arange(WX, device=s.fuel.device)
            per_row_lead = torch.where(
                consumed[band].any(dim=1),
                (consumed[band] * cols).amax(dim=1),
                torch.zeros(band.stop - band.start, dtype=cols.dtype, device=cols.device),
            )
            front_cell = float(per_row_lead.float().median())
            ts.append((k + 1) * dt)
            xs.append(front_cell * dx)

    ts, xs = np.array(ts), np.array(xs)
    # Fit ROS on the steady window: after the front clears the ignition strip, before it
    # nears the downwind edge (leave a 200 m margin).
    edge_m = (WX - 15) * dx
    steady = (xs > 12 * dx) & (xs < edge_m)
    if steady.sum() < 3:
        return {"ros": float("nan"), "n": int(steady.sum()), "reached_m": float(xs.max())}
    slope = np.polyfit(ts[steady], xs[steady], 1)[0]   # m/s
    return {"ros": slope, "n": int(steady.sum()), "reached_m": float(xs.max()),
            "t_span": (float(ts[steady].min()), float(ts[steady].max()))}


if __name__ == "__main__":
    print(f"domain {WX}x{HY} cells ({WX*10} x {HY*10} m), wind {WIND} m/s (+x), dt={DT}s/tick, {TICKS} ticks = {TICKS*DT:.0f}s sim time\n")
    print("{:>10} {:>10} {:>14} {:>12} {:>10}".format(
        "fuel", "ROS m/s", "ROS / wind", "fit pts", "reached m"))
    for name, loads, carrier in SCENARIOS:
        r = measure(loads, carrier)
        frac = r["ros"] / WIND
        print("{:>10} {:>10.3f} {:>13.1f}% {:>12} {:>9.0f}".format(
            name, r["ros"], 100 * frac, r["n"], r["reached_m"]))
