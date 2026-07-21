"""
diagnose_fire.py

Confirm *which* physical mechanism causes the two observed fire behaviours:
  (A) trees ignite but do not sustain a flame, and
  (B) a grass fire fails to burn through all the grass.

Both probes drive the real Simulation subsystems (no re-derived formulas, so they
cannot drift from the model) and print a measured verdict.

Probe A -- tree-sustain energy budget. Hold a uniform single-fuel world at a fixed
candidate flame temperature and run each combustion-relevant subsystem one tick on
the *same* input state, measuring its isolated mean fuel-temperature change:
  gain  = update_fire            (combustion self-heat kept in the fuel)
  rad   = apply_radiation        (the eps*sigma*T^4 sink + solar)
  rback = apply_radiant_heat     (super-ambient emission redistributed back from neighbours)
  cond  = exchange_fuel_air_heat (conduction to the air)
Their sum is the net per-tick drive on the fuel. Swept over temperature for grass and
for tree: a fuel sustains only where some flame temperature has net >= 0. If the tree's
net is negative everywhere above ambient, its slow low-SAV surface burn cannot outrun the
T^4 sink -- it dies once the ignition source is removed.

Probe B -- grass burn-through. Ignite the centre of a dense uniform grass field and run
the full per-tick loop. Classify every cell at the end as burnt / partially-burnt /
untouched, track the oxygen drawdown under the front, then re-run with the fresh-air
resupply boosted. If the dense fire starves its own oxygen (deep drawdown, many
partially-burnt cells) and boosting resupply lets it finish, the limit is ventilation;
if cells are either fully burnt or untouched and boosting does not help, the limit is the
short convective spread reach.

Usage:
    python tools/diagnose_fire.py            # run both probes
    python tools/diagnose_fire.py --probe A
    python tools/diagnose_fire.py --probe B
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import numpy as np  # noqa: E402
import torch        # noqa: E402

import physics_constants as pc            # noqa: E402
from config import load_config            # noqa: E402
from simulation import Simulation, SimState, lift_air_levels  # noqa: E402
from gen_maps import MapGenerator         # noqa: E402


def _single_fuel_config(cfg: dict, fuel_name: str) -> dict:
    """A copy of the production config restricted to one fuel type (keeps its real params)."""
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in cfg.items()}
    out["fuel_types"] = {fuel_name: dict(cfg["fuel_types"][fuel_name])}
    return out


def _uniform_state(sim: Simulation, grid: int, fuel_load: float, fuel_temp_k: float) -> SimState:
    """A flat SI world fully covered by the sim's (single) fuel type, all at fuel_temp_k."""
    gen = MapGenerator(_single_fuel_config(load_config(), sim.fuel_type_names[0]))
    terrain = np.zeros((grid, grid), dtype=np.float64)
    air = gen.air_temperature_profile(terrain)
    mass = gen.boundary_layer_mass(terrain)
    oxygen = gen.oxygen_profile(terrain, air)
    t = lambda a: torch.as_tensor(a, dtype=sim.dtype, device=sim.device)
    air_t, mass_t, oxy_t, terr_t = t(air), t(mass), t(oxygen), t(terrain)

    n = sim.num_fuel_types
    fuel = torch.full((n, grid, grid), fuel_load, dtype=sim.dtype, device=sim.device)
    fuel_tt = torch.full((n, grid, grid), fuel_temp_k, dtype=sim.dtype, device=sim.device)
    z = torch.zeros((1, 1), dtype=sim.dtype, device=sim.device)
    air3 = lift_air_levels(air_t, sim.num_air_levels)   # (L, H, W) per-level air stack
    return SimState(
        mass=mass_t, air_temperatures=air3, ground_temperature=air_t.clone(),
        fuel_temperatures=fuel_tt, fuel=fuel, oxygen=oxy_t, terrain=terr_t,
        temp_eq=air3.clone(), oxygen_eq=oxy_t.clone(), mass_eq=mass_t.clone(),
        x_wind_vel=torch.zeros_like(mass_t), y_wind_vel=torch.zeros_like(mass_t),
        u_amb_x=z, u_amb_y=z, radiant_flux=torch.zeros_like(mass_t),
    )


# ---------------------------------------------------------------------------
# Probe A: tree-sustain energy budget
# ---------------------------------------------------------------------------

def _budget_terms(sim: Simulation, fuel_load: float, temp_k: float) -> dict:
    """Isolated mean per-tick fuel ΔT [K] from each subsystem, all from the same input state."""
    grid = 16
    s = _uniform_state(sim, grid, fuel_load, temp_k)
    f0 = s.fuel_temperatures.clone()

    def dmean(new_fuel_temp: torch.Tensor) -> float:
        return float((new_fuel_temp - f0).mean())

    # Each subsystem is run independently on the *same* pre-step state (clones guard against
    # any in-place writes), so the four terms are directly comparable contributions for one tick.
    # air_temperatures / temp_eq are (L, H, W) stacks; update_fire and exchange take the full stack,
    # while apply_radiation and apply_radiant_heat are surface-level (level 0).
    air0 = s.air_temperatures.select(-3, 0)
    temp_eq0 = s.temp_eq.select(-3, 0)
    _, ft_fire, _, _ = sim.update_fire(
        s.air_temperatures.clone(), f0.clone(), s.fuel.clone(), s.oxygen.clone(), s.mass.clone(),
        s.x_wind_vel.clone(), s.y_wind_vel.clone(), s.terrain.clone(),
    )
    _, ft_rad, _ = sim.apply_radiation(
        air0.clone(), s.ground_temperature.clone(), f0.clone(),
        s.fuel.clone(), s.mass.clone(), temp_eq0.clone(),
    )
    ft_rback, _ = sim.apply_radiant_heat(f0.clone(), s.fuel.clone(), temp_eq0.clone())
    _, ft_cond = sim.exchange_fuel_air_heat(
        s.air_temperatures.clone(), f0.clone(), s.fuel.clone(),
    )

    gain, rad, rback, cond = dmean(ft_fire), dmean(ft_rad), dmean(ft_rback), dmean(ft_cond)
    return {"gain": gain, "rad": rad, "rback": rback, "cond": cond,
            "net": gain + rad + rback + cond}


def probe_tree_sustain(fuel_loads: dict[str, float]) -> None:
    print("=" * 78)
    print("PROBE A  --  tree-sustain energy budget (isolated per-tick fuel ΔT, K/tick)")
    print("=" * 78)
    cfg = load_config()
    temps = [400.0, 600.0, 800.0, 1000.0, 1200.0, 1500.0]

    for fuel_name in fuel_loads:
        sim = Simulation(_single_fuel_config(cfg, fuel_name))
        load = fuel_loads[fuel_name]
        a_s = float(sim.fuel_specific_surface.flatten()[0])
        print(f"\n[{fuel_name}]  fuel_load={load:.1f} kg/m^2   "
              f"a_s={a_s:.3g} m^2/kg   dt={sim.dt:.3g} s")
        print(f"  {'T[K]':>6} {'gain(burn)':>11} {'rad(T^4)':>10} "
              f"{'rback':>8} {'cond(air)':>10} {'NET':>9}")
        # A flame is only self-sustaining if the net drive stays >= 0 once the fuel is hot, i.e.
        # somewhere in the flame band (T >= 1000 K). A tiny positive net at a sub-flame smoulder
        # temperature (600-800 K) is not a flame -- the convective ignition gate and neighbour
        # cooling in the full sim will quench it -- so the band test is what discriminates.
        flame_band_net = 1e30   # the *minimum* net over the flame band: a robust flame must hold
        for T in temps:         # net >= 0 across it, not merely touch zero at its lower edge.
            b = _budget_terms(sim, load, T)
            if T >= 1000.0:
                flame_band_net = min(flame_band_net, b["net"])
            print(f"  {T:6.0f} {b['gain']:11.4f} {b['rad']:10.4f} "
                  f"{b['rback']:8.4f} {b['cond']:10.4f} {b['net']:9.4f}")
        verdict = ("SUSTAINS a flame (net>=0 across the >=1000 K band)" if flame_band_net >= 0
                   else "CANNOT hold a flame (net<0 in the >=1000 K band -> cools back down)")
        print(f"  -> worst net in the flame band (T>=1000 K) = {flame_band_net:+.4f} K/tick  ::  {verdict}")

    print("\nReading: 'gain' is combustion heat kept in the fuel; the burn rate scales with the")
    print("reactive surface a_s, so a high-SAV fuel (grass, tree canopy) holds net>0 across the flame")
    print("band and self-sustains, while a low-SAV fuel (a bare tree bole) goes net<0 and cools back")
    print("down unless something keeps feeding it heat. For the two-component tree this is the point:")
    print("the canopy sustains on its own and the bole leans on the canopy/neighbours -- so check the")
    print("bole's worst-net here and decide whether it should also self-sustain a slow burn (raise its")
    print("SAV) or only burn while fed.")


# ---------------------------------------------------------------------------
# Probe B: grass burn-through (oxygen limit vs spread reach)
# ---------------------------------------------------------------------------

def _grass_state(sim: Simulation, grid: int, fuel_load: float) -> SimState:
    s = _uniform_state(sim, grid, fuel_load, pc.T_REF)
    s.fuel_temperatures[0, grid // 2, grid // 2] = 1200.0   # one-shot central ignition
    return s


def _run_burnthrough(sim: Simulation, grid: int, fuel_load: float, ticks: int) -> dict:
    """Run a centre-ignited grass fire, following the front over time.

    Records the burnt / footprint / flaming trajectory and stops early if the fire goes out
    (no flaming cells left) -- that is the signature that distinguishes a fire that *stalls*
    with fuel remaining from one that simply burns slowly but completely.
    """
    s = _grass_state(sim, grid, fuel_load)
    total = grid * grid
    o2_ambient = float(s.oxygen.mean())
    min_o2_in_burning = o2_ambient
    burnt_seen = 0
    half_tick = None                                   # tick at which half the grid is burnt
    extinguished_at = None
    traj = []                                          # (tick, burnt, footprint)

    for k in range(1, ticks + 1):
        sim.step_fields(s)
        flaming = s.fuel_temperatures[0] > sim.flame_gate_temperature
        if bool(flaming.any()):
            min_o2_in_burning = min(min_o2_in_burning, float(s.oxygen[flaming].min()))

        fuel = s.fuel[0]
        burnt = int((fuel < sim.fuel_burnt_threshold).sum())
        footprint = int((fuel < 0.95 * fuel_load).sum())
        burnt_seen = max(burnt_seen, burnt)
        if half_tick is None and burnt >= 0.5 * total:
            half_tick = k
        if k % max(1, ticks // 8) == 0 or k == ticks:
            traj.append((k, burnt, footprint))
        # Fire is out once nothing is flaming and the central ignition has had time to catch.
        if k > 20 and not bool(flaming.any()):
            extinguished_at = k
            break

    fuel = s.fuel[0].cpu().numpy()
    burnt = int((fuel < sim.fuel_burnt_threshold).sum())
    footprint = int((fuel < 0.95 * fuel_load).sum())
    untouched = int((fuel >= 0.95 * fuel_load).sum())
    return {
        "total": total, "burnt": burnt, "footprint": footprint, "untouched": untouched,
        "burnt_frac": burnt / total, "half_tick": half_tick,
        "extinguished_at": extinguished_at, "traj": traj,
        "o2_ambient": o2_ambient, "min_o2_burning": min_o2_in_burning,
    }


def probe_grass_burnthrough(grid: int = 64, fuel_load: float = 1.5, ticks: int = 1500) -> None:
    print("\n" + "=" * 78)
    print("PROBE B  --  grass burn-through: does the front stall, or is it just slow?")
    print("=" * 78)
    cfg = _single_fuel_config(load_config(), "grass")

    base = _run_burnthrough(Simulation(cfg), grid, fuel_load, ticks)

    boosted_cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in cfg.items()}
    boosted_cfg.setdefault("relaxation", {})["enabled"] = True
    boosted_cfg["relaxation"]["oxygen_rate"] = 10.0 * float(cfg.get("relaxation", {}).get("oxygen_rate", 0.05))
    boosted = _run_burnthrough(Simulation(boosted_cfg), grid, fuel_load, ticks)

    drawdown = base["min_o2_burning"] / base["o2_ambient"] if base["o2_ambient"] else 1.0
    print(f"\ngrid={grid} ({grid * grid} cells)  fuel_load={fuel_load} kg/m^2  tick cap={ticks}  dt=0.1 s\n")
    print(f"  burnt-fraction trajectory (tick: burnt / footprint cells):")
    for k, b, fp in base["traj"]:
        print(f"    t={k:5d}   burnt={b:6d}   front+footprint={fp:6d}   ({b / base['total']:.0%} of grid burnt)")
    print(f"\n  fire went out at tick : {base['extinguished_at']}  "
          f"(None = still burning at the cap)")
    print(f"  final fully burnt     : {base['burnt']} / {base['total']}  ({base['burnt_frac']:.0%})")
    print(f"  never-lit cells left  : {base['untouched']}")
    print(f"  ticks to 50% burnt    : {base['half_tick']}")
    print(f"  min O2 under flame    : {drawdown:.2f} x ambient (1.0 = no drawdown)")
    if base["half_tick"] and boosted["half_tick"]:
        speedup = base["half_tick"] / boosted["half_tick"]
        print(f"  O2 resupply x10 -> ticks to 50% burnt {base['half_tick']} -> {boosted['half_tick']} "
              f"(speedup {speedup:.2f}x)")

    print()
    extinguished_with_fuel = base["extinguished_at"] is not None and base["burnt_frac"] < 0.9
    o2_sped_up = (base["half_tick"] and boosted["half_tick"]
                  and base["half_tick"] > 1.3 * boosted["half_tick"])
    if extinguished_with_fuel:
        print("  -> VERDICT: the front genuinely STALLS -- the fire went out with fuel still")
        print(f"     unburnt ({base['untouched']} never-lit cells). Combined with the O2 columns above,")
        print("     that points to a spread-reach / extinction limit, not a slow but complete burn.")
    elif drawdown < 0.6 or o2_sped_up:
        print("  -> VERDICT: ventilation-limited -- the fire draws its O2 down under the front and")
        print("     extra resupply speeds it up.")
    else:
        print("  -> VERDICT: NOT a barrier -- the grass burns through completely (no O2 drawdown,")
        print("     extra O2 does not help, and the front fills the grid given enough ticks). What")
        print("     looks like 'trouble burning through' is the burn RATE: the front advances slowly")
        print("     and trails a wide band of still-flaming, not-yet-consumed cells. Over a short")
        print("     episode the field looks half-burnt; run long enough and it clears entirely.")


# ---------------------------------------------------------------------------
# Probe C: tree ignition under a passing grass fire (the "trees barely catch" probe)
# ---------------------------------------------------------------------------

def _mixed_state(sim: Simulation, grid: int, grass_load: float, loads: dict[str, float]) -> SimState:
    """A flat SI world uniformly covered by grass plus the given per-type tree loads.

    Mirrors a real map cell: grass and the tree components (canopy + bole) share the cell. The fire
    is started in the grass only -- the question is whether the passing grass front lights the
    co-located / neighbouring trees. `loads` maps fuel-type name -> areal load [kg/m^2].
    """
    gen = MapGenerator(load_config())
    terrain = np.zeros((grid, grid), dtype=np.float64)
    air = gen.air_temperature_profile(terrain)
    mass = gen.boundary_layer_mass(terrain)
    oxygen = gen.oxygen_profile(terrain, air)
    t = lambda a: torch.as_tensor(a, dtype=sim.dtype, device=sim.device)
    air_t, mass_t, oxy_t, terr_t = t(air), t(mass), t(oxygen), t(terrain)

    names = sim.fuel_type_names
    gi = names.index("grass")
    fuel = torch.zeros((sim.num_fuel_types, grid, grid), dtype=sim.dtype, device=sim.device)
    fuel[gi] = grass_load
    for name, load in loads.items():
        fuel[names.index(name)] = load
    fuel_tt = torch.full_like(fuel, pc.T_REF)
    c = grid // 2
    fuel_tt[gi, c, c] = 1200.0   # ignite the grass at the centre only
    z = torch.zeros((1, 1), dtype=sim.dtype, device=sim.device)
    air3 = lift_air_levels(air_t, sim.num_air_levels)   # (L, H, W) per-level air stack
    return SimState(
        mass=mass_t, air_temperatures=air3, ground_temperature=air_t.clone(),
        fuel_temperatures=fuel_tt, fuel=fuel, oxygen=oxy_t, terrain=terr_t,
        temp_eq=air3.clone(), oxygen_eq=oxy_t.clone(), mass_eq=mass_t.clone(),
        x_wind_vel=torch.zeros_like(mass_t), y_wind_vel=torch.zeros_like(mass_t),
        u_amb_x=z, u_amb_y=z, radiant_flux=torch.zeros_like(mass_t),
    )


def probe_tree_ignition(grid: int = 64, grass_load: float = 1.5, ticks: int = 1500) -> None:
    print("\n" + "=" * 78)
    print("PROBE C  --  do trees ignite when a grass fire passes through? (sweep tree COUNT)")
    print("=" * 78)
    cfg = load_config()
    sim0 = Simulation(cfg)
    names = sim0.fuel_type_names
    gi = names.index("grass")
    tree_names = [n for n in names if n.startswith("tree")]
    per_tree = {n: float(cfg["fuel_types"][n].get("fuel_per_tree", 0.0)) for n in tree_names}
    cii = names.index("tree_canopy") if "tree_canopy" in names else None
    # Track the SURFACE bole segment (the trunk base that catches from the ground fire); the tree bole
    # is now split into tree_bole_surface + tree_bole_canopy (Phase 1).
    bii = names.index("tree_bole_surface") if "tree_bole_surface" in names else None
    max_trees = int(cfg["fuel_types"][tree_names[0]].get("max_trees_per_cell", 4))
    a_s = {n: float(sim0.fuel_specific_surface.flatten()[names.index(n)]) for n in tree_names}

    print(f"\ngrid={grid}  grass_load={grass_load} kg/m^2  ticks={ticks}  dt=0.1 s")
    print(f"flame gate = {sim0.flame_gate_temperature:.0f} K, ambient ~ {pc.T_REF:.0f} K")
    print("tree components (per tree): "
          + ", ".join(f"{n}={per_tree[n]:.2f} kg (a_s={a_s[n]:.3g})" for n in tree_names))
    print("each cell holds 1..%d identical trees; both mass and reactive surface scale with the count.\n" % max_trees)
    print(f"  {'trees':>5} {'tot tree':>8} {'grass-burnt':>11} {'canopy lit':>10} "
          f"{'bole lit':>9} {'bole burned':>11}")
    print(f"  {'/cell':>5} {'kg/m^2':>8} {'cells':>11} {'(of burnt)':>10} "
          f"{'(of burnt)':>9} {'(of burnt)':>11}")
    gate = sim0.flame_gate_temperature
    for n_trees in range(1, max_trees + 1):
        sim = Simulation(cfg)
        loads = {n: n_trees * per_tree[n] for n in tree_names}
        s = _mixed_state(sim, grid, grass_load, loads)
        canopy_peak = torch.full((grid, grid), pc.T_REF, dtype=sim.dtype, device=sim.device)
        bole_peak = canopy_peak.clone()
        for _ in range(ticks):
            sim.step_fields(s)
            if cii is not None:
                canopy_peak = torch.maximum(canopy_peak, s.fuel_temperatures[cii])
            if bii is not None:
                bole_peak = torch.maximum(bole_peak, s.fuel_temperatures[bii])

        grass = s.fuel[gi].cpu().numpy()
        grass_burnt = grass < sim.fuel_burnt_threshold
        n_burnt = int(grass_burnt.sum())
        frac = lambda mask: (mask & grass_burnt).sum() / n_burnt if n_burnt else 0.0
        canopy_lit = frac(canopy_peak.cpu().numpy() > gate)
        bole_lit = frac(bole_peak.cpu().numpy() > gate)
        bole_load = loads.get("tree_bole_surface", 0.0)
        if bii is not None and bole_load > 0 and n_burnt:
            bole = s.fuel[bii].cpu().numpy()
            bole_burned = (bole_load - bole[grass_burnt]).sum() / (bole_load * n_burnt)
        else:
            bole_burned = 0.0
        tot = sum(loads.values())
        print(f"  {n_trees:5d} {tot:8.1f} {n_burnt:11d} {canopy_lit:10.0%} "
              f"{bole_lit:9.0%} {bole_burned:11.0%}")

    print("\nReading: with the two-component tree, the light high-SAV CANOPY should ignite from the")
    print("passing grass front ('canopy lit' -> high) where the single lumped tree could not. Once the")
    print("canopy flames, its heat should bring the heavy BOLE up to its gate ('bole lit') and the bole")
    print("then burns slowly ('bole burned' rises over the run). If 'bole lit' stays low, the canopy")
    print("burns out before the bole catches -- raise the bole SAV (so it self-sustains) or the canopy")
    print("mass/coupling.")


def main() -> None:
    p = argparse.ArgumentParser(description="Diagnose fire sustain / burn-through mechanisms")
    p.add_argument("--probe", choices=["A", "B", "C", "both", "all"], default="all")
    p.add_argument("--grid", type=int, default=64, help="grid size for probes B/C (default: 64)")
    p.add_argument("--ticks", type=int, default=1500, help="tick cap for probes B/C (default: 1500)")
    args = p.parse_args()

    # Representative full-cover (4-tree) loads per type for the sustain budget (the bole is now two
    # stacked segments: ~1/3 surface, ~2/3 canopy of the old 17.2 kg/m^2).
    fuel_loads = {"grass": 1.5, "tree_canopy": 2.8, "tree_bole_surface": 5.72, "tree_bole_canopy": 11.48}

    if args.probe in ("A", "both", "all"):
        probe_tree_sustain(fuel_loads)
    if args.probe in ("B", "both", "all"):
        probe_grass_burnthrough(grid=args.grid, ticks=args.ticks)
    if args.probe in ("C", "all"):
        probe_tree_ignition(grid=args.grid, ticks=args.ticks)


if __name__ == "__main__":
    main()
