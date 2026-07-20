"""Phase 0 dynamics spike (isolated, no production changes).

Reuses the real Simulation primitives (_periodic_grad/_laplacian/_advect_periodic/_sponge_rate)
to build a 3-level step_dynamics, and compares the fire-induced wind produced by the SAME fire
heat Q against the current single-layer core. Goal: measure the fire-wind weakening (the ~89x
amplifier removal) and confirm the multi-level advection is stable (mass floor, no thin-slab
blow-up).

Levels: surface 0-5 m, canopy 5-15 m, above-canopy = rest of column.
"""
import sys, math
sys.path.insert(0, "src")
import numpy as np
import torch

from config import load_config
from simulation import Simulation, SimState
from gen_maps import MapGenerator
import physics_constants as pc

GRID = 96
DEPTHS = [5.0, 10.0, None]        # None = rest-of-column (mass = total - thin slabs)
CP, R_D, P_REF = pc.CP_AIR, pc.GAS_CONSTANT_DRY_AIR, pc.P_REF


def make_world():
    cfg = load_config()
    cfg.setdefault("units", {})["seconds_per_tick"] = 1.0
    sim = Simulation(cfg)
    terrain = np.zeros((GRID, GRID), dtype=np.float64)
    gen = MapGenerator(cfg)
    air = gen.air_temperature_profile(terrain)
    mass = gen.boundary_layer_mass(terrain)
    t = lambda a: torch.as_tensor(a, dtype=sim.dtype)
    return sim, t(terrain), t(air), t(mass)


def level_masses(total_mass):
    """Per-level mass (L,H,W): thin slabs = rho_ref*depth (uniform const), rest = total - slabs."""
    m0 = pc.RHO_REF * DEPTHS[0]
    m1 = pc.RHO_REF * DEPTHS[1]
    thin0 = torch.full_like(total_mass, m0)
    thin1 = torch.full_like(total_mass, m1)
    rest = (total_mass - m0 - m1).clamp(min=1.0)
    return torch.stack([thin0, thin1, rest], dim=0)


def step_multi(sim, m, T, u, v, terrain, temp_eq, m_eq, g_prime, mass_floor=1e-3):
    """3-level shallow-water dynamics with one shared wind. Mirrors Simulation.step_dynamics."""
    dx, dt = sim.cell_size_m, sim.dt
    C_d, nu = sim.drag_coeff, sim.viscosity
    grad, lap, adv = sim._periodic_grad, sim._periodic_laplacian, sim._advect_periodic

    eta = (m * R_D * T / P_REF).sum(0)                       # (H,W) total column thickness
    wave = float(torch.sqrt((g_prime * eta).clamp(min=0.0)).max())
    flow = float((u.abs() + v.abs()).max())
    n = max(1, int(math.ceil((flow + wave) * dt / dx / sim.cfl_target)))
    dts = dt / n

    for _ in range(n):
        eta = (m * R_D * T / P_REF).sum(0)
        surface = terrain + eta
        gx, gy = grad(surface, dx)
        u = u + dts * (-g_prime * gx - C_d * u + nu * lap(u, dx))
        v = v + dts * (-g_prime * gy - C_d * v + nu * lap(v, dx))

        dispx, dispy = u * dts / dx, v * dts / dx
        m_tot_pre = m.sum(0)
        mom_x, mom_y = m_tot_pre * u, m_tot_pre * v
        new_m = torch.empty_like(m)
        new_T = torch.empty_like(T)
        for L in range(m.shape[0]):
            energy = m[L] * CP * T[L]
            mL = adv(m[L], dispx, dispy)
            energy = adv(energy, dispx, dispy)
            new_m[L] = mL
            new_T[L] = energy / (CP * mL.clamp(min=mass_floor))   # floor guards the divide
        m, T = new_m, new_T
        mom_x, mom_y = adv(mom_x, dispx, dispy), adv(mom_y, dispx, dispy)
        m_tot = m.sum(0).clamp(min=mass_floor)
        u, v = mom_x / m_tot, mom_y / m_tot

    decay = torch.exp(-sim._sponge_rate(m.shape[-2:]) * dt)
    u = u * decay          # u_amb = 0 here
    v = v * decay
    m = m_eq + (m - m_eq) * decay
    T = temp_eq + (T - temp_eq) * decay
    return m, T, u, v


def run_single(sim, terrain, air, mass, Q, ticks):
    """Current single-layer core: inject Q via the C_plume (100 m) path, then step_dynamics."""
    C_plume = pc.RHO_REF * sim.plume_mixing_depth * CP
    air = air.clone()
    c = GRID // 2
    air[c - 1:c + 2, c - 1:c + 2] += Q / C_plume            # warm patch (one-shot heat Q)
    z = torch.zeros_like(mass)
    s = SimState(
        mass=mass.clone(), air_temperatures=air, ground_temperature=air.clone(),
        fuel_temperatures=torch.full((1, GRID, GRID), pc.T_REF, dtype=sim.dtype),
        fuel=torch.zeros((1, GRID, GRID), dtype=sim.dtype), oxygen=z.clone(), terrain=terrain,
        temp_eq=torch.full_like(mass, pc.T_REF), oxygen_eq=z.clone(), mass_eq=mass.clone(),
        x_wind_vel=z.clone(), y_wind_vel=z.clone(),
        u_amb_x=torch.zeros((1, 1), dtype=sim.dtype), u_amb_y=torch.zeros((1, 1), dtype=sim.dtype),
        radiant_flux=z.clone(),
    )
    peak = 0.0
    for _ in range(ticks):
        sim.step_dynamics(s)
        peak = max(peak, float(torch.sqrt(s.x_wind_vel ** 2 + s.y_wind_vel ** 2).max()))
    return peak, bool(torch.isfinite(s.air_temperatures).all())


def run_multi(sim, terrain, air, mass, Q, ticks, g_prime):
    m = level_masses(mass)
    T = torch.stack([torch.full_like(mass, pc.T_REF)] * 3, dim=0)
    c = GRID // 2
    C0 = m[0] * CP                                           # surface-level thermal mass
    T[0, c - 1:c + 2, c - 1:c + 2] += (Q / C0)[c - 1:c + 2, c - 1:c + 2]
    u = torch.zeros_like(mass); v = torch.zeros_like(mass)
    temp_eq = torch.stack([torch.full_like(mass, pc.T_REF)] * 3, dim=0)
    m_eq = level_masses(mass)
    peak = 0.0
    min_thin = float("inf")
    for _ in range(ticks):
        m, T, u, v = step_multi(sim, m, T, u, v, terrain, temp_eq, m_eq, g_prime)
        peak = max(peak, float(torch.sqrt(u ** 2 + v ** 2).max()))
        min_thin = min(min_thin, float(m[0].min()))
    finite = bool(torch.isfinite(T).all() and torch.isfinite(u).all())
    return peak, finite, min_thin


if __name__ == "__main__":
    sim, terrain, air, mass = make_world()
    g0 = sim.reduced_gravity
    Q = 2.0e5          # J/m^2 one-shot fire heat in the patch (small -> stay near-linear)
    TICKS = 60
    print(f"grid {GRID}, dt {sim.dt}s, reduced_gravity g'={g0}, plume_mixing_depth={sim.plume_mixing_depth} m")
    print(f"column mass ~{float(mass.mean()):.0f} kg/m2; thin slabs = {pc.RHO_REF*5:.2f}, {pc.RHO_REF*10:.2f} kg/m2\n")

    ps, fin_s = run_single(sim, terrain, air, mass, Q, TICKS)
    pm, fin_m, min_thin = run_multi(sim, terrain, air, mass, Q, TICKS, g0)
    print(f"single-layer peak fire-wind : {ps:.4e} m/s   finite={fin_s}")
    print(f"multi-level  peak fire-wind : {pm:.4e} m/s   finite={fin_m}  min thin-slab mass={min_thin:.3f}")
    ratio = ps / pm if pm > 0 else float('nan')
    print(f"weakening ratio single/multi = {ratio:.1f}x   (analytic expectation ~88.7x)\n")

    # Retune: wind ~ linear in g', so g'_multi = g0 * ratio should restore the single-layer wind.
    g_retune = g0 * ratio
    pm2, fin_m2, _ = run_multi(sim, terrain, air, mass, Q, TICKS, g_retune)
    print(f"multi-level with g'={g_retune:.2f} (=g0*ratio): peak = {pm2:.4e} m/s (target {ps:.4e})  finite={fin_m2}")
