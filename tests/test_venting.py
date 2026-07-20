"""
Plume vertical venting -- the air-temperature sink (docs/units.md, the air-temp runaway fix).

The combustion air-share heat is injected into the shallow plume slab (update_fire), but the air
field is radiatively transparent and otherwise loses heat only by weak ground exchange and
horizontal advection, so a large or sustained fire's plume heat would accumulate without bound and
conduct back into the fuel -- a temperature runaway. `Simulation.vent_plume_heat` relaxes the
super-ambient air toward the rest profile at the plume-venting rate (the inverse plume-residence
time). These tests pin the operator's exact form (one-directional relaxation, energy-only-removed)
and that, switched on, it holds a sustained fire's air temperature in a physical band while the
unvented model runs away.
"""

import math

import numpy as np
import torch

import physics_constants as pc
from conftest import make_config
from simulation import Simulation, SimState, lift_air_levels
from gen_maps import MapGenerator


def _sim(rate: float = 0.05) -> Simulation:
    cfg = make_config(8, venting=True)
    cfg["venting"]["rate"] = rate
    return Simulation(cfg)


# ---------------------------------------------------------------------------
# Operator form: one-directional exact-exponential relaxation of the TOP level toward temp_eq
# ---------------------------------------------------------------------------

def _stack(sim, base: torch.Tensor) -> torch.Tensor:
    """Lift a (H, W) field to the (L, H, W) air stack (all levels equal)."""
    return lift_air_levels(base, sim.num_air_levels)


def test_vent_cools_super_ambient_air_by_exact_factor():
    """The top level above the rest profile relaxes toward it by exactly 1 - exp(-rate*dt) of the
    excess; the surface/canopy levels are untouched (the vent acts only on the above-canopy level)."""
    sim = _sim(rate=0.05)
    eq0 = torch.full((4, 4), pc.T_REF, dtype=sim.dtype)
    temp_eq = _stack(sim, eq0)
    air = _stack(sim, eq0 + 100.0)
    out = sim.vent_plume_heat(air, temp_eq)

    factor = 1.0 - math.exp(-sim.plume_vent_rate * sim.dt)   # fraction of the excess removed per tick
    top = sim.num_air_levels - 1
    expected_top = eq0 + 100.0 * (1.0 - factor)
    assert torch.allclose(out[top], expected_top)
    # Strictly a sink: it cools the hot top air but never below the rest profile it relaxes toward.
    assert float(out[top].min()) > pc.T_REF
    # Only the top level vents; the lower levels pass through unchanged.
    assert torch.allclose(out[:top], air[:top])


def test_vent_is_one_directional_and_quiescent_safe():
    """Top air at or below the rest profile is untouched (never warmed) -- it is a heat sink only."""
    sim = _sim(rate=0.2)
    eq0 = torch.full((4, 4), pc.T_REF, dtype=sim.dtype)
    temp_eq = _stack(sim, eq0)

    # At the rest profile: a quiescent world holds station (no-op).
    assert torch.allclose(sim.vent_plume_heat(_stack(sim, eq0), temp_eq), temp_eq)
    # Below the rest profile (e.g. a cold pool): venting must not inject energy to warm it.
    cold = _stack(sim, eq0 - 30.0)
    assert torch.allclose(sim.vent_plume_heat(cold.clone(), temp_eq), cold)


def test_vent_rate_zero_is_identity():
    """A zero venting rate disables the sink (the pre-fix behavior, for isolation)."""
    sim = _sim(rate=0.0)
    eq0 = torch.full((4, 4), pc.T_REF, dtype=sim.dtype)
    temp_eq = _stack(sim, eq0)
    air = _stack(sim, eq0 + 500.0)
    assert torch.allclose(sim.vent_plume_heat(air, temp_eq), air)


# ---------------------------------------------------------------------------
# Integration: venting holds a sustained fire's air temperature bounded
# ---------------------------------------------------------------------------

def _sustained_fire_peak_air(venting: bool, grid: int = 96, half: int = 30, ticks: int = 800) -> float:
    """Peak air temperature over a large, sustained (fuel/O2-replenished) windless burn.

    A big block is lit on a flat fueled world and the fuel + oxygen are topped up every tick, so the
    fire keeps burning over the whole patch -- the regime where the plume air-share heat accumulates.
    Convection carries that heat up into the deep above-canopy level, where the vent is the only sink;
    without the vent it accumulates and the whole column (surface included) runs away, with it the air
    is held in a physical band. The horizon is longer than the single-slab model needed (~800 vs ~100
    ticks): the vertical split routes fire heat through the deep above-canopy reservoir, which must
    fill before the missing sink bites, so the runaway manifests later. Returns the peak air temp.
    """
    # Convection always carries the surface combustion heat up into the above-canopy level; venting
    # is the sink that sheds it there. Toggling only venting isolates the sink: with it off, the heat
    # convected up has nowhere to go and the top level accumulates (the runaway this fix targets).
    cfg = make_config(grid, fire=True, radiation=True, relaxation=True,
                      venting=venting, air_convection=True)
    sim = Simulation(cfg)
    gen = MapGenerator(make_config(grid))
    terrain = np.zeros((grid, grid), dtype=np.float64)
    air = gen.air_temperature_profile(terrain)
    mass = gen.boundary_layer_mass(terrain)
    oxygen = gen.oxygen_profile(terrain, air)
    t = lambda a: torch.as_tensor(a, dtype=sim.dtype, device=sim.device)

    fuel = torch.full((1, grid, grid), 1.0, dtype=sim.dtype, device=sim.device)
    fuel_t = torch.full((1, grid, grid), pc.T_REF, dtype=sim.dtype, device=sim.device)
    c = grid // 2
    fuel_t[0, c - half:c + half, c - half:c + half] = 1200.0   # a large hot ignition block
    fuel0, oxygen0 = fuel.clone(), t(oxygen).clone()
    zero = torch.zeros((1, 1), dtype=sim.dtype, device=sim.device)

    s = SimState(
        mass=t(mass),
        air_temperatures=lift_air_levels(t(air), sim.num_air_levels), ground_temperature=t(air),
        fuel_temperatures=fuel_t, fuel=fuel, oxygen=t(oxygen), terrain=t(terrain),
        temp_eq=lift_air_levels(t(air), sim.num_air_levels), oxygen_eq=t(oxygen), mass_eq=t(mass),
        x_wind_vel=torch.zeros_like(t(mass)), y_wind_vel=torch.zeros_like(t(mass)),
        u_amb_x=zero, u_amb_y=zero, radiant_flux=torch.zeros_like(t(mass)),
    )
    peak = 0.0
    for _ in range(ticks):
        sim.step_fields(s)
        s.fuel, s.oxygen = fuel0.clone(), oxygen0.clone()   # sustain the burn
        peak = max(peak, float(s.air_temperatures.max()))
    return peak


def test_venting_bounds_sustained_fire_air_temperature():
    """With venting on, a sustained fire's air stays in a physical band; without it, it runs hotter.

    The unvented air has no vertical sink, so its plume heat accumulates well past the vented case --
    the temperature runaway this fix targets (far more extreme on the full 256^2 production grid /
    longer horizons; this modest grid keeps the test fast while still separating the two clearly).
    """
    peak_on = _sustained_fire_peak_air(venting=True)
    peak_off = _sustained_fire_peak_air(venting=False)

    assert peak_on < 4000.0                  # venting holds the air in a physical band (~3000 K)
    assert peak_off > peak_on + 1000.0       # without venting the air accumulates clearly hotter (~5100 K)
