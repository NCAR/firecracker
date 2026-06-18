"""
Surface-area-controlled, oxygen-limited combustion (Phase 4 + 5.5).

update_fire burns fuel at its reactive surface, not in bulk: the areal rate is a_s*fuel times a
surface mass flux psi that puts chemistry in series with oxygen transport -- psi_kin =
B*exp(-E_a/(R*T_fuel)) (kinetic, the ignition gate; cold fuel is inert as the exponential
underflows) and psi_dif = h*(O2/O2_ref) (O2 film diffusion, the flame-hot cap). Consumption is the
exact-exponential Dfuel = fuel*(1 - exp(-a_s*psi*dt)), so a cell burns over a residence time
tau ~ 1/(a_s*psi) (Phase 5.5) instead of all at once. Burned mass releases its heat of combustion
HHV, split between fuel and air by burn_heat_fuel_fraction; oxygen [kg/m^3] also couples to the
areal burn [kg/m^2] through the shallow mixing depth d_mix as a whole-cell inventory backstop. The
air share of the heat is deposited into a shallow plume slab of depth plume_mixing_depth (Phase
5.5b), using the local near-surface density, so a burn warms the air strongly (a convective signal),
not over the full ~1 km column. These tests pin the inert-cold limit, ignition under heat, the (now
series, not linear) oxygen throttle, the inventory cap, the stoichiometric O2 budget, the HHV energy
split (against the plume capacity), the plume concentration of the air share, kinetic monotonicity
below the transport cap, the multi-tick residence time, and the surface-area (SAV) control of burn
rate. Rates are checked in regimes chosen so the inventory cap does not bind unless a test targets it.
"""

import numpy as np
import torch

import physics_constants as pc
from conftest import make_config
from simulation import Simulation


def _sim() -> Simulation:
    # grass falls to the _ft_tensor combustion defaults (B=1e8, E_a=1e5, HHV=1.6e7, s=1.4,
    # sigma=12000, rho_p=500 -> a_s=24); h=0.02, d_mix=30 m and dt=0.1 s.
    return Simulation(make_config(fuel_type_names=("grass",)))


def _sim_no_conv() -> Simulation:
    """Sim with the Phase 5.5c convective preheat disabled, to isolate the leg-b air/HHV split.

    convective_fraction=0 routes the whole air share into the plume (no part diverted to neighbour
    fuel), recovering the leg-b behaviour these split tests pin; convective spread has its own
    test_spread coverage.
    """
    cfg = make_config(fuel_type_names=("grass",))
    cfg["fire"] = {"enabled": True, "spawn_fire": False, "convective_fraction": 0.0}
    return Simulation(cfg)


def _cfg_with(*, sav: float) -> dict:
    """make_config with a chosen surface-area-to-volume for the single grass type."""
    cfg = make_config(fuel_type_names=("grass",))
    cfg["fuel_types"]["grass"]["surface_area_to_volume"] = sav
    return cfg


def _f(sim, value, shape=(8, 8)) -> torch.Tensor:
    return torch.full(shape, float(value), dtype=sim.dtype, device=sim.device)


def _burn(sim, *, temp, oxygen, fuel=1.0, mass=1500.0):
    """Run one update_fire from a uniform state; return (air, fuel_t, fuel, oxygen) tensors."""
    shape = (8, 8)
    air = _f(sim, pc.T_REF)
    fuel_t = _f(sim, temp).unsqueeze(0)
    fuel_a = _f(sim, fuel).unsqueeze(0)
    oxy = _f(sim, oxygen)
    m = _f(sim, mass)
    return sim.update_fire(air, fuel_t, fuel_a, oxy, m)


def test_cold_fuel_is_inert():
    """At ambient temperature the Arrhenius rate underflows, so fuel barely burns over a long run."""
    sim = _sim()
    air = _f(sim, pc.T_REF)
    fuel_t = _f(sim, pc.T_REF).unsqueeze(0)
    fuel = _f(sim, 1.0).unsqueeze(0)
    oxy = _f(sim, pc.O2_DENSITY_REF)
    mass = _f(sim, 1500.0)

    for _ in range(500):
        air, fuel_t, fuel, oxy = sim.update_fire(air, fuel_t, fuel, oxy, mass)

    np.testing.assert_allclose(fuel.cpu().numpy(), 1.0, atol=1e-6)   # essentially no consumption
    assert torch.isfinite(fuel_t).all()


def test_hot_fuel_ignites_and_releases_heat():
    """Fuel well above pyrolysis burns: fuel drops, oxygen is drawn down, and heat is released."""
    sim = _sim()
    _, fuel_t, fuel, oxy = _burn(sim, temp=700.0, oxygen=pc.O2_DENSITY_REF)

    assert float(fuel.mean()) < 1.0                       # fuel consumed
    assert float(oxy.mean()) < pc.O2_DENSITY_REF          # oxygen consumed
    assert float(fuel_t.mean()) > 700.0                   # combustion heated the fuel


def test_oxygen_throttles_and_extinguishes():
    """O2 enters the surface flux (psi_dif), so the rate vanishes with no oxygen and rises with it.

    The dependence is no longer linear -- O2 is one of two series resistances (film transport vs
    chemistry) -- so we pin the qualitative throttle: zero O2 burns nothing, more O2 burns strictly
    more.
    """
    sim = _sim()

    # No oxygen -> psi_dif = 0 -> the series flux is zero -> no reaction at all.
    air0, _, fuel0, oxy0 = _burn(sim, temp=700.0, oxygen=0.0)
    np.testing.assert_allclose(fuel0.cpu().numpy(), 1.0, atol=1e-12)
    np.testing.assert_allclose(air0.cpu().numpy(), pc.T_REF, atol=1e-12)
    np.testing.assert_allclose(oxy0.cpu().numpy(), 0.0, atol=1e-12)

    # More oxygen -> more burn (monotonic, via the film-transport term psi_dif = h*(O2/O2_ref)).
    burned = []
    for frac in (0.0, 0.5, 1.0):
        _, _, fuel, _ = _burn(sim, temp=600.0, oxygen=frac * pc.O2_DENSITY_REF)
        burned.append(1.0 - float(fuel.mean()))
    assert burned[0] == 0.0
    assert burned[0] < burned[1] < burned[2]


def test_oxygen_inventory_cap_binds_at_high_load():
    """The d_mix inventory backstop: a dense fuel load can out-draw the O2 present in one tick.

    The film-transport throttle (psi_dif) usually self-limits the draw below the cell's O2, but a
    heavy load (here 300 kg/m^2) demands more O2 than the d_mix slab holds, so the whole-cell
    inventory limiter scales the reaction to the oxygen available -- all O2 is consumed and the fuel
    burned matches it stoichiometrically.
    """
    sim = _sim()
    fuel0, o2_start = 300.0, 0.1                      # dense load, modest O2 -> inventory cap binds
    _, _, fuel, oxy = _burn(sim, temp=1000.0, oxygen=o2_start, fuel=fuel0)

    fuel_burned = fuel0 - float(fuel.mean())
    s, d_mix = float(sim.stoich_oxygen[0]), sim.combustion_mixing_depth

    np.testing.assert_allclose(float(oxy.mean()), 0.0, atol=1e-9)   # all oxygen consumed
    # Fuel actually burned matches the oxygen that was available (stoichiometric, not over-drawn).
    np.testing.assert_allclose(s * fuel_burned / d_mix, o2_start, rtol=1e-6)


def test_stoichiometric_oxygen_budget():
    """Oxygen drawn (density * mixing depth) equals s * fuel burned, when neither cap binds."""
    sim = _sim()
    _, _, fuel, oxy = _burn(sim, temp=640.0, oxygen=pc.O2_DENSITY_REF)

    fuel_burned = 1.0 - float(fuel.mean())
    o2_drawn_areal = (pc.O2_DENSITY_REF - float(oxy.mean())) * sim.combustion_mixing_depth
    s = float(sim.stoich_oxygen[0])
    np.testing.assert_allclose(o2_drawn_areal, s * fuel_burned, rtol=1e-6)
    assert fuel_burned > 0.0


def test_hhv_energy_split():
    """Released heat HHV*Dfuel is split between air and fuel by burn_heat_fuel_fraction."""
    sim = _sim_no_conv()
    mass = 1500.0
    air, fuel_t, fuel, _ = _burn(sim, temp=640.0, oxygen=pc.O2_DENSITY_REF, mass=mass)

    fuel_burned = 1.0 - float(fuel.mean())
    hhv = float(sim.heat_of_combustion[0])
    cp_fuel = float(sim.fuel_specific_heat[0])
    f = sim.burn_heat_fuel_fraction

    # Air share is deposited into the shallow plume slab, not the full column: its capacity is
    # C_plume = rho(T_a)*d_plume*c_p_air with the local density rho = p_ref/(R_d*T_a) at the
    # pre-step air temperature (T_REF here). So C_plume*dT_air = (1-f)*HHV*Dfuel.
    rho = pc.P_REF / (pc.GAS_CONSTANT_DRY_AIR * pc.T_REF)
    C_plume = rho * sim.plume_mixing_depth * pc.CP_AIR
    air_energy = C_plume * (float(air.mean()) - pc.T_REF)
    np.testing.assert_allclose(air_energy, (1.0 - f) * hhv * fuel_burned, rtol=1e-6)

    # Fuel share: c_p_fuel*fuel_new*dT_fuel = f*HHV*Dfuel.
    fuel_energy = cp_fuel * float(fuel.mean()) * (float(fuel_t.mean()) - 640.0)
    np.testing.assert_allclose(fuel_energy, f * hhv * fuel_burned, rtol=1e-6)


def test_arrhenius_monotonic_in_temperature():
    """Hotter fuel burns faster while the kinetic flux is limiting (below the transport cap).

    Above the kinetic->diffusion crossover the surface flux saturates at psi_dif, so the burn rate
    plateaus; here the temperatures sit below it, where psi is still climbing with the Arrhenius term.
    """
    sim = _sim()
    burned = []
    for temp in (560.0, 600.0, 640.0):
        _, _, fuel, _ = _burn(sim, temp=temp, oxygen=pc.O2_DENSITY_REF)
        burned.append(1.0 - float(fuel.mean()))

    assert burned[0] < burned[1] < burned[2]
    assert burned[0] > 0.0


def test_burn_is_not_one_tick_and_lasts_a_residence_time():
    """Phase 5.5: a flame-hot cell is consumed gradually over seconds, not all in a single tick.

    Surface-area control + O2 film transport cap the rate, so even at a high flame temperature only
    a small fraction of the fuel burns per 0.1 s tick, and full burn-down takes a multi-second
    residence time. (Fuel temperature is pinned to the flame value each tick to isolate the
    consumption rate -- in the full loop the radiative T^4 sink bounds the flame instead.)
    """
    sim = _sim()
    flame = 1200.0

    # One tick at flame temperature leaves most of the fuel: NOT the old one-tick burnout.
    _, _, fuel1, _ = _burn(sim, temp=flame, oxygen=pc.O2_DENSITY_REF, fuel=1.0)
    assert float(fuel1.mean()) > 0.9

    # Burn down with the temperature held at the flame value; count ticks to consume ~90%.
    shape = (8, 8)
    air = _f(sim, pc.T_REF)
    fuel = _f(sim, 1.0).unsqueeze(0)
    oxy = _f(sim, pc.O2_DENSITY_REF)
    mass = _f(sim, 1500.0)
    ticks = 0
    while float(fuel.mean()) > 0.1 and ticks < 2000:
        fuel_t = _f(sim, flame).unsqueeze(0)          # re-pin the flame temperature each tick
        air, _, fuel, oxy = sim.update_fire(air, fuel_t, fuel, oxy, mass)
        oxy = _f(sim, pc.O2_DENSITY_REF)              # hold ambient O2 (isolate the surface rate)
        ticks += 1

    residence_s = ticks * sim.dt
    assert residence_s > 2.0          # a real flaming residence, not a single 0.1 s tick
    assert residence_s < 120.0        # but grass-fast, not log-slow


def test_surface_area_to_volume_controls_burn_rate():
    """Phase 5.5: high-SAV fuel (grass) burns far faster than low-SAV fuel (a log) at equal state.

    Same material (HHV, density, kinetics), only sigma differs -- grass shaved fine vs a coarse log
    -- so the reactive surface area a_s = sigma/rho_p sets the rate, the physical reason grass
    flashes and logs smoulder.
    """
    grass = Simulation(_cfg_with(sav=12000.0))
    log = Simulation(_cfg_with(sav=40.0))

    _, _, fuel_g, _ = _burn(grass, temp=1200.0, oxygen=pc.O2_DENSITY_REF, fuel=1.0)
    _, _, fuel_l, _ = _burn(log, temp=1200.0, oxygen=pc.O2_DENSITY_REF, fuel=1.0)

    burned_grass = 1.0 - float(fuel_g.mean())
    burned_log = 1.0 - float(fuel_l.mean())
    assert burned_grass > 100.0 * burned_log          # ~sigma ratio (300x) -> grass burns far faster
    assert burned_log > 0.0


def test_plume_concentrates_combustion_heat():
    """Phase 5.5b: the air-share heat lands in a shallow plume, so a burn warms the air strongly.

    The heat goes into a d_plume-deep slab (capacity C_plume = rho*d_plume*c_p_air), not the full
    ~1 km boundary-layer column (m*c_p_air), so the same burn raises the air temperature by the
    column-to-plume mass ratio more -- the strong, advectable convective signal leg (c) needs. We
    pin the deposit to C_plume and check it is far larger than the full-column rise would be.
    """
    sim = _sim_no_conv()   # isolate the plume deposit (no convective diversion to neighbour fuel)
    mass = 1500.0
    air, _, fuel, _ = _burn(sim, temp=640.0, oxygen=pc.O2_DENSITY_REF, mass=mass)

    fuel_burned = 1.0 - float(fuel.mean())
    f = sim.burn_heat_fuel_fraction
    hhv = float(sim.heat_of_combustion[0])
    air_share = (1.0 - f) * hhv * fuel_burned                       # [J/m^2]

    dT_air = float(air.mean()) - pc.T_REF
    rho = pc.P_REF / (pc.GAS_CONSTANT_DRY_AIR * pc.T_REF)
    C_plume = rho * sim.plume_mixing_depth * pc.CP_AIR

    # The rise matches the plume capacity, and dwarfs what the full column would give.
    np.testing.assert_allclose(dT_air, air_share / C_plume, rtol=1e-6)
    dT_column = air_share / (mass * pc.CP_AIR)
    assert dT_air > 10.0 * dT_column                                # ~m/(rho*d_plume) stronger
    assert np.isfinite(dT_air)
