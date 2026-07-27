"""
physics_constants.py

Physical constants and reference-atmosphere scales for the SI Firecracker model.

All simulation state is in SI units (kelvin, kg, metres, seconds, joules, pascals).
These are the fixed constants of nature plus a small ISA-style reference atmosphere
the map generator and (later) the physics solvers calibrate against. Tunable,
scenario-specific values live in cfg/default.toml, not here.
"""

# --- Fundamental constants -------------------------------------------------
GRAVITY:              float = 9.81          # g       [m/s^2]
GAS_CONSTANT_DRY_AIR: float = 287.05        # R_d     [J/(kg*K)]  specific gas constant of dry air
UNIVERSAL_GAS_CONSTANT: float = 8.314462    # R       [J/(mol*K)] molar gas constant (Arrhenius)
CP_AIR:               float = 1005.0        # c_p of air at constant pressure [J/(kg*K)]
STEFAN_BOLTZMANN:     float = 5.670374e-8   # sigma   [W/(m^2*K^4)]

# --- Reference atmosphere (sea level, ISA-like) ----------------------------
T_REF:                 float = 303.15       # sea-level reference temperature [K] (30 C, fire weather)
P_REF:                 float = 101325.0     # sea-level pressure [Pa]
RHO_REF:               float = 1.165        # sea-level air density at T_REF [kg/m^3] = P_REF/(R_d*T_REF)
LAPSE_RATE:            float = 6.5e-3       # environmental lapse rate [K/m]
PRESSURE_SCALE_HEIGHT: float = 8870.0       # hydrostatic scale height H_p = R_d*T_REF/g [m]

# --- Air composition -------------------------------------------------------
O2_MASS_FRACTION: float = 0.2315            # kg O2 per kg dry air
O2_DENSITY_REF:   float = 0.2315 * 1.165    # ambient sea-level O2 partial density [kg/m^3] (~0.27)

# --- Radiation -------------------------------------------------------------
EMISSIVITY: float = 0.95                    # grey-body emissivity of the surface

# --- Water (fire-suppression moisture) -------------------------------------
# Liquid water dropped on a cell (a fire-suppression action) sits as fuel/surface moisture.
# It cannot get hotter than its boiling point: heat from the surrounding fuel and air first
# drives it to WATER_BOILING_POINT and then boils it off at WATER_LATENT_HEAT per kg -- a large
# energy sink (~5x a fuel's heat of combustion released per kg burned is offset by ~0.14 kg of
# water), which is why a wet cell stays near the boiling point and cannot reach ignition until
# the water has evaporated. See Simulation.apply_moisture.
WATER_LATENT_HEAT:   float = 2.26e6         # latent heat of vaporization of water [J/kg]
WATER_BOILING_POINT: float = 373.15         # boiling point of water at sea level [K] (100 C)

# --- Domain reference scales -----------------------------------------------
ELEV_MAX_M:     float = 500.0               # documented terrain elevation ceiling over the domain [m]
FUEL_REF_KG_M2: float = 20.0                # representative dense-forest areal biomass [kg/m^2]
WIND_REF_M_S:   float = 30.0                # solver's peak-wind calibration target [m/s] (synoptic x channeling gain)
SLOPE_REF_TAN:  float = 1.0                 # slope-observation reference: grad(z)=tan(slope); a 45-deg slope -> +/-1

# --- Observation normalization ---------------------------------------------
# The world-model observation (Simulation.build_observation) carries SI fields whose
# magnitudes span several orders (K vs kg/m^2 vs m). Each channel is mapped to roughly
# [0, 1] by an affine (value - offset) / scale with a documented, physically motivated
# window -- deterministic, not data-fitted. Keyed by channel name; every per-type fuel
# channel (fuel_<name>) shares the single "fuel" window here. The high side
# is intentionally not clamped: flames legitimately exceed the temperature window (they
# ride a little above 1) rather than saturating the model's view of fire.
OBS_NORM: dict[str, tuple[float, float]] = {
    # rest air temperature (~T_REF) -> ~0; a ~1300 K hot plume -> ~1. Air stays well below flame
    # temperature (unlike the fuel channels), so it gets a tighter window for better resolution.
    "air_temperature":  (T_REF, 1000.0),
    # rest air/fuel temperature (~T_REF) -> ~0; a ~2300 K flame -> ~1.
    "fuel_temperature": (T_REF, 2000.0),
    # bare ground -> 0; dense forest (~FUEL_REF) -> ~1.
    "fuel":             (0.0, FUEL_REF_KG_M2),
    # sea level -> 0; domain ceiling -> 1. (Terrain height is no longer an observation channel --
    # the fire reads slope, not height -- but the window is kept for the field's provenance.)
    "terrain":          (0.0, ELEV_MAX_M),
    # signed terrain slope components grad(z) = tan(slope), uphill +: calm/flat -> 0, a 45-deg slope
    # -> ~+/-1. Physically it is slope, not height, that skews fire spread (Rothermel slope effect),
    # and the derivative is hard for a conv stack to synthesize from a raw height field, so it is
    # supplied directly. Signed and zero-centered; the high side is not clamped (steeper -> above 1).
    "slope_x":          (0.0, SLOPE_REF_TAN),
    "slope_y":          (0.0, SLOPE_REF_TAN),
    # near-surface fire wind components [m/s]: signed, calm -> 0, a ~30 m/s peak -> ~+/-1.
    "wind_x":           (0.0, WIND_REF_M_S),
    "wind_y":           (0.0, WIND_REF_M_S),
    # binary "ignited"/burning mask is already 0/1, so the window is the identity.
    "ignited":          (0.0, 1.0),
}

# --- Default discretisation scales -----------------------------------------
# Overridable from cfg/default.toml ([units].cell_size_m, [simulation].steps_per_second).
DEFAULT_CELL_SIZE_M: float = 10.0           # dx = dy [m]
