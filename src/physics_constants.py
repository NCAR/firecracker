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

# --- Default discretisation scales -----------------------------------------
# Overridable from cfg/default.toml ([units].cell_size_m, [simulation].steps_per_second).
DEFAULT_CELL_SIZE_M: float = 10.0           # dx = dy [m]
