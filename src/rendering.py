"""
rendering.py

Stateless presentation layer for the Firecracker environment: the drawing primitives that turn
simulation fields into pygame surfaces. Everything here is a pure function of numpy arrays (and a
few display constants) with no reference to the env, the simulation, or torch, so it can be tested
in isolation and the env depends on it one-way (firecracker_env -> rendering).

Two pieces:
  * Colormaps + field surfaces — each field-fill view owns a Colormap (normalized [0, 1] -> RGB)
    and renders both its pixels and its colorbar from that one callable, so the bar can never
    disagree with the field. The fire view is the exception: it paints fuel mass and the burning
    overlay from several fields directly rather than mapping one scalar field through a gradient.
  * Colorbar legend — a ColorbarSpec describes one scale bar; the panel composes one or more.
"""

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pygame

# A colormap maps an array of normalized values in [0, 1] (any shape) to RGB along a new
# trailing axis, returning uint8 (..., 3). The same callable drives a view's pixel surface
# and its colorbar legend, so the bar always matches the field. See heat_colormap.
Colormap = Callable[[np.ndarray], np.ndarray]

# ---------------------------------------------------------------------------
# Display constants
# ---------------------------------------------------------------------------

MAX_CHANNEL_VALUE: int = 255
GREEN_CHANNEL: int = 1

# Per-fuel-type base colors (RGB fractions in [0, 1]) for the fire view's vegetation.
# A cell's hue is its fuel mass blended across these by type: trees read pure green, while
# grass reads yellow-green (extra red). Brightness still tracks total fuel (see
# build_fire_surface), so these only set the hue each fuel type contributes. Unknown fuel
# types fall back to plain green.
VEGETATION_COLORS: dict[str, tuple[float, float, float]] = {
    "tree":        (0.0,  1.0, 0.0),   # green
    "tree_canopy": (0.0,  1.0, 0.0),   # green (a tree's fine fuel)
    "tree_bole":   (0.5,  0.4, 0.3),   # brown (a tree's coarse fuel); only shown through a thin canopy
    "grass":       (0.75, 1.0, 0.0),   # yellow-green
    "shrub":       (0.5,  0.75, 0.25), # muted olive-green (woody scrub)
}
DEFAULT_VEGETATION_COLOR: tuple[float, float, float] = (0.0, 1.0, 0.0)

# A tree's bole (trunk/coarse fuel) is occluded by its canopy: its brown only shows through once
# the co-located canopy's fuel fraction (normalised by the map's peak canopy loading, like every
# other type's brightness) drops below this. Below it the bole reads as brown; at/above it the
# bole contributes nothing and the cell reads as green canopy.
TREE_BOLE_CANOPY_FRACTION: float = 0.10

# Biome view: one flat color per non-overlapping vegetation biome (see gen_maps.classify_biomes),
# taken straight from the fire-view vegetation hue of the fuel that dominates each biome — woodland
# reads as trees, grassland as grass, shrubland as shrub — so the biome map colors match how that
# vegetation is drawn elsewhere. Ordered by the integer label the classifier assigns (0 woodland,
# 1 grassland, 2 shrubland), so a label array indexes straight into the palette.
BIOME_NAMES: list[str] = ["Woodland", "Grassland", "Shrubland"]
_BIOME_FUELS: list[str] = ["tree", "grass", "shrub"]
BIOME_COLORS: np.ndarray = np.array(
    [[int(round(c * MAX_CHANNEL_VALUE)) for c in VEGETATION_COLORS[f]] for f in _BIOME_FUELS],
    dtype=np.uint8,
)

# ---------------------------------------------------------------------------
# Wind arrow visualization parameters
# ---------------------------------------------------------------------------

_WIND_DISPLAY_THRESHOLD:       float = 1e-4

# Temperature views map a Kelvin window onto the color ramp. The window auto-ranges to the live
# field: floor = the coldest cell, ceiling = the hottest (so the scale always reflects the real
# range -- cool shaded ground through to a flame -- instead of a fixed band). The span is clamped
# to a minimum so a near-uniform field (no fire) doesn't get over-stretched into noise. The floor
# tracks the coldest cell, which stays at ambient even during a fire (combustion only heats), so a
# flame raises the ceiling without destabilising the floor. DISPLAY_TEMP_FLOOR_K is only the
# pre-first-frame default window, before any field has been measured.
DISPLAY_TEMP_FLOOR_K:    float = 290.0
DISPLAY_MIN_TEMP_SPAN_K: float = 10.0
# Kelvin above ignition that spans the fire-overlay color ramp (ignition -> +span). Sized to
# the SI flame band: ignition ~550-600 K up through a sustained ~1540 K front to ~2400 K peaks,
# so the ramp resolves flame structure instead of saturating just above ignition.
FIRE_COLOR_TEMP_SPAN_K:  float = 1600.0

# Oxygen views map a fixed window [floor, ceiling] of O2 partial density [kg/m^3] onto the
# grayscale ramp: at/below the floor renders fully scarce (white), at/above the ceiling fully
# plentiful (black). Fixed bounds keep the scale stable frame-to-frame instead of tracking the
# live max, so the same gray always means the same O2 density.
DISPLAY_OXYGEN_FLOOR:   float = 0.20
DISPLAY_OXYGEN_CEILING: float = 0.28

WIND_ARROW_STRIDE: int = 16
WIND_ARROW_LENGTH: int = 24
WIND_ARROW_LINE_WIDTH: int = 1
WIND_ARROW_HEAD_SIZE: int = 4
WIND_ARROW_HEAD_ANGLE: float = math.pi / 6  # 30 degrees

# ---------------------------------------------------------------------------
# Colorbar legend (right-side panel)
# ---------------------------------------------------------------------------

# The legend lives in a fixed-width panel reserved on the right of the window, so the window
# is permanently this much wider than the simulation. Modes without a colorbar leave it empty.
# The panel holds up to LEGEND_PANEL_BARS bars side by side (= GUTTER + BARS * COLUMN_WIDTH).
LEGEND_PANEL_BARS:    int = 1
LEGEND_COLUMN_WIDTH:  int = 96    # one bar + its tick labels (room for ~7-char "1.0e+05" mass ticks)
LEGEND_GUTTER:        int = 8     # left pad inside the panel
LEGEND_PANEL_WIDTH:   int = LEGEND_GUTTER + LEGEND_PANEL_BARS * LEGEND_COLUMN_WIDTH

LEGEND_BAR_X:          int = 8    # gradient left edge within a column
LEGEND_BAR_WIDTH:      int = 22
LEGEND_BAR_PAD_TOP:    int = 26   # room above the bar for the title
LEGEND_BAR_PAD_BOTTOM: int = 14
LEGEND_TITLE_Y:        int = 6
LEGEND_TICKS:          int = 5    # numeric labels along each bar (top, bottom, and 3 between)
LEGEND_TICK_LEN:       int = 5    # tick mark length in px
LEGEND_LABEL_GAP:      int = 3    # gap between tick mark and its text
LEGEND_FONT_SIZE:      int = 18
LEGEND_TITLE_FONT_SIZE: int = 19

LEGEND_BG:     tuple[int, int, int] = (0, 0, 0)
LEGEND_BORDER: tuple[int, int, int] = (90, 90, 90)
LEGEND_TEXT:   tuple[int, int, int] = (220, 220, 220)

# Kelvin -> Celsius offset, for temperature tick labels.
KELVIN_TO_CELSIUS: float = 273.15

# ---------------------------------------------------------------------------
# Colormaps
#
# A Colormap maps normalized values in [0, 1] (any shape) to (..., 3) uint8 RGB. Every
# field-fill view (all modes except FIRE, which paints fuel/flames directly) owns one and
# uses it for both its pixel surface and its colorbar legend, so the bar can never disagree
# with the field. Each builder does its own normalization (fixed window, min-max, or anchored
# at zero) and hands the result to its colormap.
# ---------------------------------------------------------------------------

def heat_colormap(normalized: np.ndarray) -> np.ndarray:
    """Temperature ramp: near-black at 0, saturated red at 1 (TEMPERATURE / FUEL_TEMPERATURE)."""
    n = np.clip(normalized, 0.0, 1.0)
    return np.stack([n * 255, (1.0 - n) * 15, (1.0 - n) * 31], axis=-1).astype(np.uint8)


def oxygen_colormap(normalized: np.ndarray) -> np.ndarray:
    """Oxygen ramp (grayscale): white where O2 is scarce (0), black where it is plentiful (1)."""
    n = np.clip(normalized, 0.0, 1.0)
    v = (1.0 - n) * MAX_CHANNEL_VALUE
    return np.stack([v, v, v], axis=-1).astype(np.uint8)


def wind_speed_colormap(normalized: np.ndarray) -> np.ndarray:
    """Wind-speed ramp (grayscale): black at rest (0), white at the fastest cell (1)."""
    n = np.clip(normalized, 0.0, 1.0)
    v = n * MAX_CHANNEL_VALUE
    return np.stack([v, v, v], axis=-1).astype(np.uint8)


def pressure_colormap(normalized: np.ndarray) -> np.ndarray:
    """Pressure/mass ramp: black at 0 up to saturated blue at 1."""
    n = np.clip(normalized, 0.0, 1.0)
    zeros = np.zeros_like(n)
    return np.stack([zeros, zeros, n * MAX_CHANNEL_VALUE], axis=-1).astype(np.uint8)


def radiant_heat_colormap(normalized: np.ndarray) -> np.ndarray:
    """Radiant-heat ramp: black at 0 up to magenta at 1."""
    n = np.clip(normalized, 0.0, 1.0)
    return np.stack([n * MAX_CHANNEL_VALUE, n * 80, n * MAX_CHANNEL_VALUE], axis=-1).astype(np.uint8)


def terrain_colormap(normalized: np.ndarray) -> np.ndarray:
    """Elevation relief: dark green valleys (0) up to white peaks (1)."""
    n = np.clip(normalized, 0.0, 1.0)
    return np.stack(
        [n * MAX_CHANNEL_VALUE, 60.0 + n * (MAX_CHANNEL_VALUE - 60.0), n * MAX_CHANNEL_VALUE], axis=-1
    ).astype(np.uint8)


def sunlight_colormap(normalized: np.ndarray) -> np.ndarray:
    """Solar-exposure ramp: dark/cold in shadow (0), warming through orange to bright sunlit
    white at full exposure (1). R rises first, then G, then B, so shade -> orange -> yellow -> white."""
    n = np.clip(normalized, 0.0, 1.0)
    r = np.clip(n * 2.0, 0.0, 1.0)
    b = np.clip(n * 2.0 - 1.0, 0.0, 1.0)
    return np.stack([r * MAX_CHANNEL_VALUE, n * MAX_CHANNEL_VALUE, b * MAX_CHANNEL_VALUE], axis=-1).astype(np.uint8)


def column_height_colormap(normalized: np.ndarray) -> np.ndarray:
    """Column-top height ramp: dark (short columns, 0) up to bright cyan (tall columns, 1)."""
    n = np.clip(normalized, 0.0, 1.0)
    return np.stack([n * 40, n * MAX_CHANNEL_VALUE, n * MAX_CHANNEL_VALUE], axis=-1).astype(np.uint8)


def wind_temp_colormap(normalized: np.ndarray) -> np.ndarray:
    """Wind-arrow color ramp by air temperature: cool blue (0) through to warm red (1)."""
    n = np.clip(normalized, 0.0, 1.0)
    return np.stack([n * MAX_CHANNEL_VALUE, (1.0 - n) * 127, (1.0 - n) * MAX_CHANNEL_VALUE], axis=-1).astype(np.uint8)


def fire_overlay_colormap(normalized: np.ndarray) -> np.ndarray:
    """Burning-cell ramp by fuel temperature: red at the ignition floor (0) warming to
    yellow at the top of the span (1). Shared by the fire overlay and its colorbar."""
    n = np.clip(normalized, 0.0, 1.0)
    full = np.full_like(n, MAX_CHANNEL_VALUE)
    return np.stack([full, n * MAX_CHANNEL_VALUE, np.zeros_like(n)], axis=-1).astype(np.uint8)


# ---------------------------------------------------------------------------
# Surface builders
# ---------------------------------------------------------------------------

def _render_field(normalized: np.ndarray, scale: int, cmap: Colormap) -> pygame.Surface:
    """Map a pre-normalized (H, W) field in [0, 1] through a colormap to a scaled pygame surface.

    The single place field-fill views turn normalized values into pixels, so every mode shares
    one normalize -> colormap -> blit path (and thus the exact colors its colorbar draws).
    """
    rows, cols = normalized.shape
    rgb = cmap(np.clip(normalized, 0.0, 1.0))             # (rows, cols, 3) uint8
    rgb_scaled = np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)
    surface = pygame.Surface((cols * scale, rows * scale))
    # surfarray expects (width, height, 3); numpy is (height, width, 3)
    pygame.surfarray.blit_array(surface, rgb_scaled.transpose(1, 0, 2))
    return surface


def build_color_surface(
    field: np.ndarray, scale: int, lo: float, hi: float, cmap: Colormap = heat_colormap,
) -> pygame.Surface:
    """Map a temperature field [K] linearly over the window [lo, hi] onto a color ramp.

    Values at/below lo render dark; at/above hi saturate to the hot color. The ramp is the
    colormap cmap (default the heat ramp) — the same callable its colorbar legend uses.
    """
    span = hi - lo
    normalized = (field - lo) / span if span > 0.0 else np.zeros_like(field)
    return _render_field(normalized, scale, cmap)


def wind_temp_window(
    x_wind_vel: np.ndarray,
    y_wind_vel: np.ndarray,
    air_temperatures: np.ndarray,
    reference_magnitude: float,
) -> tuple[float, float]:
    """Air-temperature window the wind arrows are colored over: the min/max temperature across
    the cells with a visible arrow (wind >= 10% of the reference magnitude), falling back to the
    whole field when none are visible. The wind surface and its colorbar both read this, so the
    arrows and the bar share one window."""
    if reference_magnitude == 0.0:
        return float(air_temperatures.min()), float(air_temperatures.max())
    magnitudes = np.sqrt(x_wind_vel ** 2 + y_wind_vel ** 2)
    visible_temps = air_temperatures[magnitudes >= 0.1 * reference_magnitude]
    if visible_temps.size == 0:
        return float(air_temperatures.min()), float(air_temperatures.max())
    return float(visible_temps.min()), float(visible_temps.max())


def wind_arrow_color(
    air_temperature: float, temp_min: float, temp_max: float,
) -> tuple[int, int, int]:
    """Color for a wind arrow: its cell's air temperature mapped over [temp_min, temp_max] (the
    window from wind_temp_window) onto the wind-temp ramp. Shared by the field and the cursor probe
    so both tint arrows identically."""
    temp_range = temp_max - temp_min
    heat = (air_temperature - temp_min) / temp_range if temp_range > 0.0 else 0.0
    return tuple(int(c) for c in wind_temp_colormap(np.asarray(heat)))


def draw_wind_arrow(
    surface: pygame.Surface, cx: float, cy: float, vx: float, vy: float,
    magnitude: float, reference_magnitude: float, color: tuple[int, int, int],
    from_base: bool = False,
) -> None:
    """Draw one wind arrow scaled by the cell's wind magnitude (against reference_magnitude, the
    live peak) with a two-barb head. By default it is centered on (cx, cy); with from_base it
    starts at (cx, cy) and points outward (so a probe sits its base on the measured cell). Shared
    by the wind-arrow field and the cursor probe. No-op for a zero/short wind or an unset reference."""
    if magnitude <= 0.0 or reference_magnitude <= 0.0:
        return
    t = min(reference_magnitude / _WIND_DISPLAY_THRESHOLD, 1.0)
    relative = min(magnitude / reference_magnitude, 1.0)
    absolute = min(magnitude / _WIND_DISPLAY_THRESHOLD, 1.0)
    half_len = (t * relative + (1.0 - t) * absolute) * (WIND_ARROW_LENGTH / 2.0)
    if half_len < 1.0:
        return
    ux, uy = vx / magnitude, vy / magnitude
    if from_base:
        start = (int(cx), int(cy))
        tip = (int(cx + ux * 2.0 * half_len), int(cy + uy * 2.0 * half_len))
    else:
        start = (int(cx - ux * half_len), int(cy - uy * half_len))
        tip = (int(cx + ux * half_len), int(cy + uy * half_len))
    pygame.draw.line(surface, color, start, tip, WIND_ARROW_LINE_WIDTH)

    angle = math.atan2(uy, ux)
    for sign in (+1, -1):
        barb = (
            int(tip[0] - WIND_ARROW_HEAD_SIZE * math.cos(angle + sign * WIND_ARROW_HEAD_ANGLE)),
            int(tip[1] - WIND_ARROW_HEAD_SIZE * math.sin(angle + sign * WIND_ARROW_HEAD_ANGLE)),
        )
        pygame.draw.line(surface, color, tip, barb, WIND_ARROW_LINE_WIDTH)


def build_wind_surface(
    x_wind_vel: np.ndarray,
    y_wind_vel: np.ndarray,
    air_temperatures: np.ndarray,
    scale: int,
    reference_magnitude: float,
) -> pygame.Surface:
    rows, cols = x_wind_vel.shape
    surface = pygame.Surface((cols * scale, rows * scale))
    surface.set_colorkey((0, 0, 0))

    if reference_magnitude == 0.0:
        return surface

    magnitudes = np.sqrt(x_wind_vel ** 2 + y_wind_vel ** 2)
    grid_offset = WIND_ARROW_STRIDE // 2

    temp_min, temp_max = wind_temp_window(x_wind_vel, y_wind_vel, air_temperatures, reference_magnitude)

    for row in range(grid_offset, rows, WIND_ARROW_STRIDE):
        for col in range(grid_offset, cols, WIND_ARROW_STRIDE):
            magnitude = float(magnitudes[row, col])
            if magnitude < 0.1 * reference_magnitude:   # declutter: hide near-calm cells
                continue
            color = wind_arrow_color(float(air_temperatures[row, col]), temp_min, temp_max)
            draw_wind_arrow(
                surface, (col + 0.5) * scale, (row + 0.5) * scale,
                float(x_wind_vel[row, col]), float(y_wind_vel[row, col]),
                magnitude, reference_magnitude, color,
            )

    return surface


def build_oxygen_surface(oxygen: np.ndarray, scale: int) -> pygame.Surface:
    # O2 is a partial density [kg/m^3]; map the fixed window [floor, ceiling] onto the
    # grayscale ramp, so it reads black = most oxygen, white = least (see oxygen_colormap)
    # with O2 at/below the floor rendering as fully scarce.
    span = DISPLAY_OXYGEN_CEILING - DISPLAY_OXYGEN_FLOOR
    normalized = (oxygen - DISPLAY_OXYGEN_FLOOR) / span
    return _render_field(normalized, scale, oxygen_colormap)


def build_wind_speed_surface(
    x_wind_vel: np.ndarray, y_wind_vel: np.ndarray, scale: int, display_max: float | None = None
) -> pygame.Surface:
    # Wind speed [m/s] as a grayscale field: each cell's brightness tracks its wind magnitude,
    # anchored at zero (black = calm) and normalised by the current max (white = fastest cell).
    magnitude = np.sqrt(x_wind_vel ** 2 + y_wind_vel ** 2)
    v_max = float(magnitude.max()) if display_max is None else display_max
    normalized = magnitude / v_max if v_max > 0.0 else np.zeros_like(magnitude)
    return _render_field(normalized, scale, wind_speed_colormap)


def build_pressure_surface(
    pressure: np.ndarray, scale: int, display_max: float | None = None
) -> pygame.Surface:
    # Anchored at zero, normalised by the current max (like the temperature view): a
    # uniform field renders uniformly bright, and a gradient's spread visibly shrinks as
    # it equalises — unlike a min-max scale, which re-stretches the residual every frame.
    p_max = float(pressure.max()) if display_max is None else display_max
    normalized = pressure / p_max if p_max > 0.0 else np.zeros_like(pressure)
    return _render_field(normalized, scale, pressure_colormap)


def build_radiant_heat_surface(
    radiant_flux: np.ndarray, scale: int, upper_bound: float | None = None
) -> pygame.Surface:
    # EMA-smoothed upper bound when provided; otherwise fall back to the frame max.
    bound = upper_bound if upper_bound is not None else float(radiant_flux.max())
    normalized = radiant_flux / bound if bound > 0.0 else np.zeros_like(radiant_flux)
    return _render_field(normalized, scale, radiant_heat_colormap)


def build_terrain_surface(
    terrain: np.ndarray, scale: int, display_max: float | None = None
) -> pygame.Surface:
    # Terrain is elevation in metres; normalise against the current max for the relief.
    t_max = float(terrain.max()) if display_max is None else display_max
    normalized = terrain / t_max if t_max > 0.0 else np.zeros_like(terrain)
    return _render_field(normalized, scale, terrain_colormap)


def build_sunlight_surface(sunlight: np.ndarray, scale: int) -> pygame.Surface:
    # Solar exposure is already a dimensionless [0, 1] fraction, so it maps straight onto the
    # ramp over a fixed 0..1 window (no data-driven ceiling): shaded cells dark, sunlit cells bright.
    return _render_field(sunlight, scale, sunlight_colormap)


def build_biome_surface(labels: np.ndarray, scale: int) -> pygame.Surface:
    """Paint the categorical biome map: each cell's integer biome label (0 woodland, 1 grassland,
    2 shrubland) indexes the BIOME_COLORS palette, giving a flat per-biome color (no ramp)."""
    rows, cols = labels.shape
    rgb = BIOME_COLORS[np.clip(labels, 0, len(BIOME_COLORS) - 1)]   # (rows, cols, 3) uint8
    rgb_scaled = np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)
    surface = pygame.Surface((cols * scale, rows * scale))
    pygame.surfarray.blit_array(surface, rgb_scaled.transpose(1, 0, 2))
    return surface


def build_column_height_surface(
    height: np.ndarray, scale: int, display_max: float | None = None
) -> pygame.Surface:
    # Relief of the air-column top, anchored at zero and normalised by the current max
    # (like the temperature view). Column height is always non-negative.
    h_max = float(height.max()) if display_max is None else display_max
    normalized = height / h_max if h_max > 0.0 else np.zeros_like(height)
    return _render_field(normalized, scale, column_height_colormap)


def build_fire_surface(
    fuel_temperatures: np.ndarray,       # (N, H, W)
    fuel: np.ndarray,                    # (N, H, W)
    oxygen: np.ndarray,                  # (H, W)
    scale: int,
    ignition_thresholds: np.ndarray,     # (N,)
    fuel_burnt_threshold: float,
    fuel_type_names: list[str],          # (N,) names, aligned with fuel/ignition axis 0
    show_fire_overlay: bool = True,
    initial_canopy_fuel: np.ndarray | None = None,  # (H, W) each cell's original canopy load (for bole occlusion)
) -> pygame.Surface:
    rows, cols = fuel_temperatures.shape[1], fuel_temperatures.shape[2]
    rgb = np.zeros((rows, cols, 3), dtype=np.uint8)

    # Visual "burning": hot enough to pyrolyse, with fuel and oxygen present. Combustion is a
    # smooth Arrhenius rate (no extinction threshold), so this is a display/diagnostic cutoff only.
    ign = ignition_thresholds[:, np.newaxis, np.newaxis]
    burning_per_type = (
        (fuel_temperatures >= ign) &
        (fuel > fuel_burnt_threshold) &
        (oxygen[np.newaxis] > 0.0)
    )
    any_burning = burning_per_type.any(axis=0)  # (H, W)

    # Each fuel type contributes its base color (trees green, grass yellow-green) weighted by
    # its mass, so a cell's color is the mass-weighted blend of the fuels present. Mass is
    # normalised per type by that type's own peak on the map, not by a shared total: grass tops
    # out at a far lower loading than trees, so a shared scale would render grass near-black.
    # Per-type normalisation lets each vegetation type use its full brightness range (and keeps
    # the image scale-invariant: uniformly scaling one type's fuel leaves its colors unchanged).
    type_colors = np.array(
        [VEGETATION_COLORS.get(name, DEFAULT_VEGETATION_COLOR) for name in fuel_type_names],
        dtype=np.float32,
    )  # (N, 3)
    type_max = fuel.max(axis=(1, 2), keepdims=True)            # (N, 1, 1) per-type peak
    normalized = np.divide(fuel, type_max, out=np.zeros_like(fuel), where=type_max > 0.0)

    # The bole hides behind the canopy: only let its brown contribute once that cell's canopy has
    # burned down to below TREE_BOLE_CANOPY_FRACTION of the canopy load it *originally* carried
    # (per-cell, not a map-wide peak). While the canopy is still fuller than that it occludes the
    # bole, so zero the bole's contribution there. Needs the map's original canopy load; without
    # it the bole just blends in normally.
    bole_idx = [i for i, n in enumerate(fuel_type_names) if n == "tree_bole"]
    canopy_idx = [i for i, n in enumerate(fuel_type_names) if n == "tree_canopy"]
    if bole_idx and canopy_idx and initial_canopy_fuel is not None:
        canopy_now = fuel[canopy_idx].sum(axis=0)                    # (H, W) current canopy load
        # Multiplicative form (vs. dividing) avoids a divide-by-zero where no canopy ever grew;
        # there initial_canopy_fuel == 0 marks the cell occluded, but the bole load is 0 there
        # too, so zeroing it changes nothing.
        occluded = canopy_now >= TREE_BOLE_CANOPY_FRACTION * initial_canopy_fuel
        for b in bole_idx:
            normalized[b][occluded] = 0.0   # normalized[b] is a view, so this writes back

    color_accum = np.tensordot(normalized, type_colors, axes=([0], [0]))  # (H, W, 3) in [0, ~]
    rgb[:] = (np.clip(color_accum, 0.0, 1.0) * MAX_CHANNEL_VALUE).astype(np.uint8)

    if show_fire_overlay:
        # Color gradient based on the hottest fuel type; a fixed Kelvin span above the
        # minimum ignition temperature spans the ramp.
        max_fuel_temp = fuel_temperatures.max(axis=0)
        min_ign = float(ignition_thresholds.min())
        t = (max_fuel_temp - min_ign) / FIRE_COLOR_TEMP_SPAN_K
        rgb[any_burning] = fire_overlay_colormap(t[any_burning])

    rgb_scaled = np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)
    surface = pygame.Surface((cols * scale, rows * scale))
    pygame.surfarray.blit_array(surface, rgb_scaled.transpose(1, 0, 2))
    return surface


# ---------------------------------------------------------------------------
# Colorbar legend
# ---------------------------------------------------------------------------

@dataclass
class ColorbarSpec:
    """One scale bar in the legend: a colormap, the value window it spans, a title and a tick
    formatter. lo/hi are in the field's native units (the same window the surface normalizes
    over), so the bar and the field share one colormap; tick_label turns a native value into
    its display string (e.g. Kelvin -> "57°C"). Any view mode supplies a list of these."""
    title:      str
    cmap:       Colormap
    lo:         float
    hi:         float
    tick_label: Callable[[float], str]


def build_colorbar_column(
    spec: ColorbarSpec, height: int, bar_font: pygame.font.Font, title_font: pygame.font.Font,
) -> pygame.Surface:
    """Render one legend column: a title, a vertical gradient bar (lo at the bottom, hi at the
    top) and LEGEND_TICKS numeric labels along its right edge."""
    col = pygame.Surface((LEGEND_COLUMN_WIDTH, height))
    col.fill(LEGEND_BG)

    bar_x, bar_w = LEGEND_BAR_X, LEGEND_BAR_WIDTH
    bar_top = LEGEND_BAR_PAD_TOP
    bar_h = max(1, height - LEGEND_BAR_PAD_TOP - LEGEND_BAR_PAD_BOTTOM)

    # Vertical gradient: row 0 (top) = hi (norm 1), last row (bottom) = lo (norm 0).
    norm_col = np.linspace(1.0, 0.0, bar_h)            # (bar_h,)
    rgb = spec.cmap(norm_col)                          # (bar_h, 3) uint8
    grad = np.repeat(rgb[:, None, :], bar_w, axis=1)   # (bar_h, bar_w, 3)
    bar_surf = pygame.Surface((bar_w, bar_h))
    pygame.surfarray.blit_array(bar_surf, grad.transpose(1, 0, 2))
    col.blit(bar_surf, (bar_x, bar_top))
    pygame.draw.rect(col, LEGEND_BORDER, (bar_x, bar_top, bar_w, bar_h), 1)

    col.blit(title_font.render(spec.title, True, LEGEND_TEXT), (bar_x, LEGEND_TITLE_Y))

    for i in range(LEGEND_TICKS):
        frac = i / (LEGEND_TICKS - 1)                  # 0 at the bottom, 1 at the top
        y = int(bar_top + (1.0 - frac) * (bar_h - 1))
        value = spec.lo + frac * (spec.hi - spec.lo)
        pygame.draw.line(col, LEGEND_TEXT, (bar_x + bar_w, y), (bar_x + bar_w + LEGEND_TICK_LEN, y))
        label = bar_font.render(spec.tick_label(value), True, LEGEND_TEXT)
        col.blit(label, (bar_x + bar_w + LEGEND_TICK_LEN + LEGEND_LABEL_GAP, y - label.get_height() // 2))

    return col


def build_legend_panel(
    specs: list[ColorbarSpec], height: int,
    bar_font: pygame.font.Font, title_font: pygame.font.Font,
) -> pygame.Surface:
    """The full right-side legend panel: each spec drawn as an adjacent column, left to right."""
    panel = pygame.Surface((LEGEND_PANEL_WIDTH, height))
    panel.fill(LEGEND_BG)
    x = LEGEND_GUTTER
    for spec in specs:
        panel.blit(build_colorbar_column(spec, height, bar_font, title_font), (x, 0))
        x += LEGEND_COLUMN_WIDTH
    return panel


def build_biome_legend_panel(
    height: int, bar_font: pygame.font.Font, title_font: pygame.font.Font,
) -> pygame.Surface:
    """The right-side legend for the categorical biome view: a title over a color swatch and its
    name for each biome (a gradient bar makes no sense for discrete classes, so this replaces it).
    The name sits under its swatch, not beside it, so the long biome names clear the panel width."""
    panel = pygame.Surface((LEGEND_PANEL_WIDTH, height))
    panel.fill(LEGEND_BG)
    x = LEGEND_BAR_X + LEGEND_GUTTER
    panel.blit(title_font.render("Biome", True, LEGEND_TEXT), (x, LEGEND_TITLE_Y))
    sw_w, sw_h = 2 * LEGEND_BAR_WIDTH, LEGEND_BAR_WIDTH   # a short, wide swatch key
    y = LEGEND_BAR_PAD_TOP
    for name, color in zip(BIOME_NAMES, BIOME_COLORS):
        pygame.draw.rect(panel, tuple(int(c) for c in color), (x, y, sw_w, sw_h))
        pygame.draw.rect(panel, LEGEND_BORDER, (x, y, sw_w, sw_h), 1)
        label = bar_font.render(name, True, LEGEND_TEXT)
        panel.blit(label, (x, y + sw_h + LEGEND_LABEL_GAP))
        y += sw_h + label.get_height() + LEGEND_TICK_LEN * 2
    return panel
