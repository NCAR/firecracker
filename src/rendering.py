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
RED_CHANNEL: int = 0
GREEN_CHANNEL: int = 1

# ---------------------------------------------------------------------------
# Wind arrow visualization parameters
# ---------------------------------------------------------------------------

_WIND_DISPLAY_THRESHOLD:       float = 1e-4

# Temperature views map a Kelvin window onto the color ramp. The air-temperature view
# uses a fixed ambient fire-weather window (fire pushes air above the ceiling and simply
# saturates to the hot color); the fuel-temperature view shares the floor but tracks its
# ceiling to the (EMA-smoothed) peak so flame structure stays visible.
DISPLAY_TEMP_FLOOR_K:    float = 290.0
DISPLAY_AIR_TEMP_CEIL_K: float = 330.0
DISPLAY_MIN_TEMP_SPAN_K: float = 50.0
# Kelvin above ignition that spans the fire-overlay color ramp (ignition -> +span). Sized to
# the SI flame band: ignition ~550-600 K up through a sustained ~1540 K front to ~2400 K peaks,
# so the ramp resolves flame structure instead of saturating just above ignition.
FIRE_COLOR_TEMP_SPAN_K:  float = 1600.0

# EMA weight for the upper bound of the fuel-temperature and radiant-heat color
# scales. Smaller = steadier (slower to track the peak); larger = more responsive.
DISPLAY_SCALE_EMA_ALPHA: float = 0.1
# Fraction of the frame max the radiant-heat EMA tracks toward, so cells at/above this
# fraction of the smoothed peak saturate to the top color.
DISPLAY_SCALE_MAX_FRACTION: float = 0.5

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
LEGEND_PANEL_BARS:    int = 2
LEGEND_COLUMN_WIDTH:  int = 84    # one bar + its tick labels
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


def column_height_colormap(normalized: np.ndarray) -> np.ndarray:
    """Column-top height ramp: dark (short columns, 0) up to bright cyan (tall columns, 1)."""
    n = np.clip(normalized, 0.0, 1.0)
    return np.stack([n * 40, n * MAX_CHANNEL_VALUE, n * MAX_CHANNEL_VALUE], axis=-1).astype(np.uint8)


def wind_temp_colormap(normalized: np.ndarray) -> np.ndarray:
    """Wind-arrow color ramp by air temperature: cool blue (0) through to warm red (1)."""
    n = np.clip(normalized, 0.0, 1.0)
    return np.stack([n * MAX_CHANNEL_VALUE, (1.0 - n) * 127, (1.0 - n) * MAX_CHANNEL_VALUE], axis=-1).astype(np.uint8)


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

    t = min(reference_magnitude / _WIND_DISPLAY_THRESHOLD, 1.0)
    magnitudes = np.sqrt(x_wind_vel ** 2 + y_wind_vel ** 2)
    max_half_len = WIND_ARROW_LENGTH / 2.0
    grid_offset = WIND_ARROW_STRIDE // 2

    visible_temps = air_temperatures[magnitudes >= 0.1 * reference_magnitude]
    temp_min = float(visible_temps.min()) if visible_temps.size > 0 else float(air_temperatures.min())
    temp_max = float(visible_temps.max()) if visible_temps.size > 0 else float(air_temperatures.max())
    temp_range = temp_max - temp_min

    for row in range(grid_offset, rows, WIND_ARROW_STRIDE):
        for col in range(grid_offset, cols, WIND_ARROW_STRIDE):
            vx = float(x_wind_vel[row, col])
            vy = float(y_wind_vel[row, col])
            magnitude = float(magnitudes[row, col])
            if magnitude < 0.1 * reference_magnitude:
                continue

            relative = min(magnitude / reference_magnitude, 1.0)
            absolute = min(magnitude / _WIND_DISPLAY_THRESHOLD, 1.0)
            half_len = (t * relative + (1.0 - t) * absolute) * max_half_len
            if half_len < 1.0:
                continue
            ux, uy = vx / magnitude, vy / magnitude
            cx = (col + 0.5) * scale
            cy = (row + 0.5) * scale

            heat = (float(air_temperatures[row, col]) - temp_min) / temp_range if temp_range > 0.0 else 0.0
            color = tuple(int(c) for c in wind_temp_colormap(np.asarray(heat)))

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

    return surface


def build_oxygen_surface(oxygen: np.ndarray, scale: int) -> pygame.Surface:
    # O2 is a partial density [kg/m^3]; normalise against the current max so the
    # grayscale ramp reads black = most oxygen, white = least (see oxygen_colormap).
    o_max = float(oxygen.max())
    normalized = oxygen / o_max if o_max > 0.0 else np.zeros_like(oxygen)
    return _render_field(normalized, scale, oxygen_colormap)


def build_pressure_surface(pressure: np.ndarray, scale: int) -> pygame.Surface:
    # Anchored at zero, normalised by the current max (like the temperature view): a
    # uniform field renders uniformly bright, and a gradient's spread visibly shrinks as
    # it equalises — unlike a min-max scale, which re-stretches the residual every frame.
    p_max = float(pressure.max())
    normalized = pressure / p_max if p_max > 0.0 else np.zeros_like(pressure)
    return _render_field(normalized, scale, pressure_colormap)


def build_radiant_heat_surface(
    radiant_flux: np.ndarray, scale: int, upper_bound: float | None = None
) -> pygame.Surface:
    # EMA-smoothed upper bound when provided; otherwise fall back to the frame max.
    bound = upper_bound if upper_bound is not None else float(radiant_flux.max())
    normalized = radiant_flux / bound if bound > 0.0 else np.zeros_like(radiant_flux)
    return _render_field(normalized, scale, radiant_heat_colormap)


def build_terrain_surface(terrain: np.ndarray, scale: int) -> pygame.Surface:
    # Terrain is elevation in metres; normalise against the current max for the relief.
    t_max = float(terrain.max())
    normalized = terrain / t_max if t_max > 0.0 else np.zeros_like(terrain)
    return _render_field(normalized, scale, terrain_colormap)


def build_column_height_surface(height: np.ndarray, scale: int) -> pygame.Surface:
    # Relief of the air-column top, anchored at zero and normalised by the current max
    # (like the temperature view). Column height is always non-negative.
    h_max = float(height.max())
    normalized = height / h_max if h_max > 0.0 else np.zeros_like(height)
    return _render_field(normalized, scale, column_height_colormap)


def build_fire_surface(
    fuel_temperatures: np.ndarray,       # (N, H, W)
    fuel: np.ndarray,                    # (N, H, W)
    oxygen: np.ndarray,                  # (H, W)
    scale: int,
    ignition_thresholds: np.ndarray,     # (N,)
    fuel_burnt_threshold: float,
    show_fire_overlay: bool = True,
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

    # Green brightness tracks total fuel, normalised so the cell with the most fuel
    # on the map is full brightness. This keeps the image scale-invariant: uniformly
    # scaling all fuel leaves the ratios (and thus the rendered intensities) unchanged.
    total_fuel = fuel.sum(axis=0)
    max_fuel = float(total_fuel.max())
    normalized_fuel = total_fuel / max_fuel if max_fuel > 0.0 else total_fuel
    rgb[:, :, GREEN_CHANNEL] = (normalized_fuel * MAX_CHANNEL_VALUE).astype(np.uint8)

    if show_fire_overlay:
        # Color gradient based on the hottest fuel type; a fixed Kelvin span above the
        # minimum ignition temperature spans the ramp.
        max_fuel_temp = fuel_temperatures.max(axis=0)
        min_ign = float(ignition_thresholds.min())
        t = np.clip((max_fuel_temp - min_ign) / FIRE_COLOR_TEMP_SPAN_K, 0.0, 1.0)
        rgb[any_burning, RED_CHANNEL]   = MAX_CHANNEL_VALUE
        rgb[any_burning, GREEN_CHANNEL] = (t[any_burning] * MAX_CHANNEL_VALUE).astype(np.uint8)

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
