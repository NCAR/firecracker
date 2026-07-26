# The Interactive Viewer

The viewer is the window you get from `python src/main.py`. It runs the Gymnasium environment
([`src/firecracker_env.py`](../src/firecracker_env.py)) with its drawing done by stateless
primitives in [`src/rendering.py`](../src/rendering.py). This document covers the window layout,
every view mode, the controls, and the (partly experimental) action interface.

---

## Running it

```bash
python src/main.py                       # random map each reset, opens a window
python src/main.py --map map_0003        # a specific map
python src/main.py --config my.toml      # a different config
python src/main.py --maps-dir some/dir   # load maps from elsewhere
python src/main.py --steps 2000          # truncate each episode at 2000 ticks
python src/main.py --headless            # no window, run as fast as possible
python src/main.py --model a.pt [b.pt]   # show trained models beside the physics
```

`--headless` runs with no window and no real-time throttle, stepping as fast as the device allows
until it hits `max_steps` or you press Ctrl-C. There is no way to click, so set
`[fire].spawn_fire = true` in the config if you want a fire. It prints throughput and per-episode
statistics.

At startup the runner prints the full resolved configuration (device, grid, time step, wind,
combustion, every fuel type's constants, and so on), and it prints a block of world statistics at
the start and end of each episode.

---

## The window

The window is laid out as:

- a **top status bar** naming the current view mode (and, in the fuel views, which fuel types are
  selected);
- the **simulation** itself (a square, the observed grid scaled up to `window_size` pixels); and
- a **legend panel** down the right side with a color bar (and tick labels in real units) for the
  active view.

The physics grid is larger than what is drawn — the outer sponge ring is cropped off — so everything
you see is exactly the observed interior.

---

## View modes

Press `Tab` for a grouped menu (navigated with the arrow keys), or use the letter key for each mode
directly. The menu is organized into three groups (Atmosphere, Fire, Terrain) and doubles as a
keybind cheat-sheet.

| View | Key | Shows |
|------|-----|-------|
| **Air Temperature** | `T` | Surface air temperature. Auto-ranges: the color floor tracks the coldest cell and the ceiling the hottest, so a flame lifts the top of the scale without moving the bottom. |
| **Wind** | `D` | Wind direction as arrows, tinted by air temperature. This is the *fire wind* (prognostic wind × terrain-channeling gain), so the valley speed-ups are visible. |
| **Wind Speed** | `V` | Wind magnitude as a color field. |
| **Fire & Fuel** | `F` | The vegetation/fuel base, colored per fuel type, with the fire front as a toggleable overlay (key `0`). Number keys toggle which fuel types are shown; a cell reads as burning only where a *visible* fuel is alight, colored by the hottest visible fuel's temperature. |
| **Fuel Temperature** | `E` | A mass-weighted average of the selected fuel types' temperatures, over a fixed window so a flame's color reads the same every frame. |
| **Oxygen** | `O` | Near-surface oxygen density. |
| **Mass / Pressure** | `P` | Boundary-layer air mass (which doubles as surface pressure). |
| **Column Height** | `C` | Height of the top of the modeled air layer (terrain + layer thickness). |
| **Radiant Heat** | `H` | The radiant flux absorbed by fuel last tick. |
| **Terrain** | `G` | Elevation. |
| **Sunlight** | `S` | The static per-cell solar exposure baked into the map (0–100%). |
| **Biome** | `B` | The three vegetation biomes (woodland / grassland / shrubland) as flat colors, with a labeled swatch legend. |

The **Fire & Fuel** and **Fuel Temperature** views have per-fuel-type toggles: number keys `1`–`N`
turn individual fuel types on or off, a small panel (top-left, hide with `L`) lists them with
checkboxes, and the top bar names the current selection. Fire & Fuel blends the vegetation colors of
the selected types; Fuel Temperature averages their temperatures. In Fire & Fuel the `0` key toggles
the fire overlay on top of the fuel base — only cells where a shown fuel is burning light up, tinted
by the hottest shown fuel's temperature (hidden fuels are excluded from both the footprint and the
color).

A **wind-arrow overlay** (toggle with `W`) can be laid on top of any non-wind view. In the two wind
views, moving the mouse also draws a probe arrow at the hovered cell so you can read a specific
point.

---

## Controls

| Input | Action |
|-------|--------|
| **Right-click** | Ignite a fire patch at that cell |
| **Left-click** | Drive the two-step action selection (row, then column) |
| **Tab** | Toggle the view-mode menu |
| Letter keys | Switch view mode directly |
| `1`–`9` | Toggle a fuel type in the active Fire & Fuel / Fuel Temperature view |
| `0` | Toggle the fire overlay in the Fire & Fuel view |
| Arrow keys + Enter | Navigate the menu (Enter/Right selects or descends, Left goes back) |
| `Space` | Pause / resume |
| `.` | Step one frame (while paused) |
| `W` | Toggle the wind-arrow overlay |
| `L` | Show / hide the fuel-type panel |
| `R` | Reset to a new map (new episode) |
| `Esc` | Quit |

Starting a fire (right-click, or a random spawn at reset when `[fire].spawn_fire` is on) drops a
realistic **single-cell ember**: it heats the surface air and the *fine surface fuel already present*
(grass, shrub) at that cell just past ignition. It does **not** pile on extra fuel and does not light
the trunk or canopy directly — those catch only once the fire climbs the ladder. A cell with no fine
fuel simply will not light, which is realistic. A healthy simulation spreads one hot grass cell
downwind on its own.

---

## The action interface

The environment exposes a Gymnasium action space for a future fire-intervention agent. The map is
overlaid with a coarse **action grid** (32×32 cells by default, each spanning a block of simulation
cells). An action-cell is selected over **two steps** — first a row index, then a column index —
which is why left-clicking takes two clicks. A completed selection only "goes through" when the
chosen cell or one of its eight neighbors contains fire; over a fire-free neighborhood it is a
no-op. A valid selection flashes the square white.

Hovering previews what the next click will commit: a band highlights the hovered row (row phase), or
the locked row plus the hovered column (column phase), their intersection previewing the resulting
cell.

The **world effect of a selection is not wired in yet** — the chosen cell is only recorded (in the
step `info` dict) for a future reward/intervention scheme to read. The reward returned by `step` is
currently always `0`, and episodes end only by truncation at `max_steps`.

---

## Comparing against trained models

`python src/main.py --model <checkpoint> [...]` runs the same interactive world but adds one panel
per model showing that autoencoder's reconstruction of the current view, so you can eyeball how
faithfully each model reproduces the physics. Because a model only ever sees the observation, only
the observed channels can be reconstructed (air/fuel temperature, wind, fire); other views show a
"not in observation" placeholder while the physics panel still renders. Press `Tab` to flip the
model panels to a per-cell error heatmap. See [`world_model.md`](world_model.md) and
`src/model_viewer.py`.
