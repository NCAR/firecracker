# Firecracker

Firecracker is a physics-based wildfire simulator with a machine-learning "world model"
built on top of it. The simulator models how a grassfire (or a forest fire) starts, spreads,
and burns out across a patch of terrain, tracking the wind, the air and fuel temperatures,
the oxygen, and the vegetation, all in real physical units. On top of that it ships a full
neural-network pipeline that learns to *predict* how the fire will evolve, so a forecast can
run faster than the physics.

The project has two halves:

1. **The simulator** — a self-contained fire-and-atmosphere model written in PyTorch, plus an
   interactive [Gymnasium](https://gymnasium.farama.org/) environment you can watch and click
   on in a window.
2. **The world model** — tools to record millions of simulation frames, train a neural network
   to compress and forecast them, and compare the network's predictions against the real physics
   side by side.

Everything runs the same on a CPU or an NVIDIA GPU; the GPU is only ever a speed-up, never a
requirement.

---

## What it looks like

You get a live window showing the fire spreading over a randomly generated landscape. A menu of
**view modes** lets you flip between what the simulation is tracking — the flames, the wind, the
air temperature, the terrain, the vegetation, the oxygen, and more. You can pause, step frame by
frame, right-click to start a new fire, and reset to a fresh map.

Under the hood every cell of the grid carries a full physical state: air temperature (in kelvin),
wind velocity (meters per second), oxygen density, terrain height, and a stack of separate fuel
layers (grass, shrub, tree canopy, and two tree-trunk segments), each with its own temperature and
mass. The fire ignites, spreads, and dies out according to real combustion, heat-transfer, and
airflow equations rather than hand-tuned game rules.

---

## Quick start

### Requirements

- **Python 3.11 or newer** (the config loader uses the standard-library `tomllib`).
- The packages in [`requirements.txt`](requirements.txt): `numpy`, `torch`, `pygame`,
  `gymnasium`, `pytest`, and `wandb` (the last is only needed for logging during training).

```bash
pip install -r requirements.txt
```

### Generate a map, then run it

Maps are pre-baked landscapes stored on disk. A handful ship in [`maps/`](maps/), but you can
make your own:

```bash
# Generate 8 fresh landscapes into the maps/ directory
python src/gen_maps.py --count 8 --out maps

# Launch the interactive viewer (picks a random map each reset)
python src/main.py
```

Right-click anywhere in the window to start a fire, then watch it spread. Press `Tab` to open the
view-mode menu.

### Common ways to run it

```bash
# Load a specific map
python src/main.py --map map_0003

# Run without a window as fast as possible (useful for timing / batch runs).
# Set [fire].spawn_fire = true in the config first, since there is no window to click.
python src/main.py --headless

# Watch one or more trained neural models reconstruct the world beside the real physics
python src/main.py --model models/pooled_huber_deep11222_l256_14ch_best_bf16.pt
```

### Interactive controls

| Input | Action |
|-------|--------|
| **Right-click** | Start a fire at that spot |
| **Left-click** | Select a coarse action-cell (two clicks: a row, then a column — see below) |
| **Tab** | Open / close the view-mode menu |
| **Letter keys** | Jump straight to a view (see the menu for each mode's key) |
| **Number keys `1`–`9`** | In the Fire & Fuel / Fuel Temperature views, toggle individual fuel types on/off |
| **`0`** | In the Fire & Fuel view, toggle the fire overlay |
| **`Space`** | Pause / resume |
| **`.`** (period) | Advance one frame while paused |
| **`W`** | Toggle the wind-arrow overlay |
| **`L`** | Show / hide the fuel-type panel |
| **`N`** | Toggle light / dark chrome for the legend and menus (dark by default) |
| **`R`** | Reset to a new map |
| **`Esc`** | Quit |

The **view modes** are: Air Temperature, Wind, Wind Speed, Fire & Fuel, Fuel Temperature, Oxygen,
Mass / Pressure, Column Height, Radiant Heat, Terrain, Sunlight, and Biome.

---

## How the simulation works (in brief)

The world is a square grid (256×256 by default). Each tick advances the physics by a fixed amount
of simulated time (1 second by default). Every tick runs the same fixed sequence of physical
stages:

1. **Wind** — a shallow-atmosphere airflow model pushes wind over the terrain. Air is squeezed as
   it flows over ridges (speeding up) and channeled through valleys, and it carries heat with it.
2. **Sunlight & radiation** — the sun warms the ground and fuel; hot surfaces radiate heat back
   out. Sun-facing slopes settle warmer, shaded valleys cooler.
3. **Fuel ↔ air heat exchange** — fuel and the air above it trade heat by conduction.
4. **Oxygen transport** — oxygen drifts with the wind, mixes, and is replenished from aloft.
5. **Combustion** — where fuel is hot enough and oxygen is present, it burns. The burn rate comes
   from real chemistry (an Arrhenius reaction rate) limited by how fast oxygen can reach the fuel
   surface. Burning fuel releases heat, some into the fuel itself (setting the flame temperature)
   and some into the air.
6. **Fire spread** — the hot plume above a flame is carried onto neighboring fuel to ignite it,
   skewed downwind and uphill (fires run faster with the wind and up slopes). Flames also radiate
   heat to their neighbors.
7. **Vertical air structure** — the near-surface air is split into stacked layers (surface, canopy,
   and above-canopy). Heat rises through them and is finally vented to the sky, which keeps a big
   fire from overheating without bound. A dedicated "ladder" lets a surface fire climb a tree trunk
   into the crown.

Fire needs three things — **heat, fuel, and oxygen** — and goes out if any one runs low. All of the
state is in true SI units, so temperatures, wind speeds, and fire behavior land in physically
realistic ranges (grass fronts spreading at a fraction of the wind speed, flames around 1300–2300 K,
and so on).

A much deeper walk-through lives in [`docs/simulation.md`](docs/simulation.md), and the original
unit-conversion design record is in [`docs/units.md`](docs/units.md).

---

## The world model (machine learning)

The simulator doubles as a data source for training a neural network to forecast fire. The pipeline
has three stages:

1. **Record** — step many worlds in parallel and save their observation frames to disk as a
   compressed, sharded dataset (`src/rollout.py`).
2. **Compress** — train a convolutional **autoencoder** to squeeze each frame into a small latent
   vector and reconstruct it (`src/train_autoencoder.py`).
3. **Forecast** — train a **dynamics head** that steps that latent vector forward in time, so the
   network can roll the fire forward on its own without touching the physics
   (`src/train_world_model.py`, `src/train_world_model_rollout.py`).

You can then watch a trained model reconstruct or forecast the world beside the live simulation
(`python src/main.py --model ...`, or `tools/rollout_viewer.py`). Details, the training commands,
and the checkpoint catalog are in [`docs/world_model.md`](docs/world_model.md) and
[`models/README.md`](models/README.md).

---

## Configuration

All tunable parameters live in [`cfg/default.toml`](cfg/default.toml), which is heavily commented —
it is the single best reference for every knob (grid size, cell size, time step, wind strength,
combustion chemistry, fuel types, biome layout, rendering, and the ML rollout/training settings).
Anything not set there falls back to a default in the source. Point any entry point at a different
file with `--config path/to/your.toml`.

Fixed constants of nature (gravity, the gas constant, the Stefan–Boltzmann constant, the reference
atmosphere) live in [`src/physics_constants.py`](src/physics_constants.py) and are not meant to be
edited per run.

---

## Project layout

```
firecracker/
├── cfg/
│   └── default.toml          # all tunable parameters (heavily commented)
├── maps/                     # pre-baked landscapes (.npz), generated by src/gen_maps.py
├── models/                   # trained neural-network checkpoints (+ their own README)
├── fixtures/                 # small analytic test maps + a viewer config for them
├── docs/                     # in-depth documentation and design records
├── src/                      # all source code (see below)
├── tools/                    # standalone scripts: benchmarking, evaluation, visualization
├── tests/                    # pytest suite (physics + world-model tests)
└── requirements.txt
```

### Source files (`src/`)

**The simulator**

| File | What it does |
|------|--------------|
| `simulation.py` | The physics engine. One class, `Simulation`, holds no rendering code and runs every physical stage of a tick. This is the heart of the project. |
| `physics_constants.py` | Fixed constants of nature and the reference atmosphere; also the observation-normalization windows. |
| `firecracker_env.py` | The Gymnasium environment: wraps `Simulation`, owns the pygame window, view modes, controls, and the observation/action interface. |
| `config.py` | Loads the TOML config; computes the padded physics grid size. |
| `main.py` | Command-line entry point for the interactive / headless / model-comparison runners. |
| `rendering.py` | Stateless drawing primitives — color maps, color bars, wind arrows, legends — used by the environment to paint each view mode. |

**Map generation**

| File | What it does |
|------|--------------|
| `gen_maps.py` | Procedurally builds landscapes: Perlin-noise terrain, sunlight/shadow, temperature and oxygen profiles, vegetation biomes and fuel, then "spins up" the wind so a loaded map already has developed airflow. Runnable as a script. |
| `map_loader.py` | Reads, writes, validates, and selects map files. No dependency on the physics engine. |
| `perlin.py` | Vectorised Perlin-noise generator (runs batched on CPU or GPU) used by the map generator. |

**The world model (ML)**

| File | What it does |
|------|--------------|
| `rollout.py` | Steps many worlds in parallel and records their frames into a sharded on-disk dataset (`FireDataset`). |
| `traj_loader.py` | Recovers per-episode frame sequences from a flat dataset, for next-frame / rollout training. |
| `ram_loader.py` | A fast, thread-based data loader that serves batches straight from the memory-mapped shards. |
| `strided_autoencoder.py` | The convolutional autoencoder (`ConvAutoencoder`) that compresses an observation to a latent vector and back. |
| `dynamics.py` | The latent transition head (predicts the next latent) and an optional reward head. |
| `train_autoencoder.py` | Trains the autoencoder on recorded frames. |
| `train_world_model.py` | Trains the autoencoder + dynamics head together (a self-predictive / BYOL-style scheme). |
| `train_world_model_rollout.py` | An alternative "decode every rolled step" training scheme that forces the dynamics to actually propagate fire. |
| `losses.py`, `wm_losses.py` | Reconstruction and fire-weighted loss functions for the two training scripts. |
| `model_viewer.py` | Renders trained models' reconstructions beside the live physics (used by `main.py --model`). |

### Tools (`tools/`)

Standalone scripts, e.g. `bench.py` / `bench_batched.py` (throughput), `eval_full.py`
(evaluate a checkpoint over a dataset), `rollout_viewer.py` / `rollout_check.py` /
`wm_rollout_mae.py` (inspect a world model's forecast), `visualize.py` (render the test
scenarios to labeled image montages), `diagnose_fire.py` (probe specific fire behaviors),
`sweep.py` (train many replicate models in parallel), and dataset utilities
(`compute_stats.py`, `normalize_dataset.py`, `subsample_dataset.py`).

### Documentation (`docs/`)

- [`simulation.md`](docs/simulation.md) — a deep dive into the physics engine: state, the
  per-tick step sequence, and each subsystem.
- [`world_model.md`](docs/world_model.md) — the full machine-learning pipeline: observations,
  recording data, and training the autoencoder and dynamics head.
- [`interactive_viewer.md`](docs/interactive_viewer.md) — the viewer, every view mode, the
  controls, and the action interface.
- [`units.md`](docs/units.md) — the original SI-unit-conversion design record (the deepest
  reference for the physical derivations).
- [`vertical_air_levels_plan.md`](docs/vertical_air_levels_plan.md),
  [`wm_decode_rollout_plan.md`](docs/wm_decode_rollout_plan.md),
  [`wm_two_term_loss_plan.md`](docs/wm_two_term_loss_plan.md) — internal design/planning records.

---

## Running the tests

```bash
pytest
```

The suite (in [`tests/`](tests/)) covers the physics — conservation, equilibrium, combustion,
heat exchange, radiation, fire spread, venting, terrain channeling, numerical stability — and the
world-model data loaders and training paths.

---

## AI usage disclosure

All code in this project was written with [Claude Code](https://claude.ai/code). Claude was also
used to assist in writing this documentation.
```

---

## License

See [`LICENSE`](LICENSE).
