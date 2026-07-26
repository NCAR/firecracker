# The World Model (Machine Learning)

Firecracker's second half is a neural network that learns to **forecast** the fire: given the
current state of the world, predict how it will look a few ticks later — fast enough to outrun the
physics. This document walks through the whole pipeline, from recording data to training and
inspecting a model.

The physics simulator is the data source. Everything the network sees is the **observation** — the
normalized multi-channel image described at the end of [`simulation.md`](simulation.md#the-observation):
surface air temperature, per-fuel-type temperature and mass, the terrain slope, the wind, and a
binary "burning" mask. Note that some state (terrain *height*, biome labels, oxygen, pressure) is
**not** in the observation, so a model can only ever reconstruct the channels it is given.

The pipeline has three stages, which you can run separately:

```
   simulation  ──►  1. record  ──►  dataset on disk
                                          │
                                          ▼
                    2. autoencoder  (compress a frame to a small latent vector, and back)
                                          │
                                          ▼
                    3. dynamics head (step that latent vector forward in time)
```

---

## Stage 1 — Recording a dataset (`rollout.py`)

`BatchedRollout` steps many independent worlds in parallel through the exact same physics step the
interactive environment uses (the engine is batch-friendly, so `B` worlds run at once with no
special-casing). Consecutive frames barely differ, so only every `stride`-th frame is kept, which
keeps the stored samples decorrelated and the dataset small.

Two ways to use it:

- **Online** — fill an in-memory pool and draw random minibatches from it, for training in the same
  process.
- **On disk** — write a **sharded, memory-mapped dataset** (`build_dataset`), which is what the
  training scripts consume. Each shard is one `reset + collect`; the frames are stored frame-major,
  env-minor, so `traj_loader.py` can later recover each world's episode as a sequence of frames.

From the command line:

```bash
# 8 shards of igniting worlds into data/fire
python src/rollout.py --out data/fire --rounds 8 --spawn-fire
```

Useful flags: `--num-envs` (worlds in parallel), `--steps` (ticks per shard), `--stride` (keep every
Nth frame), `--buffer-dtype float16` (halve the on-disk size), `--seed`, and `--num-shards` /
`--shard-index` to split the work across processes or nodes. The `[rollout]` section of
[`cfg/default.toml`](../cfg/default.toml) documents the defaults and the memory arithmetic (the pool
can be many gigabytes, which is why it lives on the CPU by default).

Because consecutive frames are so redundant and a full float16 dataset can reach hundreds of
gigabytes, three helper tools manage datasets on disk: `tools/compute_stats.py` (per-channel
normalization stats), `tools/normalize_dataset.py` (bake a min-max `[0, 1]` scaling into the
shards), and `tools/subsample_dataset.py` (write a physically smaller copy).

### Loading the data

- **`rollout.FireDataset`** — lazily memory-maps the shards and serves flat samples; it unions
  multiple sub-directories, so a multi-process run's outputs combine into one dataset.
- **`traj_loader.TrajectoryDataset`** — recovers per-episode frame *windows* (runs of consecutive
  stored frames from one world) for next-frame / rollout training, plus `split_episodes` for a
  clean train/validation split that never leaks frames of the same episode across the split.
- **`ram_loader`** — a fast data loader that serves whole batches from the page-cache-resident
  shards using a single background thread, avoiding the crashes PyTorch's multi-process `DataLoader`
  hits under heavy node packing. It can pre-warm the shards into the OS page cache to avoid a slow
  first epoch.

---

## Stage 2 — The autoencoder (`strided_autoencoder.py`, `train_autoencoder.py`)

The autoencoder (`ConvAutoencoder`) learns to squeeze one observation frame down to a small **latent
vector** and reconstruct it. It is a symmetric convolutional network:

- The **encoder** is a stack of down-blocks. Each block does its feature learning with a residual
  pair of 3×3 convolutions and then halves the image with a parameter-free 2×2 average pool, growing
  the channel count at each stage. The final feature map is flattened and projected to a
  `latent_dim` vector.
- The **decoder** mirrors it: a dense layer back to the flattened feature map, then up-blocks that
  upsample by nearest-neighbor and convolve (a "resize-convolution", which avoids the checkerboard
  artifacts that stride-2 transposed convolutions produce).

The grid size must be divisible by `2^(number of stages)` so every pooling step lands on an integer
size. There is an optional 1×1-conv channel bottleneck around the latent and an optional
unit-length ("hypersphere") normalization of the latent; both are recorded in the checkpoint so it
rebuilds identically. The architecture and its variants are configured under `[autoencoder]` in the
config and via CLI flags.

Training minimizes the reconstruction error between an input frame and its reconstruction (MSE,
Huber, or an MS-SSIM+L1 structural loss from `losses.py`). Typical run:

```bash
# Build the dataset (igniting each world), then train:
python src/train_autoencoder.py --build --spawn-fire --data data/fire --epochs 50
```

Training runs in **bfloat16** by default (halving memory and using the tensor cores on modern GPUs,
while avoiding the underflow that plain float16 suffers), and auto-falls back to float32 on a CPU.
The chosen precision is recorded in the checkpoint. `tools/sweep.py` trains many identically-
configured replicates (differing only in random seed) in parallel and averages their learning
curves.

---

## Stage 3 — The dynamics head (`dynamics.py`)

Compressing a frame is only half the job; the point is to **forecast**. The dynamics head
(`LatentTransition`) is a small residual MLP that takes the latent for the current frame and
predicts the latent for the *next* frame. Because it is residual, it starts near the identity (a
good prior — consecutive frames are similar) and only has to learn the change. There is also an
optional `RewardHead` that reads a scalar reward (e.g. the count of newly-ignited cells) off a
latent, for a downstream agent.

There are **two training schemes**, and they exist because the first one had a specific failure:

### `train_world_model.py` — self-predictive (BYOL/SPR-style)

Trains the autoencoder and dynamics head together. For a pair of consecutive frames, the loss has
two disjoint gradient paths that meet only at the encoder:

- a **reconstruction** path (decode the current latent, compare to the current frame), which anchors
  the latent so it stays meaningful; and
- a **prediction** path (push the predicted next-latent toward the encoding of the *actual* next
  frame, produced by a slow-moving "target" copy of the encoder, with no gradient).

The second frame is never decoded — it only serves as the prediction target. This avoids
representational collapse without any explicit variance regulariser. You can train it from scratch,
or load a pretrained autoencoder and (optionally) freeze it to train only the dynamics head.

```bash
python src/train_world_model.py --data data/fire --init-from runs/ae/best.pt \
    --freeze-ae --epochs 30 --out runs/wm
```

**The catch:** matching latents by cosine similarity is minimized by a nearly *static* latent, so
the head learned to predict that the fire barely changes — rolled forward, the predicted fire
*decayed* while the real fire *grew*.

### `train_world_model_rollout.py` — decode-supervised rollout

The fix. Encode frame 0 **once**, roll the dynamics head forward several steps, **decode every
rolled step**, and supervise each decoded frame against the real future frame. Because the target
future actually contains a growing fire, the dynamics is forced to propagate fire rather than freeze
it. To keep the sparse, thin fire front from being drowned out by the large static background, the
loss (`wm_losses.py`) weights the dynamic channels and the fire front more heavily, and rescales the
temperature channels (which the min-max baking had compressed into a sliver of `[0, 1]`) on the fly.
The design reasoning is recorded in [`wm_decode_rollout_plan.md`](wm_decode_rollout_plan.md) and
[`wm_two_term_loss_plan.md`](wm_two_term_loss_plan.md).

---

## Inspecting a trained model

- **Beside the live physics** — `python src/main.py --model models/<checkpoint>.pt [more.pt ...]`
  runs the interactive world with one reconstruction panel per model next to the physics view.
  Press `Tab` to flip the model panels to a per-cell error heatmap. Only the observed channels
  (air/fuel temperature, wind, fire) can be reconstructed; other views show a placeholder.
  (See [`interactive_viewer.md`](interactive_viewer.md) and `model_viewer.py`.)
- **Rolling a forecast forward** — `tools/rollout_viewer.py` rolls the dynamics head forward in
  latent space and decodes each step, so you can watch the *neural* world evolve and see where it
  diverges from the physics.
- **Headless fidelity checks** — `tools/rollout_check.py` and `tools/wm_rollout_mae.py` print
  per-step tables of how well a forecast tracks the real future (does the fire front advance, freeze,
  or explode?). `tools/eval_full.py` scores a checkpoint's reconstruction over an entire dataset.

---

## Checkpoints

Trained models live in [`models/`](../models/), and [`models/README.md`](../models/README.md) is a
detailed catalog: which architecture and dataset each came from, its validation loss, and exact
code to rebuild and load it. Each checkpoint is a `torch.save` dict carrying the model weights and a
`model_config` describing the architecture, so the loader can reconstruct the network without you
having to remember its shape. World-model checkpoints additionally carry the `dynamics_config` and
`dynamics_state` (and, where present, a reward head).

A note on channel counts: models tagged **14ch** were trained before the tree trunk was split into
two segments (4 fuel types → `6 + 2×4 = 14` channels); **16ch** models were trained after
(5 fuel types → 16 channels). A model can only run against a config whose fuel types produce the
channel count it was trained on — `main.py --model` checks this and refuses a mismatch rather than
crashing.
