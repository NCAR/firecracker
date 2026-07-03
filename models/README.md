# Trained autoencoder checkpoints

Best replicate from each of the two 16-replicate grouped-parallel sweeps
(50 epochs, seeds 0–15). Each file is a `torch.save` dict with keys
`epoch`, `val_loss`, `model_state`, and `model_config`. The Adam
`optimizer_state` has been stripped (weights are bit-exact; only training
resumption is unavailable), which cuts each file from ~99 MiB to ~33 MiB.

| File              | Grouping | Source run                                    | Best epoch | Val loss |
|-------------------|----------|-----------------------------------------------|-----------:|---------:|
| `shuffle_best.pt` | shuffled | `sweep_shuffle_20260703_011310/run_02`        |         49 | 0.253177 |
| `strided_best.pt` | strided  | `sweep_strided_20260703_011310/run_02`        |         49 | 0.249810 |

Selected as the lowest-val-loss replicate in each sweep. Per-channel input
normalization stats live with the dataset (`<data>/stats.json`) and the
architecture is recorded in each checkpoint's `model_config`.

Load with:

```python
import torch
ckpt = torch.load("models/strided_best.pt", map_location="cpu")
model = build_model(ckpt["model_config"])   # see src/train_autoencoder.py
model.load_state_dict(ckpt["model_state"])
```
