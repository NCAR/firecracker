"""
sweep.py

Train many replicate world models in parallel and average their learning curves.

This is a replicate study, not a hyperparameter search: every run uses the *same*
architecture and training arguments and differs only in its random seed (weight init,
data shuffle, train/val split, optimiser state). Running N of them and averaging the
per-epoch validation loss gives a mean curve with error bars for how a given
architecture does on the given data.

The autoencoder is tiny, so several runs comfortably share one GPU. Each run is an
independent `src/train_autoencoder.py` subprocess; this launcher only schedules them,
injects a per-run `--out` and `--seed`, and pins each to a device.

Device packing
--------------
Training always runs on CUDA. `--gpus G` spreads the replicates across G cards and
`--per-gpu M` sets how many share each card concurrently; run i is pinned to gpu `i % G`
by exporting a single-device `CUDA_VISIBLE_DEVICES` into its subprocess (so the trainer's
`--device cuda` lands on that physical card as cuda:0), and the rest queue until a slot on
their gpu frees. With `--gpus 1` this reduces to the old single-card behaviour. Omit
`--replicas` to train exactly `G * M` runs (one full packing). Every run shares the same
architecture and training args — only the seed differs — so the average is a replicate
study across all G*M seeds regardless of which card each ran on.

The parent `CUDA_VISIBLE_DEVICES` (e.g. the 4 cards PBS allocated) is honoured: gpu index
g selects the g-th entry of that list, so this works unchanged inside a PBS job that was
given a subset of the node's GPUs.

Everything after a `--` is forwarded verbatim to every training run, so the full
train_autoencoder.py CLI is available:

    # 16 replicates over 4 GPUs, 4 sharing each card, into runs/lat1024/run_00..15:
    python tools/sweep.py --gpus 4 --per-gpu 4 --out runs/lat1024 -- \
        --data data/fire --epochs 50 --amp \
        --channels 8,16,32,64,128,256,512,1024

    # Inspect the per-run commands without launching (works anywhere, no GPU needed):
    python tools/sweep.py --gpus 4 --per-gpu 4 --out runs/lat1024 --dry-run -- \
        --data data/fire --epochs 50

When the runs finish, every `run_*/metrics.csv` is aggregated into `<out>/summary.csv`
(per-epoch mean/std across replicates) and, if matplotlib is installed, `<out>/summary.png`.
Re-run with `--aggregate-only` to rebuild those from existing run dirs without training.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"

# The device list this launcher inherited (what PBS/cgroups exposed, or unset). A run pinned
# to logical gpu `g` gets the g-th entry, so indices stay valid inside a partial allocation.
_PARENT_CVD = [d for d in (os.environ.get("CUDA_VISIBLE_DEVICES") or "").split(",") if d]


def _gpu_token(gpu: int) -> str:
    """CUDA_VISIBLE_DEVICES value pinning a subprocess to logical gpu `gpu`.

    If the parent already restricts visibility (a PBS GPU allocation), index into that list
    so `gpu` selects the g-th allocated card; otherwise fall back to the bare index.
    """
    if gpu < len(_PARENT_CVD):
        return _PARENT_CVD[gpu]
    return str(gpu)


def build_command(passthrough: list[str], out_dir: Path, seed: int) -> list[str]:
    """Assemble a train_autoencoder.py invocation: shared args plus this run's out/seed.

    The launcher owns `--out`, `--seed` and `--device` (always cuda); strip any
    user-supplied copies so they don't collide on the command line.
    """
    drop = {"--out", "--seed", "--device"}
    cmd = [sys.executable, str(_SRC / "train_autoencoder.py")]
    skip_next = False
    for tok in passthrough:
        if skip_next:
            skip_next = False
            continue
        if tok in drop:
            # These all take a value, so drop the following token too.
            skip_next = True
            continue
        cmd.append(tok)
    cmd += ["--out", str(out_dir), "--seed", str(seed), "--device", "cuda"]
    return cmd


def aggregate(out_root: Path) -> None:
    """Average every run_*/metrics.csv into <out_root>/summary.csv (+ summary.png if able)."""
    import numpy as np

    run_dirs = sorted(p for p in out_root.glob("run_*") if (p / "metrics.csv").exists())
    if not run_dirs:
        print(f"aggregate: no run_*/metrics.csv under {out_root}")
        return

    # epoch -> {"train": [...], "val": [...]} gathered across replicates.
    by_epoch: dict[int, dict[str, list[float]]] = {}
    for d in run_dirs:
        with (d / "metrics.csv").open(newline="") as f:
            for row in csv.DictReader(f):
                e = int(row["epoch"])
                slot = by_epoch.setdefault(e, {"train": [], "val": []})
                slot["train"].append(float(row["train_loss"]))
                slot["val"].append(float(row["val_loss"]))

    summary_path = out_root / "summary.csv"
    epochs = sorted(by_epoch)
    with summary_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "n", "train_mean", "train_std", "val_mean", "val_std", "val_sem"])
        rows = []
        for e in epochs:
            tr = np.array(by_epoch[e]["train"])
            va = np.array(by_epoch[e]["val"])
            sem = va.std(ddof=1) / np.sqrt(va.size) if va.size > 1 else 0.0
            row = [e, va.size,
                   f"{tr.mean():.6f}", f"{tr.std():.6f}",
                   f"{va.mean():.6f}", f"{va.std():.6f}", f"{sem:.6f}"]
            w.writerow(row)
            rows.append((e, va.mean(), va.std()))
    print(f"aggregate: {len(run_dirs)} run(s) -> {summary_path}")

    # Headline for "how good is this config": each replicate's best (min) val loss, then the
    # mean and standard error of that best across replicates. sem = std / sqrt(n) is the
    # uncertainty on the mean and is what shrinks as you add more seeds.
    best = []
    for d in run_dirs:
        with (d / "metrics.csv").open(newline="") as f:
            vals = [float(r["val_loss"]) for r in csv.DictReader(f)]
        if vals:
            best.append(min(vals))
    b = np.array(best)
    if b.size:
        std = b.std(ddof=1) if b.size > 1 else 0.0
        sem = std / np.sqrt(b.size) if b.size > 1 else 0.0
        print(f"aggregate: best-val over {b.size} replicate(s): "
              f"mean {b.mean():.6f}  std {std:.6f}  sem {sem:.6f}  "
              f"[min {b.min():.6f}, max {b.max():.6f}]")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("aggregate: matplotlib not installed; skipping summary.png")
        return

    es = np.array([r[0] for r in rows])
    mean = np.array([r[1] for r in rows])
    std = np.array([r[2] for r in rows])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for d in run_dirs:
        with (d / "metrics.csv").open(newline="") as f:
            rd = list(csv.DictReader(f))
        ax.plot([int(r["epoch"]) for r in rd], [float(r["val_loss"]) for r in rd],
                color="0.8", linewidth=0.8, zorder=1)
    ax.fill_between(es, mean - std, mean + std, alpha=0.25, color="C0", zorder=2,
                    label="±1 std")
    ax.plot(es, mean, color="C0", linewidth=2.0, zorder=3,
            label=f"mean val ({len(run_dirs)} runs)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation loss (MSE)")
    ax.set_title(out_root.name)
    ax.legend()
    fig.tight_layout()
    png_path = out_root / "summary.png"
    fig.savefig(png_path, dpi=120)
    print(f"aggregate: plot -> {png_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train replicate world models in parallel and average their curves.",
        epilog="All args after `--` are forwarded to src/train_autoencoder.py.",
    )
    parser.add_argument("--gpus", type=int, default=1,
                        help="GPUs to spread runs across; run i is pinned to gpu i%%gpus "
                             "(default: 1)")
    parser.add_argument("--replicas", type=int, default=None,
                        help="number of seeded replicate runs to train (default: gpus*per_gpu)")
    parser.add_argument("--per-gpu", type=int, default=4,
                        help="replicate runs sharing EACH gpu at once (default: 4)")
    parser.add_argument("--out", default="runs/sweep",
                        help="root dir; runs land in <out>/run_NN (default: runs/sweep)")
    parser.add_argument("--seed-base", type=int, default=0,
                        help="seed of the first replicate; run i uses seed-base+i (default: 0)")
    parser.add_argument("--poll", type=float, default=2.0,
                        help="seconds between scheduler polls (default: 2)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the per-run commands and exit without launching")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="rebuild summary.csv/png from existing <out>/run_* and exit")
    # Everything after `--` is the shared train_autoencoder.py command line.
    parser.add_argument("train_args", nargs=argparse.REMAINDER,
                        help="-- <args forwarded to train_autoencoder.py>")
    args = parser.parse_args()

    out_root = Path(args.out)
    if args.aggregate_only:
        aggregate(out_root)
        return

    if args.gpus < 1:
        parser.error("--gpus must be >= 1")
    replicas = args.replicas if args.replicas is not None else args.gpus * args.per_gpu

    # argparse.REMAINDER keeps the leading "--"; drop it.
    passthrough = args.train_args[1:] if args.train_args[:1] == ["--"] else args.train_args

    out_root.mkdir(parents=True, exist_ok=True)

    # Build the full work queue: (index, seed, gpu, out_dir, command). Runs are round-robined
    # onto gpus so an uneven replicas count still spreads evenly across the cards.
    queue: list[tuple[int, int, int, Path, list[str]]] = []
    for i in range(replicas):
        seed = args.seed_base + i
        gpu = i % args.gpus
        run_dir = out_root / f"run_{i:02d}"
        cmd = build_command(passthrough, run_dir, seed)
        queue.append((i, seed, gpu, run_dir, cmd))

    print(f"sweep: {replicas} replicate(s) across {args.gpus} gpu(s), "
          f"{args.per_gpu} per gpu; out '{out_root}/'")
    if args.dry_run:
        for i, seed, gpu, run_dir, cmd in queue:
            print(f"  run_{i:02d} (seed {seed}, gpu {gpu} -> CUDA_VISIBLE_DEVICES="
                  f"{_gpu_token(gpu)}): {' '.join(cmd)}")
        return

    pending = list(queue)
    free = [args.per_gpu] * args.gpus  # per-gpu concurrency budget
    running: dict[int, dict] = {}
    failures: list[tuple[int, int]] = []

    def launch(run):
        i, seed, gpu, run_dir, cmd = run
        run_dir.mkdir(parents=True, exist_ok=True)
        log = (run_dir / "train.log").open("w")
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = _gpu_token(gpu)
        print(f"  start run_{i:02d} (seed {seed}) on gpu {gpu} "
              f"[CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}] -> {run_dir}/train.log")
        proc = subprocess.Popen(cmd, cwd=_ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
        return {"proc": proc, "log": log, "run": run}

    while pending or running:
        # Launch any pending run whose assigned gpu has a free slot; keep the rest queued.
        still: list = []
        for run in pending:
            gpu = run[2]
            if free[gpu] > 0:
                rec = launch(run)
                running[id(rec["proc"])] = rec
                free[gpu] -= 1
            else:
                still.append(run)
        pending = still
        time.sleep(args.poll)
        for key, rec in list(running.items()):
            rc = rec["proc"].poll()
            if rc is None:
                continue
            rec["log"].close()
            i, gpu = rec["run"][0], rec["run"][2]
            status = "ok" if rc == 0 else f"FAILED (exit {rc})"
            print(f"  done  run_{i:02d} (gpu {gpu}): {status}")
            if rc != 0:
                failures.append((i, rc))
            free[gpu] += 1
            del running[key]

    print(f"sweep: all runs finished ({len(failures)} failed).")
    aggregate(out_root)
    if failures:
        ids = ", ".join(f"run_{i:02d}(exit {rc})" for i, rc in failures)
        print(f"sweep: failures: {ids}")
        sys.exit(1)


if __name__ == "__main__":
    main()
