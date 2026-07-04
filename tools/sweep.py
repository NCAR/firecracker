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

CPU binding (`--bind`, on by default)
-------------------------------------
When several lanes share a node they otherwise float across the whole allocation, so the
kernel scheduler and the RAM loader's prefetch threads contend for the same cores. On a
dual-socket box that contention is uneven: lanes whose GPU hangs off the smaller NUMA node
get starved and lag by whole epochs (observed on Casper's V100 nodes, where PBS grants 18
cores on node 0 but only 14 on node 1). `--bind` pins each lane to a *disjoint*, NUMA-local
slice of the allocated cores via `numactl --physcpubind ... --membind <node>`, so no lane is
starved and every lane's host memory stays on its GPU's socket. Each NUMA node's cores are
split as evenly as possible among the lanes on that node; because the socket core counts can
differ, lanes on a smaller node get proportionally fewer cores (this cannot be equalised
without forcing cross-socket memory access, which is the very cost we avoid). Binding is
skipped with a warning if `numactl` is missing or the topology can't be read; disable it with
`--no-bind`.

Everything after a `--` is forwarded verbatim to every training run, so the full
train_autoencoder.py CLI is available:

    # 16 replicates over 4 GPUs, 4 sharing each card, into runs/lat1024/run_00..15:
    python tools/sweep.py --gpus 4 --per-gpu 4 --out runs/lat1024 -- \
        --data data/fire --epochs 50 --weight-dtype bfloat16 \
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
import shutil
import subprocess
import sys
import time
from glob import glob
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


def _parse_cpulist(text: str) -> list[int]:
    """Expand a Linux cpulist ("0-3,8,10-11") into [0,1,2,3,8,10,11]."""
    cpus: list[int] = []
    for part in text.strip().split(","):
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-")
            cpus.extend(range(int(lo), int(hi) + 1))
        else:
            cpus.append(int(part))
    return cpus


def _numa_topology(allocated: set[int]) -> dict[int, list[int]]:
    """Map each NUMA node to the allocated cpus that live on it.

    Only cpus the job actually holds (its cpuset) are kept, so a node the allocation doesn't
    reach drops out. Returns {} if the sysfs NUMA layout can't be read.
    """
    topo: dict[int, list[int]] = {}
    for node_dir in sorted(glob("/sys/devices/system/node/node[0-9]*")):
        try:
            node = int(os.path.basename(node_dir)[len("node"):])
            node_cpus = _parse_cpulist(open(os.path.join(node_dir, "cpulist")).read())
        except (OSError, ValueError):
            continue
        here = sorted(c for c in node_cpus if c in allocated)
        if here:
            topo[node] = here
    return topo


def _gpu_numa_nodes(tokens: list[str]) -> dict[str, int]:
    """Map each CUDA_VISIBLE_DEVICES token (physical gpu) to its NUMA node via sysfs.

    Uses `nvidia-smi` to resolve index/uuid -> PCI bus id, then reads the device's
    `numa_node`. Returns {} if nvidia-smi is unavailable or any node reads as -1/unknown.
    """
    smi = shutil.which("nvidia-smi")
    if not smi:
        return {}
    try:
        out = subprocess.run(
            [smi, "--query-gpu=index,uuid,pci.bus_id", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {}

    by_key: dict[str, str] = {}  # index and uuid both -> bus id
    for line in out.strip().splitlines():
        idx, uuid, bus = (x.strip() for x in line.split(","))
        by_key[idx] = bus
        by_key[uuid] = bus

    nodes: dict[str, int] = {}
    for token in tokens:
        bus = by_key.get(token)
        if bus is None:
            return {}
        # nvidia-smi pads the PCI domain to 8 hex digits ("00000000:89:00.0"); sysfs uses 4.
        parts = bus.lower().split(":")
        sysbus = f"{parts[0][-4:]}:{parts[1]}:{parts[2]}"
        try:
            node = int(open(f"/sys/bus/pci/devices/{sysbus}/numa_node").read().strip())
        except (OSError, ValueError):
            return {}
        if node < 0:
            return {}
        nodes[token] = node
    return nodes


def _even_chunks(items: list[int], n: int) -> list[list[int]]:
    """Split `items` into `n` contiguous chunks whose sizes differ by at most one."""
    q, r = divmod(len(items), n)
    chunks, start = [], 0
    for k in range(n):
        size = q + (1 if k < r else 0)
        chunks.append(items[start:start + size])
        start += size
    return chunks


def compute_bindings(
    queue: list[tuple[int, int, int, Path, list[str]]], gpus: int
) -> dict[int, list[str]]:
    """Return {lane_index: numactl prefix} pinning each lane to disjoint, NUMA-local cores.

    Lanes are grouped by the NUMA node their gpu sits on; each node's allocated cores are
    split evenly (see `_even_chunks`) among the lanes sharing it, and each lane gets
    `numactl --physcpubind=<its cores> --membind=<node>`. Returns {} — meaning "run unbound" —
    if the cpuset, NUMA topology, or gpu->node map can't be resolved, or if any node has
    fewer allocated cores than lanes (which would leave a lane with no cpu).
    """
    try:
        allocated = set(os.sched_getaffinity(0))
    except AttributeError:  # not Linux
        return {}
    if not allocated:
        return {}
    topo = _numa_topology(allocated)
    if not topo:
        return {}
    gpu_node = _gpu_numa_nodes([_gpu_token(g) for g in range(gpus)])
    if not gpu_node:
        return {}

    # Bucket lanes by the NUMA node of their assigned gpu, preserving lane order.
    lanes_by_node: dict[int, list[int]] = {}
    for i, _seed, gpu, _out, _cmd in queue:
        node = gpu_node.get(_gpu_token(gpu))
        if node is None or node not in topo:
            return {}
        lanes_by_node.setdefault(node, []).append(i)

    binds: dict[int, list[str]] = {}
    for node, lanes in lanes_by_node.items():
        cores = topo[node]
        if len(cores) < len(lanes):
            return {}  # can't give every lane at least one core; bail to unbound
        for lane, chunk in zip(lanes, _even_chunks(cores, len(lanes))):
            cpus = ",".join(str(c) for c in chunk)
            binds[lane] = ["numactl", f"--physcpubind={cpus}", f"--membind={node}"]
    return binds


def build_command(
    passthrough: list[str], out_dir: Path, seed: int, extra: list[str] | None = None
) -> list[str]:
    """Assemble a train_autoencoder.py invocation: shared args plus this run's out/seed.

    The launcher owns `--out`, `--seed`, `--device` (always cuda) and, for a W&B run, the
    per-run `--wandb-group`/`--wandb-name`; strip any user-supplied copies so the launcher's
    values (passed via `extra`) don't collide on the command line.
    """
    drop = {"--out", "--seed", "--device", "--wandb-group", "--wandb-name"}
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
    cmd += extra or []
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
    parser.add_argument("--wandb-group", default=None,
                        help="W&B group shared by every replicate (so they aggregate); only "
                             "used when `--wandb` is in the forwarded args. Default: <out> name")
    parser.add_argument("--poll", type=float, default=2.0,
                        help="seconds between scheduler polls (default: 2)")
    parser.add_argument("--bind", dest="bind", action="store_true", default=True,
                        help="pin each lane to disjoint, NUMA-local cores via numactl "
                             "(default: on; auto-skipped if numactl/topology unavailable)")
    parser.add_argument("--no-bind", dest="bind", action="store_false",
                        help="disable CPU binding; let lanes float across the allocation")
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

    # When the forwarded args enable W&B, give every replicate a shared group (so W&B can
    # average the runs) and a distinct, seed-tagged name.
    wandb_on = "--wandb" in passthrough
    wandb_group = args.wandb_group or out_root.name

    # Build the full work queue: (index, seed, gpu, out_dir, command). Runs are round-robined
    # onto gpus so an uneven replicas count still spreads evenly across the cards.
    queue: list[tuple[int, int, int, Path, list[str]]] = []
    for i in range(replicas):
        seed = args.seed_base + i
        gpu = i % args.gpus
        run_dir = out_root / f"run_{i:02d}"
        extra = (["--wandb-group", wandb_group, "--wandb-name", f"run_{i:02d}_seed{seed}"]
                 if wandb_on else None)
        cmd = build_command(passthrough, run_dir, seed, extra)
        queue.append((i, seed, gpu, run_dir, cmd))

    # Pin each lane to disjoint, NUMA-local cores so lanes sharing a socket don't starve one
    # another. compute_bindings returns {} (run unbound) when the flag is off or the topology
    # can't be resolved; each lane's numactl prefix is prepended to its command below.
    binds: dict[int, list[str]] = {}
    if args.bind:
        if shutil.which("numactl") is None:
            print("sweep: --bind requested but numactl not found; running unbound.")
        else:
            binds = compute_bindings(queue, args.gpus)
            if not binds:
                print("sweep: --bind requested but CPU/NUMA topology unresolved; "
                      "running unbound.")

    print(f"sweep: {replicas} replicate(s) across {args.gpus} gpu(s), "
          f"{args.per_gpu} per gpu; out '{out_root}/'"
          + (f"; cpu-bound to {len(binds)} numactl slice(s)" if binds else "; unbound"))
    if args.dry_run:
        for i, seed, gpu, run_dir, cmd in queue:
            full = binds.get(i, []) + cmd
            print(f"  run_{i:02d} (seed {seed}, gpu {gpu} -> CUDA_VISIBLE_DEVICES="
                  f"{_gpu_token(gpu)}): {' '.join(full)}")
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
        prefix = binds.get(i, [])
        full = prefix + cmd
        bind_note = f" [{' '.join(prefix)}]" if prefix else ""
        print(f"  start run_{i:02d} (seed {seed}) on gpu {gpu} "
              f"[CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}]{bind_note} "
              f"-> {run_dir}/train.log")
        proc = subprocess.Popen(full, cwd=_ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
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
