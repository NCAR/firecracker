"""
ram_loader.py

A DataLoader replacement that serves whole batches from the (page-cache-resident) memmapped
FireDataset shards, using a single background *thread* for prefetch.

Why not torch.utils.data.DataLoader with workers? Under heavy process packing (many trainers
sharing a node), its worker *processes* hand tensors back through shared memory, which races
and crashes under load ("could not unlink the shared memory file"). This loader spawns no
worker processes: one background thread gathers the next batch (vectorized -- one memmap read
per shard, see FireDataset.get_batch) while the main thread runs GPU compute. numpy's memmap
copy and tensor pinning release the GIL, so the fetch overlaps the compute -- recovering the
async-prefetch speedup without any inter-process shared memory.

Interface mirrors the subset of DataLoader that train_autoencoder.py uses: it is iterable
(yielding one batch tensor per step) and supports len(). Each batch is C x N x N x B in the
dataset's stored dtype (e.g. float16); the training loop casts to float32 on device.
"""

from __future__ import annotations

import queue
import threading
from typing import Iterator

import numpy as np
import torch


def prewarm_shards(dataset, *, block: int = 8 << 20) -> threading.Thread:
    """Stream every shard front-to-back in a background thread to fault its pages into the
    OS page cache, decoupling the first read of the data from the training loop.

    Without this, the shuffled first epoch is the first touch of every page and drives a storm
    of scattered random reads across the (memmapped) shards on the parallel filesystem -- the
    ~30-minute epoch-1 tax. A plain sequential read of each shard is far cheaper on GLADE than
    that random access, and the page cache is shared node-wide, so one prewarm pass warms the
    cache for every trainer packed on the node. Reading whole files as raw bytes (no npy parse)
    releases the GIL, so the prewarm overlaps GPU compute: launch it right after the dataset is
    built and epoch 1 races the prewarm, finding pages progressively warm.

    Returns the started daemon thread (join() to block until the cache is warm, or ignore it to
    let the prewarm overlap training). `block` is the sequential read chunk in bytes.
    """

    def run() -> None:
        for p in dataset.shard_paths:
            with open(p, "rb", buffering=0) as f:
                while f.read(block):
                    pass

    thread = threading.Thread(target=run, name="prewarm", daemon=True)
    thread.start()
    return thread


class RamBatchLoader:
    """Thread-prefetched batch loader over a FireDataset's memmapped shards.

    Args:
        dataset:     a FireDataset (needs `.get_batch(indices) -> Tensor`).
        indices:     global sample indices this loader draws from (e.g. the train or val split).
        batch_size:  samples per batch.
        shuffle:     reshuffle `indices` each epoch (seeded, reproducible across epochs).
        seed:        seed for the shuffle RNG.
        drop_last:   drop a trailing partial batch (train) or keep it (val).
        pin_memory:  pin each batch for faster, overlappable host->device copies (CUDA only).
        prefetch:    max batches the background thread may stage ahead.
    """

    def __init__(
        self,
        dataset,
        indices,
        batch_size: int,
        *,
        shuffle: bool,
        seed: int,
        drop_last: bool = True,
        pin_memory: bool = False,
        prefetch: int = 3,
    ) -> None:
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.pin_memory = bool(pin_memory)
        self.prefetch = max(1, int(prefetch))
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        n = len(self.indices)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def _batch_index_chunks(self):
        order = self._rng.permutation(self.indices) if self.shuffle else self.indices
        n = len(order)
        stop = (n // self.batch_size) * self.batch_size if self.drop_last else n
        for i in range(0, stop, self.batch_size):
            yield order[i : i + self.batch_size]

    def __iter__(self) -> Iterator[torch.Tensor]:
        q: queue.Queue = queue.Queue(maxsize=self.prefetch)
        sentinel = object()

        def producer() -> None:
            try:
                for chunk in self._batch_index_chunks():
                    batch = self.dataset.get_batch(chunk)
                    if self.pin_memory:
                        batch = batch.pin_memory()
                    q.put(batch)
            except Exception as exc:  # surface a fetch error to the consumer instead of hanging
                q.put(exc)
            else:
                q.put(sentinel)

        thread = threading.Thread(target=producer, daemon=True)
        thread.start()
        while True:
            item = q.get()
            if item is sentinel:
                break
            if isinstance(item, Exception):
                raise item
            yield item
        thread.join()
