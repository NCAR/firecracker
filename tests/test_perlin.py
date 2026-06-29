"""
Tests for the vectorized octave Perlin kernel (src/perlin.py).

These pin down the contract the map builder relies on:
  * deterministic given (shape, params, base) -- maps stay reproducible,
  * scalar vs sequence `base` give a single field vs a stacked batch, and the batch
    is exactly the per-base fields stacked,
  * distinct bases give decorrelated fields (not a spatial shift of one field),
  * more octaves add finer-scale detail (higher gradient energy),
  * the field is the same on every available device -- device is just a flag.
"""

import sys
from pathlib import Path

import pytest
import torch

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from perlin import perlin_noise  # noqa: E402

PARAMS = dict(scale=32.0, octaves=4, persistence=0.5, lacunarity=2.0)


def _devices() -> list[str]:
    devs = ["cpu"]
    if torch.cuda.is_available():
        devs.append("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        devs.append("mps")
    return devs


def test_deterministic():
    a = perlin_noise(48, 48, base=7, **PARAMS)
    b = perlin_noise(48, 48, base=7, **PARAMS)
    assert torch.equal(a, b)


def test_scalar_base_shape_and_finiteness():
    field = perlin_noise(40, 56, base=0, **PARAMS)
    assert field.shape == (40, 56)
    assert torch.isfinite(field).all()


def test_sequence_base_is_batched():
    batch = perlin_noise(32, 32, base=[0, 1, 2], **PARAMS)
    assert batch.shape == (3, 32, 32)


def test_batch_equals_stacked_scalars():
    bases = [3, 11, 29]
    batch = perlin_noise(32, 40, base=bases, **PARAMS)
    stacked = torch.stack([perlin_noise(32, 40, base=b, **PARAMS) for b in bases])
    assert torch.allclose(batch, stacked)


def test_distinct_bases_are_decorrelated():
    # base is mixed into the permutation lookup, so a different base is a different
    # field -- not the same field shifted. Cross-correlation should be near zero,
    # far below a field's self-correlation (1.0).
    a = perlin_noise(64, 64, base=0, **PARAMS).flatten()
    b = perlin_noise(64, 64, base=1, **PARAMS).flatten()
    corr = torch.corrcoef(torch.stack([a, b]))[0, 1].abs()
    assert corr < 0.2


def test_more_octaves_add_detail():
    # Higher octave counts layer finer frequencies, raising neighbour-to-neighbour
    # variation. Compare mean |gradient| along a row.
    def roughness(octaves: int) -> float:
        f = perlin_noise(64, 64, base=5, scale=32.0, octaves=octaves,
                         persistence=0.5, lacunarity=2.0)
        return f.diff(dim=1).abs().mean().item()

    assert roughness(6) > roughness(1)


def test_single_octave_unaffected_by_persistence():
    # With one octave the amplitude/frequency multipliers never apply, so persistence
    # and lacunarity are inert -- a useful invariant guarding the accumulation loop.
    a = perlin_noise(32, 32, base=2, scale=24.0, octaves=1, persistence=0.5, lacunarity=2.0)
    b = perlin_noise(32, 32, base=2, scale=24.0, octaves=1, persistence=0.9, lacunarity=3.0)
    assert torch.equal(a, b)


@pytest.mark.parametrize("device", _devices())
def test_device_parity(device):
    # Device is just a flag: a backend produces the same field as the CPU at the same
    # dtype. MPS has no float64 (and the generation pipeline only targets cuda/cpu in
    # float64 anyway), so it is checked in float32 against a float32 CPU reference.
    dtype = torch.float32 if device == "mps" else torch.float64
    ref = perlin_noise(48, 48, base=13, device="cpu", dtype=dtype, **PARAMS)
    got = perlin_noise(48, 48, base=13, device=device, dtype=dtype, **PARAMS).cpu()
    assert torch.allclose(ref, got, atol=1e-6)
