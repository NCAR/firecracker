"""
perlin.py

Vectorized octave (fractional-Brownian) Perlin noise as a single torch kernel.

This replaces the per-cell scalar loop over the `noise` C library that used to drive
map generation (see gen_maps.py). The whole field -- and a whole batch of fields, one
per random `base` -- is evaluated as tensor ops, so the same code runs on CPU
(`device="cpu"`) or GPU (`device="cuda"`) with no Python pixel loop. Device is just a
flag: the two produce the same field (to floating-point tolerance), so there is one
noise implementation, not a CPU one and a GPU one.

The field is classic Ken Perlin gradient noise: a fixed 256-entry permutation table
hashes each lattice corner to one of eight gradient directions, the corner gradient
dot products are quintic-faded and bilinearly interpolated, and `octaves` such fields
are summed at geometrically increasing frequency (`lacunarity`) and decreasing
amplitude (`persistence`) -- the same fBm contract the old `pnoise2(..., octaves=...)`
calls used. The raw sum is *not* renormalised here (callers min/max-normalise the
result, as before), so amplitude is irrelevant and only the spatial statistics matter.

The per-map `base` is mixed into the *first* permutation lookup (perm[(xi+base)&255]),
not added to the coordinates, so distinct bases give decorrelated fields rather than a
whole-cell spatial shift of the same field.
"""

from __future__ import annotations

import torch

# Ken Perlin's reference 256-value permutation. A fixed shuffle of 0..255; the `base`
# argument decorrelates fields, so the table itself never needs reseeding.
_PERM_256 = [
    151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225,
    140, 36, 103, 30, 69, 142, 8, 99, 37, 240, 21, 10, 23, 190, 6, 148,
    247, 120, 234, 75, 0, 26, 197, 62, 94, 252, 219, 203, 117, 35, 11, 32,
    57, 177, 33, 88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 168, 68, 175,
    74, 165, 71, 134, 139, 48, 27, 166, 77, 146, 158, 231, 83, 111, 229, 122,
    60, 211, 133, 230, 220, 105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54,
    65, 25, 63, 161, 1, 216, 80, 73, 209, 76, 132, 187, 208, 89, 18, 169,
    200, 196, 135, 130, 116, 188, 159, 86, 164, 100, 109, 198, 173, 186, 3, 64,
    52, 217, 226, 250, 124, 123, 5, 202, 38, 147, 118, 126, 255, 82, 85, 212,
    207, 206, 59, 227, 47, 16, 58, 17, 182, 189, 28, 42, 223, 183, 170, 213,
    119, 248, 152, 2, 44, 154, 163, 70, 221, 153, 101, 155, 167, 43, 172, 9,
    129, 22, 39, 253, 19, 98, 108, 110, 79, 113, 224, 232, 178, 185, 112, 104,
    218, 246, 97, 228, 251, 34, 242, 193, 238, 210, 144, 12, 191, 179, 162, 241,
    81, 51, 145, 235, 249, 14, 239, 107, 49, 192, 214, 31, 181, 199, 106, 157,
    184, 84, 204, 176, 115, 121, 50, 45, 127, 4, 150, 254, 138, 236, 205, 93,
    222, 114, 67, 29, 24, 72, 243, 141, 128, 195, 78, 66, 215, 61, 156, 180,
]

# Eight unit-ish gradient directions selected by the low three bits of the corner hash.
_GRAD_X = [1, -1, 1, -1, 1, -1, 0, 0]
_GRAD_Y = [1, 1, -1, -1, 0, 0, 1, -1]


def _fade(t: torch.Tensor) -> torch.Tensor:
    """Quintic ease curve 6t^5 - 15t^4 + 10t^3 (zero 1st/2nd derivative at 0 and 1)."""
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _perlin_octave(
    x: torch.Tensor,          # (H, W) sample x-coordinates (already frequency-scaled)
    y: torch.Tensor,          # (H, W) sample y-coordinates
    perm: torch.Tensor,       # (256,) long permutation table
    grad_x: torch.Tensor,     # (8,) gradient x-components
    grad_y: torch.Tensor,     # (8,) gradient y-components
    base: torch.Tensor,       # (B,) long per-field base offsets
) -> torch.Tensor:
    """One octave of Perlin noise, batched over `base` -> (B, H, W)."""
    x0 = torch.floor(x)
    y0 = torch.floor(y)
    xf = x - x0                                   # (H, W) fractional offset in [0, 1)
    yf = y - y0
    xi = x0.long() & 255                          # (H, W) lattice cell index
    yi = y0.long() & 255

    B = base.shape[0]
    b = base.view(B, 1, 1)
    # Mix base into the first lookup so distinct bases decorrelate (not a spatial shift).
    xa = perm[(xi.unsqueeze(0) + b) & 255]        # (B, H, W)
    xb = perm[(xi.unsqueeze(0) + 1 + b) & 255]
    yi0 = yi.unsqueeze(0)                          # (1, H, W) broadcasts over B
    yi1 = yi0 + 1
    aa = perm[(xa + yi0) & 255]                    # corner hashes -> (B, H, W)
    ba = perm[(xb + yi0) & 255]
    ab = perm[(xa + yi1) & 255]
    bb = perm[(xb + yi1) & 255]

    xf_b = xf.unsqueeze(0)                         # (1, H, W) broadcasts over B
    yf_b = yf.unsqueeze(0)

    def grad(h: torch.Tensor, dx: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
        g = h & 7
        return grad_x[g] * dx + grad_y[g] * dy

    g_aa = grad(aa, xf_b, yf_b)
    g_ba = grad(ba, xf_b - 1.0, yf_b)
    g_ab = grad(ab, xf_b, yf_b - 1.0)
    g_bb = grad(bb, xf_b - 1.0, yf_b - 1.0)

    u = _fade(xf_b)
    v = _fade(yf_b)
    x1 = torch.lerp(g_aa, g_ba, u)
    x2 = torch.lerp(g_ab, g_bb, u)
    return torch.lerp(x1, x2, v)                   # (B, H, W)


def perlin_noise(
    height: int,
    width: int,
    *,
    scale: float,
    octaves: int,
    persistence: float,
    lacunarity: float,
    base,
    device=None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Octave (fBm) Perlin noise field(s).

    Coordinates are col/scale, row/scale (matching the old pnoise2 calls): a larger
    `scale` stretches the features. `octaves` fields are summed, each at `lacunarity`x
    the previous frequency and `persistence`x the amplitude.

    `base` is an int (-> a single (H, W) field) or a 1-D sequence/tensor of ints
    (-> a batched (B, H, W) stack, one field per base). The batched form is exactly
    the per-base fields stacked, so it is what the GPU map builder uses to evaluate a
    whole chunk of terrains at once.

    The raw octave sum is returned unnormalised (callers normalise), so the absolute
    range is unspecified; only the spatial statistics are meaningful.
    """
    device = torch.device(device) if device is not None else torch.device("cpu")
    base_t = torch.as_tensor(base, device=device, dtype=torch.long)
    scalar = base_t.ndim == 0
    base_t = base_t.reshape(-1)                    # (B,)

    perm = torch.tensor(_PERM_256, device=device, dtype=torch.long)
    grad_x = torch.tensor(_GRAD_X, device=device, dtype=dtype)
    grad_y = torch.tensor(_GRAD_Y, device=device, dtype=dtype)

    rows = torch.arange(height, device=device, dtype=dtype).view(height, 1)
    cols = torch.arange(width, device=device, dtype=dtype).view(1, width)

    total = torch.zeros((base_t.shape[0], height, width), device=device, dtype=dtype)
    freq = 1.0
    amp = 1.0
    for _ in range(int(octaves)):
        x = cols.expand(height, width) * (freq / scale)
        y = rows.expand(height, width) * (freq / scale)
        total += _perlin_octave(x, y, perm, grad_x, grad_y, base_t) * amp
        freq *= lacunarity
        amp *= persistence

    return total[0] if scalar else total
