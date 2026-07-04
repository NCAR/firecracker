"""
losses.py

Optional reconstruction losses for train_autoencoder.py beyond plain MSE / Huber.

`MSSSIML1Loss` implements the MS-SSIM + L1 objective of Zhao et al., "Loss Functions for Image
Restoration with Neural Networks" (2017): a weighted mix of `(1 - multi-scale SSIM)` and an L1
term. MS-SSIM rewards preserving local contrast and structure across a pyramid of spatial scales,
so it penalises the blur an L2 loss tolerates (averaging away uncertain high-frequency detail
lowers squared error but destroys local contrast/structure). The L1 term anchors the low-frequency
values where MS-SSIM is nearly blind (uniform offsets, flat regions). Blend:

    loss = alpha * (1 - MS-SSIM) + (1 - alpha) * L1        (Zhao et al. use alpha = 0.84)

Notes for standardised inputs
-----------------------------
SSIM's stabilising constants are C1 = (K1*L)^2, C2 = (K2*L)^2 with `L` the data's dynamic range.
For 8-bit images L = 255; for the per-channel zero-mean/unit-variance fields this trainer uses,
values are O(1), so `data_range` defaults to 6.0 (roughly a +/-3 sigma span). SSIM is computed per
channel -- each physical field (fuel, fire front, wind, ...) treated independently -- and averaged,
which is exactly "did the reconstruction keep each field's local spatial structure".

The five-level pyramid halves the image four times, so the smallest input side must be at least
`(win_size) * 2**(levels-1)` ~= 11 * 16 = 176 px; a 256x256 observation clears this comfortably.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

# Per-scale weights from Wang, Simoncelli & Bovik, "Multiscale structural similarity for image
# quality assessment" (2003), derived from a human psychophysics experiment; they sum to 1. Five
# scales (fine -> coarse) is the standard choice.
_MS_SSIM_WEIGHTS: tuple[float, ...] = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)


def _gaussian_window(win_size: int, sigma: float, channels: int,
                     device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """A (channels, 1, win_size, win_size) 2-D Gaussian kernel for depthwise (grouped) conv."""
    coords = torch.arange(win_size, device=device, dtype=dtype) - (win_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g = g / g.sum()
    kernel = (g[:, None] * g[None, :])                      # outer product -> 2-D Gaussian
    return kernel.expand(channels, 1, win_size, win_size).contiguous()


def _ssim_per_channel(x: torch.Tensor, y: torch.Tensor, window: torch.Tensor,
                      data_range: float, k1: float = 0.01, k2: float = 0.03,
                      ) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-scale SSIM over a sliding Gaussian window.

    Returns `(ssim, cs)`, each `(B, C)` -- the mean over the valid spatial window of the full SSIM
    map (luminance*contrast*structure) and of the contrast*structure term alone (which is what
    MS-SSIM combines across scales).
    """
    channels = x.shape[1]
    win_size = window.shape[-1]
    if min(x.shape[-2:]) < win_size:
        raise ValueError(
            f"MS-SSIM: spatial size {tuple(x.shape[-2:])} is smaller than the {win_size}x{win_size} "
            f"window at some pyramid scale; feed larger images (>= {win_size * 16}px side for the "
            f"5-scale default) or fewer scales.")

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    def filt(t: torch.Tensor) -> torch.Tensor:
        return F.conv2d(t, window, groups=channels)         # 'valid' (no padding), depthwise

    mu_x, mu_y = filt(x), filt(y)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sigma_x2 = filt(x * x) - mu_x2
    sigma_y2 = filt(y * y) - mu_y2
    sigma_xy = filt(x * y) - mu_xy

    cs = (2.0 * sigma_xy + c2) / (sigma_x2 + sigma_y2 + c2)          # contrast * structure
    ssim = ((2.0 * mu_xy + c1) / (mu_x2 + mu_y2 + c1)) * cs          # luminance * cs
    return ssim.mean(dim=(-1, -2)), cs.mean(dim=(-1, -2))


def ms_ssim(x: torch.Tensor, y: torch.Tensor, window: torch.Tensor, data_range: float,
            weights: Sequence[float]) -> torch.Tensor:
    """Multi-scale SSIM in ~[0, 1] (higher = more similar), averaged over batch and channels."""
    levels = len(weights)
    w = torch.tensor(weights, device=x.device, dtype=x.dtype)
    cs_per_scale: list[torch.Tensor] = []
    ssim = x.new_zeros(())
    for i in range(levels):
        ssim, cs = _ssim_per_channel(x, y, window, data_range)
        if i < levels - 1:
            cs_per_scale.append(cs)
            x = F.avg_pool2d(x, kernel_size=2)
            y = F.avg_pool2d(y, kernel_size=2)
    # MS-SSIM = prod_{i<M} cs_i^{w_i} * ssim_M^{w_M}  (Wang 2003). Clamp the base off zero so the
    # fractional powers stay finite and their gradients don't blow up at exactly-zero similarity.
    stacked = torch.stack(cs_per_scale + [ssim], dim=0).clamp(min=1e-6)   # (levels, B, C)
    msssim = torch.prod(stacked ** w.view(-1, 1, 1), dim=0)               # (B, C)
    return msssim.mean()


class MSSSIML1Loss(nn.Module):
    """`alpha * (1 - MS-SSIM) + (1 - alpha) * L1`, computed per channel in float32.

    Args:
        data_range: SSIM dynamic range `L` (default 6.0, a +/-3 sigma span for standardised inputs).
        alpha:      weight on the MS-SSIM term (Zhao et al. use 0.84).
        win_size:   Gaussian window side (default 11, the SSIM standard).
        sigma:      Gaussian window sigma (default 1.5, the SSIM standard).
        weights:    per-scale MS-SSIM weights (default the 5-scale Wang set).
    """

    def __init__(self, data_range: float = 6.0, alpha: float = 0.84, win_size: int = 11,
                 sigma: float = 1.5, weights: Sequence[float] = _MS_SSIM_WEIGHTS) -> None:
        super().__init__()
        self.data_range = float(data_range)
        self.alpha = float(alpha)
        self.win_size = int(win_size)
        self.sigma = float(sigma)
        self.weights = tuple(weights)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # SSIM's divisions and fractional powers want float32 even when the model runs in bf16/f16.
        pred = pred.float()
        target = target.float()
        window = _gaussian_window(self.win_size, self.sigma, pred.shape[1], pred.device, pred.dtype)
        msssim = ms_ssim(pred, target, window, self.data_range, self.weights)
        l1 = (pred - target).abs().mean()
        return self.alpha * (1.0 - msssim) + (1.0 - self.alpha) * l1
