"""
strided_autoencoder.py

A convolutional autoencoder in PyTorch for the Firecracker world-model observation stack.

It down/upsamples with parameter-free pooling:

    DownBlock : residual double-conv -> 2x2 average pool               (H/W halve, channels set by conv)
    UpBlock   : 2x2 nearest-neighbor upsample -> residual double-conv  (H/W double, channels set by conv)

The resample and the channel change are decoupled: a residual pair of 3x3 (same) convs
((conv -> BN -> act) x2 with a 1x1/identity skip around them) sets the per-stage width and does
the feature learning, while a parameter-free 2x2 average pool halves H/W (and nearest-neighbor
upsampling mirrors it on the way up). This "resize-convolution" ordering keeps every learned
kernel at unit stride, so it avoids the checkerboard artifacts that overlapping stride-2
transposed-conv kernels are prone to. The per-stage skip connection gives gradients an identity
path around every stage so depth can be stacked without the gradient vanishing; the final
UpBlock is a plain resize-conv reconstruction head (no BN/act/residual) that emits raw output.

The encoder is a stack of DownBlocks that halve H/W and grow channels at every stage,
producing a spatial feature map that is then flattened and projected by a dense layer to a
`latent_dim` vector; the decoder mirrors this with a dense layer back to the flattened feature
map, reshapes it, and applies a stack of UpBlocks back to the original `C x N x N` shape. This
is a baseline for compressing the world-model observation stack (see
`Simulation.build_observation` / `obs_channel_names`).

With the default six stages on a 256x256 input the encoder produces a 256x4x4 feature map
(channels x height x width), which flattens to 4096 features and projects down to a 512-d
latent vector. The grid size `N` must be divisible by `2 ** len(channels)` so every pooling
stage lands on an integer spatial size.

Quick use:

    import torch
    from strided_autoencoder import StridedConvAutoencoder

    model = StridedConvAutoencoder(in_channels=5, grid_size=256,
                                   channels=(32, 64, 128, 256, 256, 256), latent_dim=512)
    x = torch.randn(8, 5, 256, 256)       # a B x C x N x N batch
    x_hat, z = model(x)                   # reconstruction and B x 512 latent vector
    loss = torch.nn.functional.mse_loss(x_hat, x)
"""

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn


class DownBlock(nn.Module):
    """Residual double-conv stage, then a 2x2 average pool that halves H/W.

    Two 3x3 (same) convs each followed by BatchNorm + activation do the per-stage feature
    learning ((conv -> norm -> act) x2), wrapped in a skip connection: the block computes
    act(F(x) + shortcut(x)) before pooling, where F is the two-conv residual branch. The skip
    gives gradients a direct identity path around the stage, so gradient magnitude does not
    decay with depth (the vanishing-gradient mitigation as more stages are stacked). The
    shortcut is a 1x1 projection when the stage changes channel width (else a plain identity).
    The second BN's weight (gamma) is zero-initialised so the block starts as the identity map
    (shortcut only) and learns its residual from there -- a deep stack thus begins as its
    shallow self and grows depth as training needs it."""

    def __init__(self, in_ch: int, out_ch: int, activation: type[nn.Module] = nn.ReLU) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = activation()
        # 1x1 projection so the skip matches the stage's output width (identity when unchanged).
        self.shortcut: nn.Module = (
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1)
            if in_ch != out_ch else nn.Identity()
        )
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)  # H/W halved
        nn.init.zeros_(self.bn2.weight)  # block starts as identity (shortcut only)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        h = self.act(h + self.shortcut(x))
        return self.pool(h)


class UpBlock(nn.Module):
    """2x2 nearest-neighbor upsample that doubles H/W, then a residual double-conv stage that
    mirrors DownBlock (minus the pool): upsample -> act(F(x) + shortcut(x)), where F is two
    3x3 (same) convs each with BatchNorm + activation and the shortcut is a 1x1 projection (or
    identity) around them. The second BN's gamma is zero-initialised so the block starts as the
    identity, matching DownBlock.

    The final block is the raw reconstruction head: a single resize-conv with no norm,
    activation, or residual, so it can emit unbounded (including negative) reconstruction values
    -- a residual ReLU head would clamp them and a skip would not match the input-channel width."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        activation: type[nn.Module] = nn.ReLU,
        final: bool = False,
    ) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")  # H/W doubled
        self.final = final
        if final:
            self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
            return
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = activation()
        self.shortcut: nn.Module = (
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1)
            if in_ch != out_ch else nn.Identity()
        )
        nn.init.zeros_(self.bn2.weight)  # block starts as identity (shortcut only)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if self.final:
            return self.conv(x)
        h = self.act(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return self.act(h + self.shortcut(x))


class Encoder(nn.Module):
    """Stack of DownBlocks producing a spatial latent feature map (no flatten/projection)."""

    def __init__(
        self,
        in_channels: int,
        channels: Sequence[int],
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        widths = [in_channels, *channels]
        self.blocks = nn.Sequential(
            *(DownBlock(widths[i], widths[i + 1], activation) for i in range(len(channels)))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class Decoder(nn.Module):
    """Stack of UpBlocks mapping the spatial latent feature map back to the original image."""

    def __init__(
        self,
        in_channels: int,
        channels: Sequence[int],
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        rev = list(reversed(channels))
        out_widths = [*rev[1:], in_channels]
        self.blocks = nn.Sequential(
            *(
                UpBlock(rev[i], out_widths[i], activation, final=(i == len(rev) - 1))
                for i in range(len(rev))
            )
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.blocks(z)


class StridedConvAutoencoder(nn.Module):
    """A symmetric convolutional autoencoder using pool/upsample resampling.

    Emits a (reconstruction, latent) pair from its forward pass.

    Args:
        in_channels: number of input channels (e.g. len(obs_channel_names(...))).
        grid_size:   spatial size N of the square input; must be divisible by 2**len(channels).
        channels:    encoder channel widths, one per downsampling stage. The last entry is the
                     conv channel count before flattening; len(channels) sets how far H/W halve.
        latent_dim:  width of the dense latent vector the flattened feature map projects to.
        activation:  activation module class used between conv layers.
        normalize_latent: L2-normalize the latent so every vector has unit magnitude (lies on the
                     unit hypersphere). On by default; makes cosine similarity the natural latent
                     metric for the downstream world-model objectives.

    The encoder produces a `(channels[-1], N // 2**len(channels), N // 2**len(channels))`
    feature map, which is flattened and projected by a linear layer to a `latent_dim` vector.
    With the defaults (N=256, six stages, latent_dim=512) the feature map is `256 x 4 x 4`,
    flattening to 4096 features before the projection to a 512-d latent.
    """

    def __init__(
        self,
        in_channels: int = 5,
        grid_size: int = 256,
        channels: Sequence[int] = (32, 64, 128, 256, 256, 256),
        latent_dim: int = 512,
        activation: type[nn.Module] = nn.ReLU,
        normalize_latent: bool = True,
    ) -> None:
        super().__init__()
        n_stages = len(channels)
        if grid_size % (2 ** n_stages) != 0:
            raise ValueError(
                f"grid_size={grid_size} must be divisible by 2**len(channels)={2 ** n_stages}"
            )

        self.in_channels = in_channels
        self.grid_size = grid_size
        self.channels = tuple(channels)
        self.conv_channels = channels[-1]
        self.conv_spatial = grid_size // (2 ** n_stages)
        self.flat_dim = self.conv_channels * self.conv_spatial * self.conv_spatial
        self.latent_dim = latent_dim
        self.normalize_latent = normalize_latent

        self.encoder = Encoder(in_channels, channels, activation)
        self.to_latent = nn.Linear(self.flat_dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, self.flat_dim)
        self.decoder = Decoder(in_channels, channels, activation)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Map a B x C x N x N batch to its B x latent_dim latent vector.

        With `normalize_latent`, the vector is L2-normalized to unit magnitude (projected onto the
        unit hypersphere); the decoder's `from_latent` layer learns to rescale it.
        """
        h = self.encoder(x)
        z = self.to_latent(h.flatten(1))
        if self.normalize_latent:
            z = F.normalize(z, dim=-1)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Map a B x latent_dim latent vector back to a B x C x N x N reconstruction."""
        h = self.from_latent(z)
        h = h.view(-1, self.conv_channels, self.conv_spatial, self.conv_spatial)
        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (reconstruction, latent)."""
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z
