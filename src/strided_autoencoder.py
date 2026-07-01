"""
strided_autoencoder.py

A convolutional autoencoder in PyTorch, a sibling to `autoencoder.ConvAutoencoder`.

This variant down/upsamples with strided convolutions rather than pixel-unshuffle/shuffle:

    DownBlock : 3x3 (same) conv -> 2x2 stride-2 conv             (H/W halve, channels set)
    UpBlock   : 2x2 stride-2 transposed conv -> 3x3 (same) conv  (H/W double, channels set)

Where `autoencoder.ConvAutoencoder` keeps resampling lossless (a space<->depth reshape) and
lets a 1x1 conv pick the channels, here the 2x2 stride-2 conv *is* the resampler: it halves
H/W and sets the output width in one learned, parametric step (and the transposed conv mirrors
it on the way up). A 2x2 kernel at stride 2 tiles the input without overlap, so it is the
strided analog of the shuffle reshape and avoids the checkerboard artifacts that overlapping
transposed-conv kernels are prone to. The accompanying 3x3 (same) conv does the per-stage
feature learning, exactly as in the shuffle variant.

The encoder is a stack of DownBlocks that halve H/W and grow channels at every stage,
producing a spatial feature map that is then flattened and projected by a dense layer to a
`latent_dim` vector; the decoder mirrors this with a dense layer back to the flattened feature
map, reshapes it, and applies a stack of UpBlocks back to the original `C x N x N` shape. This
is a baseline for compressing the world-model observation stack (see
`Simulation.build_observation` / `obs_channel_names`).

With the default six stages on a 256x256 input the encoder produces a 512x4x4 feature map
(channels x height x width), which flattens to 8192 features and projects down to a 256-d
latent vector. The grid size `N` must be divisible by `2 ** len(channels)` so every strided
stage lands on an integer spatial size.

Quick use:

    import torch
    from strided_autoencoder import StridedConvAutoencoder

    model = StridedConvAutoencoder(in_channels=5, grid_size=256,
                                   channels=(16, 32, 64, 128, 256, 512), latent_dim=256)
    x = torch.randn(8, 5, 256, 256)       # a B x C x N x N batch
    x_hat, z = model(x)                   # reconstruction and B x 256 latent vector
    loss = torch.nn.functional.mse_loss(x_hat, x)
"""

from typing import Sequence

import torch
from torch import nn


class DownBlock(nn.Module):
    """3x3 (same) conv for per-stage feature learning, then a 2x2 stride-2 conv that halves
    H/W and sets the output width in one step. Each conv is followed by a BatchNorm2d
    (conv -> norm -> activation)."""

    def __init__(self, in_ch: int, out_ch: int, activation: type[nn.Module] = nn.ReLU) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            activation(),
            nn.Conv2d(out_ch, out_ch, kernel_size=2, stride=2),  # H/W halved
            nn.BatchNorm2d(out_ch),
            activation(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UpBlock(nn.Module):
    """2x2 stride-2 transposed conv that doubles H/W and sets the output width, then a 3x3
    (same) conv. Each conv is followed by a BatchNorm2d (conv -> norm -> activation). The final
    block omits its trailing norm and activation so it emits the raw reconstruction."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        activation: type[nn.Module] = nn.ReLU,
        final: bool = False,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),  # H/W doubled
            nn.BatchNorm2d(out_ch),
            activation(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
        ]
        if not final:
            layers += [nn.BatchNorm2d(out_ch), activation()]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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
    """A symmetric convolutional autoencoder using strided convs for resampling.

    A drop-in alternative to `autoencoder.ConvAutoencoder` with the same constructor signature
    and (reconstruction, latent) forward contract.

    Args:
        in_channels: number of input channels (e.g. len(obs_channel_names(...))).
        grid_size:   spatial size N of the square input; must be divisible by 2**len(channels).
        channels:    encoder channel widths, one per downsampling stage. The last entry is the
                     conv channel count before flattening; len(channels) sets how far H/W halve.
        latent_dim:  width of the dense latent vector the flattened feature map projects to.
        activation:  activation module class used between conv layers.

    The encoder produces a `(channels[-1], N // 2**len(channels), N // 2**len(channels))`
    feature map, which is flattened and projected by a linear layer to a `latent_dim` vector.
    With the defaults (N=256, six stages, latent_dim=256) the feature map is `512 x 4 x 4`,
    flattening to 8192 features before the projection to a 256-d latent.
    """

    def __init__(
        self,
        in_channels: int = 5,
        grid_size: int = 256,
        channels: Sequence[int] = (16, 32, 64, 128, 256, 512),
        latent_dim: int = 256,
        activation: type[nn.Module] = nn.ReLU,
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

        self.encoder = Encoder(in_channels, channels, activation)
        self.to_latent = nn.Linear(self.flat_dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, self.flat_dim)
        self.decoder = Decoder(in_channels, channels, activation)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Map a B x C x N x N batch to its B x latent_dim latent vector."""
        h = self.encoder(x)
        return self.to_latent(h.flatten(1))

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
