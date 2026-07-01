"""
autoencoder.py

A convolutional autoencoder in PyTorch.

The model is built from small, named building blocks so the architecture is legible at a
glance. Down/upsampling is done with pixel-unshuffle/shuffle (space<->depth reshapes) rather
than strided convs, which keeps the operation lossless/artifact-free and lets the adjacent
convs do all the learning:

    DownBlock : 3x3 (same) conv -> PixelUnshuffle(2) -> 1x1 conv   (H/W halve, channels set)
    UpBlock   : 1x1 conv -> PixelShuffle(2) -> 3x3 (same) conv     (H/W double, channels set)

The encoder is a stack of DownBlocks that halve H/W and fold detail into channels at every
stage, ending at a fully-convolutional spatial latent (no flatten, no dense bottleneck); the
decoder mirrors it with a stack of UpBlocks back to the original `C x N x N` shape. This is a
baseline for compressing the world-model observation stack (see
`Simulation.build_observation` / `obs_channel_names`).

With the default eight stages on a 256x256 input the latent is a 1x1x1024 feature map, fed
straight into the decoder. The grid size `N` must be divisible by `2 ** len(channels)` so every
shuffle stage lands on an integer spatial size; choose `channels` so the final spatial size is
whatever latent footprint you want (1x1 for a pure vector latent).

Quick use:

    import torch
    from autoencoder import ConvAutoencoder

    model = ConvAutoencoder(in_channels=5, grid_size=256,
                            channels=(8, 16, 32, 64, 128, 256, 512, 1024))
    x = torch.randn(8, 5, 256, 256)       # a B x C x N x N batch
    x_hat, z = model(x)                   # reconstruction and B x 1024 x 1 x 1 latent map
    loss = torch.nn.functional.mse_loss(x_hat, x)
"""

from typing import Sequence

import torch
from torch import nn


class DownBlock(nn.Module):
    """3x3 (same) conv, then PixelUnshuffle(2) to halve H/W (folding detail into channels),
    then a 1x1 conv to set the output width. Each conv is followed by a BatchNorm2d
    (conv -> norm -> activation)."""

    def __init__(self, in_ch: int, out_ch: int, activation: type[nn.Module] = nn.ReLU) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            activation(),
            nn.PixelUnshuffle(2),  # out_ch -> out_ch * 4, H/W halved
            nn.Conv2d(out_ch * 4, out_ch, kernel_size=1),
            nn.BatchNorm2d(out_ch),
            activation(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UpBlock(nn.Module):
    """1x1 conv that expands to out_ch*4 channels, then PixelShuffle(2) to double H/W
    (unfolding channels into space), then a 3x3 (same) conv. Each conv is followed by a
    BatchNorm2d (conv -> norm -> activation). The final block omits its trailing norm and
    activation so it emits the raw reconstruction."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        activation: type[nn.Module] = nn.ReLU,
        final: bool = False,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, out_ch * 4, kernel_size=1),
            nn.BatchNorm2d(out_ch * 4),
            activation(),
            nn.PixelShuffle(2),  # out_ch * 4 -> out_ch, H/W doubled
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


class ConvAutoencoder(nn.Module):
    """A symmetric convolutional autoencoder.

    Args:
        in_channels: number of input channels (e.g. len(obs_channel_names(...))).
        grid_size:   spatial size N of the square input; must be divisible by 2**len(channels).
        channels:    encoder channel widths, one per downsampling stage. The last entry is the
                     latent channel count; len(channels) sets how far H/W are halved.
        activation:  activation module class used between conv layers.

    The latent is the encoder's output feature map, of shape
    `(channels[-1], N // 2**len(channels), N // 2**len(channels))`. With the defaults
    (N=256, eight stages) that is `1024 x 1 x 1`, fed straight into the decoder.
    """

    def __init__(
        self,
        in_channels: int = 5,
        grid_size: int = 256,
        channels: Sequence[int] = (8, 16, 32, 64, 128, 256, 512, 1024),
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
        self.latent_channels = channels[-1]
        self.latent_size = grid_size // (2 ** n_stages)
        self.latent_dim = self.latent_channels * self.latent_size * self.latent_size

        self.encoder = Encoder(in_channels, channels, activation)
        self.decoder = Decoder(in_channels, channels, activation)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Map a B x C x N x N batch to its B x latent_channels x h x w latent feature map."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Map a B x latent_channels x h x w latent map back to a B x C x N x N reconstruction."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (reconstruction, latent)."""
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z
