"""
autoencoder.py

A convolutional autoencoder in PyTorch.

The model is built from small, named building blocks so the architecture is legible at a
glance. Down/upsampling is done with pixel-unshuffle/shuffle (space<->depth reshapes) rather
than strided convs, which keeps the operation lossless/artifact-free and lets the adjacent
convs do all the learning:

    DownBlock : 3x3 (same) conv -> PixelUnshuffle(2) -> 1x1 conv   (H/W halve, channels set)
    UpBlock   : 1x1 conv -> PixelShuffle(2) -> 3x3 (same) conv     (H/W double, channels set)

The encoder is a stack of DownBlocks feeding a linear bottleneck (the latent code); the
decoder mirrors it with a stack of UpBlocks back to the original `C x N x N` shape. This is a
baseline for compressing the world-model observation stack (see
`Simulation.build_observation` / `OBS_CHANNELS`).

The grid size `N` must be divisible by `2 ** len(channels)` so every shuffle stage lands on
an integer spatial size.

Quick use:

    import torch
    from autoencoder import ConvAutoencoder

    model = ConvAutoencoder(in_channels=3, grid_size=256, channels=(32, 64, 128), latent_dim=256)
    x = torch.randn(8, 3, 256, 256)       # a B x C x N x N batch
    x_hat, z = model(x)                   # reconstruction and latent code
    loss = torch.nn.functional.mse_loss(x_hat, x)
"""

from typing import Sequence

import torch
from torch import nn


class DownBlock(nn.Module):
    """3x3 (same) conv, then PixelUnshuffle(2) to halve H/W (folding detail into channels),
    then a 1x1 conv to set the output width."""

    def __init__(self, in_ch: int, out_ch: int, activation: type[nn.Module] = nn.ReLU) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            activation(),
            nn.PixelUnshuffle(2),  # out_ch -> out_ch * 4, H/W halved
            nn.Conv2d(out_ch * 4, out_ch, kernel_size=1),
            activation(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UpBlock(nn.Module):
    """1x1 conv that expands to out_ch*4 channels, then PixelShuffle(2) to double H/W
    (unfolding channels into space), then a 3x3 (same) conv. The final block omits its
    trailing activation."""

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
            activation(),
            nn.PixelShuffle(2),  # out_ch * 4 -> out_ch, H/W doubled
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
        ]
        if not final:
            layers.append(activation())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Encoder(nn.Module):
    """Stack of DownBlocks followed by a flatten + linear projection to the latent code."""

    def __init__(
        self,
        in_channels: int,
        channels: Sequence[int],
        bottleneck_numel: int,
        latent_dim: int,
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        widths = [in_channels, *channels]
        self.blocks = nn.Sequential(
            *(DownBlock(widths[i], widths[i + 1], activation) for i in range(len(channels)))
        )
        self.to_latent = nn.Sequential(nn.Flatten(), nn.Linear(bottleneck_numel, latent_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.to_latent(self.blocks(x))


class Decoder(nn.Module):
    """Linear projection from the latent code, reshaped to the bottleneck feature map, then a
    stack of UpBlocks back to the original image."""

    def __init__(
        self,
        in_channels: int,
        channels: Sequence[int],
        bottleneck_ch: int,
        bottleneck_size: int,
        latent_dim: int,
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        self._bottleneck_ch = bottleneck_ch
        self._bottleneck_size = bottleneck_size

        self.from_latent = nn.Linear(latent_dim, bottleneck_ch * bottleneck_size * bottleneck_size)

        rev = list(reversed(channels))
        out_widths = [*rev[1:], in_channels]
        self.blocks = nn.Sequential(
            *(
                UpBlock(rev[i], out_widths[i], activation, final=(i == len(rev) - 1))
                for i in range(len(rev))
            )
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.from_latent(z)
        h = h.view(-1, self._bottleneck_ch, self._bottleneck_size, self._bottleneck_size)
        return self.blocks(h)


class ConvAutoencoder(nn.Module):
    """A symmetric convolutional autoencoder.

    Args:
        in_channels: number of input channels (e.g. len(OBS_CHANNELS)).
        grid_size:   spatial size N of the square input; must be divisible by 2**len(channels).
        channels:    encoder channel widths, one per downsampling stage.
        latent_dim:  size of the bottleneck (the latent code).
        activation:  activation module class used between conv layers.
    """

    def __init__(
        self,
        in_channels: int = 5,
        grid_size: int = 256,
        channels: Sequence[int] = (32, 64, 128),
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
        self.latent_dim = latent_dim
        bottleneck_size = grid_size // (2 ** n_stages)
        bottleneck_ch = channels[-1]
        bottleneck_numel = bottleneck_ch * bottleneck_size * bottleneck_size

        self.encoder = Encoder(in_channels, channels, bottleneck_numel, latent_dim, activation)
        self.decoder = Decoder(
            in_channels, channels, bottleneck_ch, bottleneck_size, latent_dim, activation
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Map a B x C x N x N batch to its B x latent_dim code."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Map a B x latent_dim code back to a B x C x N x N reconstruction."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (reconstruction, latent)."""
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z
