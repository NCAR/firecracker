"""
vae.py

A convolutional variational autoencoder (VAE) in PyTorch, the probabilistic sibling of
`autoencoder.ConvAutoencoder`.

The architecture is identical to `ConvAutoencoder` -- the same pixel-unshuffle/shuffle
(space<->depth) resampling blocks -- so the only change is the bottleneck. Where the plain
autoencoder projects the flattened encoder feature map to a single deterministic `latent_dim`
vector, the VAE projects it to a *distribution*: two heads emit a per-dimension mean and
log-variance, a latent is drawn with the reparameterization trick, and the decoder maps that
sample back to the `C x N x N` reconstruction:

    DownBlock : 3x3 (same) conv -> PixelUnshuffle(2) -> 1x1 conv   (H/W halve, channels set)
    UpBlock   : 1x1 conv -> PixelShuffle(2) -> 3x3 (same) conv     (H/W double, channels set)

    encode : feature map -> (mu, logvar)               two Linear heads on the flattened map
    sample : z = mu + exp(0.5 * logvar) * eps          reparameterization (eps ~ N(0, I))
    decode : z -> reconstruction                        mirror of ConvAutoencoder's decoder

Training a VAE minimises reconstruction error *plus* the KL divergence of the approximate
posterior N(mu, sigma^2) from the standard-normal prior, which regularises the latent space
toward a smooth, samplable N(0, I). Use `ConvVAE.kl_divergence(mu, logvar)` for that term:

    x_hat, z, mu, logvar = model(x)
    recon = torch.nn.functional.mse_loss(x_hat, x)
    kl = model.kl_divergence(mu, logvar)               # per-sample mean, summed over latent dims
    loss = recon + beta * kl                            # beta weights the KL (beta-VAE)

At inference the mean is used directly (no sampling) whenever the module is in eval mode, so
`model.eval()` gives a deterministic encode/decode. The grid size `N` must be divisible by
`2 ** len(channels)` so every shuffle stage lands on an integer spatial size.

Quick use:

    import torch
    from vae import ConvVAE

    model = ConvVAE(in_channels=5, grid_size=256,
                    channels=(16, 32, 64, 128, 256, 512), latent_dim=256)
    x = torch.randn(8, 5, 256, 256)              # a B x C x N x N batch
    x_hat, z, mu, logvar = model(x)              # reconstruction, sampled latent, posterior params
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


class ConvVAE(nn.Module):
    """A symmetric convolutional variational autoencoder with a dense latent bottleneck.

    Shares `autoencoder.ConvAutoencoder`'s constructor signature and encoder/decoder, but the
    bottleneck is probabilistic: the flattened encoder feature map is projected by two linear
    heads to a per-dimension mean (`mu`) and log-variance (`logvar`), a latent is drawn with the
    reparameterization trick, and the decoder maps that sample back to the input shape.

    Args:
        in_channels: number of input channels (e.g. len(obs_channel_names(...))).
        grid_size:   spatial size N of the square input; must be divisible by 2**len(channels).
        channels:    encoder channel widths, one per downsampling stage. The last entry is the
                     conv channel count before flattening; len(channels) sets how far H/W halve.
        latent_dim:  width of the latent distribution the flattened feature map projects to.
        activation:  activation module class used between conv layers.

    The encoder produces a `(channels[-1], N // 2**len(channels), N // 2**len(channels))`
    feature map, which is flattened and projected by two linear heads to `latent_dim`-wide `mu`
    and `logvar` vectors. With the defaults (N=256, six stages, latent_dim=256) the feature map
    is `512 x 4 x 4`, flattening to 8192 features before the projections to a 256-d latent.

    `forward` returns `(reconstruction, z, mu, logvar)`. During training a latent is sampled;
    in eval mode (`model.eval()`) the mean is used directly, giving a deterministic pass.
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
        self.to_mu = nn.Linear(self.flat_dim, latent_dim)
        self.to_logvar = nn.Linear(self.flat_dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, self.flat_dim)
        self.decoder = Decoder(in_channels, channels, activation)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map a B x C x N x N batch to the posterior parameters (mu, logvar), each B x latent_dim."""
        h = self.encoder(x).flatten(1)
        return self.to_mu(h), self.to_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Draw z ~ N(mu, exp(logvar)) with the reparameterization trick while training; return
        the mean `mu` unchanged in eval mode so encode/decode is deterministic at inference."""
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Map a B x latent_dim latent vector back to a B x C x N x N reconstruction."""
        h = self.from_latent(z)
        h = h.view(-1, self.conv_channels, self.conv_spatial, self.conv_spatial)
        return self.decoder(h)

    @staticmethod
    def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """KL(N(mu, sigma^2) || N(0, I)), summed over latent dims and averaged over the batch."""
        kl_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return kl_per_sample.mean()

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (reconstruction, sampled latent, mu, logvar)."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, z, mu, logvar
