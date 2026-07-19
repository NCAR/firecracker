"""
strided_autoencoder.py

A convolutional autoencoder in PyTorch for the Firecracker world-model observation stack.

The stage does its feature learning with a residual pair of 3x3 (same) convs and then resamples;
the resample and the channel change are decoupled:

    DownBlock : residual double-conv -> 2x2 average pool               (H/W halve, channels set by conv)
    UpBlock   : 2x2 nearest-neighbor upsample -> residual double-conv  (H/W double, channels set by conv)

A parameter-free 2x2 average pool halves H/W on the way down (an anti-aliasing low-pass before
subsampling), and nearest-neighbor upsampling mirrors it on the way up. The decoder's upsampling is
a "resize-convolution" (nearest-neighbor upsample then a unit-stride conv), which keeps every
decoder kernel at unit stride and so cannot produce the checkerboard artifacts that overlapping
stride-2 transposed-conv kernels are prone to. The per-stage skip connection gives gradients an
identity path around every stage so depth can be stacked without the gradient vanishing; the
final UpBlock is a plain resize-conv reconstruction head (no BN/act/residual) that emits raw
output.

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
    from strided_autoencoder import ConvAutoencoder

    model = ConvAutoencoder(in_channels=5, grid_size=256,
                            channels=(32, 64, 128, 256, 256, 256), latent_dim=512)
    x = torch.randn(8, 5, 256, 256)       # a B x C x N x N batch
    x_hat, z = model(x)                   # reconstruction and B x 512 latent vector
    loss = torch.nn.functional.mse_loss(x_hat, x)
"""

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn


class ResidualBlock(nn.Module):
    """A single post-activation residual double-conv (no resample).

    Computes `act(F(x) + shortcut(x))` with `F = conv -> BN -> act -> conv -> BN`. It is the
    resample-free unit that DownBlock/UpBlock stack when a stage carries more than one residual
    block (`num_blocks > 1`): the stage's first block changes channel width and lives inline on the
    Down/UpBlock (so single-block stages keep their original parameter names), and every *extra*
    block is one of these at the stage's output width (in_ch == out_ch, so the shortcut is a plain
    identity). The second BN's gamma is zero-initialised so the block starts as its shortcut (see
    DownBlock's note on why the activation must sit after the add, not end the branch)."""

    def __init__(self, in_ch: int, out_ch: int, activation: type[nn.Module] = nn.ReLU) -> None:
        super().__init__()
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
        h = self.act(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return self.act(h + self.shortcut(x))


class DownBlock(nn.Module):
    """`num_blocks` residual double-convs, then a 2x2 average pool that halves H/W.

    The block computes act(F(x) + shortcut(x)) and then pools, where F is the two-conv residual
    branch: conv -> BN -> act -> conv -> BN. The activation is applied AFTER the addition
    (post-activation ResNet ordering), so F's last op is the (linear) second BN and the activation
    sits on the residual sum. This ordering matters for the zero-init trick below: the second BN's
    weight (gamma) is zero-initialised so F(x) = 0 and the block starts as the shortcut alone (a
    deep stack thus begins as its shallow self and grows depth as training needs it). Because F
    ends in a *linear* BN, zero-init zeroes F's output value but not the gradient flowing into F --
    the branch is dormant in value yet fully alive in gradient, so it wakes up over the first few
    steps. (Do NOT move the activation to the end of F: act(BN) on a zero-init BN is act(0), which
    parks the branch on the dead side of the ReLU kink -- zero derivative -- and the whole conv
    branch never receives gradient. That variant was tried and stalled the model.) The skip gives
    gradients a direct path around the stage, so gradient magnitude does not decay with depth. The
    shortcut is a 1x1 projection when the stage changes channel width (else a plain identity).

    The 2x2 average pool is a parameter-free fixed low-pass, applied to the activated residual sum,
    so the identity/gradient path above is unaffected by it."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        activation: type[nn.Module] = nn.ReLU,
        num_blocks: int = 1,
    ) -> None:
        super().__init__()
        if num_blocks < 1:
            raise ValueError(f"num_blocks must be >= 1, got {num_blocks}")
        # First residual block: it carries the stage's channel change (in_ch -> out_ch). Its layers
        # stay inline (not wrapped in a ResidualBlock) so a single-block stage keeps the original
        # parameter names and older checkpoints load unchanged.
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
        # Extra same-width residual blocks (empty for num_blocks=1, so no extra state_dict keys).
        self.extra = nn.ModuleList(
            ResidualBlock(out_ch, out_ch, activation) for _ in range(num_blocks - 1)
        )
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)  # parameter-free H/W halving
        nn.init.zeros_(self.bn2.weight)  # block starts as identity (shortcut only)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        h = self.act(h + self.shortcut(x))
        for blk in self.extra:
            h = blk(h)
        return self.pool(h)


class UpBlock(nn.Module):
    """2x2 nearest-neighbor upsample that doubles H/W, then a residual double-conv stage that
    mirrors DownBlock (minus the pool): upsample -> act(F(x) + shortcut(x)), where F is two 3x3
    (same) convs each with BatchNorm (conv -> BN -> act -> conv -> BN) and the shortcut is a 1x1
    projection (or identity) around them. The activation is applied after the addition (post-
    activation ordering), so F ends in a linear BN whose gamma is zero-initialised: F(x) = 0 at
    init and gradient still flows into F, matching DownBlock (see its note on why the activation
    must not end the branch).

    The final block is the raw reconstruction head: a single resize-conv with no norm,
    activation, or residual, so it can emit unbounded (including negative) reconstruction values
    -- a residual ReLU head would clamp them and a skip would not match the input-channel width."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        activation: type[nn.Module] = nn.ReLU,
        final: bool = False,
        num_blocks: int = 1,
    ) -> None:
        super().__init__()
        if num_blocks < 1:
            raise ValueError(f"num_blocks must be >= 1, got {num_blocks}")
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
        # Extra same-width residual blocks (empty for num_blocks=1, so no extra state_dict keys).
        self.extra = nn.ModuleList(
            ResidualBlock(out_ch, out_ch, activation) for _ in range(num_blocks - 1)
        )
        nn.init.zeros_(self.bn2.weight)  # block starts as identity (shortcut only)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if self.final:
            return self.conv(x)
        h = self.act(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        h = self.act(h + self.shortcut(x))
        for blk in self.extra:
            h = blk(h)
        return h


class Encoder(nn.Module):
    """Stack of DownBlocks producing a spatial latent feature map (no flatten/projection)."""

    def __init__(
        self,
        in_channels: int,
        channels: Sequence[int],
        activation: type[nn.Module] = nn.ReLU,
        blocks_per_stage: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        bps = list(blocks_per_stage) if blocks_per_stage is not None else [1] * len(channels)
        widths = [in_channels, *channels]
        self.blocks = nn.Sequential(
            *(DownBlock(widths[i], widths[i + 1], activation, num_blocks=bps[i])
              for i in range(len(channels)))
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
        blocks_per_stage: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        rev = list(reversed(channels))
        out_widths = [*rev[1:], in_channels]
        # Mirror the encoder's per-stage depth: decoder stage i corresponds to the encoder stage at
        # the same resolution/width, i.e. the reversed blocks_per_stage. The final reconstruction
        # head is a single plain conv, so its block count is ignored.
        bps = list(blocks_per_stage) if blocks_per_stage is not None else [1] * len(channels)
        rev_bps = list(reversed(bps))
        self.blocks = nn.Sequential(
            *(
                UpBlock(rev[i], out_widths[i], activation, final=(i == len(rev) - 1),
                        num_blocks=rev_bps[i])
                for i in range(len(rev))
            )
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.blocks(z)


class ConvAutoencoder(nn.Module):
    """A symmetric convolutional autoencoder: 2x2 avg-pool encoder / resize-conv decoder.

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
        bottleneck_channels: int | None = None,
        latent_bn: bool = False,
        blocks_per_stage: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        n_stages = len(channels)
        if grid_size % (2 ** n_stages) != 0:
            raise ValueError(
                f"grid_size={grid_size} must be divisible by 2**len(channels)={2 ** n_stages}"
            )
        # Per-stage residual depth (one entry per channel width). None = one block per stage (the
        # original architecture). The decoder mirrors this list (reversed) so the net stays
        # symmetric. Extra blocks are same-width residual units inserted before each pool.
        if blocks_per_stage is not None:
            blocks_per_stage = tuple(int(b) for b in blocks_per_stage)
            if len(blocks_per_stage) != n_stages:
                raise ValueError(
                    f"blocks_per_stage must have one entry per stage (len(channels)={n_stages}), "
                    f"got {len(blocks_per_stage)}"
                )
            if any(b < 1 for b in blocks_per_stage):
                raise ValueError(f"blocks_per_stage entries must be >= 1, got {blocks_per_stage}")
        self.blocks_per_stage = blocks_per_stage or (1,) * n_stages

        self.in_channels = in_channels
        self.grid_size = grid_size
        self.channels = tuple(channels)
        self.conv_channels = channels[-1]
        self.conv_spatial = grid_size // (2 ** n_stages)
        self.latent_dim = latent_dim
        self.normalize_latent = normalize_latent
        # Optional 1x1-conv channel bottleneck around the latent: compress the encoder's
        # conv_channels x S x S map to bottleneck_channels x S x S with a 1x1 conv (a per-pixel
        # channel projection, shared across all S*S positions) BEFORE flattening, and expand back
        # with a mirrored 1x1 conv AFTER the latent. This preserves the S x S spatial layout while
        # shrinking the flattened width (and hence the two dense latent projections) by
        # conv_channels/bottleneck_channels. flat_channels is what gets flattened/reshaped. Each 1x1
        # conv is followed by BatchNorm (no activation), so the projections stay linear in
        # representation: the encoder-side BN normalizes the channel stats feeding the dense latent
        # layer, and the decoder-side BN mirrors it by normalizing the input to the decoder's first
        # conv (a different tensor than that conv's own BN, so not redundant with it).
        self.bottleneck_channels = bottleneck_channels
        self.flat_channels = bottleneck_channels if bottleneck_channels is not None else self.conv_channels
        self.flat_dim = self.flat_channels * self.conv_spatial * self.conv_spatial

        self.encoder = Encoder(in_channels, channels, activation, self.blocks_per_stage)
        if bottleneck_channels is not None:
            self.enc_project = nn.Sequential(
                nn.Conv2d(self.conv_channels, bottleneck_channels, kernel_size=1),
                nn.BatchNorm2d(bottleneck_channels),
            )
            self.dec_project = nn.Sequential(
                nn.Conv2d(bottleneck_channels, self.conv_channels, kernel_size=1),
                nn.BatchNorm2d(self.conv_channels),
            )
        self.to_latent = nn.Linear(self.flat_dim, latent_dim)
        # Optional BatchNorm on the latent (SimSiam/BYOL projector-output BN). It centers each latent
        # dim across the batch, which makes a constant embedding impossible -- the direction-collapse
        # (all z parallel, participation_ratio -> 0) that the L2-norm alone cannot prevent. affine=False
        # (as in SimSiam's output projector) so the net can't learn a scale/shift that re-enables it.
        self.latent_bn = latent_bn
        if latent_bn:
            self.latent_norm = nn.BatchNorm1d(latent_dim, affine=False)
        self.from_latent = nn.Linear(latent_dim, self.flat_dim)
        self.decoder = Decoder(in_channels, channels, activation, self.blocks_per_stage)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Map a B x C x N x N batch to its B x latent_dim latent vector.

        With `normalize_latent`, the vector is L2-normalized to unit magnitude (projected onto the
        unit hypersphere); the decoder's `from_latent` layer learns to rescale it.
        """
        h = self.encoder(x)
        if self.bottleneck_channels is not None:
            h = self.enc_project(h)  # conv_channels -> bottleneck_channels (1x1), spatial preserved
        z = self.to_latent(h.flatten(1))
        if self.latent_bn:
            z = self.latent_norm(z)     # batch-center each dim (anti-collapse) before the L2 projection
        if self.normalize_latent:
            z = F.normalize(z, dim=-1)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Map a B x latent_dim latent vector back to a B x C x N x N reconstruction."""
        h = self.from_latent(z)
        h = h.view(-1, self.flat_channels, self.conv_spatial, self.conv_spatial)
        if self.bottleneck_channels is not None:
            h = self.dec_project(h)  # bottleneck_channels -> conv_channels (1x1) before the decoder
        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (reconstruction, latent)."""
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z
