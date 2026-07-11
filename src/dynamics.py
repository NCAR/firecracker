"""
dynamics.py

Latent transition (dynamics) head for the Firecracker world model.

The autoencoder (strided_autoencoder.ConvAutoencoder) maps an observation to a unit-norm
latent (normalize_latent=True projects it onto the unit hypersphere). The world model adds a
one-step predictor over that latent: given z_t, predict the next latent z_{t+1}. Trained with
a cosine-similarity objective against a stop-gradient target (SimSiam/BYOL-style), the L2
normalization on both the encoder output and this head's output -- together with the decoder
anchoring the latent to stay reconstructable -- keeps the representation from collapsing without
any explicit variance/covariance regularizer.

The head is a residual MLP: it predicts a delta added to z_t, then re-normalizes, so it starts
near the identity (a good prior: consecutive stored frames are close on the sphere) and only has
to learn the motion off it.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentTransition(nn.Module):
    """Residual MLP predictor z_t -> z_{t+1} on the unit hypersphere.

    Args:
        latent_dim:       width of the latent vector (matches the autoencoder's latent_dim).
        hidden_dim:       width of the hidden layers (defaults to 1024).
        depth:            number of hidden Linear+BN+ReLU blocks before the output projection.
        residual:         predict a delta added to z_t (True) rather than z_{t+1} outright.
        normalize_output: L2-normalize the prediction so it lies on the same unit sphere as the
                          encoder's latents, making cosine similarity the natural training metric.
    """

    def __init__(
        self,
        latent_dim: int,
        *,
        hidden_dim: int | None = None,
        depth: int = 2,
        residual: bool = True,
        normalize_output: bool = True,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        hidden_dim = hidden_dim if hidden_dim is not None else 1024
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim          # resolved width (record this, not the raw None default)
        self.residual = bool(residual)
        self.normalize_output = bool(normalize_output)

        layers: list[nn.Module] = []
        width = latent_dim
        for _ in range(depth):
            layers += [nn.Linear(width, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True)]
            width = hidden_dim
        layers.append(nn.Linear(width, latent_dim))       # output projection back to latent width
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Predict the next latent from `z` (B x latent_dim), on the unit sphere if normalized."""
        out = self.net(z)
        if self.residual:
            out = z + out
        if self.normalize_output:
            out = F.normalize(out, dim=-1)
        return out
