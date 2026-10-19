"""
DocROI-Former: Intra-Document Font Contrastive Embedding Head
============================================================
Projects character tokens from Stage 1 into an L2-normalized typographical
latent space Z_font in R^D (D=128).

Supervised via InfoNCE loss to enforce that authentic characters printed on the
same card cluster tightly together, while digitally spliced characters from
Canva, Figma, or Photoshop register as out-of-distribution typographical anomalies.

Design Pattern: Self-Supervised Projection Head (SOLID: Single Responsibility)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FontContrastiveHead(nn.Module):
    """
    Typographical Representation Head.
    Maps character-level spatial tokens into a normalized unit hypersphere.
    """

    def __init__(self, in_channels: int = 64, embedding_dim: int = 128):
        super().__init__()
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim

        # 2-layer MLP projection head (SimCLR / InfoNCE style)
        self.projector = nn.Sequential(
            nn.Linear(in_channels, in_channels * 2),
            nn.LayerNorm(in_channels * 2),
            nn.GELU(),
            nn.Linear(in_channels * 2, embedding_dim)
        )

    def forward(self, f_stage1: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f_stage1 (torch.Tensor): Stage 1 spatial feature map [B, C, H, W] (stride 4).
        Returns:
            torch.Tensor: L2-normalized character token embeddings [B, N, embedding_dim]
                          where N = H * W.
        """
        B, C, H, W = f_stage1.shape
        # Flatten spatial dimensions: [B, N, C]
        tokens = f_stage1.flatten(2).transpose(1, 2)

        # Project to typographical embedding space
        z = self.projector(tokens)

        # L2 normalize onto unit hypersphere (critical for cosine similarity / InfoNCE)
        z_normalized = F.normalize(z, p=2, dim=-1)
        return z_normalized
