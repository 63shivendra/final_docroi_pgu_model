"""
DocROI-Former: Global Document Decision Head
============================================
Outputs a scalar global fraud probability P_doc in [0.0, 1.0] for instant
automated Straight-Through Processing (STP) and enterprise triage.

Design Pattern: Decoupled Task Head (SOLID: Single Responsibility)
"""

import torch
import torch.nn as nn


class GlobalDocumentClassifierHead(nn.Module):
    """
    Decoupled Global Document Fraud Classifier.
    Fuses deep spatial semantic tokens and deep frequency noise anomalies
    to produce a calibrated global probability of document tampering.
    """

    def __init__(self, in_features: int = 512 + 512, hidden_dim: int = 256):
        super().__init__()
        self.in_features = in_features

        # Global average + max pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)

        # Multi-layer classification perceptron with dropout
        self.mlp = nn.Sequential(
            nn.Linear(in_features * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, f_spatial_deep: torch.Tensor, f_freq_deep: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f_spatial_deep (torch.Tensor): Deepest spatial features [B, C_s, H_d, W_d].
            f_freq_deep (torch.Tensor): Deepest frequency features [B, C_f, H_d, W_d].
        Returns:
            torch.Tensor: Global document classification logits [B, 1].
        """
        # Pool spatial and frequency streams
        s_avg = self.gap(f_spatial_deep).flatten(1)
        s_max = self.gmp(f_spatial_deep).flatten(1)

        f_avg = self.gap(f_freq_deep).flatten(1)
        f_max = self.gmp(f_freq_deep).flatten(1)

        pooled = torch.cat([s_avg, s_max, f_avg, f_max], dim=1)
        return self.mlp(pooled)
