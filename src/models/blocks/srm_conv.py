"""
DocROI-Former: Spatial Rich Model (SRM) High-Pass Filter Banks
=============================================================
Implements deterministic, fixed SRM filters derived from Steganalysis and
Digital Image Forensics (Fridrich & Kodovsky, IEEE T-IFS 2012).

Filters compute 1st-order, 2nd-order, and 3rd-order spatial finite difference
residuals, highly sensitive to boundary pasting, cloning, and local blur anomalies.

Design Pattern: Fixed Feature Extractor (SOLID: Single Responsibility)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SRMConv2d(nn.Module):
    """
    Fixed Spatial Rich Model (SRM) Filter Bank.
    Constructs 3 non-trainable, mathematically fixed high-pass kernels
    operating across the RGB channels.
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = 5

        # Define canonical SRM high-pass kernels (normalized)
        # 1. 1st order derivative (KB filter)
        k1 = np.array([
            [ 0,  0,  0,  0,  0],
            [ 0, -1,  2, -1,  0],
            [ 0,  2, -4,  2,  0],
            [ 0, -1,  2, -1,  0],
            [ 0,  0,  0,  0,  0]
        ], dtype=np.float32) / 4.0

        # 2. 2nd order derivative (Laplacian KV filter)
        k2 = np.array([
            [-1,  2, -2,  2, -1],
            [ 2, -6,  8, -6,  2],
            [-2,  8, -12, 8, -2],
            [ 2, -6,  8, -6,  2],
            [-1,  2, -2,  2, -1]
        ], dtype=np.float32) / 12.0

        # 3. 3rd order derivative (Square edge filter)
        k3 = np.array([
            [ 0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0],
            [ 0,  1, -2,  1,  0],
            [ 0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0]
        ], dtype=np.float32) / 2.0

        # Stack into [3, 1, 5, 5]
        kernels = np.stack([k1, k2, k3], axis=0)[:, np.newaxis, :, :]

        # Tile across in_channels (e.g. RGB) -> [3, 3, 5, 5]
        kernels = np.tile(kernels, (1, in_channels, 1, 1))

        # Register as fixed buffer (non-trainable)
        self.register_buffer("srm_weights", torch.from_numpy(kernels).float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward convolution with fixed SRM filter bank.
        Args:
            x (torch.Tensor): Input document RGB tensor [B, 3, H, W].
        Returns:
            torch.Tensor: SRM residual maps [B, 3, H, W].
        """
        return F.conv2d(x, self.srm_weights, padding=2)
