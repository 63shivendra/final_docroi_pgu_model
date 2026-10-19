"""
DocROI-Former: Bayar Constrained Convolutional Operator
======================================================
Implements the mathematically constrained convolutional layer from Bayar & Stamm
(Information Forensics and Security, 2018).

The constraint forces the kernel to compute prediction error residuals rather than
semantic image representations by enforcing:
    sum_{i, j != 0} w_{i, j} = -w_{0, 0} = 1.0 (with central weight set to -1)

Design Pattern: Custom Convolutional Operator (SOLID: Single Responsibility)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BayarConv2d(nn.Module):
    """
    Bayar Constrained Convolution for forensic prediction residual extraction.

    Mathematically enforces zero-sum constraints on filter weights during
    every forward pass, stripping away image semantic content and extracting
    micro-manipulation noise and resampling traces.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        kernel_size: int = 5,
        stride: int = 1,
        padding: int = 2
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd for symmetric centering."
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Unconstrained learnable weights
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        nn.init.xavier_uniform_(self.weight)

        # Precompute central mask index
        self.center_idx = kernel_size // 2

    def _get_constrained_weights(self) -> torch.Tensor:
        """
        Normalizes weights so that off-center elements sum to 1.0,
        and sets the central kernel weight to -1.0.
        """
        w = self.weight.clone()

        # Zero out center temporarily to compute off-center sum
        w[:, :, self.center_idx, self.center_idx] = 0.0

        # Sum of all off-center weights across spatial dimensions
        sum_off_center = w.sum(dim=(2, 3), keepdim=True)
        # Sign-preserving stable denominator: prevents zero-sum division and FP16 overflow
        sign = torch.sign(sum_off_center)
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        denom = torch.where(sum_off_center.abs() < 1e-4, sign * 1e-4, sum_off_center)

        # Normalize off-center weights with clamp
        w = torch.clamp(w / denom, min=-30.0, max=30.0)

        # Enforce center element strictly to -1.0
        w[:, :, self.center_idx, self.center_idx] = -1.0
        return w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward convolution using dynamically constrained kernels.
        Args:
            x (torch.Tensor): Input document tensor [B, C, H, W].
        Returns:
            torch.Tensor: High-pass prediction error residual map [B, out_channels, H, W].
        """
        constrained_weight = self._get_constrained_weights()
        return F.conv2d(x, constrained_weight, stride=self.stride, padding=self.padding)
