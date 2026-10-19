"""
DocROI-Former: Multi-Scale Frequency Feature Pyramid with Fast 8x8 Block-DCT
=============================================================================
Fuses:
  1. Bayar constrained prediction residuals (micro-resampling anomalies)
  2. Spatial Rich Model (SRM) high-pass filter banks (physical splice edges)
  3. Fast GPU 8x8 Block Discrete Cosine Transform (DCT) stream (JPEG compression
     quantization table and grid alignment mismatches)

Outputs a 4-stage hierarchical frequency feature pyramid matching spatial strides
(4, 8, 16, 32), plus optional native-resolution SRM residuals for PGU edge snapping.

Design Pattern: Multi-Source Feature Pyramid (SOLID: Single Responsibility)
"""

from typing import List, Tuple, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.interfaces import IFrequencyEncoder
from src.models.blocks.bayar_conv import BayarConv2d
from src.models.blocks.srm_conv import SRMConv2d


class ConvBNReLU(nn.Module):
    """Basic convolutional block with Batch Normalization and GELU activation."""

    def __init__(self, in_c: int, out_c: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


def generate_dct_basis_filters() -> torch.Tensor:
    """
    Constructs the 64 orthogonal 2D-DCT basis kernels of size 8x8.
    Returns:
        torch.Tensor: [64, 1, 8, 8] float32 tensor.
    """
    filters = np.zeros((64, 1, 8, 8), dtype=np.float32)
    for u in range(8):
        for v in range(8):
            alpha_u = 1.0 / np.sqrt(8.0) if u == 0 else np.sqrt(2.0 / 8.0)
            alpha_v = 1.0 / np.sqrt(8.0) if v == 0 else np.sqrt(2.0 / 8.0)
            for x in range(8):
                for y in range(8):
                    filters[u * 8 + v, 0, x, y] = (
                        alpha_u * alpha_v *
                        np.cos((2.0 * x + 1.0) * u * np.pi / 16.0) *
                        np.cos((2.0 * y + 1.0) * v * np.pi / 16.0)
                    )
    return torch.from_numpy(filters)


class BlockDCTStream(nn.Module):
    """
    Fast GPU-Accelerated 8x8 Block-DCT Compression Analyzer.
    Projects image luminance into 64 frequency basis coefficients per 8x8 block.
    """

    def __init__(self, stage1_dim: int = 64, stage2_dim: int = 128):
        super().__init__()
        # Register non-trainable orthogonal 8x8 DCT filter weights
        self.register_buffer("dct_weights", generate_dct_basis_filters())

        # Luminance projection weights: Y = 0.299 R + 0.587 G + 0.114 B
        luma = np.array([0.299, 0.587, 0.114], dtype=np.float32).reshape(1, 3, 1, 1)
        self.register_buffer("luma_weights", torch.from_numpy(luma))

        # Lightweight projections to frequency pyramid stages
        self.proj_stage2 = nn.Sequential(
            nn.Conv2d(64, stage2_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(stage2_dim),
            nn.GELU()
        )
        self.proj_stage1 = nn.Sequential(
            nn.Conv2d(64, stage1_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(stage1_dim),
            nn.GELU()
        )

    def forward(self, rgb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            rgb (torch.Tensor): RGB input tensor [B, 3, H, W].
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - dct_stage1: [B, stage1_dim, H/4, W/4] (upsampled 2x)
                - dct_stage2: [B, stage2_dim, H/8, W/8] (native 8x8 block grid)
        """
        # Convert RGB to Luminance Y
        y = F.conv2d(rgb, self.luma_weights)  # [B, 1, H, W]

        # Extract 64-channel 2D-DCT coefficients per 8x8 block (Stride 8)
        dct_grid = F.conv2d(y, self.dct_weights, stride=8)  # [B, 64, H/8, W/8]

        # Stage 2 injection (Stride 8)
        dct_stage2 = self.proj_stage2(dct_grid)  # [B, stage2_dim, H/8, W/8]

        # Stage 1 injection (Stride 4: 2x bilinear upsampling)
        h4, w4 = rgb.shape[-2] // 4, rgb.shape[-1] // 4
        dct_up = F.interpolate(dct_grid, size=(h4, w4), mode="bilinear", align_corners=False)
        dct_stage1 = self.proj_stage1(dct_up)  # [B, stage1_dim, H/4, W/4]

        return dct_stage1, dct_stage2


class FrequencyEncoder(IFrequencyEncoder):
    """
    Hierarchical Frequency Residual Feature Pyramid Encoder with Block-DCT.
    Extracts multi-scale forensic anomalies from spatial residuals and frequency grids.
    """

    def __init__(
        self,
        in_channels: int = 3,
        stage_channels: Tuple[int, int, int, int] = (64, 128, 320, 512),
        use_bayar: bool = True,
        use_srm: bool = True,
        use_dct: bool = True
    ):
        super().__init__()
        self.use_bayar = use_bayar
        self.use_srm = use_srm
        self.use_dct = use_dct
        self._stage_channels = stage_channels

        # Forensic filter front-ends
        front_channels = 0
        if use_bayar:
            self.bayar = BayarConv2d(in_channels=in_channels, out_channels=3)
            front_channels += 3
        if use_srm:
            self.srm = SRMConv2d(in_channels=in_channels)
            front_channels += 3

        # Fast 8x8 Block-DCT Stream
        c1, c2, c3, c4 = stage_channels
        if use_dct:
            self.dct_stream = BlockDCTStream(stage1_dim=c1, stage2_dim=c2)

        # Initial stem: Project residuals to Stage 1 (stride 4)
        self.stem = nn.Sequential(
            ConvBNReLU(front_channels, c1 // 2, kernel_size=3, stride=2, padding=1),
            ConvBNReLU(c1 // 2, c1, kernel_size=3, stride=2, padding=1)
        )  # Stride 4

        # Stage 2: Stride 8
        self.stage2 = nn.Sequential(
            ConvBNReLU(c1, c2, kernel_size=3, stride=2, padding=1),
            ConvBNReLU(c2, c2, kernel_size=3, stride=1, padding=1)
        )

        # Stage 3: Stride 16
        self.stage3 = nn.Sequential(
            ConvBNReLU(c2, c3, kernel_size=3, stride=2, padding=1),
            ConvBNReLU(c3, c3, kernel_size=3, stride=1, padding=1)
        )

        # Stage 4: Stride 32
        self.stage4 = nn.Sequential(
            ConvBNReLU(c3, c4, kernel_size=3, stride=2, padding=1),
            ConvBNReLU(c4, c4, kernel_size=3, stride=1, padding=1)
        )

    def forward(
        self,
        x: torch.Tensor,
        return_srm: bool = False
    ) -> Union[List[torch.Tensor], Tuple[List[torch.Tensor], Optional[torch.Tensor]]]:
        """
        Extracts multi-stage high-pass frequency residual pyramid.
        Args:
            x (torch.Tensor): Input document RGB tensor [B, 3, H, W].
            return_srm (bool): If True, returns ([F1, F2, F3, F4], srm_res).
        Returns:
            List[torch.Tensor] or Tuple: Multi-stage features and optional SRM residuals.
        """
        residuals = []
        srm_res = None
        if self.use_bayar:
            residuals.append(self.bayar(x))
        if self.use_srm:
            srm_res = self.srm(x)
            residuals.append(srm_res)

        f_res = torch.cat(residuals, dim=1) if len(residuals) > 1 else residuals[0]

        # Forward through hierarchical pyramid
        f1 = self.stem(f_res)    # Stride 4
        f2 = self.stage2(f1)     # Stride 8

        # Inject Fast 8x8 Block-DCT Compression features if enabled
        if self.use_dct:
            dct_f1, dct_f2 = self.dct_stream(x)
            f1 = f1 + dct_f1
            f2 = f2 + dct_f2

        f3 = self.stage3(f2)     # Stride 16
        f4 = self.stage4(f3)     # Stride 32

        stage_feats = [f1, f2, f3, f4]
        if return_srm:
            return stage_feats, srm_res
        return stage_feats

    @property
    def out_channels(self) -> Tuple[int, int, int, int]:
        return self._stage_channels
