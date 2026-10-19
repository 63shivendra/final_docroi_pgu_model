"""
DocROI-Former: Progressive Guided Upsampling (PGU) Boundary-Refining Mask Head
=============================================================================
Replaces the naive 4x bilinear interpolation with a 2-stage Progressive Guided
Upsampling (PGU) decoder:
  - Stage A (96x96 -> 192x192, Stride 2): Fuses high-resolution Stage-1 spatial
    features via lightweight 3x3 Depthwise-Separable Convolutions. Outputs an
    auxiliary prediction for Deep Boundary Supervision during training.
  - Stage B (192x192 -> 384x384, Stride 1): Injects native-resolution Spatial Rich
    Model (SRM) high-pass residuals through a learned dynamic Edge Gate, snapping
    the boundary tightly onto 1-pixel font character ink strokes.

Design Pattern: Progressive Decoder / Deep Supervision (SOLID: Single Responsibility)
"""

from typing import List, Tuple, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """Lightweight 3x3 Depthwise-Separable Convolutional Block."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.dw = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        self.bn_dw = nn.BatchNorm2d(in_channels)
        self.act_dw = nn.GELU()
        self.pw = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn_pw = nn.BatchNorm2d(out_channels)
        self.act_pw = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act_dw(self.bn_dw(self.dw(x)))
        x = self.act_pw(self.bn_pw(self.pw(x)))
        return x


class BoundaryRefiningMaskHead(nn.Module):
    """
    Multi-Scale Feature Aggregation with Progressive Guided Upsampling (PGU).
    Eliminates the spatial Nyquist blur bottleneck at sub-character boundaries.
    """

    def __init__(
        self,
        in_channels_list: Tuple[int, int, int, int] = (64, 128, 320, 512),
        decoder_channels: int = 256,
        target_resolution: Tuple[int, int] = (384, 384),
        srm_channels: int = 3
    ):
        super().__init__()
        self.target_resolution = target_resolution
        c1, c2, c3, c4 = in_channels_list

        # 1. Multi-Stage Feature Projectors to uniform decoder channels
        self.proj1 = nn.Conv2d(c1, decoder_channels, kernel_size=1)
        self.proj2 = nn.Conv2d(c2, decoder_channels, kernel_size=1)
        self.proj3 = nn.Conv2d(c3, decoder_channels, kernel_size=1)
        self.proj4 = nn.Conv2d(c4, decoder_channels, kernel_size=1)

        # 2. Multi-scale Stride-4 Fusion Block (96x96)
        self.fusion = nn.Sequential(
            nn.Conv2d(decoder_channels * 4, decoder_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.GELU()
        )

        # Stride-4 boundary context extraction
        self.boundary_refine_stride4 = nn.Sequential(
            nn.Conv2d(decoder_channels, 128, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU()
        )

        # 3. Stage A: Progressive Guided Upsampling (96x96 -> 192x192, Stride 2)
        self.up_stage_a = nn.Sequential(
            nn.Conv2d(128, 96, kernel_size=1, bias=False),
            nn.BatchNorm2d(96),
            nn.GELU()
        )
        self.spatial_skip_proj = nn.Sequential(
            nn.Conv2d(c1, 96, kernel_size=1, bias=False),
            nn.BatchNorm2d(96),
            nn.GELU()
        )
        self.fuse_stage_a = DepthwiseSeparableConv(in_channels=96 + 96, out_channels=96)

        # Auxiliary boundary head at 192x192 for deep supervision (active in training only)
        self.aux_head = nn.Conv2d(96, 1, kernel_size=1)
        nn.init.normal_(self.aux_head.weight, std=0.01)
        nn.init.constant_(self.aux_head.bias, -2.19)

        # 4. Stage B: Native Guided Upsampling (192x192 -> 384x384, Stride 1)
        self.up_stage_b = nn.Sequential(
            nn.Conv2d(96, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU()
        )
        self.fuse_stage_b = DepthwiseSeparableConv(in_channels=48, out_channels=48)

        # Native SRM high-pass residual edge injection
        self.srm_edge_proj = nn.Sequential(
            nn.Conv2d(srm_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48)
        )
        self.edge_gate = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size=1),
            nn.Tanh()
        )

        # Final 1-pixel stroke boundary predictor
        self.final_conv = nn.Sequential(
            nn.Conv2d(48, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.GELU(),
            nn.Conv2d(24, 1, kernel_size=1)
        )

        # Scientific Focal Prior Initialization: Pre-bias output to -2.19 (pi = 0.10)
        final_layer = self.final_conv[-1]
        nn.init.normal_(final_layer.weight, std=0.01)
        nn.init.constant_(final_layer.bias, -2.19)

    def forward(
        self,
        features: List[torch.Tensor],
        stage1_spatial: Optional[torch.Tensor] = None,
        srm_residuals: Optional[torch.Tensor] = None,
        return_aux: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            features (List[torch.Tensor]): [F1 (stride 4), F2 (stride 8), F3 (stride 16), F4 (stride 32)].
            stage1_spatial (torch.Tensor, optional): Stage 1 spatial feature map [B, C1, H/4, W/4].
            srm_residuals (torch.Tensor, optional): Full-resolution SRM residuals [B, 3, H, W].
            return_aux (bool): If True, returns (mask_logits, aux_mask_logits).
        Returns:
            torch.Tensor or Tuple: High-resolution mask logits [B, 1, H, W] (and optional aux [B, 1, H/2, W/2]).
        """
        f1, f2, f3, f4 = features
        h1, w1 = f1.shape[-2:]

        # Step 1: Project multi-scale features to common dimension at Stride 4
        p1 = self.proj1(f1)
        p2 = F.interpolate(self.proj2(f2), size=(h1, w1), mode="bilinear", align_corners=False)
        p3 = F.interpolate(self.proj3(f3), size=(h1, w1), mode="bilinear", align_corners=False)
        p4 = F.interpolate(self.proj4(f4), size=(h1, w1), mode="bilinear", align_corners=False)

        fused = self.fusion(torch.cat([p1, p2, p3, p4], dim=1))
        refined_stride4 = self.boundary_refine_stride4(fused)  # [B, 128, H/4, W/4]

        # Step 2: Stage A (Stride 4 -> Stride 2: 96x96 -> 192x192)
        target_stage_a = (h1 * 2, w1 * 2)
        up_a = F.interpolate(refined_stride4, size=target_stage_a, mode="bilinear", align_corners=False)
        up_a = self.up_stage_a(up_a)  # [B, 96, H/2, W/2]

        # Skip-fuse Stage 1 spatial features
        spatial_ref = stage1_spatial if stage1_spatial is not None else f1
        skip_spatial = F.interpolate(self.spatial_skip_proj(spatial_ref), size=target_stage_a, mode="bilinear", align_corners=False)
        feat_stage_a = self.fuse_stage_a(torch.cat([up_a, skip_spatial], dim=1))  # [B, 96, H/2, W/2]

        aux_logits = self.aux_head(feat_stage_a)  # [B, 1, H/2, W/2]

        # Step 3: Stage B (Stride 2 -> Stride 1: 192x192 -> 384x384)
        up_b = F.interpolate(feat_stage_a, size=self.target_resolution, mode="bilinear", align_corners=False)
        feat_stage_b = self.fuse_stage_b(self.up_stage_b(up_b))  # [B, 48, H, W]

        # Inject native SRM high-pass residual edge contours if provided
        if srm_residuals is not None:
            if srm_residuals.shape[-2:] != self.target_resolution:
                # FIX (Issue #4): Use nearest-neighbour resampling, NOT bilinear.
                # SRM residuals are high-frequency edge maps; bilinear resampling applies
                # Gaussian smoothing and destroys the sub-pixel sharpness that Stage B
                # was specifically designed to preserve. nearest preserves edge crispness.
                import warnings
                warnings.warn(
                    f"[MaskHead] SRM residuals size {srm_residuals.shape[-2:]} != "
                    f"target {self.target_resolution}. Resampling with nearest. "
                    "Ensure dataset returns native-resolution inputs for best results.",
                    stacklevel=2
                )
                srm_res = F.interpolate(srm_residuals, size=self.target_resolution, mode="nearest")
            else:
                srm_res = srm_residuals
            edge_feats = self.srm_edge_proj(srm_res)
            gate = self.edge_gate(feat_stage_b)
            feat_stage_b = feat_stage_b + gate * edge_feats

        # Final 1-pixel character contour prediction
        out_mask = self.final_conv(feat_stage_b)  # [B, 1, H, W]

        if return_aux:
            return out_mask, aux_logits
        return out_mask
