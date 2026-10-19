"""
DocROI-Former: Semantic Relevance Gating Module
===============================================
Attenuates forensic anomaly signals according to legal semantic importance:
  - Anomaly on empty margin, staple hole, or paper fold -> Gated to 0.0 (Ignored)
  - Anomaly touching Name, DOB, ID number, Photo, or QR -> Passed with full 1.0 weight

Design Pattern: Gating Mechanism (SOLID: Single Responsibility)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticRelevanceGate(nn.Module):
    """
    Applies continuous spatial relevance attenuation:
        F_gated = F_in * (alpha + (1 - alpha) * W_relevance)
    Where alpha in [0.0, 0.1] allows a tiny baseline signal through for global context
    while effectively suppressing harmless margin lighting and dust.
    """

    def __init__(self, channels: int, alpha: float = 0.15):
        super().__init__()
        self.channels = channels
        self.alpha = alpha

        # Lightweight channel-wise calibration
        self.gate_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )

    def forward(self, feat: torch.Tensor, relevance_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat (torch.Tensor): Feature map [B, C, H, W].
            relevance_map (torch.Tensor): Continuous relevance weight in [0.0, 1.0] [B, 1, H, W].
        Returns:
            torch.Tensor: Attenuated feature map [B, C, H, W].
        """
        # Resize relevance map if spatial dimensions differ
        if relevance_map.shape[-2:] != feat.shape[-2:]:
            rel_resized = F.interpolate(relevance_map, size=feat.shape[-2:], mode="bilinear", align_corners=False)
        else:
            rel_resized = relevance_map

        # Apply continuous attenuation
        effective_gate = self.alpha + (1.0 - self.alpha) * rel_resized
        channel_mod = self.gate_proj(feat)
        return feat * effective_gate * channel_mod
