"""
DocROI-Former: Dynamic Semantic Entity Token Router
===================================================
Decomposes document visual tokens into legal semantic entities:
  - Class 0: Paper Substrate (Inert background, margins, table surface)
  - Class 1: Text & Digits (Name, DOB, PAN/Aadhaar/GST numbers)
  - Class 2: Portrait Photo Box (Face image)
  - Class 3: QR Code / Barcode
  - Class 4: Official Seal / Signature

Implements: IEntityRouter (SOLID: Interface Segregation Principle)
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.interfaces import IEntityRouter


class EntityTokenRouter(IEntityRouter):
    """
    Learns to dynamically route spatial tokens into semantic categories
    and produces a continuous relevance gating map W_semantic in [0.0, 1.0].
    """

    def __init__(
        self,
        in_channels: int = 64,
        num_classes: int = 5,
        enable_relevance_gating: bool = True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.enable_relevance_gating = enable_relevance_gating

        # Lightweight spatial semantic classifier
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.GELU(),
            nn.Conv2d(in_channels // 2, num_classes, kernel_size=1)
        )

        # Relevance gate projection (0.0 for paper, 1.0 for legal entities)
        self.relevance_conv = nn.Sequential(
            nn.Conv2d(num_classes, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(
        self,
        x: torch.Tensor,
        entity_prior: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for semantic token routing.
        Args:
            x (torch.Tensor): Feature map from Stage 1 [B, C, H, W].
            entity_prior (torch.Tensor, optional): External binary prior [B, 1, H, W].
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - routing_logits: Semantic category logits [B, num_classes, H, W].
                - relevance_map: Relevance weight in [0.0, 1.0] of shape [B, 1, H, W].
        """
        logits = self.classifier(x)

        if entity_prior is not None:
            # Upsample/downsample prior if spatial dimensions differ
            if entity_prior.shape[-2:] != logits.shape[-2:]:
                prior_resized = F.interpolate(entity_prior, size=logits.shape[-2:], mode="nearest")
            else:
                prior_resized = entity_prior

            # Boost text/entity logit channel with prior
            # Class 0 is paper, Class 1 is text
            logits[:, 1:2, :, :] = logits[:, 1:2, :, :] + prior_resized * 2.0

        if self.enable_relevance_gating:
            relevance = self.relevance_conv(logits)
        else:
            relevance = torch.ones(
                (logits.shape[0], 1, logits.shape[2], logits.shape[3]),
                device=logits.device,
                dtype=logits.dtype
            )

        return logits, relevance
