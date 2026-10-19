"""
DocROI-Former: Hierarchical Spatial Vision Transformer Backbone
==============================================================
Encapsulates a multi-scale Pyramid Vision Transformer (PVTv2-B2) backbone,
producing 4 hierarchical feature stages with Stage-1 operating at stride 4
(96x96 tokens at 384x384 input), essential for resolving fine character strokes.

Implements: ISpatialEncoder (SOLID: Liskov Substitution Principle)
"""

from typing import List, Tuple
import torch
import torch.nn as nn
import timm

from src.core.interfaces import ISpatialEncoder


class SpatialVisionTransformer(ISpatialEncoder):
    """
    Hierarchical Vision Transformer Spatial Backbone.
    Extracts multi-stage visual representations:
      - Stage 1: Stride 4  (C=64,  H/4,  W/4)  - Sub-character stroke resolution
      - Stage 2: Stride 8  (C=128, H/8,  W/8)  - Word & field geometry
      - Stage 3: Stride 16 (C=320, H/16, W/16) - Document section layout
      - Stage 4: Stride 32 (C=512, H/32, W/32) - Global document context
    """

    def __init__(
        self,
        model_name: str = "pvt_v2_b2",
        pretrained: bool = True,
        in_channels: int = 3,
        drop_path_rate: float = 0.1
    ):
        super().__init__()
        self.model_name = model_name
        self._out_channels = (64, 128, 320, 512)

        # Create hierarchical backbone extracting multi-stage features
        self.encoder = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_channels,
            features_only=True,
            drop_path_rate=drop_path_rate
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Input document RGB image [B, 3, H, W].
        Returns:
            List[torch.Tensor]: Multi-stage features [F1, F2, F3, F4].
        """
        return self.encoder(x)

    @property
    def out_channels(self) -> Tuple[int, int, int, int]:
        return self._out_channels
