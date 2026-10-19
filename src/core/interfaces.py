"""
DocROI-Former: Core Abstract Interfaces & Protocols
===================================================
Defines formal architectural contracts across all subsystems in compliance
with the Interface Segregation Principle (ISP) and Dependency Inversion Principle (DIP).

Components adhere strictly to these interfaces to guarantee low coupling and seamless
swapping of backbones, loss functions, and evaluation engines.
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any, Optional, List
import torch
import torch.nn as nn


class ISpatialEncoder(ABC, nn.Module):
    """
    Contract for Hierarchical Spatial Representation Backbones.
    Must output multi-scale feature pyramids (Stage 1 to Stage 4).
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass.
        Args:
            x (torch.Tensor): Input document RGB tensor of shape [B, 3, H, W].
        Returns:
            List[torch.Tensor]: Multi-stage feature maps [F1, F2, F3, F4]
                                where F1 has stride 4 (vital for character strokes).
        """
        pass

    @property
    @abstractmethod
    def out_channels(self) -> Tuple[int, int, int, int]:
        """Returns channel dimensions for all 4 hierarchical stages."""
        pass


class IFrequencyEncoder(ABC, nn.Module):
    """
    Contract for High-Pass Residual & Forensic Sensor Noise Stream.
    Extracts micro-resampling artifacts, PRNU, and compression noise.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass.
        Args:
            x (torch.Tensor): Input document RGB tensor of shape [B, 3, H, W].
        Returns:
            List[torch.Tensor]: Multi-stage frequency feature pyramid matching spatial strides.
        """
        pass


class IEntityRouter(ABC, nn.Module):
    """
    Contract for Dynamic Semantic Entity Token Routing.
    Decomposes the canvas into Text, Photo, QR, Seal, and Paper Substrate.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor, entity_prior: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts entity routing masks and relevance weights.
        Args:
            x (torch.Tensor): Feature map or raw image [B, C, H, W].
            entity_prior (torch.Tensor, optional): Precomputed heuristic entity mask [B, 1, H, W].
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - routing_mask: Binary or soft entity assignment [B, Num_Classes, H, W].
                - relevance_weights: Attenuation weights for margin glare [B, 1, H, W].
        """
        pass


class ILossCriterion(ABC, nn.Module):
    """
    Contract for Modular Mathematical Optimization Objectives.
    """

    @abstractmethod
    def forward(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Computes structured loss terms.
        Args:
            predictions (Dict[str, torch.Tensor]): Model outputs (mask_logits, global_logits, font_embeddings).
            targets (Dict[str, torch.Tensor]): Ground truth masks and document labels.
        Returns:
            Dict[str, torch.Tensor]: Breakdown of individual loss terms and total loss.
        """
        pass


class ITrustReceiptEngine(ABC):
    """
    Contract for Cryptographic Governance & Document Trust Receipt Generation.
    NPCI compliance standard.
    """

    @abstractmethod
    def generate_receipt(
        self,
        image_bytes: bytes,
        document_type: str,
        global_risk_score: float,
        pixel_mask: torch.Tensor,
        suspicious_fields: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generates an immutable, cryptographically hashed trust receipt.
        """
        pass
