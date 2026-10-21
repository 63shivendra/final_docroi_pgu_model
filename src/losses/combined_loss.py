"""
DocROI-Former: Unified Multi-Task Loss Criterion
================================================
Coordinates all 5 mathematical optimization objectives:
    L_Total = lambda_focal * L_Focal +
              lambda_dice * L_Dice +
              lambda_boundary * L_Boundary +
              lambda_font * L_Font +
              lambda_global * L_Global

Implements: ILossCriterion (SOLID: Liskov Substitution Principle)
"""

from typing import Dict, Tuple, Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.interfaces import ILossCriterion
from configs.docroi_config import LossWeightsConfig
from src.losses.boundary_loss import BoundarySignedDistanceLoss
from src.losses.font_contrastive_loss import IntraDocumentFontContrastiveLoss
from src.losses.focal_dice_loss import EntityNormalizedFocalDiceLoss


class DocROICombinedLoss(ILossCriterion):
    """
    Master Multi-Task Loss Criterion for DocROI-Former.
    """

    def __init__(self, config: Optional[LossWeightsConfig] = None):
        super().__init__()
        self.config = config or LossWeightsConfig()

        self.focal_dice_criterion = EntityNormalizedFocalDiceLoss(
            pos_weight=self.config.pos_weight,
            gamma=self.config.focal_gamma,
            smooth=self.config.dice_smooth
        )

        self.boundary_criterion = BoundarySignedDistanceLoss()
        self.font_criterion = IntraDocumentFontContrastiveLoss(temperature=0.07)

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            predictions (Dict[str, torch.Tensor]):
                - 'mask_logits': [B, 1, H, W]
                - 'global_logits': [B, 1]
                - 'font_embeddings': [B, N, D]
                - 'entity_mask': [B, 1, H_low, W_low]
            targets (Dict[str, torch.Tensor]):
                - 'mask': Ground truth binary mask [B, 1, H, W]
                - 'label': Document binary label [B, 1] (0.0 for clean, 1.0 for fraud)
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing individual loss components and 'loss_total'.
        """
        mask_logits = predictions["mask_logits"]
        global_logits = predictions["global_logits"]
        font_embeddings = predictions["font_embeddings"]
        entity_mask = predictions["entity_mask"]

        target_mask = targets["mask"]
        target_label = targets["label"]

        # 1. Entity-Normalized Focal & Dice Loss
        loss_focal, loss_dice = self.focal_dice_criterion(mask_logits, target_mask, entity_mask)

        # 2. Boundary Signed Distance Loss
        dist_map = targets.get("dist_map", None)
        loss_boundary = self.boundary_criterion(mask_logits, target_mask, dist_map=dist_map)

        # 3. Intra-Document Font Contrastive Loss
        loss_font = self.font_criterion(font_embeddings, target_mask, entity_mask)

        # 4. Global Document Classification Loss (Class-Balanced)
        # In-domain DocTamper is ~98% positive; downweighting pos_weight to 0.20 balances the gradient
        # so Head A does not develop an artificial positive bias on authentic documents.
        pos_weight_global = torch.tensor([0.20], device=global_logits.device, dtype=global_logits.dtype)
        loss_global = F.binary_cross_entropy_with_logits(global_logits, target_label, pos_weight=pos_weight_global)

        # 5. Auxiliary Deep Boundary Supervision Loss (192x192, Stride 2)
        loss_aux = torch.tensor(0.0, device=mask_logits.device, dtype=mask_logits.dtype)
        if "aux_mask_logits" in predictions and predictions["aux_mask_logits"] is not None:
            aux_logits = predictions["aux_mask_logits"]
            target_mask_aux = F.interpolate(target_mask, size=aux_logits.shape[-2:], mode="nearest")
            entity_mask_aux = F.interpolate(entity_mask, size=aux_logits.shape[-2:], mode="nearest")
            aux_focal, aux_dice = self.focal_dice_criterion(aux_logits, target_mask_aux, entity_mask_aux)
            loss_aux = 0.5 * aux_focal + 0.5 * aux_dice

        # 6. Master Weighted Sum
        lambda_aux = getattr(self.config, "lambda_aux", 0.35)
        loss_total = (
            self.config.lambda_focal * loss_focal +
            self.config.lambda_dice * loss_dice +
            self.config.lambda_boundary * loss_boundary +
            lambda_aux * loss_aux +
            self.config.lambda_font * loss_font +
            self.config.lambda_global * loss_global
        )

        return {
            "loss_total": loss_total,
            "loss_focal": loss_focal,
            "loss_dice": loss_dice,
            "loss_boundary": loss_boundary,
            "loss_aux": loss_aux,
            "loss_font": loss_font,
            "loss_global": loss_global
        }
