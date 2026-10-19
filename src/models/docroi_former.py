"""
DocROI-Former: Unified Entity-Gated Frequency Vision Transformer
================================================================
Master architecture integrating:
  1. Hierarchical Spatial Vision Transformer Backbone (PVTv2-B2, Stage 1 stride 4).
  2. Multi-Scale Frequency Forensic Residual Stream (Bayar + SRM filter banks).
  3. Dynamic Semantic Entity Token Router (Text, Photo, QR, Seal vs Paper Substrate).
  4. Entity-Gated Cross-Attention Engine (Masking inert background paper).
  5. Semantic Relevance Gating (Muting margin folds, creases, and lighting glare).
  6. Decoupled Dual Output Heads:
     - Global Document Decision Head (P_doc for automated triage).
     - High-Resolution Boundary-Refining Mask Head (55%-70% IoU character outlines).
     - Intra-Document Font Contrastive Embedding Head (Canva/digital font verification).

Design Pattern: Facade / Composite (SOLID: Single Responsibility & Open/Closed)
"""

from typing import Dict, List, Tuple, Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.docroi_config import DocROIConfig
from src.models.backbones.spatial_encoder import SpatialVisionTransformer
from src.models.backbones.frequency_encoder import FrequencyEncoder
from src.models.attention.entity_router import EntityTokenRouter
from src.models.attention.cross_attention import EntityGatedCrossAttention
from src.models.attention.relevance_gating import SemanticRelevanceGate
from src.models.heads.classification_head import GlobalDocumentClassifierHead
from src.models.heads.mask_head import BoundaryRefiningMaskHead
from src.models.heads.font_head import FontContrastiveHead


class DocROIFormer(nn.Module):
    """
    Unified DocROI-Former Architecture for High-Precision Document Forgery Localization
    and Enterprise Operational Decisioning.
    """

    def __init__(self, config: Optional[DocROIConfig] = None):
        super().__init__()
        self.config = config or DocROIConfig()

        # 1. Dual-Stream Encoders
        self.spatial_encoder = SpatialVisionTransformer(
            model_name=self.config.spatial.backbone_name,
            pretrained=self.config.spatial.pretrained,
            in_channels=self.config.spatial.in_channels,
            drop_path_rate=self.config.spatial.drop_path_rate
        )

        self.frequency_encoder = FrequencyEncoder(
            in_channels=self.config.spatial.in_channels,
            stage_channels=self.config.spatial.stage_channels,
            use_bayar=self.config.frequency.use_bayar,
            use_srm=self.config.frequency.use_srm,
            use_dct=getattr(self.config.frequency, "use_dct", True)
        )

        channels = self.config.spatial.stage_channels  # (64, 128, 320, 512)

        # 2. Dynamic Semantic Entity Router & Relevance Gate
        self.entity_router = EntityTokenRouter(
            in_channels=channels[0],
            num_classes=self.config.router.num_classes,
            enable_relevance_gating=self.config.router.enable_relevance_gating
        )

        self.relevance_gates = nn.ModuleList([
            SemanticRelevanceGate(channels=c) for c in channels
        ])

        # 3. Cross-Attention Fusion Modules with Spatial Reduction (4, 2, 1, 1)
        sr_ratios = (4, 2, 1, 1)
        self.cross_attentions = nn.ModuleList([
            EntityGatedCrossAttention(
                dim=c,
                num_heads=self.config.fusion.num_heads if c % self.config.fusion.num_heads == 0 else 4,
                sr_ratio=sr_ratios[i],
                mask_background_value=self.config.router.mask_background_value
            ) for i, c in enumerate(channels)
        ])

        # Feature combination layers
        self.fuse_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c * 2, c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c),
                nn.GELU()
            ) for c in channels
        ])

        # 4. Decoupled Heads
        self.global_classifier = GlobalDocumentClassifierHead(
            in_features=channels[3] * 2,
            hidden_dim=self.config.heads.global_hidden_dim
        )

        self.mask_head = BoundaryRefiningMaskHead(
            in_channels_list=channels,
            decoder_channels=self.config.heads.mask_decoder_channels,
            target_resolution=self.config.spatial.input_resolution
        )

        self.font_head = FontContrastiveHead(
            in_channels=channels[0],
            embedding_dim=self.config.heads.font_embedding_dim
        )

    def forward(
        self,
        x: torch.Tensor,
        entity_prior: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for end-to-end inference and training.
        Args:
            x (torch.Tensor): Input document RGB tensor [B, 3, H, W].
            entity_prior (torch.Tensor, optional): Optional heuristic entity prior [B, 1, H, W].
        Returns:
            Dict[str, torch.Tensor]:
                - 'mask_logits': Pixel forgery logits [B, 1, H, W].
                - 'global_logits': Document fraud logits [B, 1].
                - 'global_prob': Calibrated probability P_doc in [0.0, 1.0].
                - 'font_embeddings': L2-normalized character tokens [B, N_1, 128].
                - 'routing_logits': Entity category logits [B, 5, H/4, W/4].
                - 'relevance_map': Semantic attenuation weights [B, 1, H/4, W/4].
        """
        # Step 1: Extract multi-stage hierarchical features from both streams
        spatial_feats = self.spatial_encoder(x)       # [S1, S2, S3, S4]
        freq_feats, srm_res = self.frequency_encoder(x, return_srm=True)  # [F1, F2, F3, F4], [B, 3, H, W]

        # Step 2: Route semantic entity tokens from Stage 1 spatial features
        routing_logits, relevance_map = self.entity_router(spatial_feats[0], entity_prior)

        # Derive entity mask: 1.0 for legal entities (classes 1..4), 0.0 for paper (class 0)
        # Class 0: Paper. If argmax is 0, it is paper substrate.
        pred_classes = torch.argmax(routing_logits, dim=1, keepdim=True)
        entity_mask_2d = (pred_classes > 0).float()  # [B, 1, H/4, W/4]

        # Step 3: Fused multi-stage feature aggregation
        fused_features = []
        for i in range(len(spatial_feats)):
            s_i = spatial_feats[i]
            f_i = freq_feats[i]

            # Apply semantic relevance gating to frequency noise (suppresses margin glare)
            f_i_gated = self.relevance_gates[i](f_i, relevance_map)

            # Resize entity mask to current stage resolution
            if entity_mask_2d.shape[-2:] != s_i.shape[-2:]:
                cur_mask = F.interpolate(entity_mask_2d, size=s_i.shape[-2:], mode="nearest")
            else:
                cur_mask = entity_mask_2d

            # Entity-gated cross-attention with spatial reduction: [B, C, H, W]
            cross_feat = self.cross_attentions[i](s_i, f_i_gated, cur_mask)

            # Fuse with residual spatial representation
            stage_fused = self.fuse_convs[i](torch.cat([s_i, cross_feat], dim=1))
            fused_features.append(stage_fused)

        # Step 4: Execute Decoupled Heads
        # Head B: Progressive Guided Upsampling (PGU) Mask Head
        if self.training:
            mask_logits, aux_mask_logits = self.mask_head(
                fused_features,
                stage1_spatial=spatial_feats[0],
                srm_residuals=srm_res,
                return_aux=True
            )
        else:
            mask_logits = self.mask_head(
                fused_features,
                stage1_spatial=spatial_feats[0],
                srm_residuals=srm_res,
                return_aux=False
            )
            aux_mask_logits = None

        # Head A: Global Document Fraud Decision Head (Pure End-to-End Neural Classifier)
        global_logits = self.global_classifier(spatial_feats[-1], freq_feats[-1])
        global_prob = torch.sigmoid(global_logits)

        # Head C: Intra-Document Font Contrastive Head
        font_embeddings = self.font_head(spatial_feats[0])

        res = {
            "mask_logits": mask_logits,
            "global_logits": global_logits,
            "global_prob": global_prob,
            "font_embeddings": font_embeddings,
            "routing_logits": routing_logits,
            "relevance_map": relevance_map,
            "entity_mask": entity_mask_2d
        }
        if aux_mask_logits is not None:
            res["aux_mask_logits"] = aux_mask_logits
        return res
