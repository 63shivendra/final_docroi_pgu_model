"""
DocROI-Former: Strongly-Typed Enterprise Configuration Module
============================================================
This module encapsulates all hyperparameter, architectural, loss weighting,
and governance configurations for the DocROI-Former framework using immutable
and validated Python dataclasses.

Design Pattern: Configuration Object / Parameter Object (SOLID: Single Responsibility)
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
import os


@dataclass(frozen=True)
class SpatialEncoderConfig:
    """Configuration for the Hierarchical Spatial Transformer Backbone."""
    backbone_name: str = "pvt_v2_b2"
    pretrained: bool = True  # Loaded from ImageNet-1k pre-trained weights for rich visual features
    in_channels: int = 3
    input_resolution: Tuple[int, int] = (384, 384)
    stage_strides: Tuple[int, int, int, int] = (4, 8, 16, 32)
    stage_channels: Tuple[int, int, int, int] = (64, 128, 320, 512)
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.1


@dataclass(frozen=True)
class FrequencyEncoderConfig:
    """Configuration for the Micro-Noise High-Pass Frequency Residual Stream."""
    use_bayar: bool = True
    use_srm: bool = True
    use_dct: bool = True
    bayar_kernel_size: int = 5
    srm_kernel_size: int = 5
    pyramid_channels: Tuple[int, int, int, int] = (64, 128, 320, 512)
    normalize_residuals: bool = True


@dataclass(frozen=True)
class EntityRouterConfig:
    """Configuration for the Dynamic Semantic Entity Token Router."""
    num_classes: int = 5  # 0: Paper, 1: Text, 2: Photo, 3: QR, 4: Stamp
    mask_background_value: float = -1e4  # Softmax -inf masking value
    enable_relevance_gating: bool = True
    otsu_kernel_size: int = 3
    dilation_iterations: int = 2


@dataclass(frozen=True)
class AttentionFusionConfig:
    """Configuration for the Entity-Gated Cross-Attention Engine."""
    embed_dim: int = 256
    num_heads: int = 8
    qkv_bias: bool = True
    attn_drop: float = 0.0
    proj_drop: float = 0.0


@dataclass(frozen=True)
class DecoupledHeadsConfig:
    """Configuration for Decoupled Global and Local Task Heads."""
    num_global_classes: int = 1
    font_embedding_dim: int = 128
    font_temperature: float = 0.07
    mask_decoder_channels: int = 256
    global_hidden_dim: int = 512


@dataclass(frozen=True)
class LossWeightsConfig:
    """
    Calibrated Multi-Task Loss Hyperparameters.
    Total Loss = lambda_focal * L_Focal + lambda_dice * L_Dice +
                 lambda_boundary * L_Boundary + lambda_font * L_Font +
                 lambda_global * L_Global
    """
    lambda_focal: float = 0.30
    lambda_dice: float = 0.40
    lambda_boundary: float = 0.20
    lambda_aux: float = 0.35  # Deep auxiliary supervision weight at 192x192
    lambda_font: float = 0.10
    lambda_global: float = 0.30
    pos_weight: float = 1.80  # Entity-normalized positive penalty (no bloat)
    focal_gamma: float = 2.00
    dice_smooth: float = 1e-6


@dataclass(frozen=True)
class GovernanceConfig:
    """Configuration for NPCI Compliance, PII Handling, and Trust Receipts."""
    policy_version: str = "NPCI_KYC_FORENSICS_V2.4"
    model_version: str = "DocROI-Former_v1.0.0_fp16"
    agent_id: str = "NPCI_GOVERNED_MERCHANT_VERIFIER_01"
    auto_approval_threshold: float = 0.15  # P_doc < 0.15 -> Auto-Approve
    escalation_threshold: float = 0.50     # P_doc >= 0.50 -> Escalate to Officer
    min_tamper_pixel_evidence: int = 35   # Minimum pixels to flag suspicious field
    redact_aadhaar_pii: bool = True
    enable_sha256_receipts: bool = True


@dataclass(frozen=True)
class TrainingHardwareConfig:
    """Hardware and Optimization Parameters for RTX 5080."""
    device: str = "cuda"
    batch_size: int = 12
    num_workers: int = 4
    epochs: int = 20
    base_lr_encoder: float = 6e-5
    base_lr_heads: float = 3e-4
    weight_decay: float = 1e-2
    warmup_epochs: int = 3
    use_fp16: bool = True
    pin_memory: bool = True
    gradient_clip_val: float = 1.0


@dataclass
class DocROIConfig:
    """Master Unified Configuration for DocROI-Former Framework."""
    spatial: SpatialEncoderConfig = field(default_factory=SpatialEncoderConfig)
    frequency: FrequencyEncoderConfig = field(default_factory=FrequencyEncoderConfig)
    router: EntityRouterConfig = field(default_factory=EntityRouterConfig)
    fusion: AttentionFusionConfig = field(default_factory=AttentionFusionConfig)
    heads: DecoupledHeadsConfig = field(default_factory=DecoupledHeadsConfig)
    loss: LossWeightsConfig = field(default_factory=LossWeightsConfig)
    governance: GovernanceConfig = field(default_factory=GovernanceConfig)
    training: TrainingHardwareConfig = field(default_factory=TrainingHardwareConfig)

    # Base workspace directories resolved dynamically to repo root
    _root: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    workspace_dir: str = _root
    academic_data_dir: str = os.path.join(_root, "data")
    real_kyc_data_dir: str = os.path.join(_root, "Real_dataset_pan_adhaar")
    output_dir: str = os.path.join(_root, "checkpoints")
    reports_dir: str = os.path.join(_root, "reports")

    def __post_init__(self):
        """Ensure necessary output and artifact directories exist and auto-detect Kaggle paths if present."""
        # 1. Environment Variable overrides
        if os.environ.get("DOCROI_DATA_DIR"):
            self.academic_data_dir = os.environ["DOCROI_DATA_DIR"]
        if os.environ.get("DOCROI_KYC_DIR"):
            self.real_kyc_data_dir = os.environ["DOCROI_KYC_DIR"]
        if os.environ.get("DOCROI_OUTPUT_DIR"):
            self.output_dir = os.environ["DOCROI_OUTPUT_DIR"]
        if os.environ.get("DOCROI_REPORTS_DIR"):
            self.reports_dir = os.environ["DOCROI_REPORTS_DIR"]

        # 2. Kaggle Cloud Environment Auto-Detection
        if os.path.exists("/kaggle/input"):
            # Check for attached DocTamper dataset in Kaggle input
            for root, dirs, _ in os.walk("/kaggle/input"):
                if "DocTamperV1-TrainingSet" in dirs or "Doctamper" in dirs:
                    cand = os.path.join(root, "Doctamper") if "Doctamper" in dirs else root
                    if os.path.exists(cand):
                        self.academic_data_dir = cand
                        break
            # Check for Real KYC dataset in Kaggle input if not in local repo
            if not os.path.exists(self.real_kyc_data_dir):
                for root, dirs, _ in os.walk("/kaggle/input"):
                    if "Real_dataset_pan_adhaar" in dirs:
                        self.real_kyc_data_dir = os.path.join(root, "Real_dataset_pan_adhaar")
                        break
                    elif "pan_card" in dirs and "adhaar_card" in dirs:
                        self.real_kyc_data_dir = root
                        break
            # On Kaggle, ensure writable working output directory
            if not os.path.exists(self.output_dir) or not os.access(os.path.dirname(self.output_dir) or ".", os.W_OK):
                self.output_dir = "/kaggle/working/checkpoints"
                self.reports_dir = "/kaggle/working/reports"

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
