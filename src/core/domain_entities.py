"""
DocROI-Former: Domain Entities & Data Transfer Objects (DTOs)
============================================================
Defines immutable data models representing document findings, forensic evidence,
NPCI trust receipts, and per-sample benchmark metrics.

Design Pattern: Domain Model / Value Object (SOLID: Single Responsibility)
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
import time


@dataclass(frozen=True)
class EntityRegion:
    """Represents a localized legal entity zone on a document."""
    field_id: str
    field_name: str           # e.g., 'date_of_birth', 'pan_number', 'photo_box'
    bbox_xyxy: List[int]      # [x1, y1, x2, y2]
    confidence: float
    category: str             # 'TEXT', 'PHOTO', 'QR_CODE', 'OFFICIAL_STAMP'

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FieldFinding:
    """Represents forensic evidence detected within a specific document field."""
    field_name: str
    reason_code: str          # e.g., 'TAMPER_FONT_INCONSISTENCY', 'TAMPER_PHOTO_SPLICING'
    anomaly_score: float      # Normalized confidence in [0.0, 1.0]
    tampered_pixels: int
    mask_iou: Optional[float]
    detail: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ForensicEvidence:
    """Complete forensic breakdown for an analyzed document."""
    document_hash: str
    document_type: str        # 'PAN_CARD', 'AADHAAR_CARD', 'GST_CERTIFICATE', etc.
    global_fraud_score: float # In [0.0, 1.0]
    decision: str             # 'APPROVED', 'REJECTED', 'ESCALATED'
    is_tampered: bool
    tampered_pixel_count: int
    findings: List[FieldFinding] = field(default_factory=list)
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DocumentTrustReceipt:
    """
    Official NPCI-Compliant Document Trust Receipt.
    Immutable, audit-grade verification artifact.
    """
    receipt_id: str
    document_hash: str
    document_type: str
    decision: str
    risk_score: float
    confidence_level: float
    suspicious_fields: List[Dict[str, Any]]
    supporting_evidence: Dict[str, Any]
    governance_checks: Dict[str, Any]
    model_version: str
    policy_version: str
    agent_identity: str
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    def to_dict(self) -> Dict[str, Any]:
        """Convert receipt to clean serializable dictionary."""
        return asdict(self)


@dataclass
class EvaluationSampleRecord:
    """
    Individual sample record for full-dataset benchmark verification.
    Written directly into master_individual_test_results.csv.
    """
    sample_id: str
    dataset_name: str
    file_path: str
    ground_truth_label: str   # 'AUTHENTIC' or 'TAMPERED'
    predicted_label: str      # 'AUTHENTIC' or 'TAMPERED'
    confusion_class: str      # 'TP', 'TN', 'FP', 'FN'
    global_fraud_score: float
    pixel_iou: float
    pixel_f1: float
    pixel_precision: float
    pixel_recall: float
    inference_latency_ms: float
    margin_glare_suppressed: bool = True
