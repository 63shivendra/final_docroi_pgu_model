"""
DocROI-Former Core Package Initialization
"""

from .interfaces import (
    ISpatialEncoder,
    IFrequencyEncoder,
    IEntityRouter,
    ILossCriterion,
    ITrustReceiptEngine
)
from .domain_entities import (
    EntityRegion,
    FieldFinding,
    ForensicEvidence,
    DocumentTrustReceipt,
    EvaluationSampleRecord
)
from .reason_codes import (
    ForensicReasonCode,
    GovernanceDecision,
    DocumentCategory
)
from .vision_preprocessor import (
    DocumentRectifier,
    PerimeterGuardband,
    SemanticRoIMasker,
    RectificationResult
)

__all__ = [
    "ISpatialEncoder",
    "IFrequencyEncoder",
    "IEntityRouter",
    "ILossCriterion",
    "ITrustReceiptEngine",
    "EntityRegion",
    "FieldFinding",
    "ForensicEvidence",
    "DocumentTrustReceipt",
    "EvaluationSampleRecord",
    "ForensicReasonCode",
    "GovernanceDecision",
    "DocumentCategory",
    "DocumentRectifier",
    "PerimeterGuardband",
    "SemanticRoIMasker",
    "RectificationResult"
]
