"""
Layer 2: Universal Structural, Grammatical & Mathematical Document Integrity Layer
==================================================================================
Package exports for generalized KYC and KYB validation.
"""

from src.layer2.base import (
    BaseStructuralValidator,
    DocumentValidationReport,
    ValidationCheckResult,
    SeverityLevel
)
from src.layer2.pan_structural_validator import PANStructuralValidator
from src.layer2.aadhaar_validator import AadhaarStructuralValidator, VerhoeffAlgorithm
from src.layer2.passport_mrz_validator import PassportMRZValidator, ICAOChecksum
from src.layer2.kyb_business_validator import KYBStructuralValidator, GSTINChecksum
from src.layer2.generic_document_validator import GenericDocumentValidator
from src.layer2.qr_cryptographic_validator import QRCryptographicCrossValidator, QRStatus, DecodedQRPayload
from src.layer2.document_validator_registry import UnifiedStructuralAuditEngine

__all__ = [
    "BaseStructuralValidator",
    "DocumentValidationReport",
    "ValidationCheckResult",
    "SeverityLevel",
    "PANStructuralValidator",
    "AadhaarStructuralValidator",
    "VerhoeffAlgorithm",
    "PassportMRZValidator",
    "ICAOChecksum",
    "KYBStructuralValidator",
    "GSTINChecksum",
    "GenericDocumentValidator",
    "QRCryptographicCrossValidator",
    "QRStatus",
    "DecodedQRPayload",
    "UnifiedStructuralAuditEngine"
]
