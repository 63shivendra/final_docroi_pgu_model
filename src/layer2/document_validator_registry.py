"""
Layer 2: Master Structural Validator Registry & Arbitration Facade
==================================================================
Unified entrypoint that orchestrates document type auto-detection, executes
specialized and generic grammatical/mathematical validators, and synthesizes
an audit-grade structural integrity report compatible with NPCI trust receipts.

Design Pattern: Registry / Facade / Chain of Responsibility (SOLID Principles)
Academic Standard: 100% Generalized, Extensible, Zero Hardcoding.
"""

from typing import Dict, List, Optional, Any, Type

from src.layer2.base import (
    BaseStructuralValidator,
    DocumentValidationReport,
    ValidationCheckResult,
    SeverityLevel
)
from src.layer2.pan_structural_validator import PANStructuralValidator
from src.layer2.aadhaar_validator import AadhaarStructuralValidator
from src.layer2.passport_mrz_validator import PassportMRZValidator
from src.layer2.kyb_business_validator import KYBStructuralValidator
from src.layer2.generic_document_validator import GenericDocumentValidator
from src.layer2.qr_cryptographic_validator import QRCryptographicCrossValidator
from src.core.domain_entities import FieldFinding


class UnifiedStructuralAuditEngine:
    """
    Master facade for Layer 2 structural, lexical, and mathematical audit.
    Auto-detects document classes or dispatches to explicitly targeted validators,
    and conducts optical QR cryptographic & verifiable hash cross-checks.
    """

    def __init__(self):
        self._validators: Dict[str, BaseStructuralValidator] = {}
        self._generic_validator = GenericDocumentValidator()
        self._qr_validator = QRCryptographicCrossValidator()
        self._register_default_validators()

    def _register_default_validators(self):
        self.register(PANStructuralValidator())
        self.register(AadhaarStructuralValidator())
        self.register(PassportMRZValidator())
        self.register(KYBStructuralValidator())

    def register(self, validator: BaseStructuralValidator):
        """Registers a specialized document validator."""
        self._validators[validator.supported_document_type.upper()] = validator

    def auto_detect_document_type(self, fields: Dict[str, Any]) -> str:
        """
        Infers document category based on the presence of canonical key fields.
        Returns one of: 'PAN_CARD', 'AADHAAR_CARD', 'PASSPORT', 'KYB_CORPORATE', 'GENERIC'.
        """
        keys = set(k.lower() for k in fields.keys())

        if any(k in keys for k in ("pan", "pan_number", "permanent_account_number")):
            return "PAN_CARD"
        elif any(k in keys for k in ("aadhaar", "aadhaar_number", "uid", "uidai")):
            return "AADHAAR_CARD"
        elif any(k in keys for k in ("mrz", "mrz_line1", "mrz_1", "passport_number")):
            return "PASSPORT"
        elif any(k in keys for k in ("gstin", "gst_number", "cin", "corporate_id", "opening_balance")):
            return "KYB_CORPORATE"

        return "GENERIC"

    def audit(
        self,
        fields: Dict[str, Any],
        doc_type: Optional[str] = None
    ) -> DocumentValidationReport:
        """
        Executes complete structural validation across specialized and generic rules.
        Args:
            fields (Dict[str, Any]): Dictionary of extracted document fields.
            doc_type (str, optional): Target document category. If omitted, auto-detected.
        Returns:
            DocumentValidationReport: Unified audit report.
        """
        target_type = doc_type.upper() if doc_type else self.auto_detect_document_type(fields)
        # Normalize common aliases
        type_alias_map = {
            "PAN": "PAN_CARD", "PANCARD": "PAN_CARD", "PAN_CARD": "PAN_CARD",
            "AADHAAR": "AADHAAR_CARD", "AADHAR": "AADHAAR_CARD", "UID": "AADHAAR_CARD", "AADHAAR_CARD": "AADHAAR_CARD",
            "PASSPORT": "PASSPORT",
            "GST": "KYB_CORPORATE", "GSTIN": "KYB_CORPORATE", "CIN": "KYB_CORPORATE", "KYB": "KYB_CORPORATE", "KYB_CORPORATE": "KYB_CORPORATE"
        }
        target_type = type_alias_map.get(target_type, target_type)
        all_checks: List[ValidationCheckResult] = []
        metadata: Dict[str, Any] = {"detected_document_type": target_type}

        # 1. Execute Specialized Validator if available
        validator = self._validators.get(target_type)
        if validator:
            spec_report = validator.validate(fields)
            all_checks.extend(spec_report.checks)
            metadata.update(spec_report.metadata)

        # 2. Execute Universal Generic Document Validator
        generic_report = self._generic_validator.validate(fields)
        all_checks.extend(generic_report.checks)

        # 3. Execute 2D Barcode & Cryptographic Hash Cross-Validator (if image is available)
        image = fields.get("image")
        if image is None and "image_path" in fields:
            import cv2
            image = cv2.imread(fields["image_path"])
        if image is not None:
            qr_checks = self._qr_validator.validate_document(image, fields, doc_type=target_type)
            all_checks.extend(qr_checks)

        # 4. Compute Composite Integrity Score
        score, crit, warn = BaseStructuralValidator.compute_integrity_score(all_checks)
        is_valid = (crit == 0)

        return DocumentValidationReport(
            document_type=target_type,
            is_structurally_valid=is_valid,
            structural_integrity_score=score,
            critical_violations_count=crit,
            warning_violations_count=warn,
            checks=all_checks,
            metadata=metadata
        )

    def to_domain_field_findings(self, report: DocumentValidationReport) -> List[FieldFinding]:
        """
        Converts Layer 2 structural validation failures into official FieldFinding
        objects that plug directly into ForensicEvidence and DocumentTrustReceipt.
        """
        findings: List[FieldFinding] = []
        for check in report.checks:
            if not check.passed:
                anomaly_weight = 1.0 if check.severity == SeverityLevel.CRITICAL else 0.5
                findings.append(FieldFinding(
                    field_name=check.check_name,
                    reason_code=check.reason_code,
                    anomaly_score=anomaly_weight,
                    tampered_pixels=0,  # Layer 2 is symbolic/grammatical (0 pixels)
                    mask_iou=None,
                    detail=check.detail
                ))
        return findings
