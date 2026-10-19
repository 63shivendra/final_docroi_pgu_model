"""
DocROI-Former: Standardized Forensic Reason Codes & Policy Registry
==================================================================
Defines immutable, machine-readable reason codes compliant with NPCI,
RBI digital onboarding norms, and ISO/IEC international fraud taxonomies.
"""

from enum import Enum, unique


@unique
class ForensicReasonCode(str, Enum):
    """Forensic Anomaly & Manipulation Reason Codes."""
    # Authentic / Baseline
    DOC_CLEAN = "DOC_CLEAN_NO_TAMPERING_DETECTED"
    MARGIN_GLARE_SUPPRESSED = "INFO_MARGIN_GLARE_SUPPRESSED_BY_ENTITY_GATING"

    # Text & Character Alterations
    TAMPER_TEXT_FONT_INCONSISTENCY = "TAMPER_TEXT_FONT_INCONSISTENCY"
    TAMPER_TEXT_KERNING_ANOMALY = "TAMPER_TEXT_KERNING_ANOMALY"
    TAMPER_TEXT_RESAMPLING_EDGES = "TAMPER_TEXT_RESAMPLING_EDGES"
    TAMPER_DIGIT_SUBSTITUTION = "TAMPER_DIGIT_SUBSTITUTION"

    # Photo & Portrait Alterations
    TAMPER_PHOTO_SPLICING_BORDER = "TAMPER_PHOTO_SPLICING_BORDER"
    TAMPER_PHOTO_FACE_SWAP = "TAMPER_PHOTO_FACE_SWAP"
    TAMPER_PHOTO_RECOMPRESSION_HALO = "TAMPER_PHOTO_RECOMPRESSION_HALO"

    # Security & Structural Elements
    TAMPER_QR_SUBSTRATE_MISMATCH = "TAMPER_QR_SUBSTRATE_MISMATCH"
    TAMPER_SIGNATURE_COPY_MOVE = "TAMPER_SIGNATURE_COPY_MOVE"
    TAMPER_OFFICIAL_SEAL_ALTERATION = "TAMPER_OFFICIAL_SEAL_ALTERATION"
    TAMPER_STRUCTURAL_LAYOUT_VIOLATION = "TAMPER_STRUCTURAL_LAYOUT_VIOLATION"

    # Benign Real-World Artifacts
    BENIGN_STAPLE_PIN_HOLE = "BENIGN_STAPLE_PIN_HOLE_ON_MARGIN"
    BENIGN_PAPER_FOLD_CREASE = "BENIGN_PAPER_FOLD_CREASE"
    BENIGN_SCANNER_GLASS_DUST = "BENIGN_SCANNER_GLASS_DUST"


@unique
class GovernanceDecision(str, Enum):
    """Final Operational Triage Decisions."""
    AUTO_APPROVED = "APPROVED"
    AUTO_REJECTED = "REJECTED"
    ESCALATE_TO_HUMAN = "ESCALATE_TO_HUMAN_REVIEW"


@unique
class DocumentCategory(str, Enum):
    """Supported Merchant Identity & Financial Document Classes."""
    PAN_CARD = "PAN_CARD"
    AADHAAR_CARD = "AADHAAR_CARD"
    BANK_CHEQUE = "BANK_CHEQUE"
    GST_CERTIFICATE = "GST_CERTIFICATE"
    PASSPORT_INDIA = "PASSPORT_INDIA"
    DRIVING_LICENSE = "DRIVING_LICENSE"
    VOTER_ID = "VOTER_ID"
    UNKNOWN_DOCUMENT = "UNKNOWN_DOCUMENT"
