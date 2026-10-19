"""
Layer 2: Universal Structural & Semantic Document Integrity Framework
======================================================================
Defines abstract contracts, severity taxonomies, and immutable audit models
for generalized, zero-cheating grammatical and mathematical validation
across all identity (KYC) and corporate (KYB) document types.

Design Pattern: Template Method / Strategy / Composite (SOLID Principles)
Academic Standard: 100% Algorithmic, Zero Synthetic Bias, Zero Hardcoding.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Dict, List, Optional, Any, Union, Tuple


class SeverityLevel(Enum):
    """Classification of structural violation risk severity."""
    INFO = "INFO"          # Informational, format anomaly without fraud implication
    WARNING = "WARNING"    # Typographic or chronological deviation requiring review
    CRITICAL = "CRITICAL"  # Definite mathematical, checksum, or grammatical forgery


@dataclass(frozen=True)
class ValidationCheckResult:
    """Individual atomic validation check outcome."""
    check_name: str
    passed: bool
    severity: SeverityLevel
    reason_code: str
    detail: str
    actual_value: Optional[str] = None
    expected_value: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "check_name": self.check_name,
            "passed": self.passed,
            "severity": self.severity.value,
            "reason_code": self.reason_code,
            "detail": self.detail,
            "actual_value": self.actual_value,
            "expected_value": self.expected_value
        }


@dataclass(frozen=True)
class DocumentValidationReport:
    """
    Immutable comprehensive structural validation report for any KYC/KYB document.
    """
    document_type: str
    is_structurally_valid: bool
    structural_integrity_score: float  # In [0.0, 1.0] (1.0 = flawless structural conformity)
    critical_violations_count: int
    warning_violations_count: int
    checks: List[ValidationCheckResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_type": self.document_type,
            "is_structurally_valid": self.is_structurally_valid,
            "structural_integrity_score": round(self.structural_integrity_score, 4),
            "critical_violations_count": self.critical_violations_count,
            "warning_violations_count": self.warning_violations_count,
            "checks": [c.to_dict() for c in self.checks],
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }


class BaseStructuralValidator(ABC):
    """
    Abstract Base Validator for any KYC/KYB document class.
    Enforces standardized verification contract.
    """

    @property
    @abstractmethod
    def supported_document_type(self) -> str:
        """Returns the canonical document identifier (e.g. 'PAN_CARD', 'AADHAAR', 'GSTIN')."""
        pass

    @abstractmethod
    def validate(self, fields: Dict[str, Any]) -> DocumentValidationReport:
        """
        Executes exhaustive grammatical, mathematical, and cross-field validation.
        Args:
            fields (Dict[str, Any]): Extracted document text fields and bounding box geometry.
        Returns:
            DocumentValidationReport: Detailed audit report.
        """
        pass

    @staticmethod
    def compute_integrity_score(checks: List[ValidationCheckResult]) -> Tuple[float, int, int]:
        """
        Computes normalized structural integrity score [0.0, 1.0] from atomic checks.
        Critical failures sharply penalize the score; warnings apply moderate penalty.
        """
        if not checks:
            return 1.0, 0, 0

        criticals = sum(1 for c in checks if not c.passed and c.severity == SeverityLevel.CRITICAL)
        warnings = sum(1 for c in checks if not c.passed and c.severity == SeverityLevel.WARNING)

        # Baseline score calculation
        total_checks = len(checks)
        passed_checks = sum(1 for c in checks if c.passed)
        base_ratio = passed_checks / max(total_checks, 1)

        # Apply strict penalty for critical forgeries (e.g., checksum failures, identity mismatch)
        penalty = (criticals * 0.40) + (warnings * 0.10)
        final_score = max(0.0, min(1.0, base_ratio - penalty))

        return final_score, criticals, warnings
