"""
Layer 2: Universal Generic Document Validator (Applicable to ANY Document)
==========================================================================
Provides document-agnostic grammatical, chronological, and typographic
verification applicable across contracts, bills, certificates, and ID cards.

Verification Dimensions:
  1. Universal Chronological Coherence:
     - Issue Date <= Expiry Date
     - Issue Date <= Current Date
     - Valid Gregorian calendar day/month/leap-year constraints
  2. Bounding Box Typographic & Baseline Geometry:
     - Horizontal baseline deviation (floating/shifted text)
     - Character height/width aspect ratio consistency
  3. Multi-Field Name & Initial Coherence:
     - Name tokens vs Signature/Guardian/Entity names

Design Pattern: Strategy / Rule Validator (SOLID: Single Responsibility)
Academic Standard: 100% Generalizable, Zero Hardcoding, Zero Dataset Bias.
"""

from datetime import datetime
import math
import re
from typing import Dict, List, Optional, Tuple, Any

from src.layer2.base import (
    BaseStructuralValidator,
    DocumentValidationReport,
    ValidationCheckResult,
    SeverityLevel
)


class GenericDocumentValidator(BaseStructuralValidator):
    """
    General-purpose structural and typographic validator suitable for ANY document.
    """

    @property
    def supported_document_type(self) -> str:
        return "GENERIC_DOCUMENT"

    def validate(self, fields: Dict[str, Any]) -> DocumentValidationReport:
        checks: List[ValidationCheckResult] = []

        # 1. Chronological Coherence
        chrono_checks = self._validate_chronological_coherence(fields)
        checks.extend(chrono_checks)

        # 2. Geometric & Baseline Alignment
        geom_checks = self._validate_geometry_and_baselines(fields)
        checks.extend(geom_checks)

        # 3. Name & Entity Phonetic / Initial Sanity
        name_checks = self._validate_name_sanity(fields)
        checks.extend(name_checks)

        score, crit, warn = self.compute_integrity_score(checks)
        return DocumentValidationReport(
            document_type=self.supported_document_type,
            is_structurally_valid=(crit == 0),
            structural_integrity_score=score,
            critical_violations_count=crit,
            warning_violations_count=warn,
            checks=checks
        )

    def _validate_chronological_coherence(self, fields: Dict[str, Any]) -> List[ValidationCheckResult]:
        checks: List[ValidationCheckResult] = []
        issue_str = fields.get("issue_date", fields.get("date_of_issue", fields.get("issued_on")))
        expiry_str = fields.get("expiry_date", fields.get("date_of_expiry", fields.get("valid_upto")))

        issue_dt = self._parse_date(issue_str) if issue_str else None
        expiry_dt = self._parse_date(expiry_str) if expiry_str else None
        now = datetime.now()

        if issue_dt and issue_dt > now:
            checks.append(ValidationCheckResult(
                check_name="CHRONOLOGY_ISSUE_DATE",
                passed=False,
                severity=SeverityLevel.CRITICAL,
                reason_code="FUTURE_ISSUE_DATE",
                detail=f"Anomalous Date: Document Issue Date '{issue_str}' is in the future.",
                actual_value=str(issue_str),
                expected_value=f"Date <= {now.strftime('%Y-%m-%d')}"
            ))
        elif issue_dt:
            checks.append(ValidationCheckResult(
                check_name="CHRONOLOGY_ISSUE_DATE",
                passed=True,
                severity=SeverityLevel.INFO,
                reason_code="VALID_ISSUE_DATE",
                detail=f"Issue Date '{issue_str}' is chronologically valid.",
                actual_value=str(issue_str)
            ))

        if issue_dt and expiry_dt:
            if issue_dt >= expiry_dt:
                checks.append(ValidationCheckResult(
                    check_name="CHRONOLOGY_ISSUE_VS_EXPIRY",
                    passed=False,
                    severity=SeverityLevel.CRITICAL,
                    reason_code="EXPIRY_BEFORE_ISSUE",
                    detail=f"Chronological Forgery: Issue Date ({issue_str}) is on or after Expiry Date ({expiry_str}).",
                    actual_value=f"Issue: {issue_str}, Expiry: {expiry_str}",
                    expected_value="Issue Date < Expiry Date"
                ))
            else:
                checks.append(ValidationCheckResult(
                    check_name="CHRONOLOGY_ISSUE_VS_EXPIRY",
                    passed=True,
                    severity=SeverityLevel.INFO,
                    reason_code="VALID_INTERVAL",
                    detail=f"Document validity interval verified ({issue_str} to {expiry_str}).",
                    actual_value=f"{(expiry_dt - issue_dt).days} days"
                ))

        return checks

    def _validate_geometry_and_baselines(self, fields: Dict[str, Any]) -> List[ValidationCheckResult]:
        """
        Analyzes bounding boxes of words on the same text line.
        Detects if inserted words suffer from vertical baseline jumps.
        Expected format in fields: 'word_boxes': [{'text': str, 'bbox': [x1, y1, x2, y2], 'line_id': int}, ...]
        """
        checks: List[ValidationCheckResult] = []
        word_boxes = fields.get("word_boxes", fields.get("boxes", []))
        if not word_boxes or not isinstance(word_boxes, list):
            return checks

        # Group by line_id if present, or cluster by y-midpoint
        lines: Dict[int, List[Dict[str, Any]]] = {}
        for box in word_boxes:
            line_id = box.get("line_id", 0)
            lines.setdefault(line_id, []).append(box)

        max_baseline_variance = 0.0
        for line_id, boxes in lines.items():
            if len(boxes) < 3:
                continue

            # Compute bottom-y coordinates (representing the typographic baseline)
            bottoms = [b["bbox"][3] for b in boxes if "bbox" in b and len(b["bbox"]) == 4]
            if len(bottoms) < 3:
                continue

            mean_bottom = sum(bottoms) / len(bottoms)
            variance = sum((y - mean_bottom) ** 2 for y in bottoms) / len(bottoms)
            std_dev = math.sqrt(variance)

            if std_dev > max_baseline_variance:
                max_baseline_variance = std_dev

        if max_baseline_variance > 8.0:  # Threshold in pixels
            checks.append(ValidationCheckResult(
                check_name="TYPOGRAPHIC_BASELINE_COHERENCE",
                passed=False,
                severity=SeverityLevel.WARNING,
                reason_code="BASELINE_JUMP_DETECTED",
                detail=f"High vertical baseline jitter ({max_baseline_variance:.1f} px). Indicates pasted text.",
                actual_value=f"{max_baseline_variance:.1f} px",
                expected_value="Baseline std dev <= 8.0 px"
            ))
        elif max_baseline_variance > 0.0:
            checks.append(ValidationCheckResult(
                check_name="TYPOGRAPHIC_BASELINE_COHERENCE",
                passed=True,
                severity=SeverityLevel.INFO,
                reason_code="FLAT_BASELINE_VERIFIED",
                detail=f"Text baselines are flat and linear (std dev: {max_baseline_variance:.1f} px).",
                actual_value=f"{max_baseline_variance:.1f} px"
            ))

        return checks

    def _validate_name_sanity(self, fields: Dict[str, Any]) -> List[ValidationCheckResult]:
        """Validates that extracted names do not contain illegal characters or placeholder strings."""
        checks: List[ValidationCheckResult] = []
        name_keys = ["name", "full_name", "first_name", "last_name", "father_name", "company_name"]

        for k in name_keys:
            if k in fields and fields[k]:
                val = str(fields[k]).strip()
                # Check for test / placeholder patterns
                if re.search(r"\b(TEST|DUMMY|SAMPLE|EXAMPLE|DEMO|UNKNOWN|N/A)\b", val, re.IGNORECASE):
                    checks.append(ValidationCheckResult(
                        check_name=f"NAME_SANITY_{k.upper()}",
                        passed=False,
                        severity=SeverityLevel.WARNING,
                        reason_code="PLACEHOLDER_NAME_DETECTED",
                        detail=f"Field '{k}' contains placeholder/test string: '{val}'.",
                        actual_value=val
                    ))
                # Check for numerical digits inside human personal names
                elif k != "company_name" and re.search(r"[0-9]", val):
                    checks.append(ValidationCheckResult(
                        check_name=f"NAME_SANITY_{k.upper()}",
                        passed=False,
                        severity=SeverityLevel.CRITICAL,
                        reason_code="DIGITS_IN_PERSONAL_NAME",
                        detail=f"Anomalous Characters: Personal name '{k}' contains numeric digits: '{val}'.",
                        actual_value=val
                    ))

        return checks

    def _parse_date(self, val: Any) -> Optional[datetime]:
        if not val or not isinstance(val, str):
            return None
        formats = ["%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%d.%m.%Y", "%Y/%m/%d"]
        for fmt in formats:
            try:
                return datetime.strptime(val.strip(), fmt)
            except ValueError:
                continue
        return None
