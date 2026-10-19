"""
Layer 2: Corporate KYB (GSTIN, CIN & Financial Ledger) Validator
================================================================
Implements official statutory and mathematical validation for Indian business
entities under the Goods and Services Tax Act, 2017 and Companies Act, 2013.

Supported KYB Documents:
  1. GSTIN (15-character Goods & Services Tax Identification Number):
     - State Census Code (01-38)
     - Embedded 10-character Income Tax PAN
     - Entity registration counter
     - Default statutory constant 'Z'
     - Modulo-36 Luhn-variant checksum verification
  2. CIN (21-character Corporate Identification Number):
     - Listing status (L/U)
     - 5-digit NIC Economic Activity Code
     - 2-letter State Code
     - 4-digit Year of Incorporation (1850 <= Year <= Current)
     - Company Classification (PTC, PLC, FTC, GOI, SGC, NPL, ULL)
     - 6-digit Registrar of Companies (RoC) sequential number
  3. Financial Statement & Bank Ledger Mathematical Reconciliation:
     - Opening Balance + Total Inflows (Credits) - Total Outflows (Debits) == Closing Balance

Design Pattern: Strategy / Rule Validator (SOLID: Single Responsibility)
Academic Standard: 100% Algorithmic, Statutory Standard, Zero Hardcoding.
"""

from datetime import datetime
import re
from typing import Dict, List, Optional, Tuple, Any

from src.layer2.base import (
    BaseStructuralValidator,
    DocumentValidationReport,
    ValidationCheckResult,
    SeverityLevel
)
from src.layer2.pan_structural_validator import PANStructuralValidator


# Official Indian State & Union Territory GST Census Codes
VALID_GST_STATE_CODES: Dict[str, str] = {
    "01": "Jammu and Kashmir", "02": "Himachal Pradesh", "03": "Punjab", "04": "Chandigarh",
    "05": "Uttarakhand", "06": "Haryana", "07": "Delhi", "08": "Rajasthan",
    "09": "Uttar Pradesh", "10": "Bihar", "11": "Sikkim", "12": "Arunachal Pradesh",
    "13": "Nagaland", "14": "Manipur", "15": "Mizoram", "16": "Tripura",
    "17": "Meghalaya", "18": "Assam", "19": "West Bengal", "20": "Jharkhand",
    "21": "Odisha", "22": "Chhattisgarh", "23": "Madhya Pradesh", "24": "Gujarat",
    "26": "Dadra and Nagar Haveli & Daman and Diu", "27": "Maharashtra",
    "29": "Karnataka", "30": "Goa", "31": "Lakshadweep", "32": "Kerala",
    "33": "Tamil Nadu", "34": "Puducherry", "35": "Andaman and Nicobar Islands",
    "36": "Telangana", "37": "Andhra Pradesh", "38": "Ladakh", "97": "Other Territory"
}

# Ministry of Corporate Affairs (MCA) 2-letter State Identifiers
VALID_MCA_STATE_CODES = {
    "AP", "AR", "AS", "BR", "CT", "CG", "DL", "GA", "GJ", "HR", "HP", "JK", "JH",
    "KA", "KL", "MP", "MH", "MN", "ML", "MZ", "NL", "OR", "PB", "RJ", "SK", "TN",
    "TG", "TR", "UP", "UT", "UK", "WB", "CH", "DH", "DD", "LD", "PY", "LA"
}

VALID_COMPANY_TYPES = {"PTC", "PLC", "FTC", "GOI", "SGC", "NPL", "ULL", "OPC"}


class GSTINChecksum:
    """Computes and validates the official GSTIN Modulo-36 check digit."""

    CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    @classmethod
    def calculate_check_char(cls, gstin_14: str) -> Optional[str]:
        if len(gstin_14) != 14:
            return None

        total = 0
        for i, ch in enumerate(gstin_14.upper()):
            idx = cls.CHARS.find(ch)
            if idx == -1:
                return None

            factor = 2 if (i % 2 == 1) else 1
            product = idx * factor
            quotient = product // 36
            remainder = product % 36
            total += quotient + remainder

        check_idx = (36 - (total % 36)) % 36
        return cls.CHARS[check_idx]


class KYBStructuralValidator(BaseStructuralValidator):
    """
    Validates structural, lexical, and mathematical integrity of corporate
    KYB identity artifacts (GSTIN, CIN, Financial Ledger Balances).
    """

    GSTIN_REGEX = re.compile(r"^([0-9]{2})([A-Z]{5}[0-9]{4}[A-Z]{1})([1-9A-Z]{1})(Z)([0-9A-Z]{1})$")
    CIN_REGEX = re.compile(r"^([LU])([0-9]{5})([A-Z]{2})([0-9]{4})([A-Z]{3})([0-9]{6})$")

    def __init__(self):
        super().__init__()
        self.pan_validator = PANStructuralValidator()

    @property
    def supported_document_type(self) -> str:
        return "KYB_CORPORATE"

    def validate(self, fields: Dict[str, Any]) -> DocumentValidationReport:
        checks: List[ValidationCheckResult] = []
        doc_subtype = "UNKNOWN"

        # Route 1: GSTIN Validation
        if "gstin" in fields or "gst_number" in fields:
            doc_subtype = "GSTIN"
            raw_gstin = str(fields.get("gstin", fields.get("gst_number", ""))).strip().upper()
            gst_checks = self._validate_gstin(raw_gstin, fields)
            checks.extend(gst_checks)

        # Route 2: Corporate CIN Validation
        if "cin" in fields or "corporate_id" in fields:
            doc_subtype = "CIN" if doc_subtype == "UNKNOWN" else f"{doc_subtype}_AND_CIN"
            raw_cin = str(fields.get("cin", fields.get("corporate_id", ""))).strip().upper()
            cin_checks = self._validate_cin(raw_cin, fields)
            checks.extend(cin_checks)

        # Route 3: Financial Statement Ledger Mathematical Reconciliation
        if all(k in fields for k in ("opening_balance", "closing_balance")) and (
            "total_credits" in fields or "total_debits" in fields
        ):
            ledger_checks = self._validate_ledger_math(fields)
            checks.extend(ledger_checks)

        if not checks:
            checks.append(ValidationCheckResult(
                check_name="KYB_PAYLOAD_CHECK",
                passed=False,
                severity=SeverityLevel.WARNING,
                reason_code="NO_KYB_FIELDS_FOUND",
                detail="No recognized KYB corporate identifiers (GSTIN, CIN, Ledger) present in payload."
            ))

        score, crit, warn = self.compute_integrity_score(checks)
        return DocumentValidationReport(
            document_type=f"KYB_{doc_subtype}",
            is_structurally_valid=(crit == 0),
            structural_integrity_score=score,
            critical_violations_count=crit,
            warning_violations_count=warn,
            checks=checks
        )

    def _validate_gstin(self, gstin: str, fields: Dict[str, Any]) -> List[ValidationCheckResult]:
        checks: List[ValidationCheckResult] = []
        match = self.GSTIN_REGEX.match(gstin)

        if not match or len(gstin) != 15:
            checks.append(ValidationCheckResult(
                check_name="GSTIN_LEXICAL_FORMAT",
                passed=False,
                severity=SeverityLevel.CRITICAL,
                reason_code="INVALID_GSTIN_FORMAT",
                detail=f"GSTIN '{gstin}' violates statutory 15-character structure.",
                actual_value=gstin,
                expected_value="15-character alphanumeric (2 State + 10 PAN + 1 Entity + Z + Check)"
            ))
            return checks

        checks.append(ValidationCheckResult(
            check_name="GSTIN_LEXICAL_FORMAT",
            passed=True,
            severity=SeverityLevel.INFO,
            reason_code="VALID_GSTIN_FORMAT",
            detail=f"GSTIN '{gstin}' conforms to 15-character statutory structure.",
            actual_value=gstin
        ))

        state_code = match.group(1)
        embedded_pan = match.group(2)
        entity_code = match.group(3)
        constant_z = match.group(4)
        check_char = match.group(5)

        # Sub-check 1: State Code Validity
        state_name = VALID_GST_STATE_CODES.get(state_code)
        if state_name:
            checks.append(ValidationCheckResult(
                check_name="GSTIN_STATE_CODE",
                passed=True,
                severity=SeverityLevel.INFO,
                reason_code="VALID_STATE_CODE",
                detail=f"State code '{state_code}' resolves to: {state_name}.",
                actual_value=state_code
            ))
        else:
            checks.append(ValidationCheckResult(
                check_name="GSTIN_STATE_CODE",
                passed=False,
                severity=SeverityLevel.CRITICAL,
                reason_code="INVALID_STATE_CODE",
                detail=f"State code '{state_code}' is not a valid Indian census GST code.",
                actual_value=state_code,
                expected_value="Valid 2-digit state code (01-38)"
            ))

        # Sub-check 2: Embedded PAN Validation
        pan_report = self.pan_validator.validate({"pan_number": embedded_pan, "name": fields.get("trade_name", "")})
        checks.append(ValidationCheckResult(
            check_name="GSTIN_EMBEDDED_PAN",
            passed=pan_report.is_structurally_valid,
            severity=SeverityLevel.CRITICAL if not pan_report.is_structurally_valid else SeverityLevel.INFO,
            reason_code="VALID_EMBEDDED_PAN" if pan_report.is_structurally_valid else "INVALID_EMBEDDED_PAN",
            detail=f"Embedded PAN '{embedded_pan}' inside GSTIN verified." if pan_report.is_structurally_valid else
                   f"Embedded PAN '{embedded_pan}' inside GSTIN is structurally invalid.",
            actual_value=embedded_pan
        ))

        # Sub-check 3: Modulo-36 Checksum Verification
        expected_check = GSTINChecksum.calculate_check_char(gstin[:14])
        if expected_check and check_char == expected_check:
            checks.append(ValidationCheckResult(
                check_name="GSTIN_MOD36_CHECKSUM",
                passed=True,
                severity=SeverityLevel.INFO,
                reason_code="VALID_GSTIN_CHECKSUM",
                detail="Satisfies official statutory Modulo-36 Luhn-variant checksum.",
                actual_value=check_char
            ))
        else:
            checks.append(ValidationCheckResult(
                check_name="GSTIN_MOD36_CHECKSUM",
                passed=False,
                severity=SeverityLevel.CRITICAL,
                reason_code="GSTIN_CHECKSUM_FAILED",
                detail=f"GSTIN failed Modulo-36 checksum. Found '{check_char}', expected '{expected_check}'.",
                actual_value=check_char,
                expected_value=expected_check
            ))

        return checks

    def _validate_cin(self, cin: str, fields: Dict[str, Any]) -> List[ValidationCheckResult]:
        checks: List[ValidationCheckResult] = []
        match = self.CIN_REGEX.match(cin)

        if not match or len(cin) != 21:
            checks.append(ValidationCheckResult(
                check_name="CIN_LEXICAL_FORMAT",
                passed=False,
                severity=SeverityLevel.CRITICAL,
                reason_code="INVALID_CIN_FORMAT",
                detail=f"CIN '{cin}' violates statutory 21-character structure.",
                actual_value=cin,
                expected_value="21-character alphanumeric (Listing + NIC + State + Year + Type + Sequence)"
            ))
            return checks

        checks.append(ValidationCheckResult(
            check_name="CIN_LEXICAL_FORMAT",
            passed=True,
            severity=SeverityLevel.INFO,
            reason_code="VALID_CIN_FORMAT",
            detail=f"CIN '{cin}' satisfies statutory 21-character MCA structure.",
            actual_value=cin
        ))

        listing_status = match.group(1)
        nic_code = match.group(2)
        state_code = match.group(3)
        year_str = match.group(4)
        comp_type = match.group(5)
        seq_num = match.group(6)

        # Check State
        if state_code in VALID_MCA_STATE_CODES:
            checks.append(ValidationCheckResult(
                check_name="CIN_STATE_CODE",
                passed=True,
                severity=SeverityLevel.INFO,
                reason_code="VALID_MCA_STATE",
                detail=f"MCA State identifier '{state_code}' verified.",
                actual_value=state_code
            ))
        else:
            checks.append(ValidationCheckResult(
                check_name="CIN_STATE_CODE",
                passed=False,
                severity=SeverityLevel.CRITICAL,
                reason_code="INVALID_MCA_STATE",
                detail=f"Invalid state code '{state_code}' in CIN.",
                actual_value=state_code
            ))

        # Check Incorporation Year
        current_year = datetime.now().year
        incorp_year = int(year_str)
        if 1850 <= incorp_year <= current_year:
            checks.append(ValidationCheckResult(
                check_name="CIN_INCORPORATION_YEAR",
                passed=True,
                severity=SeverityLevel.INFO,
                reason_code="VALID_INCORP_YEAR",
                detail=f"Incorporation year '{incorp_year}' verified.",
                actual_value=str(incorp_year)
            ))
        else:
            checks.append(ValidationCheckResult(
                check_name="CIN_INCORPORATION_YEAR",
                passed=False,
                severity=SeverityLevel.CRITICAL,
                reason_code="INVALID_INCORP_YEAR",
                detail=f"Incorporation year '{incorp_year}' is chronologically invalid.",
                actual_value=str(incorp_year),
                expected_value=f"1850 <= Year <= {current_year}"
            ))

        # Check Company Classification
        if comp_type in VALID_COMPANY_TYPES:
            checks.append(ValidationCheckResult(
                check_name="CIN_COMPANY_TYPE",
                passed=True,
                severity=SeverityLevel.INFO,
                reason_code="VALID_COMPANY_TYPE",
                detail=f"Company classification '{comp_type}' verified.",
                actual_value=comp_type
            ))
        else:
            checks.append(ValidationCheckResult(
                check_name="CIN_COMPANY_TYPE",
                passed=False,
                severity=SeverityLevel.WARNING,
                reason_code="UNRECOGNIZED_COMPANY_TYPE",
                detail=f"Company classification '{comp_type}' not standard in MCA taxonomy.",
                actual_value=comp_type,
                expected_value=f"One of {VALID_COMPANY_TYPES}"
            ))

        return checks

    def _validate_ledger_math(self, fields: Dict[str, Any]) -> List[ValidationCheckResult]:
        """Validates opening + credits - debits == closing balance."""
        checks: List[ValidationCheckResult] = []
        try:
            opening = float(fields.get("opening_balance", 0.0))
            closing = float(fields.get("closing_balance", 0.0))
            credits = float(fields.get("total_credits", 0.0))
            debits = float(fields.get("total_debits", 0.0))

            expected_closing = round(opening + credits - debits, 2)
            actual_closing = round(closing, 2)
            discrepancy = abs(expected_closing - actual_closing)

            if discrepancy < 0.05:  # Tolerance for minor rounding
                checks.append(ValidationCheckResult(
                    check_name="LEDGER_ARITHMETIC_RECONCILIATION",
                    passed=True,
                    severity=SeverityLevel.INFO,
                    reason_code="LEDGER_MATH_BALANCED",
                    detail=f"Ledger balances reconciled: Opening ({opening}) + Credits ({credits}) - Debits ({debits}) == Closing ({closing}).",
                    actual_value=str(actual_closing)
                ))
            else:
                checks.append(ValidationCheckResult(
                    check_name="LEDGER_ARITHMETIC_RECONCILIATION",
                    passed=False,
                    severity=SeverityLevel.CRITICAL,
                    reason_code="LEDGER_MATH_DISCREPANCY",
                    detail=(
                        f"Financial Ledger Tampering Detected: Closing balance {actual_closing} violates "
                        f"arithmetic ledger equation. Expected {expected_closing} (Discrepancy: {discrepancy:.2f})."
                    ),
                    actual_value=str(actual_closing),
                    expected_value=str(expected_closing)
                ))
        except (ValueError, TypeError) as e:
            checks.append(ValidationCheckResult(
                check_name="LEDGER_ARITHMETIC_RECONCILIATION",
                passed=False,
                severity=SeverityLevel.WARNING,
                reason_code="NON_NUMERIC_LEDGER_VALUES",
                detail=f"Could not parse ledger balances into floating-point numbers: {e}"
            ))

        return checks
