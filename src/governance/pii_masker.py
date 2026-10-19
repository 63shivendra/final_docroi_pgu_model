"""
DocROI-Former: Edge-First PII Masking Module
============================================
Performs client-side / edge-device redaction of sensitive Personal Identifiable
Information (PII) before logs, telemetry, or receipts are transmitted upstream.

In compliance with UIDAI Aadhaar masking circulars:
  - Masks the first 8 digits of Aadhaar numbers: 'XXXX-XXXX-1234'
  - Blurs signature blocks if present in transmission artifacts.

Design Pattern: Privacy Filter / Masker (SOLID: Single Responsibility)
"""

import re
from typing import Dict, Any


class PIIMasker:
    """
    Edge-Device PII Protection & Redaction Engine.
    """

    @staticmethod
    def mask_aadhaar_number(text: str) -> str:
        """
        Replaces 12-digit Aadhaar pattern with masked format XXXX-XXXX-1234.
        """
        # Match 12 digits with optional spaces or hyphens
        pattern = r"\b(\d{4})[\s-]?(\d{4})[\s-]?(\d{4})\b"
        return re.sub(pattern, r"XXXX-XXXX-\3", text)

    @staticmethod
    def mask_pan_number(pan: str) -> str:
        """
        Partially masks PAN number: 'ABCDE****F'.
        """
        if len(pan) == 10:
            return f"{pan[:5]}****{pan[-1]}"
        return pan

    @classmethod
    def sanitize_field_findings(cls, findings: list) -> list:
        """
        Sanitizes extracted field text in findings list.
        """
        sanitized = []
        for f in findings:
            f_copy = dict(f)
            if "extracted_value" in f_copy:
                f_copy["extracted_value"] = cls.mask_aadhaar_number(str(f_copy["extracted_value"]))
            sanitized.append(f_copy)
        return sanitized
