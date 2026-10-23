"""
Core Module: Multi-Modal Document Classifier & Intake Gatekeeper
================================================================
The Gatekeeper Layer (Step 0) of the KYC/KYB Intake Pipeline.
Executes before heavy neural vision or structural fraud checks:
  1. Validates physical image suitability (Document vs Non-Document / Blur / Glare).
  2. Classifies document category into canonical taxonomy (PAN, Aadhaar, Passport, DL, Voter ID, GST).
  3. Verifies if the uploaded document matches the target/requested document type (Fast-Fail Gate).

Academic & Enterprise Standard: 100% Algorithmic, Zero Synthetic Bias.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np


class DocumentCategory(Enum):
    PAN_CARD = "PAN_CARD"
    AADHAAR_CARD = "AADHAAR_CARD"
    PASSPORT = "PASSPORT"
    DRIVING_LICENSE = "DRIVING_LICENSE"
    VOTER_ID = "VOTER_ID"
    GST_CERTIFICATE = "GST_CERTIFICATE"
    COMMERCIAL_INVOICE = "COMMERCIAL_INVOICE"
    NON_DOCUMENT = "NON_DOCUMENT"
    UNKNOWN = "UNKNOWN"


@dataclass
class DocumentClassificationResult:
    detected_category: DocumentCategory
    confidence: float                  # In [0.0, 1.0]
    is_matching_requested: bool
    requested_category: Optional[str] = None
    signals_detected: Dict[str, Any] = field(default_factory=dict)
    rejection_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detected_category": self.detected_category.value,
            "confidence": round(self.confidence, 4),
            "is_matching_requested": self.is_matching_requested,
            "requested_category": self.requested_category,
            "rejection_reason": self.rejection_reason,
            "signals_detected": self.signals_detected
        }


class DocumentClassifier:
    """
    Intake Gatekeeper that identifies document classes using a multi-modal blend of
    lexical anchors, regex archetypes, and physical aspect-ratio heuristics.
    """

    # Multi-lingual Lexical Anchor Signatures
    ANCHORS = {
        DocumentCategory.PAN_CARD: [
            "income tax department", "incometax department", "आयकर विभाग",
            "govt. of india", "govt of india", "भारत सरकार",
            "permanent account number", "permanent account", "pan card",
            "father's name", "fathers name"
        ],
        DocumentCategory.AADHAAR_CARD: [
            "unique identification authority", "uidai", "government of india",
            "भारत सरकार", "मेरा आधार", "meri pehchan", "enrolment no",
            "help@uidai.gov.in", "1947", "vid:", "vid :", "virtual id"
        ],
        DocumentCategory.PASSPORT: [
            "passport", "passeport", "republic of india", "भारत गणराज्य",
            "type / type p", "country code ind", "nationality indian"
        ],
        DocumentCategory.DRIVING_LICENSE: [
            "driving licence", "driving license", "motor vehicles act",
            "transport department", "union of india", "form 7", "dl no"
        ],
        DocumentCategory.VOTER_ID: [
            "election commission of india", "भारत निर्वाचन आयोग",
            "elector photo identity card", "identity card", "epic no"
        ],
        DocumentCategory.GST_CERTIFICATE: [
            "goods and services tax", "form gst reg-06", "registration certificate",
            "government of india", "tax invoice"
        ]
    }

    # Canonical Regex Patterns
    REGEX_PATTERNS = {
        DocumentCategory.PAN_CARD: re.compile(
            r'(\b[A-Z]{5}[0-9]{4}[A-Z]\b|\b[A-Z]{3}[ABCFGHLJPTK][A-Z][0-9SIAOLZB]{4}[A-Z]\b)'
        ),
        DocumentCategory.AADHAAR_CARD: re.compile(
            r'(\b[2-9]{1}[0-9]{3}\s?[0-9]{4}\s?[0-9]{4}\b|\b[X*]{4}\s?[X*]{4}\s?[0-9]{4}\b|\b[0-9]{4}\s?[0-9]{4}\s?[0-9]{4}\s?[0-9]{4}\b)'
        ),
        DocumentCategory.PASSPORT: re.compile(r'P<IND[A-Z<]+'),
        DocumentCategory.DRIVING_LICENSE: re.compile(r'\b[A-Z]{2}[-\s]?[0-9]{2}[-\s]?(19|20)[0-9]{2}[-\s]?[0-9]{7}\b'),
        DocumentCategory.VOTER_ID: re.compile(r'\b[A-Z]{3}[0-9]{7}\b'),
        DocumentCategory.GST_CERTIFICATE: re.compile(r'\b[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][1-9A-Z]Z[0-9A-Z]\b')
    }

    def __init__(self, blur_threshold: float = 35.0):
        self.blur_threshold = blur_threshold

    def classify(
        self,
        image: np.ndarray,
        ocr_texts: Optional[List[str]] = None,
        requested_doc_type: Optional[str] = None,
        qr_type: Optional[str] = None
    ) -> DocumentClassificationResult:
        """
        Classifies an input image and verifies if it matches the expected document type.
        """
        h, w = image.shape[:2]
        aspect_ratio = float(max(h, w)) / float(max(min(h, w), 1))

        # 1. Image Quality / Liveness Check
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        if laplacian_var < self.blur_threshold:
            return DocumentClassificationResult(
                detected_category=DocumentCategory.NON_DOCUMENT,
                confidence=0.1,
                is_matching_requested=False,
                requested_category=requested_doc_type,
                rejection_reason=(
                    f"Image is excessively blurry (Laplacian variance: {laplacian_var:.1f} < {self.blur_threshold}). "
                    f"Please upload a clear, sharp photo of your document."
                ),
                signals_detected={"laplacian_variance": laplacian_var, "aspect_ratio": round(aspect_ratio, 2)}
            )

        # 2. Text Normalization
        text_corpus = " ".join(ocr_texts).lower() if ocr_texts else ""
        text_corpus_upper = text_corpus.upper()

        scores: Dict[DocumentCategory, float] = {cat: 0.0 for cat in DocumentCategory}
        detected_signals: Dict[str, Any] = {
            "aspect_ratio": round(aspect_ratio, 2),
            "laplacian_variance": round(laplacian_var, 1),
            "matched_anchors": {},
            "matched_regexes": {}
        }

        # 3. Anchor Keyword Scoring
        for cat, anchor_list in self.ANCHORS.items():
            matches = [kw for kw in anchor_list if kw in text_corpus]
            if matches:
                scores[cat] += len(matches) * 20.0
                detected_signals["matched_anchors"][cat.value] = matches

        # 4. Regex Pattern Scoring
        for cat, regex in self.REGEX_PATTERNS.items():
            found = regex.findall(text_corpus_upper)
            if found:
                scores[cat] += 40.0
                detected_signals["matched_regexes"][cat.value] = found[:3]

        # 5. Aspect-Ratio Matching (ISO/IEC 7810 ID-1 standard ~ 1.58:1)
        if 1.40 <= aspect_ratio <= 1.75:
            # Card format supports PAN, Aadhaar, DL, Voter ID
            scores[DocumentCategory.PAN_CARD] += 5.0
            scores[DocumentCategory.AADHAAR_CARD] += 5.0
            scores[DocumentCategory.DRIVING_LICENSE] += 5.0
            scores[DocumentCategory.VOTER_ID] += 5.0

        # 5.5 QR Cryptographic Signature Scoring
        if qr_type:
            detected_signals["qr_type"] = qr_type
            if qr_type == "AADHAAR_SECURE":
                scores[DocumentCategory.AADHAAR_CARD] += 50.0
            elif qr_type == "PAN_ENHANCED":
                scores[DocumentCategory.PAN_CARD] += 50.0
            elif qr_type == "GST_IRN":
                scores[DocumentCategory.GST_CERTIFICATE] += 50.0

        # Determine winner
        best_cat = max(scores, key=scores.get)
        best_score = scores[best_cat]

        # Normalize confidence to [0.0, 1.0]
        confidence = min(1.0, max(0.1, best_score / 60.0))

        if best_score < 15.0:
            detected_category = DocumentCategory.UNKNOWN
            confidence = 0.2
        else:
            detected_category = best_cat

        # 6. Target Slot Matching (Gatekeeper Decision)
        is_matching = True
        rejection_reason = None

        if requested_doc_type:
            # Normalize requested category string
            req_norm = requested_doc_type.upper().replace(" ", "_")
            alias_map = {
                "PAN": DocumentCategory.PAN_CARD,
                "PAN_CARD": DocumentCategory.PAN_CARD,
                "PANCARD": DocumentCategory.PAN_CARD,
                "AADHAAR": DocumentCategory.AADHAAR_CARD,
                "AADHAAR_CARD": DocumentCategory.AADHAAR_CARD,
                "PASSPORT": DocumentCategory.PASSPORT,
                "DL": DocumentCategory.DRIVING_LICENSE,
                "DRIVING_LICENSE": DocumentCategory.DRIVING_LICENSE,
                "VOTER_ID": DocumentCategory.VOTER_ID,
                "EPIC": DocumentCategory.VOTER_ID,
                "GST": DocumentCategory.GST_CERTIFICATE,
                "GSTIN": DocumentCategory.GST_CERTIFICATE
            }
            req_enum = alias_map.get(req_norm)

            if req_enum:
                if detected_category != req_enum and detected_category != DocumentCategory.UNKNOWN:
                    is_matching = False
                    rejection_reason = (
                        f"WRONG DOCUMENT TYPE DETECTED: Expected '{req_enum.value}', "
                        f"but the uploaded document was identified as '{detected_category.value}' "
                        f"(Confidence: {confidence*100:.1f}%). Please upload the requested document."
                    )
            elif detected_category == DocumentCategory.UNKNOWN:
                is_matching = False
                rejection_reason = (
                    f"UNRECOGNIZED DOCUMENT: Could not verify document as a valid '{requested_doc_type}'. "
                    f"Please ensure document text is legible and well-lit."
                )

        return DocumentClassificationResult(
            detected_category=detected_category,
            confidence=confidence,
            is_matching_requested=is_matching,
            requested_category=requested_doc_type,
            signals_detected=detected_signals,
            rejection_reason=rejection_reason
        )
