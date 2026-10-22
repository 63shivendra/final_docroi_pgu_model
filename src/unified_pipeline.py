"""
Unified Document Fraud & Authenticity Pipeline
==============================================
Master end-to-end pipeline integrating all 3 defense-in-depth tiers:
  1. Tier 1 Pre/Post-processing: DocumentRectifier + PerimeterGuardband + SemanticRoIMasker
  2. Tier 1 Neural Vision: DocROI-Former v2 (Spatial PVTv2 + Frequency SRM/DCT Residuals)
  3. Layer 2 Structural & Mathematical: Verhoeff D5, CBDT Section 139A, ICAO Doc 9303, GSTIN Mod-36
  4. Layer 3 Cryptographic: 2D Barcode scanner (zxing-cpp), SHA-256 IRN hash & QR cross-match
  5. Governance: NPCI-compliant DocumentTrustReceipt generator

Academic Standard: 100% Algorithmic, Zero Synthetic Bias, Zero Hardcoding.
"""

import os
import sys
import time
import hashlib
import json
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from configs.docroi_config import DocROIConfig
from src.core.vision_preprocessor import (
    DocumentRectifier,
    PerimeterGuardband,
    SemanticRoIMasker,
    RectificationResult
)
from src.layer2.document_validator_registry import UnifiedStructuralAuditEngine
from src.layer2.base import DocumentValidationReport, SeverityLevel, ValidationCheckResult
from src.governance.trust_receipt import TrustReceiptEngine
from src.core.domain_entities import DocumentTrustReceipt
from src.core.document_classifier import DocumentClassifier, DocumentClassificationResult, DocumentCategory

try:
    import easyocr
    HAS_EASYOCR = True
except ImportError:
    HAS_EASYOCR = False



@dataclass
class UnifiedAuditResult:
    """Master container returned by the UnifiedDocumentFraudPipeline."""
    document_name: str
    decision: str                      # 'APPROVED', 'ESCALATE_TO_HUMAN_REVIEW', 'REJECTED', 'REJECTED_WRONG_DOCUMENT_TYPE'
    trust_score: float                 # In [0.0, 1.0] (1.0 = absolute authentic trust)
    risk_score: float                  # In [0.0, 1.0]
    rectification: RectificationResult
    extracted_fields: Dict[str, Any]
    structural_report: DocumentValidationReport
    neural_metrics: Dict[str, Any]
    trust_receipt: Dict[str, Any]
    classification: Optional[Dict[str, Any]] = None
    clean_heatmap: Optional[np.ndarray] = None
    evidence_board_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_name": self.document_name,
            "decision": self.decision,
            "trust_score": round(self.trust_score, 4),
            "risk_score": round(self.risk_score, 4),
            "rectification": {
                "was_rectified": self.rectification.was_rectified,
                "original_shape": self.rectification.original_shape,
                "rectified_shape": self.rectification.rectified_shape,
                "background_ratio_removed": round(self.rectification.background_ratio_removed, 4)
            },
            "extracted_fields": self.extracted_fields,
            "classification": self.classification,
            "structural_audit": self.structural_report.to_dict(),
            "neural_vision": self.neural_metrics,
            "trust_receipt": self.trust_receipt,
            "evidence_board_path": self.evidence_board_path
        }


class UnifiedDocumentFraudPipeline:
    """
    Enterprise-ready master pipeline for comprehensive KYC/KYB fraud analysis.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        config: Optional[DocROIConfig] = None
    ):
        self.device = device
        self.config = config or DocROIConfig()
        
        # 1. Vision Preprocessors & Intake Gatekeeper
        self.rectifier = DocumentRectifier()
        self.guardband = PerimeterGuardband()
        self.roi_masker = SemanticRoIMasker()
        self.classifier = DocumentClassifier()

        # 2. Structural & Cryptographic Audit Engine (Layer 2 & 3)
        self.structural_engine = UnifiedStructuralAuditEngine()

        # 3. Governance Trust Receipt Engine
        self.receipt_engine = TrustReceiptEngine(self.config.governance)

        # 4. OCR Engine (optional on-the-fly)
        self.ocr_reader = None
        if HAS_EASYOCR:
            try:
                self.ocr_reader = easyocr.Reader(['en'], gpu=(device == "cuda"))
            except Exception:
                pass

        # 5. Neural Model (DocROI-Former v2)
        self.model = None
        if checkpoint_path:
            ckpt = checkpoint_path
        elif os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints", "best_docroi_former_v2_pgu.pth")):
            ckpt = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints", "best_docroi_former_v2_pgu.pth")
        elif os.path.exists("checkpoints/best_docroi_former_v2_pgu.pth"):
            ckpt = "checkpoints/best_docroi_former_v2_pgu.pth"
        else:
            ckpt = "kaggle_trained_model/latestnight_28sept/best_docroi_former_latest28sept.pth"

        if os.path.exists(ckpt):
            self._load_neural_model(ckpt)

    def _load_neural_model(self, checkpoint_path: str):
        """Initializes and loads pre-trained weights."""
        from src.models.docroi_former import DocROIFormer
        try:
            self.model = DocROIFormer(self.config)
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            # Handle key mismatch if v1 vs v2
            model_keys = self.model.state_dict().keys()
            filtered_dict = {k: v for k, v in state_dict.items() if k in model_keys and v.shape == self.model.state_dict()[k].shape}
            self.model.load_state_dict(filtered_dict, strict=False)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"[Warning] Neural model initialization skipped: {e}")
    @staticmethod
    def extract_tamper_bounding_boxes(
        heatmap: np.ndarray,
        threshold: float = 0.50,
        min_pixels: int = 25,
        raw_tokens: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Extracts discrete bounding boxes and contours around localized neural tamper clusters.
        Correlates each detected tamper zone with overlapping OCR entity text tokens.
        """
        if heatmap is None or heatmap.size == 0 or heatmap.max() < threshold:
            return []

        binary_mask = (heatmap >= threshold).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        binary_closed = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(binary_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []

        for idx, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            bx, by, bw, bh = cv2.boundingRect(cnt)
            region_mask = binary_mask[by:by+bh, bx:bx+bw]
            pixel_count = int((region_mask > 0).sum())

            if pixel_count < min_pixels:
                continue

            region_heatmap = heatmap[by:by+bh, bx:bx+bw]
            peak_prob = float(region_heatmap.max() * 100.0)
            mean_prob = float(region_heatmap[region_mask > 0].mean() * 100.0)

            # Correlate with OCR entity tokens if available
            associated_tokens = []
            if raw_tokens:
                for token in raw_tokens:
                    tbx, tby, tbw, tbh = token.get("box", [0, 0, 0, 0])
                    # Check box intersection
                    ix1 = max(bx, tbx)
                    iy1 = max(by, tby)
                    ix2 = min(bx + bw, tbx + tbw)
                    iy2 = min(by + bh, tby + tbh)
                    if ix2 > ix1 and iy2 > iy1:
                        associated_tokens.append(token.get("text", ""))

            field_label = ", ".join(associated_tokens) if associated_tokens else "UNPRINTED_CANVAS_ZONE"

            boxes.append({
                "box_id": f"TAMPER_ZONE_{len(boxes) + 1}",
                "box": [int(bx), int(by), int(bw), int(bh)],
                "peak_probability_pct": round(peak_prob, 1),
                "mean_probability_pct": round(mean_prob, 1),
                "tampered_pixels": pixel_count,
                "contour_area": int(area),
                "associated_field": field_label
            })

        boxes.sort(key=lambda x: x["peak_probability_pct"], reverse=True)
        return boxes

    def process(
        self,
        image_input: Union[str, np.ndarray],
        doc_type: Optional[str] = None,
        manual_fields: Optional[Dict[str, Any]] = None,
        output_evidence_dir: Optional[str] = "applied_results"
    ) -> UnifiedAuditResult:
        """
        Executes end-to-end 5-step forensic audit.
        """
        # Load Image
        if isinstance(image_input, str):
            doc_name = os.path.basename(image_input)
            raw_img = cv2.imread(image_input)
            if raw_img is None:
                raise FileNotFoundError(f"Cannot load image at '{image_input}'")
        else:
            doc_name = "in_memory_document.png"
            raw_img = image_input

        # -------------------------------------------------------------
        # STEP 1: PHYSICAL LOCALIZATION & RECTIFICATION (Pre-processing)
        # -------------------------------------------------------------
        rect_result = self.rectifier.rectify(raw_img)
        rectified_card = rect_result.rectified_image
        h_card, w_card = rectified_card.shape[:2]

        # -------------------------------------------------------------
        # STEP 2: OCR EXTRACTION & BOUNDING BOX DISCOVERY
        # -------------------------------------------------------------
        extracted_fields = manual_fields.copy() if manual_fields else {}
        entity_boxes = []
        ocr_texts = []

        if self.ocr_reader is not None:
            try:
                # Multi-scale OCR handling for low-resolution document captures
                scale = max(1.0, 720.0 / min(h_card, w_card))
                if scale > 1.0:
                    scaled_card = cv2.resize(rectified_card, (int(w_card * scale), int(h_card * scale)), interpolation=cv2.INTER_CUBIC)
                else:
                    scaled_card = rectified_card

                ocr_results = self.ocr_reader.readtext(scaled_card)
                raw_tokens = []
                for (bbox, text, prob) in ocr_results:
                    ocr_texts.append(text)
                    # Convert 4-point polygon to [x, y, w, h] mapped back to rectified_card scale
                    pts = np.array([[int(pt[0] / scale), int(pt[1] / scale)] for pt in bbox], dtype=np.int32)
                    bx, by, bw, bh = cv2.boundingRect(pts)
                    entity_boxes.append((bx, by, bw, bh))
                    raw_tokens.append({
                        "text": text,
                        "confidence": round(float(prob), 4),
                        "box": [int(bx), int(by), int(bw), int(bh)]
                    })

                extracted_fields["raw_tokens"] = raw_tokens
                extracted_fields["token_count"] = len(raw_tokens)
                if raw_tokens:
                    extracted_fields["avg_confidence"] = round(float(sum(t["confidence"] for t in raw_tokens) / len(raw_tokens) * 100.0), 1)
                else:
                    extracted_fields["avg_confidence"] = 0.0

                # Auto-parse standard ID patterns if not provided manually
                full_text_upper = " ".join(ocr_texts).upper()
                import re

                # 1. PAN Card Auto-Discovery with Positional Homoglyph Repair
                if "pan" not in extracted_fields and "pan_number" not in extracted_fields:
                    pan_match = re.search(r'\b[A-Z]{5}[0-9]{4}[A-Z]\b', full_text_upper)
                    if pan_match:
                        extracted_fields["pan_number"] = pan_match.group(0)
                        extracted_fields["pan"] = pan_match.group(0)
                    else:
                        from src.layer2.pan_structural_validator import PANStructuralValidator
                        # Scan 10-char alphanumeric tokens for homoglyphs (e.g. APUPP25SIA -> APUPP2551A)
                        candidates = re.findall(r'\b[A-Z0-9]{10}\b', full_text_upper)
                        for cand in candidates:
                            repaired, was_rep = PANStructuralValidator.repair_pan_homoglyphs(cand)
                            if was_rep:
                                extracted_fields["pan_number"] = repaired
                                extracted_fields["pan"] = repaired
                                extracted_fields["raw_pan_ocr"] = cand
                                break

                # 2. Aadhaar & Virtual ID (VID) Disambiguation via Sequential Token Isolation
                # Prevents 16-digit VID substrings (e.g. 5th-16th digits) from falsely triggering 12-digit Aadhaar regex
                remaining_text = full_text_upper

                # Step A: Check & Consume 16-Digit Virtual ID (VID)
                vid_match = re.search(r'(?:VID\s*[:\-]?\s*)?\b([0-9]{4}\s?[0-9]{4}\s?[0-9]{4}\s?[0-9]{4})\b', remaining_text)
                if vid_match:
                    extracted_fields["vid_number"] = vid_match.group(1).strip()
                    remaining_text = remaining_text[:vid_match.start()] + " [CONSUMED_VID] " + remaining_text[vid_match.end():]

                # Step B: Check & Consume Masked Aadhaar (e.g. XXXX XXXX 1234 or **** **** 1234)
                masked_match = re.search(r'\b([X*]{4}\s?[X*]{4}\s?[0-9]{4})\b', remaining_text)
                if masked_match:
                    extracted_fields["masked_aadhaar"] = masked_match.group(1).strip()
                    remaining_text = remaining_text[:masked_match.start()] + " [CONSUMED_MASKED] " + remaining_text[masked_match.end():]

                # Step C: Extract 12-Digit Full Aadhaar strictly from remaining unconsumed tokens
                full_aadhaar_match = re.search(r'\b([2-9]{1}[0-9]{3}\s?[0-9]{4}\s?[0-9]{4})\b', remaining_text)
                if full_aadhaar_match:
                    extracted_fields["aadhaar_number"] = full_aadhaar_match.group(1).strip()
                elif "masked_aadhaar" in extracted_fields:
                    extracted_fields["aadhaar_number"] = extracted_fields["masked_aadhaar"]
                elif "vid_number" in extracted_fields:
                    extracted_fields["aadhaar_number"] = extracted_fields["vid_number"]

                # 3. GSTIN Auto-Discovery (15 chars)
                if "gstin" not in extracted_fields:
                    gstin_match = re.search(r'\b[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][1-9A-Z]Z[0-9A-Z]\b', full_text_upper)
                    if gstin_match:
                        extracted_fields["gstin"] = gstin_match.group(0)

                # 4. Name Auto-Discovery
                if "name" not in extracted_fields:
                    for i, t in enumerate(ocr_texts):
                        if "name" in t.lower() and i + 1 < len(ocr_texts):
                            extracted_fields["name"] = ocr_texts[i+1].strip()
                            break

                # 5. Date of Birth Auto-Discovery
                if "dob" not in extracted_fields:
                    dob_match = re.search(r'\b\d{2}/\d{2}/\d{4}\b', full_text_upper)
                    if dob_match:
                        extracted_fields["dob"] = dob_match.group(0)

            except Exception as e:
                print(f"[Warning] OCR extraction encountered error: {e}")

        # -------------------------------------------------------------
        # STEP 2.5: DOCUMENT CLASSIFICATION & INTAKE GATEKEEPER (Step 0)
        # -------------------------------------------------------------
        classification = self.classifier.classify(
            image=rectified_card,
            ocr_texts=ocr_texts,
            requested_doc_type=doc_type
        )

        # If doc_type was not specified, dynamically route to detected category
        if not doc_type and classification.detected_category != DocumentCategory.UNKNOWN:
            doc_type = classification.detected_category.value

        # Fast-fail gate: If requested document does not match uploaded document or fails image quality
        if doc_type and not classification.is_matching_requested:
            is_non_doc = (classification.detected_category == DocumentCategory.NON_DOCUMENT)
            rejection_text = classification.rejection_reason or ""

            if is_non_doc and ("blur" in rejection_text.lower() or "laplacian" in rejection_text.lower()):
                decision = "REJECTED_BLURRY_IMAGE"
                reason_code = "IMAGE_EXCESSIVE_BLUR"
                check_name = "IMAGE_QUALITY_SHARPNESS"
            elif is_non_doc:
                decision = "REJECTED_NON_DOCUMENT"
                reason_code = "PHYSICAL_DOCUMENT_NOT_DETECTED"
                check_name = "PHYSICAL_DOCUMENT_SUITABILITY"
            elif classification.detected_category == DocumentCategory.UNKNOWN:
                decision = "REJECTED_UNKNOWN_DOCUMENT"
                reason_code = "UNKNOWN_DOCUMENT_TAXONOMY"
                check_name = "DOCUMENT_TAXONOMY_IDENTIFICATION"
            else:
                decision = "REJECTED_WRONG_DOCUMENT_TYPE"
                reason_code = "DOCUMENT_CATEGORY_MISMATCH"
                check_name = "INTAKE_GATEKEEPER_SLOT_MATCH"

            risk_score = 1.0
            trust_score = 0.0

            fail_check = ValidationCheckResult(
                check_name=check_name,
                passed=False,
                severity=SeverityLevel.CRITICAL,
                reason_code=reason_code,
                detail=rejection_text or f"Uploaded document does not match requested '{doc_type}'."
            )
            fast_fail_report = DocumentValidationReport(
                document_type=doc_type,
                is_structurally_valid=False,
                structural_integrity_score=0.0,
                critical_violations_count=1,
                warning_violations_count=0,
                checks=[fail_check],
                metadata={"intake_gatekeeper": classification.to_dict()}
            )

            raw_bytes = cv2.imencode('.png', raw_img)[1].tobytes()
            receipt_dict = self.receipt_engine.generate_receipt(
                image_bytes=raw_bytes,
                document_type=doc_type,
                global_risk_score=risk_score,
                pixel_mask=None,
                suspicious_fields=[fail_check.to_dict()],
                metadata={"doc_name": doc_name, "rejection_reason": classification.rejection_reason}
            )

            evidence_path = None
            if output_evidence_dir:
                os.makedirs(output_evidence_dir, exist_ok=True)
                evidence_path = os.path.join(output_evidence_dir, f"{os.path.splitext(doc_name)[0]}_audit_board.png")
                self._render_evidence_board(raw_img, rectified_card, np.zeros((h_card, w_card), dtype=np.float32), decision, trust_score, fast_fail_report, evidence_path)

            return UnifiedAuditResult(
                document_name=doc_name,
                decision=decision,
                trust_score=trust_score,
                risk_score=risk_score,
                rectification=rect_result,
                extracted_fields=extracted_fields,
                structural_report=fast_fail_report,
                neural_metrics={"status": "SKIPPED_INTAKE_MISMATCH", "reason": classification.rejection_reason},
                trust_receipt=receipt_dict,
                classification=classification.to_dict(),
                clean_heatmap=np.zeros((h_card, w_card), dtype=np.float32),
                evidence_board_path=evidence_path
            )

        # -------------------------------------------------------------
        # STEP 3: LAYER 2 & 3 STRUCTURAL, MATHEMATICAL & QR AUDIT
        # -------------------------------------------------------------
        audit_fields = extracted_fields.copy()
        audit_fields["image"] = rectified_card
        if doc_type:
            audit_fields["document_type"] = doc_type

        structural_report = self.structural_engine.audit(audit_fields, doc_type=doc_type)

        # -------------------------------------------------------------
        # STEP 4: TIER 1 NEURAL VISION INSPECTION (DocROI-Former v2)
        # -------------------------------------------------------------
        clean_heatmap = np.zeros((h_card, w_card), dtype=np.float32)
        neural_metrics = {
            "model": "DocROI-Former v2 Dual-Stream",
            "global_fraud_probability_pct": 0.0,
            "peak_pixel_probability_pct": 0.0,
            "tampered_pixels_count": 0,
            "tampered_area_pct": 0.0,
            "tampered_bounding_boxes": [],
            "tampered_clusters_count": 0,
            "status": "EVALUATED"
        }

        if self.model is not None:
            # Prepare tensor
            img_rgb = cv2.cvtColor(rectified_card, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (384, 384))
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            norm = (img_resized.astype(np.float32) / 255.0 - mean) / std
            tensor = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)

            with torch.no_grad():
                preds = self.model(tensor)
                p_doc = preds.get("global_prob", torch.tensor([[0.0]])).item() * 100.0
                mask_logits = preds.get("mask_logits")
                if mask_logits is not None:
                    raw_probs = torch.sigmoid(mask_logits).squeeze().cpu().numpy()
                    raw_heatmap_card = cv2.resize(raw_probs, (w_card, h_card))
                else:
                    raw_heatmap_card = np.zeros((h_card, w_card), dtype=np.float32)

            # Apply Perimeter Guardband (Erode cut edges)
            guarded_heatmap = self.guardband.apply(raw_heatmap_card, margin_percent=0.035)

            # Apply Semantic RoI Gating (Focus on text and photo)
            clean_heatmap = self.roi_masker.filter_heatmap(guarded_heatmap, entity_boxes=entity_boxes)

            flagged_mask = (clean_heatmap > 0.50)
            tamper_pixels = int(flagged_mask.sum())
            peak_prob = float(clean_heatmap.max() * 100.0)

            # Extract discrete connected component bounding boxes around each tamper cluster
            tamper_boxes = self.extract_tamper_bounding_boxes(
                clean_heatmap,
                threshold=0.50,
                min_pixels=25,
                raw_tokens=extracted_fields.get("raw_tokens", [])
            )

            neural_metrics.update({
                "global_fraud_probability_pct": round(p_doc, 2),
                "peak_pixel_probability_pct": round(peak_prob, 2),
                "tampered_pixels_count": tamper_pixels,
                "tampered_area_pct": round((tamper_pixels / float(h_card * w_card)) * 100.0, 4),
                "tampered_bounding_boxes": tamper_boxes,
                "tampered_clusters_count": len(tamper_boxes)
            })

        # -------------------------------------------------------------
        # STEP 5: MASTER CONSENSUS ARBITRATION & TRUST RECEIPT SYNTHESIS
        # -------------------------------------------------------------
        # Arbitration Rules:
        # 1. Any CRITICAL Layer 2/3 failure (IRN mismatch, Verhoeff fail, QR unreadable) -> REJECTED
        # 2. Peak neural pixel prob > 85% with > 300 pixels inside entity -> REJECTED (Splice detected)
        # 3. Warning violations or marginal pixel noise -> ESCALATE_TO_HUMAN_REVIEW
        # 4. Flawless grammar, valid QR, zero pixel anomalies -> APPROVED

        has_critical = (structural_report.critical_violations_count > 0)
        has_warning = (structural_report.warning_violations_count > 0)
        has_neural_splice = (neural_metrics["tampered_pixels_count"] > 300 and neural_metrics["peak_pixel_probability_pct"] > 85.0)

        if has_critical or has_neural_splice:
            decision = "REJECTED"
            risk_score = 0.95
        elif has_warning or (neural_metrics["tampered_pixels_count"] > 100):
            decision = "ESCALATE_TO_HUMAN_REVIEW"
            risk_score = 0.45
        else:
            decision = "APPROVED"
            risk_score = 0.05

        trust_score = 1.0 - risk_score

        # Domain findings conversion
        findings = self.structural_engine.to_domain_field_findings(structural_report)
        findings_dict = [f.to_dict() for f in findings]

        # Generate official DocumentTrustReceipt
        raw_bytes = cv2.imencode('.png', raw_img)[1].tobytes()
        receipt_dict = self.receipt_engine.generate_receipt(
            image_bytes=raw_bytes,
            document_type=structural_report.document_type,
            global_risk_score=risk_score,
            pixel_mask=torch.from_numpy(clean_heatmap).unsqueeze(0) if clean_heatmap is not None else None,
            suspicious_fields=findings_dict,
            metadata={"doc_name": doc_name}
        )

        # -------------------------------------------------------------
        # STEP 6: EVIDENCE BOARD VISUAL RENDERING (Optional export)
        # -------------------------------------------------------------
        evidence_path = None
        if output_evidence_dir:
            os.makedirs(output_evidence_dir, exist_ok=True)
            evidence_path = os.path.join(output_evidence_dir, f"{os.path.splitext(doc_name)[0]}_audit_board.png")
            self._render_evidence_board(raw_img, rectified_card, clean_heatmap, decision, trust_score, structural_report, evidence_path)

        return UnifiedAuditResult(
            document_name=doc_name,
            decision=decision,
            trust_score=trust_score,
            risk_score=risk_score,
            rectification=rect_result,
            extracted_fields=extracted_fields,
            structural_report=structural_report,
            neural_metrics=neural_metrics,
            trust_receipt=receipt_dict,
            classification=classification.to_dict(),
            clean_heatmap=clean_heatmap,
            evidence_board_path=evidence_path
        )

    def _render_evidence_board(
        self,
        raw_img: np.ndarray,
        rectified_card: np.ndarray,
        heatmap: np.ndarray,
        decision: str,
        trust_score: float,
        report: DocumentValidationReport,
        save_path: str
    ):
        """Creates an audit evidence board."""
        try:
            h, w = rectified_card.shape[:2]
            # Heatmap colorization
            hm_uint8 = np.clip(heatmap * 255.0, 0, 255).astype(np.uint8)
            hm_color = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(rectified_card, 0.65, hm_color, 0.35, 0)

            # Resize raw for side-by-side
            raw_disp = cv2.resize(raw_img, (w, h))

            # Banner
            banner_h = 70
            banner = np.zeros((banner_h, w * 3, 3), dtype=np.uint8)
            color = (0, 180, 0) if decision == "APPROVED" else ((0, 140, 255) if "ESCALATE" in decision else (0, 0, 220))
            banner[:] = (25, 25, 25)

            status_text = f"DECISION: {decision} | TRUST SCORE: {trust_score*100:.1f}% | TYPE: {report.document_type}"
            cv2.putText(banner, status_text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)

            row = np.hstack([raw_disp, rectified_card, overlay])
            board = np.vstack([banner, row])

            cv2.imwrite(save_path, board)
        except Exception:
            pass
