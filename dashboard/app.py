"""
DocROI-Former v2 Forensic Intelligence Suite - Local Web Dashboard
===================================================================
A self-contained, high-performance forensic inspection server for
visualizing multi-tier fraud detection, Step 0 Gatekeeper classification,
structural mathematical checks, and neural heatmaps in real-time.
"""

import os
import sys
import base64
import json
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.unified_pipeline import UnifiedDocumentFraudPipeline

app = Flask(__name__, template_folder="templates", static_folder="static")

# Initialize master pipeline once at startup
print("[Forensic Suite] Initializing DocROI-Former Master Pipeline...")
pipeline = UnifiedDocumentFraudPipeline(device="cuda" if os.environ.get("USE_CUDA") == "1" else "cpu")
print("[Forensic Suite] Pipeline initialized successfully!")

# Preset sample definitions for instant 1-click testing
PRESETS = {
    "aadhaar_01_clean": {
        "name": "Authentic Aadhaar (Secure QR)",
        "category": "AADHAAR_CARD",
        "description": "Authentic Aadhaar card with valid UIDAI Secure QR payload, 13 OCR tokens, and valid Verhoeff checksum.",
        "path": os.path.join(REPO_ROOT, "test_samples", "aadhaar", "aadhaar_01_clean_secure_qr.jpg")
    },
    "aadhaar_03_tampered_name": {
        "name": "Tampered Aadhaar (Name Altered)",
        "category": "AADHAAR_CARD",
        "description": "Digitally modified Cardholder Name field ('KUMAR S M'). Triggers Layer 3 QR signature cross-check mismatch.",
        "path": os.path.join(REPO_ROOT, "test_samples", "aadhaar", "aadhaar_03_tampered_name_text.jpg")
    },
    "aadhaar_04_tampered_uid": {
        "name": "Tampered Aadhaar (UID Digit Forged)",
        "category": "AADHAAR_CARD",
        "description": "Single-digit alteration in 12-digit UID. Breaks Layer 2 Dihedral D5 Verhoeff checksum.",
        "path": os.path.join(REPO_ROOT, "test_samples", "aadhaar", "aadhaar_04_tampered_uid_checksum.jpg")
    },
    "pan_01_clean": {
        "name": "Authentic PAN (High-Res)",
        "category": "PAN_CARD",
        "description": "Pristine authentic PAN card with standard 10-char alphanumeric structure and valid Individual 'P' entity code.",
        "path": os.path.join(REPO_ROOT, "test_samples", "pan", "pan_01_clean_highres.jpg")
    },
    "pan_03_tampered_char": {
        "name": "Tampered PAN ('P' altered to 'B')",
        "category": "PAN_CARD",
        "description": "Surgical 1-character semantic forgery modifying 4th character of PAN. PGU-v3 circles tampered letter.",
        "path": os.path.join(REPO_ROOT, "test_samples", "pan", "pan_03_tampered_single_char.jpg")
    },
    "pan_04_tampered_photo": {
        "name": "Tampered PAN (Photo Spliced)",
        "category": "PAN_CARD",
        "description": "Cardholder portrait photo replaced / spliced from different camera sensor with boundary seam.",
        "path": os.path.join(REPO_ROOT, "test_samples", "pan", "pan_04_tampered_photo_splice.jpg")
    }
}



def bgr_to_base64_png(bgr_img: np.ndarray) -> str:
    """Encodes BGR numpy image to base64 data URI string."""
    if bgr_img is None or bgr_img.size == 0:
        return ""
    success, buffer = cv2.imencode('.png', bgr_img)
    if not success:
        return ""
    b64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{b64}"


def draw_forensic_tamper_circles(canvas: np.ndarray, tampered_boxes: list = None, heatmap: np.ndarray = None) -> np.ndarray:
    """
    Renders high-precision glowing forensic target circles, concentric reticle rings,
    and label callouts directly over localized forgery regions.
    Guarantees that if a neural heatmap contains hot anomalies (>=0.35),
    at least one targeting circle is placed at the peak anomaly centroid.
    """
    if canvas is None or canvas.size == 0:
        return canvas
    out = canvas.copy()
    h, w = out.shape[:2]

    boxes_to_draw = list(tampered_boxes) if tampered_boxes else []

    # Fallback: if no discrete bounding box was formed but heatmap has high anomaly peak
    if not boxes_to_draw and heatmap is not None and heatmap.size > 0 and heatmap.max() >= 0.35:
        hm_res = cv2.resize(heatmap, (w, h))
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(hm_res)
        if max_val >= 0.35:
            px, py = max_loc
            boxes_to_draw.append({
                "box_id": "TAMPER_ZONE_1",
                "box": [max(0, px - 16), max(0, py - 16), 32, 32],
                "center": [px, py],
                "radius": 24,
                "peak_probability_pct": round(float(max_val) * 100.0, 1),
                "tampered_pixels": int((hm_res >= 0.35).sum()),
                "associated_field": "NEURAL_ANOMALY"
            })

    is_rgba = (out.shape[2] == 4) if len(out.shape) == 3 else False
    color_red = (0, 0, 255, 255) if is_rgba else (0, 0, 255)
    color_halo = (0, 0, 160, 220) if is_rgba else (0, 0, 160)
    color_amber = (0, 180, 255, 255) if is_rgba else (0, 180, 255)
    color_bg = (12, 16, 26, 240) if is_rgba else (12, 16, 26)
    color_white = (255, 255, 255, 255) if is_rgba else (255, 255, 255)

    for t_box in boxes_to_draw:
        bx, by, bw, bh = t_box.get("box", [0, 0, 0, 0])
        peak_pct = t_box.get("peak_probability_pct", 0.0)
        pixels = t_box.get("tampered_pixels", 0)
        field = t_box.get("associated_field", "")

        # Compute centroid & adaptive radius
        center = t_box.get("center")
        if center and len(center) == 2:
            cx, cy = int(center[0]), int(center[1])
        else:
            cx, cy = int(bx + bw / 2.0), int(by + bh / 2.0)

        rad = t_box.get("radius")
        if rad is None:
            rad = int(max(np.hypot(bw, bh) / 2.0 + 8.0, 20.0))
        rad = max(16, min(rad, min(w, h) // 3))

        # 1. Subtle Outer Glow Halo Circle
        cv2.circle(out, (cx, cy), rad + 4, color_halo, 1, cv2.LINE_AA)
        # 2. Main High-Visibility Forensic Targeting Circle (Neon Red)
        cv2.circle(out, (cx, cy), rad, color_red, 2, cv2.LINE_AA)
        # 3. Inner Concentric Reticle Ring (Amber / Gold)
        cv2.circle(out, (cx, cy), max(3, rad - 6), color_amber, 1, cv2.LINE_AA)
        # 4. Center Micro-Dot Target
        cv2.circle(out, (cx, cy), 2, color_red, -1, cv2.LINE_AA)

        # 5. Forensic Reticle Crosshair Ticks on 4 quadrants
        tick_len = 6
        cv2.line(out, (cx, cy - rad - tick_len), (cx, cy - rad + 2), color_red, 2, cv2.LINE_AA)
        cv2.line(out, (cx, cy + rad - 2), (cx, cy + rad + tick_len), color_red, 2, cv2.LINE_AA)
        cv2.line(out, (cx - rad - tick_len, cy), (cx - rad + 2, cy), color_red, 2, cv2.LINE_AA)
        cv2.line(out, (cx + rad - 2, cy), (cx + rad + tick_len, cy), color_red, 2, cv2.LINE_AA)

        # 6. Sleek Callout Badge with Forgery % & Entity Name
        label_text = f"FORGERY: {peak_pct}% ({pixels} px)"
        if field and field != "UNPRINTED_CANVAS_ZONE":
            label_text = f"FORGERY: {field} ({peak_pct}%)"

        font_scale = 0.40
        thickness = 1
        (lw, lh), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

        # Position label above circle (or below if near top border)
        badge_y = cy - rad - 10
        if badge_y - lh - 4 < 4:
            badge_y = cy + rad + lh + 14

        badge_x = max(4, min(w - lw - 10, cx - lw // 2))

        # Badge pill background & neon border
        cv2.rectangle(out, (badge_x - 3, badge_y - lh - 4), (badge_x + lw + 5, badge_y + 3), color_bg, -1)
        cv2.rectangle(out, (badge_x - 3, badge_y - lh - 4), (badge_x + lw + 5, badge_y + 3), color_red, 1)
        # Text
        cv2.putText(out, label_text, (badge_x + 1, badge_y - 1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_white, thickness, cv2.LINE_AA)

    return out


def create_heatmap_overlay(rectified_img: np.ndarray, heatmap: np.ndarray, thresh: float = 0.25, max_alpha: float = 0.80) -> np.ndarray:
    """
    Renders thermal heatmap STRICTLY over tampered regions (anomaly >= thresh).
    All clean, non-tampered regions remain 100% untouched original document pixels (zero blue tint).
    """
    if rectified_img is None or rectified_img.size == 0 or heatmap is None or heatmap.size == 0:
        return rectified_img
    h, w = rectified_img.shape[:2]
    hm_resized = cv2.resize(heatmap, (w, h)).astype(np.float32)

    # If document has no anomalies above threshold, return pristine image directly
    if float(hm_resized.max()) < thresh:
        return rectified_img.copy()

    hm_uint8 = np.clip(hm_resized * 255.0, 0, 255).astype(np.uint8)
    hm_color = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)

    # Localized alpha mask: strictly 0.0 below threshold, ramping smoothly up to max_alpha
    alpha_mask = np.clip((hm_resized - thresh) / max(1.0 - thresh, 1e-5), 0.0, 1.0)
    alpha_mask = np.power(alpha_mask, 0.65) * max_alpha
    alpha_3d = np.repeat(alpha_mask[:, :, np.newaxis], 3, axis=2)

    overlay = (rectified_img.astype(np.float32) * (1.0 - alpha_3d) + hm_color.astype(np.float32) * alpha_3d)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def create_pure_heatmap_image(heatmap: np.ndarray, target_shape: tuple, thresh: float = 0.25) -> np.ndarray:
    """
    Creates RGBA image where alpha is strictly 0 for non-tampered pixels (< thresh),
    and glowing thermal color strictly for tampered pixels (>= thresh).
    """
    h, w = target_shape[:2]
    if heatmap is None or heatmap.size == 0:
        return np.zeros((h, w, 4), dtype=np.uint8)
    hm_resized = cv2.resize(heatmap, (w, h)).astype(np.float32)

    hm_uint8 = np.clip(hm_resized * 255.0, 0, 255).astype(np.uint8)
    hm_color = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)

    alpha_mask = np.clip((hm_resized - thresh) / max(1.0 - thresh, 1e-5), 0.0, 1.0)
    alpha_uint8 = (np.power(alpha_mask, 0.65) * 255.0).astype(np.uint8)

    rgba = cv2.merge([hm_color[:, :, 0], hm_color[:, :, 1], hm_color[:, :, 2], alpha_uint8])
    return rgba


def create_sharpness_edge_map(image: np.ndarray) -> np.ndarray:
    """Computes Laplacian high-frequency edge variance map to visually show sharpness vs blur."""
    if image is None or image.size == 0:
        return image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_abs = np.absolute(lap)
    p99 = np.percentile(lap_abs, 99.5) if lap_abs.max() > 0 else 1.0
    lap_norm = np.clip((lap_abs / max(p99, 1e-5)) * 255.0, 0, 255).astype(np.uint8)
    color_map = cv2.applyColorMap(lap_norm, cv2.COLORMAP_INFERNO)
    return color_map


def create_ocr_annotated_image(rectified_img: np.ndarray, raw_tokens: list) -> np.ndarray:
    """Draws sleek forensic bounding boxes and text labels around each detected OCR token."""
    if rectified_img is None or rectified_img.size == 0:
        return rectified_img
    annotated = rectified_img.copy()
    h, w = annotated.shape[:2]

    for token in raw_tokens:
        bx, by, bw, bh = token.get("box", [0, 0, 0, 0])
        text = str(token.get("text", "")).strip()
        conf = float(token.get("confidence", 0.0))

        # Color based on confidence: cyan for high confidence, amber for medium
        color = (0, 220, 255) if conf >= 0.65 else (0, 165, 255)

        # Draw bounding box rectangle
        cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), color, 2)

        # Background badge for text legibility
        if text:
            label = f"{text} ({int(conf * 100)}%)"
            font_scale = 0.38
            thickness = 1
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            label_y = max(lh + 4, by - 4)
            cv2.rectangle(annotated, (bx, label_y - lh - 4), (bx + lw + 4, label_y + 3), (12, 16, 26), -1)
            cv2.putText(annotated, label, (bx + 2, label_y - 1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

    return annotated


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/presets", methods=["GET"])
def get_presets():
    preset_list = []
    for pid, meta in PRESETS.items():
        exists = os.path.exists(meta["path"])
        preset_list.append({
            "id": pid,
            "name": meta["name"],
            "category": meta["category"],
            "description": meta["description"],
            "available": exists
        })
    return jsonify({"presets": preset_list})


@app.route("/api/preset_image/<preset_id>", methods=["GET"])
def get_preset_image(preset_id):
    if preset_id not in PRESETS:
        return jsonify({"error": "Unknown preset ID"}), 404
    meta = PRESETS[preset_id]
    if not os.path.exists(meta["path"]):
        return jsonify({"error": f"Image file not found at {meta['path']}"}), 404
    img = cv2.imread(meta["path"])
    return jsonify({
        "id": preset_id,
        "name": meta["name"],
        "category": meta["category"],
        "image_data": bgr_to_base64_png(img)
    })


@app.route("/api/analyze", methods=["POST"])
def analyze_document():
    try:
        requested_doc_type = request.form.get("doc_type", "").strip() or None
        if requested_doc_type == "AUTO":
            requested_doc_type = None

        preset_id = request.form.get("preset_id")
        file = request.files.get("file")

        print(f"\n[Dashboard API] /api/analyze called:")
        print(f"  - Files in request: {list(request.files.keys())}")
        print(f"  - Form values: {request.form.to_dict()}")

        if file and file.filename:
            file_bytes = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img is None:
                return jsonify({"error": f"Failed to decode uploaded image: '{file.filename}'"}), 400
            input_source = img
            doc_name = file.filename
            print(f"  -> SUCCESS: Using CUSTOM UPLOADED IMAGE: '{doc_name}' ({img.shape[1]}x{img.shape[0]} px, {len(file_bytes)} bytes)")
        elif preset_id and preset_id in PRESETS:
            preset_meta = PRESETS[preset_id]
            if not os.path.exists(preset_meta["path"]):
                return jsonify({"error": f"Preset image file missing: {preset_meta['path']}"}), 404
            img = cv2.imread(preset_meta["path"])
            input_source = img
            doc_name = os.path.basename(preset_meta["path"])
            print(f"  -> Using PRESET IMAGE: '{preset_id}' -> '{doc_name}'")
            if not requested_doc_type and preset_meta.get("category"):
                requested_doc_type = preset_meta["category"]
        else:
            return jsonify({"error": "No image file uploaded and no valid preset selected."}), 400

        # Run master forensic pipeline
        result = pipeline.process(
            image_input=input_source,
            doc_type=requested_doc_type,
            output_evidence_dir=None
        )

        # Prepare visual artifacts
        rect_img = result.rectification.rectified_image
        heatmap = result.clean_heatmap if result.clean_heatmap is not None else np.zeros(rect_img.shape[:2], dtype=np.float32)
        tamper_boxes = result.neural_metrics.get("tampered_bounding_boxes", [])

        # Fallback: if no discrete bounding box was formed but heatmap has high anomaly peak (>=0.35)
        if not tamper_boxes and heatmap.size > 0 and heatmap.max() >= 0.35:
            hm_res = cv2.resize(heatmap, (rect_img.shape[1], rect_img.shape[0]))
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(hm_res)
            if max_val >= 0.35:
                px, py = max_loc
                peak_pct = round(float(max_val) * 100.0, 1)
                tamper_boxes = [{
                    "box_id": "TAMPER_ZONE_1",
                    "box": [max(0, px - 16), max(0, py - 16), 32, 32],
                    "center": [px, py],
                    "radius": 24,
                    "peak_probability_pct": peak_pct,
                    "tampered_pixels": int((hm_res >= 0.35).sum()),
                    "associated_field": "NEURAL_ANOMALY"
                }]
                result.neural_metrics["tampered_bounding_boxes"] = tamper_boxes
                result.neural_metrics["tampered_clusters_count"] = len(tamper_boxes)

        overlay_img = create_heatmap_overlay(rect_img, heatmap)
        rect_annotated_img = rect_img.copy()
        raw_tokens = result.extracted_fields.get("raw_tokens", [])
        ocr_annotated_img = create_ocr_annotated_image(rect_img, raw_tokens)

        response_data = {
            "success": True,
            "document_name": doc_name,
            "decision": result.decision,
            "trust_score": round(result.trust_score * 100.0, 1),
            "risk_score": round(result.risk_score * 100.0, 1),
            "rectification": {
                "was_rectified": result.rectification.was_rectified,
                "original_shape": list(result.rectification.original_shape),
                "rectified_shape": list(result.rectification.rectified_shape),
                "background_removed_pct": round(result.rectification.background_ratio_removed * 100.0, 1)
            },
            "classification": result.classification,
            "structural_audit": result.structural_report.to_dict(),
            "neural_vision": result.neural_metrics,
            "extracted_fields": result.extracted_fields,
            "trust_receipt": result.trust_receipt,
            "visuals": {
                "raw_image": bgr_to_base64_png(input_source),
                "rectified_image": bgr_to_base64_png(rect_img),
                "rectified_annotated": bgr_to_base64_png(rect_annotated_img),
                "ocr_image": bgr_to_base64_png(ocr_annotated_img),
                "blur_edge_map": bgr_to_base64_png(create_sharpness_edge_map(rect_img)),
                "pure_heatmap": bgr_to_base64_png(create_pure_heatmap_image(heatmap, rect_img.shape)),
                "heatmap_overlay": bgr_to_base64_png(overlay_img)
            }
        }
        return jsonify(response_data)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n=======================================================")
    print(f" DOCROI-FORMER FORENSIC INTELLIGENCE SUITE READY")
    print(f" Access URL: http://127.0.0.1:{port}")
    print(f"=======================================================\n")
    app.run(host="127.0.0.1", port=port, debug=False)
