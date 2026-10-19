"""
Automated OCR & Template Extraction Visualizer
=============================================
Extracts OCR text and field bounding boxes from Indian Identity Documents
(PAN & Aadhaar) based on canonical templates, and generates visually annotated
verification boards in a separate folder for easy manual and automated review.

Usage:
  python extract_and_visualize_ocr.py --image "D:/shivendra pratap singh/image.png"
  python extract_and_visualize_ocr.py --batch 5
"""

import os
import sys
import glob
import json
import csv
import re
import cv2
import numpy as np

# Ensure docroi_former is in sys.path
repo_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, repo_dir)

from src.core.vision_preprocessor import DocumentRectifier
from src.layer2.aadhaar_validator import AadhaarStructuralValidator, VerhoeffAlgorithm
from src.layer2.pan_structural_validator import PANStructuralValidator

try:
    import easyocr
    HAS_EASYOCR = True
except ImportError:
    HAS_EASYOCR = False


def normalize_pan_homoglyphs(text: str) -> str:
    """Corrects OCR homoglyphs for 10-char PAN string."""
    clean = re.sub(r'[^A-Z0-9]', '', text.upper())
    if len(clean) != 10:
        return clean
    char_map_for_digits = {'O': '0', 'I': '1', 'Z': '2', 'S': '5', 'B': '8', 'Q': '0', 'D': '0'}
    char_map_for_letters = {'0': 'O', '1': 'I', '2': 'Z', '5': 'S', '8': 'B'}
    
    chars = list(clean)
    for i in range(5):
        if chars[i] in char_map_for_letters:
            chars[i] = char_map_for_letters[chars[i]]
    for i in range(5, 9):
        if chars[i] in char_map_for_digits:
            chars[i] = char_map_for_digits[chars[i]]
    if chars[9] in char_map_for_letters:
        chars[9] = char_map_for_letters[chars[9]]
    return "".join(chars)


def normalize_aadhaar_digits(text: str) -> str:
    """Normalizes digits for 12-digit Aadhaar candidate."""
    char_map = {'O': '0', 'I': '1', 'Z': '2', 'S': '5', 'B': '8', 'Q': '0', 'D': '0', 'L': '1', 'G': '6'}
    cleaned = []
    for ch in text.upper():
        if ch.isdigit():
            cleaned.append(ch)
        elif ch in char_map:
            cleaned.append(char_map[ch])
    digits = "".join(cleaned)
    return digits


class DocumentOCRVisualizer:
    def __init__(self, output_dir: str = r"D:\shivendra pratap singh\ocr_visual_verification"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.rectifier = DocumentRectifier()
        self.aadhaar_val = AadhaarStructuralValidator()
        self.pan_val = PANStructuralValidator()
        
        self.reader = None
        if HAS_EASYOCR:
            import torch
            device_cuda = torch.cuda.is_available()
            self.reader = easyocr.Reader(['en'], gpu=device_cuda)

    def process_image(self, img_path: str, doc_type: str = "AUTO") -> dict:
        bgr = cv2.imread(img_path)
        if bgr is None:
            return {"error": f"Cannot load {img_path}"}
        
        doc_name = os.path.splitext(os.path.basename(img_path))[0]
        
        # 1. Perspective Rectification
        rect_res = self.rectifier.rectify(bgr)
        working_img = rect_res.rectified_image.copy()
        h, w = working_img.shape[:2]

        # 2. Run OCR
        ocr_boxes = []
        full_text_list = []
        if self.reader:
            results = self.reader.readtext(working_img)
            for (bbox, text, conf) in results:
                pts = np.array(bbox, dtype=np.int32)
                bx, by, bw, bh = cv2.boundingRect(pts)
                ocr_boxes.append({
                    "text": text.strip(),
                    "conf": float(conf),
                    "box": (bx, by, bw, bh)
                })
                full_text_list.append(text.strip())

        full_text = " ".join(full_text_list)
        
        # 3. Detect Document Type
        path_norm = img_path.replace("\\", "/").lower()
        if doc_type and doc_type != "AUTO":
            detected_type = doc_type
        elif "adhaar_card" in path_norm or "aadhaar" in path_norm:
            detected_type = "AADHAAR_CARD"
        elif "pan_card" in path_norm:
            detected_type = "PAN_CARD"
        else:
            is_pan = ("INCOME" in full_text.upper() or "PERMANENT" in full_text.upper() or "ACCOUNT" in full_text.upper())
            is_aadhaar = ("GOVERNMENT OF INDIA" in full_text.upper() or "AADHAAR" in full_text.upper() or "UIDAI" in full_text.upper())
            detected_type = "AADHAAR_CARD" if is_aadhaar else "PAN_CARD"

        # 4. Extract Key Fields
        extracted = {
            "document_type": detected_type,
            "id_number": "NOT_FOUND",
            "id_status": "MISSING",
            "name": "NOT_FOUND",
            "dob": "NOT_FOUND"
        }
        
        matched_box_ids = set()

        if detected_type == "PAN_CARD":
            # Search for PAN
            for i, item in enumerate(ocr_boxes):
                candidate = item["text"].replace(" ", "").upper()
                norm = normalize_pan_homoglyphs(candidate)
                if re.match(r'^[A-Z]{5}[0-9]{4}[A-Z]$', norm):
                    extracted["id_number"] = norm
                    extracted["id_status"] = "VALID (CBDT Format)"
                    matched_box_ids.add((i, "ID_NUMBER"))
                    break
            
            # If not found directly, look for any 10-char token
            if extracted["id_number"] == "NOT_FOUND":
                tokens = re.findall(r'[A-Za-z0-9]{10}', full_text)
                for tok in tokens:
                    norm = normalize_pan_homoglyphs(tok)
                    if re.match(r'^[A-Z]{5}[0-9]{4}[A-Z]$', norm):
                        extracted["id_number"] = norm
                        extracted["id_status"] = "VALID (CBDT Format)"
                        break

        elif detected_type == "AADHAAR_CARD":
            # Search for 12 digits
            digits_pool = []
            for i, item in enumerate(ocr_boxes):
                d = normalize_aadhaar_digits(item["text"])
                if len(d) == 12:
                    is_v = VerhoeffAlgorithm.validate_checksum(d)
                    extracted["id_number"] = f"{d[:4]} {d[4:8]} {d[8:]}"
                    extracted["id_status"] = "VALID (Verhoeff Pass)" if is_v else "INVALID CHECKSUM"
                    matched_box_ids.add((i, "ID_NUMBER"))
                    break
                elif len(d) == 4:
                    digits_pool.append((i, d))
            
            # If split into 3 groups of 4 (e.g. 1234 5678 9012)
            if extracted["id_number"] == "NOT_FOUND" and len(digits_pool) >= 3:
                combined_12 = "".join([d[1] for d in digits_pool[:3]])
                is_v = VerhoeffAlgorithm.validate_checksum(combined_12)
                extracted["id_number"] = f"{combined_12[:4]} {combined_12[4:8]} {combined_12[8:]}"
                extracted["id_status"] = "VALID (Verhoeff Pass)" if is_v else "INVALID CHECKSUM"
                for (idx_box, _) in digits_pool[:3]:
                    matched_box_ids.add((idx_box, "ID_NUMBER"))

        # Search for Name & DOB
        for i, item in enumerate(ocr_boxes):
            t = item["text"]
            # DOB regex
            m_dob = re.search(r'\b\d{2}/\d{2}/\d{4}\b', t)
            if m_dob and extracted["dob"] == "NOT_FOUND":
                extracted["dob"] = m_dob.group(0)
                matched_box_ids.add((i, "DOB"))
            
            # Name heuristic
            if "name" in t.lower() and i + 1 < len(ocr_boxes) and extracted["name"] == "NOT_FOUND":
                candidate_name = ocr_boxes[i+1]["text"].strip()
                if len(candidate_name) > 3 and not candidate_name.isdigit():
                    extracted["name"] = candidate_name
                    matched_box_ids.add((i+1, "NAME"))

        # 5. Create Annotated Visual Verification Board
        annotated = working_img.copy()
        
        # Draw all OCR bounding boxes
        for i, item in enumerate(ocr_boxes):
            bx, by, bw, bh = item["box"]
            box_type = None
            for (mb_idx, b_type) in matched_box_ids:
                if mb_idx == i:
                    box_type = b_type
                    break
            
            if box_type == "ID_NUMBER":
                color = (0, 220, 0) if "VALID" in extracted["id_status"] else (0, 0, 255)
                thick = 2
            elif box_type in ("NAME", "DOB"):
                color = (255, 200, 0)
                thick = 2
            else:
                color = (200, 150, 150)
                thick = 1
            
            cv2.rectangle(annotated, (bx, by), (bx+bw, by+bh), color, thick)
            label_text = f"{item['text']} ({item['conf']*100:.0f}%)"
            cv2.putText(annotated, label_text[:25], (bx, max(by - 5, 12)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

        # Create Top and Bottom Info Banners
        header_h = 75
        board = np.zeros((h + header_h, w, 3), dtype=np.uint8)
        board[header_h:, :] = annotated
        
        # Banner background
        cv2.rectangle(board, (0, 0), (w, header_h), (20, 24, 30), -1)
        cv2.line(board, (0, header_h), (w, header_h), (60, 70, 80), 1)

        # Write Extracted Fields on Banner
        cv2.putText(board, f"DOC: {doc_name[:28]} | TYPE: {extracted['document_type']}", 
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 255), 1, cv2.LINE_AA)
        
        status_color = (0, 255, 0) if "VALID" in extracted["id_status"] else (0, 140, 255)
        cv2.putText(board, f"ID NUM: {extracted['id_number']} [{extracted['id_status']}]", 
                    (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.46, status_color, 1, cv2.LINE_AA)
        
        cv2.putText(board, f"NAME: {extracted['name']} | DOB: {extracted['dob']}", 
                    (10, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220, 220, 220), 1, cv2.LINE_AA)

        # Save to Output Folder
        out_file_name = f"annotated_{doc_name}.png"
        out_path = os.path.join(self.output_dir, out_file_name)
        cv2.imwrite(out_path, board)

        return {
            "document_name": doc_name,
            "image_path": img_path,
            "annotated_path": out_path,
            "extracted": extracted,
            "total_boxes_found": len(ocr_boxes)
        }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract OCR and generate verification output images")
    parser.add_argument("--image", type=str, default=None, help="Path to single image")
    parser.add_argument("--batch", type=int, default=None, help="Process N sample images from Real KYC dataset")
    parser.add_argument("--output_dir", type=str, default=r"D:\shivendra pratap singh\ocr_visual_verification")
    args = parser.parse_args()

    visualizer = DocumentOCRVisualizer(output_dir=args.output_dir)

    target_images = []
    if args.image:
        target_images.append(args.image)
    else:
        # Default: Test on image.png + 2 real PAN + 2 real Aadhaar
        if os.path.exists(r"D:\shivendra pratap singh\image.png"):
            target_images.append(r"D:\shivendra pratap singh\image.png")
        
        kyc_root = os.path.join(repo_dir, "Real_dataset_pan_adhaar")
        pan_samples = glob.glob(os.path.join(kyc_root, "pan_card", "**", "*.jpg"), recursive=True)[:2]
        aadhaar_samples = glob.glob(os.path.join(kyc_root, "adhaar_card", "**", "*.jpg"), recursive=True)[:2]
        target_images.extend(pan_samples)
        target_images.extend(aadhaar_samples)

    print("=" * 80)
    print(f"  OCR & TEMPLATE EXTRACTION VISUALIZER")
    print(f"  Saving annotated verification images to: {args.output_dir}")
    print("=" * 80)

    summary_records = []
    for img_p in target_images:
        print(f"\nProcessing: {os.path.basename(img_p)} ...")
        res = visualizer.process_image(img_p)
        if "error" in res:
            print(f"  Error: {res['error']}")
            continue
        ext = res["extracted"]
        print(f"  [Extracted Type]: {ext['document_type']}")
        print(f"  [ID Number]:      {ext['id_number']} ({ext['id_status']})")
        print(f"  [Name]:           {ext['name']}")
        print(f"  [DOB]:            {ext['dob']}")
        print(f"  [Saved Board]:    {res['annotated_path']}")
        
        summary_records.append({
            "File": res["document_name"],
            "Document_Type": ext["document_type"],
            "ID_Number": ext["id_number"],
            "ID_Status": ext["id_status"],
            "Name": ext["name"],
            "DOB": ext["dob"],
            "Boxes_Found": res["total_boxes_found"],
            "Annotated_Image": res["annotated_path"]
        })

    # Save CSV Summary
    csv_path = os.path.join(args.output_dir, "ocr_verification_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["File", "Document_Type", "ID_Number", "ID_Status", "Name", "DOB", "Boxes_Found", "Annotated_Image"])
        writer.writeheader()
        writer.writerows(summary_records)

    print("\n" + "=" * 80)
    print(f"  ALL DONE! Visual verification summary saved to: {csv_path}")
    print(f"  You can now open folder: '{args.output_dir}' to inspect all annotated images.")
    print("=" * 80)


if __name__ == "__main__":
    main()
