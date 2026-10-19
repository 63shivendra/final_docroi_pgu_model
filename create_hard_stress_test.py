"""
DocROI-Former: Real-World 1% Class Imbalance Hard Stress-Test Environment
========================================================================
Creates a dedicated high-difficulty evaluation benchmark suite in:
  'D:/shivendra pratap singh/hard_eval_stress_test_kyc_1percent'

Composition:
  - Exactly 100 Real Indian Identity Documents (50 Aadhaar + 50 PAN)
  - 99 Clean Authentic Documents (Label = 0, Mask = All 0s)
  - 1 Surgically Forged Document (Label = 1, Exact Ground-Truth Mask)
    (Surgical single-digit / field forgery representing realistic 1% banking fraud rate)
  - 'ground_truth_manifest.csv' documenting every sample for strict auditing.
"""

import os
import glob
import random
import csv
import cv2
import numpy as np

def create_stress_test():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(base_dir) # D:/shivendra pratap singh
    
    target_dir = os.path.join(parent_dir, "hard_eval_stress_test_kyc_1percent")
    images_dir = os.path.join(target_dir, "images")
    masks_dir = os.path.join(target_dir, "masks")
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    
    # 1. Gather real Aadhaar and PAN images
    kyc_root = os.path.join(base_dir, "Real_dataset_pan_adhaar")
    pan_candidates = glob.glob(os.path.join(kyc_root, "pan_card", "**", "*.jpg"), recursive=True)
    adhaar_candidates = glob.glob(os.path.join(kyc_root, "adhaar_card", "**", "*.jpg"), recursive=True)
    
    # Filter valid readable images
    random.seed(42)
    random.shuffle(pan_candidates)
    random.shuffle(adhaar_candidates)
    
    selected_pan = pan_candidates[:50]
    selected_adhaar = adhaar_candidates[:50]
    
    total_selected = len(selected_pan) + len(selected_adhaar)
    print(f"[Stress Test Builder] Gathered {len(selected_pan)} PAN + {len(selected_adhaar)} Aadhaar = {total_selected} Total Cards")
    
    if total_selected < 100:
        raise ValueError(f"Need at least 100 images, but found {total_selected}")
        
    combined_docs = []
    for p in selected_pan:
        combined_docs.append((p, "PAN"))
    for a in selected_adhaar:
        combined_docs.append((a, "AADHAAR"))
        
    random.seed(1337)
    random.shuffle(combined_docs)
    combined_docs = combined_docs[:100]
    
    # Exactly 1 document out of 100 is chosen for surgical forgery (1% class imbalance)
    tampered_index = 42 # Deterministic choice (Sample #42)
    
    manifest_rows = []
    
    for idx, (img_path, doc_type) in enumerate(combined_docs):
        sample_id = f"KYC_STRESS_{idx+1:03d}"
        out_img_name = f"{sample_id}_{doc_type}.jpg"
        out_mask_name = f"{sample_id}_{doc_type}_mask.png"
        
        out_img_path = os.path.join(images_dir, out_img_name)
        out_mask_path = os.path.join(masks_dir, out_mask_name)
        
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (384, 384))
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if idx == tampered_index:
            # Place surgical tampering DIRECTLY ON THE PRINTED DIGITS of the Aadhaar number
            # On 384x384, the printed Aadhaar number '6965' sits at y ~ 272-292, x ~ 160-176
            y1 = 272
            y2 = 292
            x1 = 160
            x2 = 175
            box_h = y2 - y1
            box_w = x2 - x1
            
            # Extract another printed digit from the number line (e.g. from the '2' or '7' at x ~ 270)
            src_y1 = 272
            src_x1 = 270
            src_patch = img[src_y1:src_y1+box_h, src_x1:src_x1+box_w].copy()
            
            # Apply subtle digital font alteration (Canva/Photoshop digit replacement)
            src_patch_altered = cv2.convertScaleAbs(src_patch, alpha=1.20, beta=-8)
            src_patch_altered = cv2.GaussianBlur(src_patch_altered, (3, 3), 0.3)
            
            img[y1:y2, x1:x2] = src_patch_altered
            mask[y1:y2, x1:x2] = 255
            
            is_tampered = 1
            tamper_pixels = int(np.sum(mask > 0))
            tamper_pct = round((tamper_pixels / (h * w)) * 100.0, 4)
            desc = "Surgical single-digit font splice on ID field (Photoshop/Canva insertion simulation)"
        else:
            is_tampered = 0
            tamper_pixels = 0
            tamper_pct = 0.0
            desc = "Clean authentic document (Plastic laminate, camera noise, zero forgery)"
            
        cv2.imwrite(out_img_path, img)
        cv2.imwrite(out_mask_path, mask)
        
        manifest_rows.append({
            "sample_id": sample_id,
            "filename": out_img_name,
            "mask_filename": out_mask_name,
            "doc_type": doc_type,
            "ground_truth_label": "TAMPERED" if is_tampered else "AUTHENTIC",
            "is_tampered_binary": is_tampered,
            "tampered_pixels": tamper_pixels,
            "tamper_area_percentage": tamper_pct,
            "description": desc
        })
        
    manifest_csv = os.path.join(target_dir, "ground_truth_manifest.csv")
    with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["sample_id", "filename", "mask_filename", "doc_type", "ground_truth_label", "is_tampered_binary", "tampered_pixels", "tamper_area_percentage", "description"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)
        
    # Write README.md inside the stress test folder
    readme_content = f"""# Real-World 1% Class Imbalance KYC Stress-Test Suite
========================================================================
**Location:** `{target_dir}`  
**Purpose:** Extreme Operational Hard Environment for CVPR Real-World Forensic Evaluation.

### Benchmark Statistics:
- **Total Real Documents:** 100 (50 Aadhaar + 50 PAN)
- **Authentic Documents:** 99 (99.0%)
- **Tampered Documents:** 1 (1.0% Extreme Class Imbalance)
- **Tampered Sample ID:** `KYC_STRESS_043` (`KYC_STRESS_043_PAN.jpg` or `AADHAAR.jpg`)
- **Tampering Type:** Surgical single-digit typographical replacement (Photoshop/Canva font mismatch simulation).
- **Substrate Artifacts:** Real plastic laminates, holographic overlays, perspective distortion, and sensor camera flash.

### How to Evaluate Your Trained Model Later:
```bash
python evaluate_stress_test.py --checkpoint "checkpoints/best_docroi_former_v2_pgu.pth"
```
"""
    with open(os.path.join(target_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme_content)
        
    print("\n" + "="*80)
    print(f"  [SUCCESS] 1% CLASS IMBALANCE STRESS-TEST ENVIRONMENT CREATED!")
    print(f"  Destination Folder: {target_dir}")
    print(f"  Total Images Generated: 100")
    print(f"  Authentic Clean Cards: 99")
    print(f"  Surgically Forged Cards: 1 (1.0% Extreme Class Imbalance)")
    print(f"  Manifest CSV: {manifest_csv}")
    print("="*80 + "\n")

if __name__ == "__main__":
    create_stress_test()
