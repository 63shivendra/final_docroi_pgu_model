"""
DocROI-Former: 1% Class Imbalance KYC Stress-Test Evaluator
===========================================================
Evaluates a model checkpoint on the extreme hard benchmark suite:
  'D:/shivendra pratap singh/hard_eval_stress_test_kyc_1percent'

Computes:
  1. False Alarm Rate (FAR) across 99 authentic clean cards
  2. Fraud Catch Rate (Recall) on the 1 surgically forged card
  3. Pixel IoU and F1-score on the tampered character region
  4. Enterprise Automated Triage Accuracy
"""

import os
import sys
import argparse
import csv
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.docroi_config import DocROIConfig
from src.models.docroi_former import DocROIFormer
from src.dataset.entity_tokenizer import OtsuEntityTokenizer

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

def evaluate_stress_test(checkpoint_path: str, device: str = "cuda"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(base_dir)
    target_dir = os.path.join(parent_dir, "hard_eval_stress_test_kyc_1percent")
    images_dir = os.path.join(target_dir, "images")
    masks_dir = os.path.join(target_dir, "masks")
    manifest_csv = os.path.join(target_dir, "ground_truth_manifest.csv")
    
    if not os.path.exists(manifest_csv):
        print(f"[Error] Manifest CSV not found at {manifest_csv}. Please run create_hard_stress_test.py first.")
        return
        
    cfg = DocROIConfig()
    device = device if torch.cuda.is_available() else "cpu"
    model = DocROIFormer(cfg).to(device)
    
    if os.path.exists(checkpoint_path):
        print(f"[Model Loader] Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict, strict=False)
    else:
        print(f"[Warning] Checkpoint {checkpoint_path} not found. Running with initialized weights.")
        
    model.eval()
    tokenizer = OtsuEntityTokenizer()
    
    with open(manifest_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        samples = list(reader)
        
    print(f"\n[Evaluating Stress Test] Total Samples: {len(samples)} (99 Authentic, 1 Tampered)")
    print("=" * 80)
    
    tp, fp, tn, fn = 0, 0, 0, 0
    tampered_iou = 0.0
    tampered_f1 = 0.0
    
    results = []
    
    for row in samples:
        img_p = os.path.join(images_dir, row["filename"])
        mask_p = os.path.join(masks_dir, row["mask_filename"])
        is_gt_tampered = int(row["is_tampered_binary"])
        
        img = cv2.imread(img_p)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_norm = (img_rgb.astype(np.float32) / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
        img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
        
        gt_mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
        gt_mask_bin = (gt_mask > 127).astype(np.float32)
        
        entity_prior = tokenizer(img_tensor[0]).unsqueeze(0)
        
        with torch.no_grad():
            preds = model(img_tensor, entity_prior=entity_prior)
            prob = preds["global_prob"].item()
            pred_mask_sig = torch.sigmoid(preds["mask_logits"])[0, 0].cpu().numpy()
            pred_mask_bin = (pred_mask_sig > 0.50).astype(np.float32)
            
        peak_pixel = float(pred_mask_sig.max())
        doc_risk = max(prob, peak_pixel)
        is_pred_tampered = int(doc_risk >= 0.50)
        
        # Classification confusion
        if is_gt_tampered and is_pred_tampered:
            tp += 1
            confusion = "TP (Caught Fraud)"
        elif not is_gt_tampered and not is_pred_tampered:
            tn += 1
            confusion = "TN (True Authentic)"
        elif not is_gt_tampered and is_pred_tampered:
            fp += 1
            confusion = "FP (False Alarm)"
        else:
            fn += 1
            confusion = "FN (Missed Fraud)"
            
        # If tampered sample, compute pixel IoU and F1
        p_iou, p_f1 = 0.0, 0.0
        if is_gt_tampered:
            inter = np.logical_and(pred_mask_bin, gt_mask_bin).sum()
            union = np.logical_or(pred_mask_bin, gt_mask_bin).sum()
            p_iou = float(inter / (union + 1e-6)) * 100.0
            p_f1 = float(2 * inter / (pred_mask_bin.sum() + gt_mask_bin.sum() + 1e-6)) * 100.0
            tampered_iou = p_iou
            tampered_f1 = p_f1
            print(f"  >>> [TAMPERED CARD DETECTED] Sample: {row['sample_id']} | Risk Score: {doc_risk:.4f} | Pixel IoU: {p_iou:.2f}% | F1: {p_f1:.2f}%")
            
        results.append({
            "sample_id": row["sample_id"],
            "filename": row["filename"],
            "doc_type": row["doc_type"],
            "ground_truth": row["ground_truth_label"],
            "predicted": "TAMPERED" if is_pred_tampered else "AUTHENTIC",
            "global_risk_score": round(doc_risk, 4),
            "confusion": confusion,
            "pixel_iou": round(p_iou, 2) if is_gt_tampered else "N/A"
        })
        
    far = (fp / max(tn + fp, 1)) * 100.0
    catch_rate = (tp / max(tp + fn, 1)) * 100.0
    accuracy = ((tp + tn) / len(samples)) * 100.0
    
    print("=" * 80)
    print("  STRESS TEST EVALUATION SUMMARY (1% CLASS IMBALANCE KYC)")
    print("=" * 80)
    print(f"  Total Authentic Cards Tested: 99")
    print(f"  True Negatives (Correctly Approved): {tn}/99")
    print(f"  False Positives (False Alarms on Plastic/Glare): {fp}/99")
    print(f"  False Alarm Rate (FAR): {far:.2f}% (Industry Target: < 5%)")
    print(f"  Fraud Catch Rate (Recall): {catch_rate:.2f}% ({tp}/1 tampered card caught)")
    print(f"  Tampered Pixel IoU: {tampered_iou:.2f}%")
    print(f"  Overall Document Decision Accuracy: {accuracy:.2f}%")
    print("=" * 80)
    
    out_csv = os.path.join(base_dir, "reports", "kyc_stress_test_1percent_results.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"  Detailed individual report saved to: {out_csv}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_docroi_former_v2_pgu.pth")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    evaluate_stress_test(args.checkpoint, args.device)
