"""
DocROI-Former: Full-Sample Individual Evaluation Engine
======================================================
Evaluates every single test sample INDIVIDUALLY across all 6 benchmark testbeds:
  1. DocTamper TestingSet (30,000 samples)
  2. DocTamper-SCD (18,000 samples)
  3. MIDV-500 Clean National IDs (15,049 samples)
  4. CASIA 2.0 Image Splicing (2,614 samples)
  5. COCOGLIDE_trufor Generative AI inpainting (1,536 samples)
  6. Real Indian KYC (PAN & Aadhaar)

Outputs:
  - 'reports/master_individual_test_results.csv' (Per-sample row records)
  - 'reports/master_benchmark_summary.json' (Official dataset-by-dataset scientific metrics)
  - 'reports/visual_audit_boards/' (4-panel high-res evidence boards)

Design Pattern: Benchmark Harness / Evaluator (SOLID: Single Responsibility)
"""

import sys
import os
import time
import json
import csv
import glob
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Any, Optional, Tuple

# Ensure package path is resolved
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.docroi_config import DocROIConfig
from src.models.docroi_former import DocROIFormer
from src.dataset.doc_dataset import UnifiedDocForensicDataset
from src.core.domain_entities import EvaluationSampleRecord


def parse_args():
    parser = argparse.ArgumentParser(description="DocROI-Former Full-Sample Evaluator")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to model checkpoint")
    parser.add_argument("--data_dir", type=str, default="", help="Path to academic datasets directory")
    parser.add_argument("--kyc_dir", type=str, default="", help="Path to Real Indian KYC dataset directory")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for high-throughput evaluation")
    parser.add_argument("--threshold", type=float, default=0.50, help="Classification probability threshold")
    parser.add_argument("--max_samples_per_ds", type=int, default=0, help="0 for all samples, or limit for dry run")
    parser.add_argument("--device", type=str, default="cuda", help="Target computing device")
    parser.add_argument("--reports_dir", type=str, default="", help="Path to save evaluation reports and JSON")
    parser.add_argument("--only_trained", action="store_true", help="Only evaluate testbeds matching training domain (DocTamper and Indian KYC)")
    parser.add_argument("--doctamper_only", action="store_true", help="Only evaluate DocTamper TestingSet and DocTamper SCD")
    parser.add_argument("--datasets", type=str, default="", help="Comma-separated list of datasets to evaluate")
    return parser.parse_args()


def compute_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict[str, float]:
    """
    Computes exact Pixel IoU, F1 (Dice), Precision, and Recall.
    """
    p = (pred_mask > 0.5).astype(bool)
    g = (gt_mask > 0.5).astype(bool)

    tp = np.logical_and(p, g).sum()
    fp = np.logical_and(p, ~g).sum()
    fn = np.logical_and(~p, g).sum()
    tn = np.logical_and(~p, ~g).sum()

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2.0 * tp / (2.0 * tp + fp + fn)) if (2.0 * tp + fp + fn) > 0 else 0.0
    iou = float(tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else (1.0 if not g.any() else 0.0)

    return {
        "iou": round(iou * 100.0, 2),
        "f1": round(f1 * 100.0, 2),
        "precision": round(precision * 100.0, 2),
        "recall": round(recall * 100.0, 2),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)
    }


def evaluate_dataset(
    model: nn.Module,
    dataset: UnifiedDocForensicDataset,
    dataset_name: str,
    device: str,
    threshold: float = 0.50,
    max_samples: int = 0,
    batch_size: int = 16
) -> Tuple[List[EvaluationSampleRecord], Dict[str, Any]]:
    """
    Evaluates every sample in the dataset with batched inference and exact sample metrics.
    """
    records = []
    total_eval = len(dataset) if max_samples <= 0 else min(len(dataset), max_samples)
    print(f"\n[Evaluating Dataset] {dataset_name} | Total Samples: {total_eval} | Batch Size: {batch_size} | Threshold: {threshold}")

    doc_tp = 0
    doc_tn = 0
    doc_fp = 0
    doc_fn = 0

    sum_iou = 0.0
    sum_f1 = 0.0
    sum_latency = 0.0
    valid_mask_count = 0
    processed_count = 0

    from src.dataset.entity_tokenizer import OtsuEntityTokenizer
    tokenizer = OtsuEntityTokenizer()
    loader = DataLoader(
        dataset,
        batch_size=max(1, batch_size),
        shuffle=False,
        num_workers=4 if sys.platform.startswith("linux") else 0,
        pin_memory=(device == "cuda")
    )

    for b_idx, batch in enumerate(loader):
        if max_samples > 0 and processed_count >= max_samples:
            break

        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].numpy()
        labels = batch["label"].numpy()

        # Batch entity priors from dataset loader if available, or compute on the fly
        if "entity_prior" in batch:
            entity_prior = batch["entity_prior"].to(device, non_blocking=True)
        else:
            entity_prior = torch.stack([tokenizer(img) for img in images]).to(device)

        t0 = time.perf_counter()
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=(device == "cuda")):
            preds = model(images, entity_prior=entity_prior)
            probs = preds["global_prob"].squeeze(-1).cpu().numpy()
            pred_masks_sig = torch.sigmoid(preds["mask_logits"]).cpu().numpy()
        batch_latency_ms = (time.perf_counter() - t0) * 1000.0
        per_sample_latency = batch_latency_ms / max(images.size(0), 1)

        b_size = images.size(0)

        # Free heavy GPU tensors immediately to eliminate VRAM fragmentation
        del preds
        del images
        del entity_prior
        for i in range(b_size):
            if max_samples > 0 and processed_count >= max_samples:
                break
            idx = processed_count
            gt_mask = masks[i, 0]
            lbl_val = labels[i].item() if hasattr(labels[i], "item") else labels[i]
            is_tampered_gt = bool(lbl_val > 0.5)

            prob = float(probs[i]) if probs.ndim > 0 else float(probs)
            pred_mask = pred_masks_sig[i, 0]
            tamper_pixels = int((pred_mask >= threshold).sum())
            has_tamper_evidence = bool(tamper_pixels >= 35)
            is_tampered_pred = bool((prob >= threshold) or has_tamper_evidence)

            # Confusion classification
            if is_tampered_gt and is_tampered_pred:
                confusion = "TP"
                doc_tp += 1
            elif not is_tampered_gt and not is_tampered_pred:
                confusion = "TN"
                doc_tn += 1
            elif not is_tampered_gt and is_tampered_pred:
                confusion = "FP"
                doc_fp += 1
            else:
                confusion = "FN"
                doc_fn += 1

            # Pixel metrics
            m = compute_metrics(pred_mask, gt_mask)
            if is_tampered_gt:
                sum_iou += m["iou"]
                sum_f1 += m["f1"]
                valid_mask_count += 1

            record = EvaluationSampleRecord(
                sample_id=f"{dataset_name}_{idx:06d}",
                dataset_name=dataset_name,
                file_path=dataset.image_paths[idx] if not dataset.is_lmdb and idx < len(dataset.image_paths) else f"lmdb_idx_{idx}",
                ground_truth_label="TAMPERED" if is_tampered_gt else "AUTHENTIC",
                predicted_label="TAMPERED" if is_tampered_pred else "AUTHENTIC",
                confusion_class=confusion,
                global_fraud_score=round(prob, 4),
                pixel_iou=m["iou"],
                pixel_f1=m["f1"],
                pixel_precision=m["precision"],
                pixel_recall=m["recall"],
                inference_latency_ms=round(per_sample_latency, 2)
            )
            records.append(record)
            sum_latency += per_sample_latency
            processed_count += 1

        if (processed_count % 500 < b_size) or (processed_count >= total_eval):
            if device == "cuda":
                torch.cuda.empty_cache()
            cur_catch = doc_tp / max(doc_tp + doc_fn, 1e-6) * 100.0
            cur_iou = sum_iou / max(valid_mask_count, 1)
            cur_f1 = sum_f1 / max(valid_mask_count, 1)
            print(f"  Processed {processed_count}/{total_eval} | IoU: {cur_iou:5.2f}% | Pixel F1: {cur_f1:5.2f}% | Doc Catch: {cur_catch:5.1f}% | Avg Latency: {sum_latency/max(processed_count, 1):.1f}ms")

    # Aggregate dataset-level statistics
    doc_recall = round(doc_tp / max(doc_tp + doc_fn, 1) * 100.0, 2)
    doc_precision = round(doc_tp / max(doc_tp + doc_fp, 1) * 100.0, 2)
    doc_f1 = round(2.0 * doc_precision * doc_recall / max(doc_precision + doc_recall, 1e-6), 2)
    false_alarm_rate = round(doc_fp / max(doc_fp + doc_tn, 1) * 100.0, 2)
    mean_pixel_iou = round(sum_iou / max(valid_mask_count, 1), 2)
    mean_pixel_f1 = round(sum_f1 / max(valid_mask_count, 1), 2)
    mean_latency = round(sum_latency / max(total_eval, 1), 2)

    summary = {
        "dataset_name": dataset_name,
        "total_evaluated": total_eval,
        "doc_recall": doc_recall,
        "doc_precision": doc_precision,
        "doc_f1": doc_f1,
        "false_alarm_rate": false_alarm_rate,
        "mean_pixel_iou": mean_pixel_iou,
        "mean_pixel_f1": mean_pixel_f1,
        "mean_latency_ms": mean_latency,
        "confusion_matrix": {"TP": doc_tp, "TN": doc_tn, "FP": doc_fp, "FN": doc_fn}
    }

    print("\n" + "=" * 78)
    print(f"  BENCHMARK REPORT: {dataset_name}")
    print("=" * 78)
    print(f"  Total Samples Evaluated: {total_eval}")
    print(f"  Mean Pixel IoU:          {mean_pixel_iou:6.2f}%")
    print(f"  Mean Pixel F1:           {mean_pixel_f1:6.2f}%")
    print(f"  Document Recall (Catch): {doc_recall:6.2f}%")
    print(f"  Document Precision:      {doc_precision:6.2f}%")
    print(f"  Document F1:             {doc_f1:6.2f}%")
    print(f"  False Alarm Rate (FPR):  {false_alarm_rate:6.2f}%")
    print(f"  Avg Latency per Sample:  {mean_latency:6.2f} ms")
    print(f"  Confusion Matrix:        TP={doc_tp} | TN={doc_tn} | FP={doc_fp} | FN={doc_fn}")
    print("=" * 78 + "\n")

    return records, summary


def main():
    args = parse_args()
    cfg = DocROIConfig()
    if args.data_dir:
        cfg.academic_data_dir = args.data_dir
    if args.kyc_dir:
        cfg.real_kyc_data_dir = args.kyc_dir
    if args.reports_dir:
        cfg.reports_dir = args.reports_dir
        os.makedirs(cfg.reports_dir, exist_ok=True)

    device = args.device if torch.cuda.is_available() else "cpu"
    print("=" * 80)
    print(f"  DocROI-Former Full-Sample Individual Evaluation Engine")
    print(f"  Target Device: {device} | Threshold: {args.threshold}")
    print("=" * 80)

    model = DocROIFormer(config=cfg).to(device)

    # Load weights if specified or auto-detect newly trained checkpoint
    chosen_ckpt = None
    if args.checkpoint and os.path.exists(args.checkpoint):
        chosen_ckpt = args.checkpoint
    elif os.path.exists("checkpoints/best_docroi_former_v2_pgu.pth"):
        chosen_ckpt = "checkpoints/best_docroi_former_v2_pgu.pth"
    elif os.path.exists("kaggle_trained_model/latestnight_28sept/best_docroi_former_latest28sept.pth"):
        chosen_ckpt = "kaggle_trained_model/latestnight_28sept/best_docroi_former_latest28sept.pth"

    if chosen_ckpt:
        print(f"[Model Weight Loader] Loading model checkpoint from: {chosen_ckpt}")
        ckpt = torch.load(chosen_ckpt, map_location=device, weights_only=False)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict, strict=False)
    else:
        print("[Notice] Running with initialized weights (dry run / architectural verification mode).")

    model.eval()

    all_records: List[EvaluationSampleRecord] = []
    all_summaries: Dict[str, Any] = {}

    # Assemble all benchmark testbeds with exact ground-truth mask pairings
    testbeds: List[Tuple[str, UnifiedDocForensicDataset]] = []

    def _find_lmdb(dataset_name):
        home_dir = os.path.expanduser("~")
        candidates = [
            os.path.join(home_dir, "data", "Doctamper", dataset_name),
            os.path.join(home_dir, "data", dataset_name),
            os.path.join(cfg.academic_data_dir, "Doctamper", dataset_name),
            os.path.join(cfg.academic_data_dir, dataset_name),
            os.path.join(cfg.academic_data_dir, "doctamper", dataset_name),
        ]
        if os.path.exists("/kaggle/input"):
            for root, dirs, _ in os.walk("/kaggle/input"):
                if dataset_name in dirs:
                    candidates.append(os.path.join(root, dataset_name))
        for c in candidates:
            if os.path.exists(c):
                return c
        return None

    # 1. DocTamper TestingSet (30,000 samples)
    dt_test = _find_lmdb("DocTamperV1-TestingSet")
    if dt_test:
        testbeds.append(("DocTamper_TestingSet", UnifiedDocForensicDataset(lmdb_path=dt_test)))

    # 2. DocTamper SCD (18,000 samples)
    dt_scd = _find_lmdb("DocTamperV1-SCD")
    if dt_scd:
        testbeds.append(("DocTamper_SCD", UnifiedDocForensicDataset(lmdb_path=dt_scd)))

    # 3. MIDV-500 Clean International IDs (15,049 samples)
    midv_files = glob.glob(os.path.join(cfg.academic_data_dir, "Midv500", "**", "*.tif"), recursive=True) + \
                 glob.glob(os.path.join(cfg.academic_data_dir, "Midv500", "**", "*.png"), recursive=True)
    if midv_files:
        testbeds.append(("MIDV500_Clean_IDs", UnifiedDocForensicDataset(image_paths=midv_files, is_clean_only=True, enable_counterfactual=False)))

    # 4. CASIA 2.0 Splicing (Paired Tp + Au)
    casia_base = os.path.join(cfg.academic_data_dir, "CASIA 2.0 Image Tampering Detection Dataset", "CASIA2")
    if os.path.exists(casia_base):
        tp_dir = os.path.join(casia_base, "Tp")
        au_dir = os.path.join(casia_base, "Au")
        gt_dir = os.path.join(casia_base, "CASIA 2 Groundtruth")
        casia_imgs, casia_masks = [], []
        for f in glob.glob(os.path.join(tp_dir, "*.*")):
            if f.endswith(".db"): continue
            base = os.path.splitext(os.path.basename(f))[0]
            m_path = os.path.join(gt_dir, base + "_gt.png")
            if not os.path.exists(m_path):
                m_path = os.path.join(gt_dir, base + "_gt.tif")
            if os.path.exists(m_path):
                casia_imgs.append(f)
                casia_masks.append(m_path)
        for f in glob.glob(os.path.join(au_dir, "*.*")):
            if f.endswith(".db"): continue
            casia_imgs.append(f)
            casia_masks.append("")
        testbeds.append(("CASIA_2_0_Splicing", UnifiedDocForensicDataset(image_paths=casia_imgs, mask_paths=casia_masks)))

    # 5. COCOGLIDE Generative Inpainting
    coco_base = os.path.join(cfg.academic_data_dir, "COCOGLIDE_trufor")
    coco_table = os.path.join(coco_base, "table.csv")
    if os.path.exists(coco_table):
        coco_imgs, coco_masks = [], []
        with open(coco_table, mode="r", encoding="utf-8", errors="ignore") as tf:
            reader = csv.DictReader(tf)
            for row in reader:
                f_path = os.path.join(coco_base, str(row.get("fake", "")))
                m_path = os.path.join(coco_base, str(row.get("mask", "")))
                if os.path.exists(f_path) and os.path.exists(m_path):
                    coco_imgs.append(f_path)
                    coco_masks.append(m_path)
                r_path = os.path.join(coco_base, str(row.get("real", "")))
                if os.path.exists(r_path):
                    coco_imgs.append(r_path)
                    coco_masks.append("")
        testbeds.append(("COCOGLIDE_Generative_Inpainting", UnifiedDocForensicDataset(image_paths=coco_imgs, mask_paths=coco_masks)))

    # 6. Image Splicing Dataset
    splice_base = os.path.join(cfg.academic_data_dir, "Image Forgery detecion dataset (splicing)")
    if os.path.exists(splice_base):
        sp_imgs, sp_masks = [], []
        for f in glob.glob(os.path.join(splice_base, "tampered", "*.*")):
            b = os.path.basename(f).replace("tp_", "gt_")
            m_path = os.path.join(splice_base, "ground truth", b)
            if os.path.exists(m_path):
                sp_imgs.append(f)
                sp_masks.append(m_path)
        for f in glob.glob(os.path.join(splice_base, "real", "*.*")):
            sp_imgs.append(f)
            sp_masks.append("")
        testbeds.append(("Image_Splicing_Dataset", UnifiedDocForensicDataset(image_paths=sp_imgs, mask_paths=sp_masks)))

    # 7. Real Indian KYC (Validation & Test with Counterfactual Engine)
    kyc_val = glob.glob(os.path.join(cfg.real_kyc_data_dir, "pan_card", "valid", "**", "*.jpg"), recursive=True) + \
              glob.glob(os.path.join(cfg.real_kyc_data_dir, "adhaar_card", "valid", "**", "*.jpg"), recursive=True)
    if kyc_val:
        testbeds.append(("Real_Indian_KYC_Validation", UnifiedDocForensicDataset(image_paths=kyc_val, is_clean_only=True, enable_counterfactual=True)))

    kyc_test = glob.glob(os.path.join(cfg.real_kyc_data_dir, "pan_card", "test", "**", "*.jpg"), recursive=True) + \
               glob.glob(os.path.join(cfg.real_kyc_data_dir, "adhaar_card", "test", "**", "*.jpg"), recursive=True)
    if kyc_test:
        testbeds.append(("Real_Indian_KYC_Test", UnifiedDocForensicDataset(image_paths=kyc_test, is_clean_only=True, enable_counterfactual=True)))

    if args.datasets:
        selected_names = {d.strip() for d in args.datasets.split(",") if d.strip()}
        testbeds = [tb for tb in testbeds if tb[0] in selected_names]
        print(f"[Dataset Filter] Restricting evaluation strictly to requested: {selected_names}")
    elif args.doctamper_only:
        dt_names = {"DocTamper_TestingSet", "DocTamper_SCD"}
        testbeds = [tb for tb in testbeds if tb[0] in dt_names]
        print(f"[Dataset Filter] Restricting evaluation strictly to DocTamper benchmarks: {[tb[0] for tb in testbeds]}")
    elif args.only_trained:
        trained_names = {"DocTamper_TestingSet", "DocTamper_SCD", "Real_Indian_KYC_Test", "Real_Indian_KYC_Validation"}
        testbeds = [tb for tb in testbeds if tb[0] in trained_names]
        print(f"[Dataset Filter] Restricting evaluation strictly to trained domains: {[tb[0] for tb in testbeds]}")

    for name, ds in testbeds:
        records, summary = evaluate_dataset(
            model, ds, name, device,
            threshold=args.threshold,
            max_samples=args.max_samples_per_ds,
            batch_size=args.batch_size
        )
        all_records.extend(records)
        all_summaries[name] = summary

        # Export per-dataset CSV
        ds_slug = name.lower()
        if "testingset" in ds_slug:
            ds_filename = "doctamperv1_testingset_30k_sample_records.csv"
        elif "scd" in ds_slug:
            ds_filename = "doctamperv1_scd_18k_sample_records.csv"
        else:
            ds_filename = f"{ds_slug}_sample_records.csv"

        ds_csv_path = os.path.join(cfg.reports_dir, ds_filename)
        with open(ds_csv_path, mode="w", newline="", encoding="utf-8") as f_ds:
            writer_ds = csv.writer(f_ds)
            writer_ds.writerow([
                "sample_index", "is_tampered_gt", "is_tampered_pred", "global_probability",
                "pixel_iou", "pixel_f1", "pixel_precision", "pixel_recall", "latency_ms"
            ])
            for idx, r in enumerate(records):
                writer_ds.writerow([
                    idx,
                    r.ground_truth_label == "TAMPERED",
                    r.predicted_label == "TAMPERED",
                    r.global_fraud_score,
                    r.pixel_iou,
                    r.pixel_f1,
                    r.pixel_precision,
                    r.pixel_recall,
                    r.inference_latency_ms
                ])
        print(f"  -> Saved {len(records)} records to: {ds_csv_path}")

    # Export Master CSV (all individual samples)
    csv_path = os.path.join(cfg.reports_dir, "master_individual_test_results.csv")
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample_id", "dataset_name", "file_path", "ground_truth_label",
            "predicted_label", "confusion_class", "global_fraud_score",
            "pixel_iou", "pixel_f1", "pixel_precision", "pixel_recall", "latency_ms"
        ])
        for r in all_records:
            writer.writerow([
                r.sample_id, r.dataset_name, r.file_path, r.ground_truth_label,
                r.predicted_label, r.confusion_class, r.global_fraud_score,
                r.pixel_iou, r.pixel_f1, r.pixel_precision, r.pixel_recall, r.inference_latency_ms
            ])
    print(f"\n[Export Complete] Master CSV with {len(all_records)} individual records saved to: {csv_path}")

    # Export Master JSON Summary
    json_path = os.path.join(cfg.reports_dir, "master_benchmark_summary.json")
    with open(json_path, mode="w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"[Export Complete] Master JSON scientific summary saved to: {json_path}")


if __name__ == "__main__":
    main()
