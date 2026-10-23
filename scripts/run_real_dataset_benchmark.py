import os
import sys
import time
import json
import csv
import argparse
from typing import Dict, Any, List
import cv2
import numpy as np

# Ensure project root in python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.unified_pipeline import UnifiedDocumentFraudPipeline
from dashboard.app import create_heatmap_overlay, create_ocr_annotated_image

DEFAULT_DATASET_DIR = r"D:\shivendra pratap singh\docroi_former\Real_dataset_pan_adhaar"
DEFAULT_OUTPUT_DIR = r"D:\shivendra pratap singh\docroi_former\testingrealpanaddahronnpipeline"


def parse_args():
    parser = argparse.ArgumentParser(description="Run DocROI-Former Pipeline on Real PAN & Aadhaar Datasets")
    parser.add_argument("--dataset-dir", type=str, default=DEFAULT_DATASET_DIR, help="Path to Real_dataset_pan_adhaar")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output destination folder")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum number of samples to process (None for all)")
    parser.add_argument("--splits", type=str, default="all", help="Splits to process: 'test', 'valid', 'test,valid', or 'all'")
    parser.add_argument("--categories", type=str, default="adhaar_card,pan_card", help="Comma-separated categories")
    parser.add_argument("--device", type=str, default="cuda" if os.environ.get("USE_CUDA", "1") == "1" else "cpu")
    return parser.parse_args()


def gather_dataset_files(dataset_dir: str, categories: List[str], target_splits: List[str]) -> List[Dict[str, str]]:
    """Discovers all images matching specified categories and splits."""
    samples = []
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for cat in categories:
        cat_dir = os.path.join(dataset_dir, cat)
        if not os.path.exists(cat_dir):
            print(f"[Warning] Category directory not found: {cat_dir}")
            continue

        for root, dirs, files in os.walk(cat_dir):
            rel_path = os.path.relpath(root, cat_dir)
            split_name = rel_path.split(os.sep)[0].lower() if rel_path != "." else "root"
            
            # Check if this split is in target_splits
            if "all" not in target_splits and split_name not in target_splits:
                continue

            for f in sorted(files):
                ext = os.path.splitext(f)[1].lower()
                if ext in valid_exts:
                    full_path = os.path.join(root, f)
                    sample_id = f"{cat}_{split_name}_{os.path.splitext(f)[0]}"
                    # Sanitize sample_id for filenames
                    clean_id = "".join([c if c.isalnum() or c in ("-", "_") else "_" for c in sample_id])
                    samples.append({
                        "sample_id": clean_id,
                        "category": cat,
                        "split": split_name,
                        "filename": f,
                        "file_path": full_path
                    })

    return samples


def init_master_csv(csv_path: str) -> set:
    """Initializes master CSV if not present, and returns set of already processed sample_ids."""
    processed = set()
    headers = [
        "sample_id",
        "category",
        "split",
        "filename",
        "decision",
        "trust_score_pct",
        "risk_score_pct",
        "doc_type_predicted",
        "id_number_extracted",
        "name_extracted",
        "dob_extracted",
        "qr_detected",
        "qr_type",
        "qr_is_signed",
        "structural_violations_count",
        "structural_critical_count",
        "neural_peak_prob_pct",
        "tampered_pixels_count",
        "tampered_clusters_count",
        "latency_sec",
        "overlay_image",
        "rectified_image",
        "ocr_image",
        "report_json"
    ]

    if os.path.exists(csv_path):
        try:
            with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if "sample_id" in row:
                        processed.add(row["sample_id"])
            print(f"[Resume] Found existing CSV with {len(processed)} already processed samples.")
        except Exception as e:
            print(f"[Warning] Could not read existing CSV: {e}")

    if not os.path.exists(csv_path):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    return processed


def append_csv_row(csv_path: str, row: Dict[str, Any]):
    """Appends a single audited record to master CSV."""
    headers = [
        "sample_id", "category", "split", "filename", "decision",
        "trust_score_pct", "risk_score_pct", "doc_type_predicted",
        "id_number_extracted", "name_extracted", "dob_extracted",
        "qr_detected", "qr_type", "qr_is_signed",
        "structural_violations_count", "structural_critical_count",
        "neural_peak_prob_pct", "tampered_pixels_count", "tampered_clusters_count",
        "latency_sec", "overlay_image", "rectified_image", "ocr_image", "report_json"
    ]
    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writerow(row)


def update_summary_statistics(csv_path: str, stats_json_path: str):
    """Computes aggregate analytics from the master CSV and dumps a live summary JSON."""
    if not os.path.exists(csv_path):
        return

    total = 0
    decisions = {}
    categories = {}
    splits = {}
    latencies = []
    tampered_counts = []
    peak_probs = []

    with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            dec = row.get("decision", "UNKNOWN")
            decisions[dec] = decisions.get(dec, 0) + 1

            cat = row.get("category", "UNKNOWN")
            categories[cat] = categories.get(cat, 0) + 1

            sp = row.get("split", "UNKNOWN")
            splits[sp] = splits.get(sp, 0) + 1

            try:
                latencies.append(float(row.get("latency_sec", 0)))
            except:
                pass

            try:
                tampered_counts.append(int(row.get("tampered_pixels_count", 0)))
            except:
                pass

            try:
                peak_probs.append(float(row.get("neural_peak_prob_pct", 0)))
            except:
                pass

    if total == 0:
        return

    approved = decisions.get("APPROVED", 0)
    rejected = sum(v for k, v in decisions.items() if "REJECT" in k)
    escalated = sum(v for k, v in decisions.items() if "ESCALATE" in k)

    stats = {
        "total_documents_processed": total,
        "decisions": {
            "approved": {"count": approved, "percentage": round((approved / total) * 100.0, 2)},
            "rejected": {"count": rejected, "percentage": round((rejected / total) * 100.0, 2)},
            "escalated_to_human_review": {"count": escalated, "percentage": round((escalated / total) * 100.0, 2)},
            "all_breakdown": decisions
        },
        "by_category": categories,
        "by_split": splits,
        "performance": {
            "avg_latency_sec": round(float(np.mean(latencies)), 3) if latencies else 0,
            "min_latency_sec": round(float(np.min(latencies)), 3) if latencies else 0,
            "max_latency_sec": round(float(np.max(latencies)), 3) if latencies else 0,
            "p95_latency_sec": round(float(np.percentile(latencies, 95)), 3) if latencies else 0
        },
        "neural_metrics": {
            "avg_peak_tamper_prob_pct": round(float(np.mean(peak_probs)), 2) if peak_probs else 0,
            "docs_with_zero_tamper_pixels": sum(1 for t in tampered_counts if t == 0),
            "docs_with_anomalies": sum(1 for t in tampered_counts if t > 0)
        },
        "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(stats_json_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    args = parse_args()
    print("=" * 80)
    print(" DOCROI-FORMER ENTERPRISE DATASET FORENSIC AUDIT RUNNER")
    print("=" * 80)
    print(f" Source Dataset Directory : {args.dataset_dir}")
    print(f" Target Output Directory  : {args.output_dir}")
    print(f" Categories               : {args.categories}")
    print(f" Target Splits            : {args.splits}")
    print(f" Execution Device         : {args.device}")
    print(f" Sample Limit             : {args.max_samples or 'ALL Available'}")
    print("=" * 80)

    # Prepare directories
    visuals_dir = os.path.join(args.output_dir, "visuals")
    reports_dir = os.path.join(args.output_dir, "reports")
    os.makedirs(visuals_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    csv_path = os.path.join(args.output_dir, "master_summary.csv")
    stats_path = os.path.join(args.output_dir, "summary_statistics.json")

    # Discover files
    cats = [c.strip() for c in args.categories.split(",") if c.strip()]
    splits = [s.strip().lower() for s in args.splits.split(",") if s.strip()]
    samples = gather_dataset_files(args.dataset_dir, cats, splits)
    print(f"[Discovery] Found {len(samples)} total valid images across requested splits.")

    if args.max_samples and len(samples) > args.max_samples:
        samples = samples[:args.max_samples]
        print(f"[Limit] Capped dataset to {len(samples)} samples as requested.")

    # Init CSV
    processed_ids = init_master_csv(csv_path)

    # Filter out already processed
    remaining_samples = [s for s in samples if s["sample_id"] not in processed_ids]
    print(f"[Status] {len(processed_ids)} already processed. Remaining to process: {len(remaining_samples)}")

    if not remaining_samples:
        print("[Complete] All samples have already been processed! Exiting.")
        update_summary_statistics(csv_path, stats_path)
        return

    # Initialize Master Pipeline
    print("\n[Pipeline] Initializing UnifiedDocumentFraudPipeline on device:", args.device)
    pipeline = UnifiedDocumentFraudPipeline(device=args.device)
    print("[Pipeline] Ready! Beginning forensic analysis loop...\n")

    start_all_time = time.time()
    success_count = 0
    error_count = 0

    for idx, item in enumerate(remaining_samples, 1):
        sample_id = item["sample_id"]
        file_path = item["file_path"]
        cat = item["category"]
        split = item["split"]
        filename = item["filename"]

        t_start = time.time()
        try:
            # 1. Execute unified pipeline
            result = pipeline.process(file_path)
            t_elapsed = time.time() - t_start

            rect_img = result.rectification.rectified_image
            clean_hm = result.clean_heatmap
            ext_fields = result.extracted_fields
            struct_rep = result.structural_report
            neural = result.neural_metrics

            # 2. Render Visual Artifacts
            # A. Clean Rectified Image
            rect_filename = f"{sample_id}_rectified.jpg"
            rect_save_path = os.path.join(visuals_dir, rect_filename)
            if rect_img is not None and rect_img.size > 0:
                cv2.imwrite(rect_save_path, rect_img)

            # B. Localized Thresholded Heatmap Overlay (No circles, strictly localized)
            overlay_filename = f"{sample_id}_overlay.jpg"
            overlay_save_path = os.path.join(visuals_dir, overlay_filename)
            if rect_img is not None and clean_hm is not None:
                overlay_img = create_heatmap_overlay(rect_img, clean_hm, thresh=0.25, max_alpha=0.80)
                cv2.imwrite(overlay_save_path, overlay_img)

            # C. OCR Annotated Boxes
            ocr_filename = f"{sample_id}_ocr.jpg"
            ocr_save_path = os.path.join(visuals_dir, ocr_filename)
            raw_tokens = ext_fields.get("raw_tokens", [])
            if rect_img is not None and raw_tokens:
                ocr_img = create_ocr_annotated_image(rect_img, raw_tokens)
                cv2.imwrite(ocr_save_path, ocr_img)

            # 3. Save Detailed Document Report JSON
            report_filename = f"{sample_id}_report.json"
            report_save_path = os.path.join(reports_dir, report_filename)
            report_data = result.to_dict()
            with open(report_save_path, "w", encoding="utf-8") as rf:
                json.dump(report_data, rf, indent=2)

            # 4. Extract Key Fields for Master CSV
            id_num = (
                ext_fields.get("aadhaar_number")
                or ext_fields.get("pan_number")
                or ext_fields.get("pan")
                or ext_fields.get("gstin")
                or ""
            )
            name_val = ext_fields.get("name") or ""
            dob_val = ext_fields.get("dob") or ""
            has_qr = bool(ext_fields.get("has_qr", False))
            qr_type = str(ext_fields.get("qr_type", ""))
            qr_signed = bool(ext_fields.get("qr_is_signed", False))

            crit_count = struct_rep.critical_violations_count if struct_rep else 0
            warn_count = struct_rep.warning_violations_count if struct_rep else 0
            total_viol = crit_count + warn_count

            peak_prob = float(neural.get("peak_pixel_probability_pct", 0.0))
            tamper_pixels = int(neural.get("tampered_pixels_count", 0))
            tamper_clusters = int(neural.get("tampered_clusters_count", 0))
            pred_doc = result.classification.get("predicted_category") if result.classification else ""

            # 5. Append to Master CSV
            csv_row = {
                "sample_id": sample_id,
                "category": cat,
                "split": split,
                "filename": filename,
                "decision": result.decision,
                "trust_score_pct": round(result.trust_score * 100.0, 1),
                "risk_score_pct": round(result.risk_score * 100.0, 1),
                "doc_type_predicted": pred_doc,
                "id_number_extracted": id_num,
                "name_extracted": name_val,
                "dob_extracted": dob_val,
                "qr_detected": has_qr,
                "qr_type": qr_type,
                "qr_is_signed": qr_signed,
                "structural_violations_count": total_viol,
                "structural_critical_count": crit_count,
                "neural_peak_prob_pct": peak_prob,
                "tampered_pixels_count": tamper_pixels,
                "tampered_clusters_count": tamper_clusters,
                "latency_sec": round(t_elapsed, 3),
                "overlay_image": os.path.relpath(overlay_save_path, args.output_dir),
                "rectified_image": os.path.relpath(rect_save_path, args.output_dir),
                "ocr_image": os.path.relpath(ocr_save_path, args.output_dir),
                "report_json": os.path.relpath(report_save_path, args.output_dir)
            }
            append_csv_row(csv_path, csv_row)
            success_count += 1

            # Progress log
            status_symbol = "[OK]  " if result.decision == "APPROVED" else ("[REJ] " if "REJECT" in result.decision else "[REV] ")
            print(f"[{idx}/{len(remaining_samples)}] {status_symbol} {sample_id[:35]:<35} | {result.decision:<18} | Trust: {result.trust_score * 100:5.1f}% | Latency: {t_elapsed:.2f}s | TamperPx: {tamper_pixels}")

        except Exception as e:
            error_count += 1
            t_elapsed = time.time() - t_start
            print(f"[{idx}/{len(remaining_samples)}] [ERR] ERROR on {sample_id}: {e}")
            error_row = {
                "sample_id": sample_id,
                "category": cat,
                "split": split,
                "filename": filename,
                "decision": "ERROR_PROCESSING_FAILED",
                "trust_score_pct": 0.0,
                "risk_score_pct": 100.0,
                "doc_type_predicted": "ERROR",
                "id_number_extracted": "",
                "name_extracted": "",
                "dob_extracted": "",
                "qr_detected": False,
                "qr_type": "",
                "qr_is_signed": False,
                "structural_violations_count": 0,
                "structural_critical_count": 0,
                "neural_peak_prob_pct": 0.0,
                "tampered_pixels_count": 0,
                "tampered_clusters_count": 0,
                "latency_sec": round(t_elapsed, 3),
                "overlay_image": "",
                "rectified_image": "",
                "ocr_image": "",
                "report_json": ""
            }
            append_csv_row(csv_path, error_row)

        # Update summary statistics JSON every 10 samples
        if idx % 10 == 0 or idx == len(remaining_samples):
            update_summary_statistics(csv_path, stats_path)

    total_time = time.time() - start_all_time
    update_summary_statistics(csv_path, stats_path)

    print("\n" + "=" * 80)
    print(" BENCHMARK EXECUTION COMPLETED")
    print("=" * 80)
    print(f" Total Processed in this run : {success_count + error_count}")
    print(f" Successful Audits           : {success_count}")
    print(f" Errors Encountered          : {error_count}")
    print(f" Total Wall Time             : {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")
    if success_count > 0:
        print(f" Average Latency per Sample  : {total_time / (success_count + error_count):.2f} seconds")
    print(f" Master CSV Summary Saved    : {csv_path}")
    print(f" Summary Statistics JSON     : {stats_path}")
    print(f" Visual Artifacts Folder     : {visuals_dir}")
    print(f" Reports Folder              : {reports_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
