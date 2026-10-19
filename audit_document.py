"""
Command-Line Interface for Unified Document Fraud & Authenticity Audit
======================================================================
Usage:
  python audit_document.py --image "d:/shivendra pratap singh/image.png"
  python audit_document.py --image "d:/shivendra pratap singh/modifciedone.jpg"
  python audit_document.py --image "path/to/any_id.jpg" --type PAN_CARD
"""

import argparse
import json
import os
import sys

# Ensure docroi_former is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.unified_pipeline import UnifiedDocumentFraudPipeline


def main():
    parser = argparse.ArgumentParser(description="Unified KYC/KYB Document Fraud Audit Pipeline")
    parser.add_argument("--image", type=str, required=True, help="Path to input document image")
    parser.add_argument("--type", type=str, default=None, help="Document type (PAN_CARD, AADHAAR_CARD, PASSPORT, KYB_CORPORATE, or auto-detect)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to custom model checkpoint (.pth)")
    parser.add_argument("--device", type=str, default="cuda", help="Computing device (cuda or cpu)")
    parser.add_argument("--output_dir", type=str, default="applied_results", help="Directory to save visual evidence board and JSON")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Image '{args.image}' not found.")
        sys.exit(1)

    print("=" * 70)
    print("  UNIFIED DOCUMENT FRAUD & AUTHENTICITY AUDIT PIPELINE")
    print("=" * 70)
    print(f"Target Image:  {args.image}")
    print(f"Device:        {args.device}")
    print(f"Specified Type:{args.type or 'Auto-Detect'}")
    if args.checkpoint:
        print(f"Checkpoint:    {args.checkpoint}")
    print("-" * 70)

    pipeline = UnifiedDocumentFraudPipeline(checkpoint_path=args.checkpoint, device=args.device)
    result = pipeline.process(args.image, doc_type=args.type, output_evidence_dir=args.output_dir)

    print("\n" + "=" * 70)
    print(f"AUDIT DECISION:      {result.decision}")
    print(f"TRUST SCORE:         {result.trust_score * 100:.2f}% / 100%")
    print(f"RISK SCORE:          {result.risk_score * 100:.2f}%")
    print(f"DOCUMENT TYPE:       {result.structural_report.document_type}")
    print(f"RECTIFICATION:       {'Applied (Background stripped)' if result.rectification.was_rectified else 'Passed-through (Clean crop)'}")
    print(f"NEURAL ANOMALY AREA: {result.neural_metrics.get('tampered_area_pct', 0.0)}% ({result.neural_metrics.get('tampered_pixels_count', 0)} pixels)")
    print("=" * 70)

    print("\n[Layer 2 & 3 Structural & Cryptographic Checks]:")
    for c in result.structural_report.checks:
        status_tag = "PASS" if c.passed else f"FAIL ({c.severity.value})"
        print(f"  [{status_tag}] {c.check_name}: {c.detail[:80]}...")

    # Save detailed JSON report
    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, f"{os.path.splitext(os.path.basename(args.image))[0]}_audit_receipt.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2)

    print(f"\nOfficial Document Trust Receipt saved to: {json_path}")
    if result.evidence_board_path:
        print(f"Visual Evidence Board saved to:           {result.evidence_board_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
