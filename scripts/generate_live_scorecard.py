import os
import csv
import json
import time

CSV_PATH = r"D:\shivendra pratap singh\docroi_former\testingrealpanaddahronnpipeline\master_summary.csv"
SCORECARD_PATH = r"D:\shivendra pratap singh\docroi_former\testingrealpanaddahronnpipeline\AUDIT_SCORECARD.md"
TOTAL_DATASET_IMAGES = 4372


def generate_scorecard():
    if not os.path.exists(CSV_PATH):
        print("CSV does not exist yet.")
        return

    with open(CSV_PATH, mode="r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    total = len(reader)
    if total == 0:
        print("CSV is empty.")
        return

    # Counts
    approved_count = 0
    rejected_count = 0
    escalated_count = 0

    cat_counts = {}
    split_counts = {}

    for row in reader:
        dec = row.get("decision", "")
        cat = row.get("category", "UNKNOWN")
        sp = row.get("split", "UNKNOWN")

        if cat not in cat_counts:
            cat_counts[cat] = {"total": 0, "approved": 0, "rejected": 0, "escalated": 0}
        cat_counts[cat]["total"] += 1

        if sp not in split_counts:
            split_counts[sp] = {"total": 0, "approved": 0, "rejected": 0, "escalated": 0}
        split_counts[sp]["total"] += 1

        if dec == "APPROVED":
            approved_count += 1
            cat_counts[cat]["approved"] += 1
            split_counts[sp]["approved"] += 1
        elif "REJECT" in dec:
            rejected_count += 1
            cat_counts[cat]["rejected"] += 1
            split_counts[sp]["rejected"] += 1
        else:
            escalated_count += 1
            cat_counts[cat]["escalated"] += 1
            split_counts[sp]["escalated"] += 1

    app_pct = (approved_count / total) * 100.0
    rej_pct = (rejected_count / total) * 100.0
    esc_pct = (escalated_count / total) * 100.0
    total_non_approved = rejected_count + escalated_count
    total_non_app_pct = (total_non_approved / total) * 100.0
    progress_pct = (total / TOTAL_DATASET_IMAGES) * 100.0

    print("=" * 75)
    print(f" LIVE FORENSIC AUDIT SCORECARD ({total}/{TOTAL_DATASET_IMAGES} Images - {progress_pct:.1f}% Complete)")
    print("=" * 75)
    print(f" TOTAL ACCEPTED (APPROVED) : {approved_count:5d}  ({app_pct:5.1f}%)")
    print(f" TOTAL REJECTED            : {rejected_count:5d}  ({rej_pct:5.1f}%)")
    print(f" ESCALATED TO HUMAN REVIEW : {escalated_count:5d}  ({esc_pct:5.1f}%)")
    print(f" TOTAL NON-ACCEPTED (Rej+Esc): {total_non_approved:5d}  ({total_non_app_pct:5.1f}%)")
    print("-" * 75)
    for cat, data in cat_counts.items():
        print(f" [{cat.upper()}] Total: {data['total']} | Approved: {data['approved']} | Rejected: {data['rejected']} | Escalated: {data['escalated']}")
    print("=" * 75)

    # Generate Markdown Scorecard
    md_content = f"""# Forensic Audit Scorecard: Real PAN & Aadhaar Benchmark

**Audit Timestamp**: {time.strftime('%Y-%m-%d %H:%M:%S')}  
**Overall Dataset Progress**: **{total:,} / {TOTAL_DATASET_IMAGES:,} Images ({progress_pct:.1f}%)**

---

## 1. Executive Binary Verdict (Accepted vs. Rejected)

| Verdict Class | Document Count | Percentage | Operational Meaning |
| :--- | :---: | :---: | :--- |
| **ACCEPTED (Passed)** | **{approved_count:,}** | **{app_pct:.1f}%** | Verified genuine: Flawless structural checksums and clean neural pixel integrity. |
| **REJECTED (Hard Fail)** | **{rejected_count:,}** | **{rej_pct:.1f}%** | Tampered / Forged / Structural failure: Breaks Verhoeff/CBDT checksum or detected neural fraud. |
| **ESCALATED TO REVIEW** | **{escalated_count:,}** | **{esc_pct:.1f}%** | Borderline anomaly: Flagged for Level 2 human compliance officer review. |
| **TOTAL AUDITED** | **{total:,}** | **100.0%** | Cumulative documents analyzed across the dataset. |

> **Strict Binary Breakdown**:
> - **Total Accepted**: **{approved_count:,} ({app_pct:.1f}%)**
> - **Total Not Accepted (Rejected + Under Review)**: **{total_non_approved:,} ({total_non_app_pct:.1f}%)**

---

## 2. Category Breakdown (PAN vs. Aadhaar)

| Document Category | Total Scanned | Accepted (Pass) | Rejected (Fail) | Human Review | Pass Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
"""
    for cat, data in cat_counts.items():
        c_tot = data["total"]
        c_app = data["approved"]
        c_rej = data["rejected"]
        c_esc = data["escalated"]
        c_rate = (c_app / c_tot * 100.0) if c_tot > 0 else 0
        md_content += f"| **`{cat}`** | {c_tot:,} | {c_app:,} | {c_rej:,} | {c_esc:,} | **{c_rate:.1f}%** |\n"

    md_content += """
---

## 3. Dataset Split Breakdown

| Split | Total Scanned | Accepted | Rejected | Escalated |
| :--- | :---: | :---: | :---: | :---: |
"""
    for sp, data in split_counts.items():
        md_content += f"| **`{sp}`** | {data['total']:,} | {data['approved']:,} | {data['rejected']:,} | {data['escalated']:,} |\n"

    md_content += f"""
---

## 4. Output Artifacts Navigation
- **Master Spreadsheet**: [`master_summary.csv`](file:///D:/shivendra%20pratap%20singh/docroi_former/testingrealpanaddahronnpipeline/master_summary.csv)
- **Visual Heatmap Overlays**: [`visuals/`](file:///D:/shivendra%20pratap%20singh/docroi_former/testingrealpanaddahronnpipeline/visuals)
- **Detailed JSON Reports**: [`reports/`](file:///D:/shivendra%20pratap%20singh/docroi_former/testingrealpanaddahronnpipeline/reports)
- **Aggregate Statistics**: [`summary_statistics.json`](file:///D:/shivendra%20pratap%20singh/docroi_former/testingrealpanaddahronnpipeline/summary_statistics.json)
"""

    with open(SCORECARD_PATH, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Scorecard written to: {SCORECARD_PATH}")


if __name__ == "__main__":
    generate_scorecard()
