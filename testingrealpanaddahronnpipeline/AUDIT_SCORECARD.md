# Forensic Audit Scorecard: Real PAN & Aadhaar Benchmark

**Audit Timestamp**: 2026-10-23 16:25:49  
**Overall Dataset Progress**: **4,372 / 4,372 Images (100.0%)**

---

## 1. Executive Binary Verdict (Accepted vs. Rejected)

| Verdict Class | Document Count | Percentage | Operational Meaning |
| :--- | :---: | :---: | :--- |
| **ACCEPTED (Passed)** | **2,009** | **46.0%** | Verified genuine: Flawless structural checksums and clean neural pixel integrity. |
| **REJECTED (Hard Fail)** | **1,862** | **42.6%** | Tampered / Forged / Structural failure: Breaks Verhoeff/CBDT checksum or detected neural fraud. |
| **ESCALATED TO REVIEW** | **501** | **11.5%** | Borderline anomaly: Flagged for Level 2 human compliance officer review. |
| **TOTAL AUDITED** | **4,372** | **100.0%** | Cumulative documents analyzed across the dataset. |

> **Strict Binary Breakdown**:
> - **Total Accepted**: **2,009 (46.0%)**
> - **Total Not Accepted (Rejected + Under Review)**: **2,363 (54.0%)**

---

## 2. Category Breakdown (PAN vs. Aadhaar)

| Document Category | Total Scanned | Accepted (Pass) | Rejected (Fail) | Human Review | Pass Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`adhaar_card`** | 2,646 | 1,195 | 1,032 | 419 | **45.2%** |
| **`pan_card`** | 1,726 | 814 | 830 | 82 | **47.2%** |

---

## 3. Dataset Split Breakdown

| Split | Total Scanned | Accepted | Rejected | Escalated |
| :--- | :---: | :---: | :---: | :---: |
| **`test`** | 265 | 128 | 97 | 40 |
| **`train`** | 3,310 | 1,512 | 1,423 | 375 |
| **`valid`** | 797 | 369 | 342 | 86 |

---

## 4. Output Artifacts Navigation
- **Master Spreadsheet**: [`master_summary.csv`](file:///D:/shivendra%20pratap%20singh/docroi_former/testingrealpanaddahronnpipeline/master_summary.csv)
- **Visual Heatmap Overlays**: [`visuals/`](file:///D:/shivendra%20pratap%20singh/docroi_former/testingrealpanaddahronnpipeline/visuals)
- **Detailed JSON Reports**: [`reports/`](file:///D:/shivendra%20pratap%20singh/docroi_former/testingrealpanaddahronnpipeline/reports)
- **Aggregate Statistics**: [`summary_statistics.json`](file:///D:/shivendra%20pratap%20singh/docroi_former/testingrealpanaddahronnpipeline/summary_statistics.json)
