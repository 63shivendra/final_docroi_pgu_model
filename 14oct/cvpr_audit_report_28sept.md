# DocROI-Former 28-Sept Baseline Forensic Audit Report (14-Oct-2026)

### Evaluation Details
- **Model Evaluated:** `kaggle_trained_model/latestnight_28sept/best_docroi_former_latest28sept.pth`
- **Mask Head Architecture:** Original 28-Sept V1 Bilinear Head
- **Inference Device:** CPU (0 MB VRAM impact on active training)
- **Evaluation Threshold:** Fixed $\tau = 0.50$ (Zero test-time tuning)
- **Post-Processing:** None (Zero morphological dilation)

### Sample-by-Sample Results

| Dataset | Sample ID | Type | Ground Truth | Global P_doc | Pixel IoU | Pixel F1 | Precision | Recall | Audit Verdict |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| DocTamper-TestingSet | 15 | Clean | CLEAN | 99.99% | 0.00% | 0.00% | 0.00% | 100.00% | **CLEAN_FALSE_ALARM** |
| DocTamper-SCD | 15 | Clean | CLEAN | 99.97% | 0.00% | 0.00% | 0.00% | 100.00% | **CLEAN_FALSE_ALARM** |
