import sys, os, torch, cv2, shutil, json
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

repo_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, repo_dir)
from src.dataset.doc_dataset import UnifiedDocForensicDataset
from configs.docroi_config import DocROIConfig
from src.models.docroi_former import DocROIFormer
from src.dataset.entity_tokenizer import OtsuEntityTokenizer

# Target Output Directories
dest_dir1 = os.path.join(os.path.dirname(repo_dir), "14oct")
dest_dir2 = os.path.join(repo_dir, "14oct")

os.makedirs(dest_dir1, exist_ok=True)
os.makedirs(dest_dir2, exist_ok=True)
print(f"[Init] Output directories verified:\n  1. {dest_dir1}\n  2. {dest_dir2}")

# V1 Mask Head from 28-Sept Baseline Architecture
class V1MaskHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj1 = nn.Conv2d(64, 256, 1)
        self.proj2 = nn.Conv2d(128, 256, 1)
        self.proj3 = nn.Conv2d(320, 256, 1)
        self.proj4 = nn.Conv2d(512, 256, 1)
        self.fusion = nn.Sequential(
            nn.Conv2d(1024, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.boundary_refine = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1, bias=True)
        )

    def forward(self, feats, orig_size=(384, 384)):
        p1 = self.proj1(feats[0])
        p2 = F.interpolate(self.proj2(feats[1]), size=p1.shape[2:], mode='bilinear', align_corners=False)
        p3 = F.interpolate(self.proj3(feats[2]), size=p1.shape[2:], mode='bilinear', align_corners=False)
        p4 = F.interpolate(self.proj4(feats[3]), size=p1.shape[2:], mode='bilinear', align_corners=False)
        fused = self.fusion(torch.cat([p1, p2, p3, p4], dim=1))
        logits = self.boundary_refine(fused)
        return F.interpolate(logits, size=orig_size, mode='bilinear', align_corners=False)

# SAFETY FIRST: Run strictly on CPU to protect training GPU memory from collision
device = torch.device("cpu")
print(f"[Device Selection] Running strictly on {device} (0 MB VRAM impact on active training GPU).")

cfg = DocROIConfig()
model = DocROIFormer(config=cfg).to(device)

ckpt_path = os.path.join(repo_dir, "kaggle_trained_model", "latestnight_28sept", "best_docroi_former_latest28sept.pth")
print(f"[Loading Weights] {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
model.load_state_dict(state_dict, strict=False)

v1_head = V1MaskHead().to(device)
v1_weights = {k.replace("mask_head.", ""): v for k, v in state_dict.items() if k.startswith("mask_head.")}
v1_head.load_state_dict(v1_weights)
v1_head.eval()
model.eval()
print("[Model Ready] 28-Sept Baseline Model + V1 Mask Head successfully loaded.")

tokenizer = OtsuEntityTokenizer()

# Find dataset paths (supporting both Linux and Windows paths)
def find_dataset(name):
    home = os.path.expanduser("~")
    cands = [
        os.path.join(repo_dir, "data", "Doctamper", name),
        os.path.join(home, "data", "Doctamper", name),
        os.path.join("/home/ogmailab/data/Doctamper", name),
        os.path.join(cfg.academic_data_dir, "Doctamper", name),
        os.path.join(cfg.academic_data_dir, name)
    ]
    for c in cands:
        if os.path.exists(c):
            return c
    return None

dt_test_path = find_dataset("DocTamperV1-TestingSet")
scd_path = find_dataset("DocTamperV1-SCD")
print(f"[Dataset Locations]\n  TestingSet: {dt_test_path}\n  SCD: {scd_path}")

ds_test = UnifiedDocForensicDataset(lmdb_path=dt_test_path)
ds_scd = UnifiedDocForensicDataset(lmdb_path=scd_path)
print(f"[Datasets Loaded] DocTamper-TestingSet: {len(ds_test)} samples | DocTamper-SCD: {len(ds_scd)} samples")

def find_test_samples(ds, tampered_count=3, clean_count=1, stride=37):
    tampered_indices = []
    clean_indices = []
    for idx in range(15, len(ds), stride):
        sample = ds[idx]
        gt = sample["mask"][0].numpy()
        p_sum = (gt > 0.5).sum()
        if p_sum > 200 and len(tampered_indices) < tampered_count:
            tampered_indices.append(idx)
        elif p_sum == 0 and len(clean_indices) < clean_count:
            clean_indices.append(idx)
        if len(tampered_indices) >= tampered_count and len(clean_indices) >= clean_count:
            break
    return tampered_indices, clean_indices

test_t, test_c = find_test_samples(ds_test, 3, 1, stride=31)
scd_t, scd_c = find_test_samples(ds_scd, 3, 1, stride=43)
print(f"DocTamper-TestingSet selected: Tampered={test_t}, Clean={test_c}")
print(f"DocTamper-SCD selected: Tampered={scd_t}, Clean={scd_c}")

all_results = []

def evaluate_and_plot(ds, idx, ds_title, out_prefix, is_clean=False):
    sample = ds[idx]
    img = sample["image"].unsqueeze(0).to(device)
    gt = sample["mask"][0].numpy()
    
    # RGB recovery
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
    img_np = sample["image"].permute(1, 2, 0).numpy()
    img_rgb = np.clip((img_np * std + mean) * 255.0, 0, 255).astype(np.uint8)
    
    prior = tokenizer(img[0]).unsqueeze(0).to(device)
    with torch.no_grad():
        spatial_feats = model.spatial_encoder(img)
        freq_feats, _ = model.frequency_encoder(img, return_srm=True)
        routing_logits, relevance_map = model.entity_router(spatial_feats[0], prior)
        pred_classes = torch.argmax(routing_logits, dim=1, keepdim=True)
        entity_mask_2d = (pred_classes > 0).float()
        
        fused_features = []
        for s_idx in range(4):
            s_i = spatial_feats[s_idx]
            f_i = freq_feats[s_idx]
            f_i_gated = model.relevance_gates[s_idx](f_i, relevance_map)
            if entity_mask_2d.shape[-2:] != s_i.shape[-2:]:
                cur_mask = F.interpolate(entity_mask_2d, size=s_i.shape[-2:], mode="nearest")
            else:
                cur_mask = entity_mask_2d
            cross = model.cross_attentions[s_idx](s_i, f_i_gated, cur_mask)
            fused = model.fuse_convs[s_idx](torch.cat([s_i, cross], dim=1))
            fused_features.append(fused)
            
        logits = v1_head(fused_features, orig_size=(384, 384))
        sig = torch.sigmoid(logits)[0, 0].numpy()
        
        g_logits = model.global_classifier(spatial_feats[-1], freq_feats[-1])
        p_doc = float(torch.sigmoid(g_logits).item())
        
    p_bin = sig > 0.50
    g_bin = gt > 0.50
    
    gt_pixels = int(g_bin.sum())
    pred_pixels = int(p_bin.sum())
    
    tp = int(np.logical_and(p_bin, g_bin).sum())
    fp = int(np.logical_and(p_bin, ~g_bin).sum())
    fn = int(np.logical_and(~p_bin, g_bin).sum())
    tn = int(np.logical_and(~p_bin, ~g_bin).sum())
    
    if gt_pixels > 0:
        iou = float(tp / (tp + fp + fn + 1e-8) * 100.0)
        f1 = float(2 * tp / (2 * tp + fp + fn + 1e-8) * 100.0)
        precision = float(tp / (tp + fp + 1e-8) * 100.0)
        recall = float(tp / (tp + fn + 1e-8) * 100.0)
        verdict = "CORRECT" if (p_doc > 0.50 and iou > 0.0) else ("FALSE_ALARM_OVERSEG" if (p_doc > 0.50 and iou == 0.0) else "MISSED")
    else:
        # Authentic clean document
        iou = 100.0 if pred_pixels == 0 else 0.0
        f1 = 100.0 if pred_pixels == 0 else 0.0
        precision = 100.0 if pred_pixels == 0 else 0.0
        recall = 100.0
        verdict = "CORRECT_CLEAN" if (p_doc < 0.50 and pred_pixels == 0) else "CLEAN_FALSE_ALARM"

    rec = {
        "dataset": ds_title,
        "sample_id": int(idx),
        "is_authentic": is_clean,
        "ground_truth": "CLEAN" if is_clean else "TAMPERED",
        "predicted_document_verdict": "TAMPERED" if p_doc >= 0.50 else "AUTHENTIC",
        "global_fraud_prob": float(p_doc * 100.0),
        "gt_pixels": gt_pixels,
        "pred_pixels": pred_pixels,
        "pixel_iou": iou,
        "pixel_f1": f1,
        "precision": precision,
        "recall": recall,
        "verdict_audit": verdict,
        "file": f"{out_prefix}.png"
    }
    all_results.append(rec)
    
    # 4-Panel High-Resolution Visual Audit
    fig, axes = plt.subplots(1, 4, figsize=(22, 6), dpi=140)
    fig.patch.set_facecolor('#0d1117')
    
    # 1. Input Image
    axes[0].imshow(img_rgb)
    axes[0].set_title(f"1. Input Document\n[{ds_title} Sample #{idx}]", color='white', fontsize=12, fontweight='bold', pad=8)
    axes[0].axis('off')
    
    # 2. Ground Truth Mask
    gt_overlay = img_rgb.copy()
    gt_color = np.zeros_like(img_rgb)
    gt_color[g_bin] = [0, 255, 0] # Green
    gt_overlay = cv2.addWeighted(gt_overlay, 0.70, gt_color, 0.30, 0)
    axes[1].imshow(gt_overlay)
    status_str = f"[{gt_pixels} Tampered Pixels]" if not is_clean else "[0 Pixels - Authentic Clean]"
    axes[1].set_title(f"2. Ground Truth Mask\n{status_str}", color='#7ee787', fontsize=12, fontweight='bold', pad=8)
    axes[1].axis('off')
    
    # 3. Model Predicted Heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * np.clip(sig, 0, 1)), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heat_overlay = cv2.addWeighted(img_rgb, 0.65, heatmap, 0.35, 0)
    axes[2].imshow(heat_overlay)
    axes[2].set_title(f"3. 28-Sept V1 Heatmap\n(Global P_doc: {p_doc*100:.1f}%)", color='#58a6ff', fontsize=12, fontweight='bold', pad=8)
    axes[2].axis('off')
    
    # 4. Forensic Overlap Audit
    audit_overlay = img_rgb.copy()
    color_map = np.zeros_like(img_rgb)
    color_map[np.logical_and(p_bin, g_bin)] = [0, 255, 0]     # Green = TP (Accurate)
    color_map[np.logical_and(p_bin, ~g_bin)] = [255, 60, 60]  # Red = FP (False Alarm)
    color_map[np.logical_and(~p_bin, g_bin)] = [60, 150, 255] # Blue = FN (Missed)
    audit_overlay = cv2.addWeighted(audit_overlay, 0.60, color_map, 0.40, 0)
    axes[3].imshow(audit_overlay)
    axes[3].set_title(f"4. Model vs GT Overlap Audit\nIoU: {iou:.2f}% | F1: {f1:.2f}%\n[Green=TP, Red=FP, Blue=FN]", color='#ffa657', fontsize=12, fontweight='bold', pad=8)
    axes[3].axis('off')
    
    plt.suptitle(f"DocROI-Former 28-Sept Baseline Audit — {ds_title} (Sample #{idx})\nPixel IoU: {iou:.2f}% | Pixel F1: {f1:.2f}% | Global Fraud Prob: {p_doc*100:.2f}% | Audit: {verdict}",
                 color='#f0f6fc', fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    out_filename = f"{out_prefix}.png"
    p1 = os.path.join(dest_dir1, out_filename)
    p2 = os.path.join(dest_dir2, out_filename)
    plt.savefig(p1, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    shutil.copy(p1, p2)
    plt.close()
    print(f"  [Saved] {out_filename} -> IoU: {iou:.2f}%, F1: {f1:.2f}%, P_doc: {p_doc*100:.2f}%, Audit: {verdict}")

print("\n--- Running Evaluation on DocTamper-TestingSet ---")
for i, idx in enumerate(test_t):
    evaluate_and_plot(ds_test, idx, "DocTamper-TestingSet", f"doctamper_testingset_sample_{i+1}")
for idx in test_c:
    evaluate_and_plot(ds_test, idx, "DocTamper-TestingSet", "doctamper_testingset_authentic_clean", is_clean=True)

print("\n--- Running Evaluation on DocTamper-SCD (Single Character Deletion) ---")
for i, idx in enumerate(scd_t):
    evaluate_and_plot(ds_scd, idx, "DocTamper-SCD", f"doctamper_scd_sample_{i+1}")
for idx in scd_c:
    evaluate_and_plot(ds_scd, idx, "DocTamper-SCD", "doctamper_scd_authentic_clean", is_clean=True)

# Save JSON Summary and Markdown Audit Report
summary_json1 = os.path.join(dest_dir1, "results_summary.json")
summary_json2 = os.path.join(dest_dir2, "results_summary.json")
with open(summary_json1, "w") as f:
    json.dump(all_results, f, indent=2)
shutil.copy(summary_json1, summary_json2)

md_report = os.path.join(dest_dir1, "cvpr_audit_report_28sept.md")
with open(md_report, "w") as f:
    f.write("# DocROI-Former 28-Sept Baseline Forensic Audit Report (14-Oct-2026)\n\n")
    f.write("### Evaluation Details\n")
    f.write("- **Model Evaluated:** `kaggle_trained_model/latestnight_28sept/best_docroi_former_latest28sept.pth`\n")
    f.write("- **Mask Head Architecture:** Original 28-Sept V1 Bilinear Head\n")
    f.write("- **Inference Device:** CPU (0 MB VRAM impact on active training)\n")
    f.write("- **Evaluation Threshold:** Fixed $\\tau = 0.50$ (Zero test-time tuning)\n")
    f.write("- **Post-Processing:** None (Zero morphological dilation)\n\n")
    f.write("### Sample-by-Sample Results\n\n")
    f.write("| Dataset | Sample ID | Type | Ground Truth | Global P_doc | Pixel IoU | Pixel F1 | Precision | Recall | Audit Verdict |\n")
    f.write("|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
    for r in all_results:
        f.write(f"| {r['dataset']} | {r['sample_id']} | {'Clean' if r['is_authentic'] else 'Tampered'} | {r['ground_truth']} | {r['global_fraud_prob']:.2f}% | {r['pixel_iou']:.2f}% | {r['pixel_f1']:.2f}% | {r['precision']:.2f}% | {r['recall']:.2f}% | **{r['verdict_audit']}** |\n")

shutil.copy(md_report, os.path.join(dest_dir2, "cvpr_audit_report_28sept.md"))
print(f"\n[Audit Completed] Results and report successfully saved in {dest_dir1} and {dest_dir2}!")
