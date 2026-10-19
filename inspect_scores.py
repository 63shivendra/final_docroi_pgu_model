import os
import sys
import cv2
import numpy as np
import torch

repo_dir = r"d:\shivendra pratap singh\docroi_former"
sys.path.insert(0, repo_dir)

from configs.docroi_config import DocROIConfig
from src.models.docroi_former import DocROIFormer
from src.dataset.entity_tokenizer import OtsuEntityTokenizer

img_path = r"D:\shivendra pratap singh\finalaltrerecipt.png"
device = "cuda" if torch.cuda.is_available() else "cpu"

cfg = DocROIConfig()
model = DocROIFormer(config=cfg).to(device)
ckpt = torch.load(os.path.join(repo_dir, "checkpoints", "best_docroi_former_v2_pgu.pth"), map_location=device, weights_only=False)
state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
model.load_state_dict(state_dict, strict=False)
model.eval()

orig_bgr = cv2.imread(img_path)
orig_h, orig_w = orig_bgr.shape[:2]
orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)

resized_rgb = cv2.resize(orig_rgb, (384, 384), interpolation=cv2.INTER_LINEAR)
norm_tensor = torch.from_numpy(resized_rgb).permute(2, 0, 1).float() / 255.0
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
norm_tensor = ((norm_tensor - mean) / std).unsqueeze(0).to(device)

with torch.no_grad():
    preds = model(norm_tensor)
    prob_384 = torch.sigmoid(preds["mask_logits"])[0, 0].cpu().numpy()
    global_prob = torch.sigmoid(preds["global_logits"])[0, 0].item()

prob_orig = cv2.resize(prob_384, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
tokenizer = OtsuEntityTokenizer()
entity_mask = tokenizer.extract_entity_mask(orig_rgb)
gated_prob = prob_orig * entity_mask

print(f"Global Document Fraud Probability: {global_prob*100:.2f}%")
print(f"Overall Image Max:                  {prob_orig.max()*100:.2f}%\n")

# Detailed regions
regions = [
    ("Receipt Header (Script font)", 0, 90, 200, 750),
    ("Word 'Adress:'", 110, 165, 40, 240),
    ("Number '1284' Total", 110, 165, 260, 380),
    ("  - Digit '1' (x=260..290)", 110, 165, 260, 290),
    ("  - Digit '2' (x=290..318)", 110, 165, 290, 318),
    ("  - Digit '8' (x=318..345)", 110, 165, 318, 345),
    ("  - Digit '4' (x=345..378)", 110, 165, 345, 378),
    ("Word 'Lorem' (Address line)", 110, 165, 380, 540),
    ("Word 'Ipsum' (Address line)", 110, 165, 545, 700),
    ("Word 'Dolor' (Address line)", 110, 165, 730, 890),
    ("Telephone: 'Tel: 123-456-7890'", 180, 240, 40, 550),
    ("Date: 'Date: 01-01-2018'", 290, 350, 40, 550),
    ("Time: '10:35' Total", 290, 350, 700, 890),
    ("  - Digits '10:' (x=700..780)", 290, 350, 700, 780),
    ("  - Digits '35'  (x=780..890)", 290, 350, 780, 890),
    ("Bottom Item: 'Lorem'", 430, 485, 10, 220),
    ("Price: '6.50' Total", 430, 485, 780, 900),
    ("  - Digit '6.' (x=780..825)", 430, 485, 780, 825),
    ("  - Digits '50' (x=825..895)", 430, 485, 825, 895),
]

print(f"{'Region Evaluated':<32} | {'Raw Max %':<10} | {'Gated Max %':<12} | {'Mean %':<10}")
print("-" * 72)
for name, y1, y2, x1, x2 in regions:
    sub_raw = prob_orig[y1:y2, x1:x2]
    sub_gate = gated_prob[y1:y2, x1:x2]
    r_max = sub_raw.max() * 100.0 if sub_raw.size > 0 else 0.0
    g_max = sub_gate.max() * 100.0 if sub_gate.size > 0 else 0.0
    g_mean = sub_gate[sub_gate > 0].mean() * 100.0 if (sub_gate > 0).any() else 0.0
    print(f"{name:<32} | {r_max:<10.2f} | {g_max:<12.2f} | {g_mean:<10.2f}")
