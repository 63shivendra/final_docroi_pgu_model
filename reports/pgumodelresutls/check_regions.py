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

def inspect(img_path):
    if not os.path.exists(img_path):
        print(f"File not found: {img_path}")
        return

    print("=" * 70)
    print(f"Inspecting File: {img_path}")
    print("=" * 70)

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

    # Specific regions on the card:
    # 1. Photo region: x: 10..110, y: 70..210
    photo_prob = prob_orig[70:210, 10:110].max() * 100.0

    # 2. PAN Number region ('IFAPD0845E'): x: 140..280, y: 120..160
    pan_slice = prob_orig[120:165, 140:280]
    pan_max = pan_slice.max() * 100.0
    pan_mean = pan_slice.mean() * 100.0

    # 3. Specifically the letter 'E' (last char of PAN): x: 250..280, y: 125..155
    e_slice = prob_orig[125:155, 250:280]
    e_max = e_slice.max() * 100.0 if e_slice.size > 0 else 0.0

    # 4. Name region ('Wandhekar Krushna Dadasaheb'): x: 10..250, y: 210..250
    name_slice = prob_orig[210:250, 10:250]
    name_max = name_slice.max() * 100.0

    # 5. QR code region: x: 280..420, y: 150..300
    qr_slice = prob_orig[150:300, 280:420]
    qr_max = qr_slice.max() * 100.0

    print(f"Global Document Fraud Probability: {global_prob*100:.2f}%")
    print(f"Overall Image Max:                  {prob_orig.max()*100:.2f}%")
    print(f"\nRegional Max Breakdown:")
    print(f"  - Spliced Photo Box (x:10..110):  {photo_prob:.2f}%  <-- MASSIVE SPLICED PHOTO")
    print(f"  - PAN Text Field ('IFAPD0845E'):  {pan_max:.2f}%")
    print(f"  - Last Letter ('E' on 'F'):       {e_max:.2f}%")
    print(f"  - Name Field:                     {name_max:.2f}%")
    print(f"  - QR Code:                        {qr_max:.2f}%")

if __name__ == "__main__":
    inspect(r"D:\shivendra pratap singh\fulll.png")
    if os.path.exists(r"E:\shivendra pratap singh\image.png"):
        inspect(r"E:\shivendra pratap singh\image.png")
