import sys
import os
import cv2
import numpy as np
import torch

repo_dir = r"d:\shivendra pratap singh\docroi_former"
sys.path.insert(0, repo_dir)

from configs.docroi_config import DocROIConfig
from src.models.docroi_former import DocROIFormer

def main():
    img_path = r"E:\shivendra pratap singh\image.png"
    if not os.path.exists(img_path):
        img_path = r"D:\shivendra pratap singh\image.png"

    ckpt_path = os.path.join(repo_dir, "checkpoints", "best_docroi_former_v2_pgu.pth")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = DocROIConfig()
    model = DocROIFormer(config=cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
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
        prob_map_384 = torch.sigmoid(preds["mask_logits"])[0, 0].cpu().numpy()
        global_prob = torch.sigmoid(preds["global_logits"])[0, 0].item()

    prob_map_orig = cv2.resize(prob_map_384, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    # Find top 5 local maxima coordinates
    flat_indices = np.argsort(prob_map_orig.ravel())[::-1][:10]
    print(f"Global Fraud Probability: {global_prob*100:.2f}%")
    print(f"Overall Max Probability: {prob_map_orig.max()*100:.2f}%")
    print(f"Overall Mean Probability: {prob_map_orig.mean()*100:.2f}%")
    print(f"Overall 99th percentile: {np.percentile(prob_map_orig, 99)*100:.2f}%")

    print("\nTop 5 Peak Coordinates in Original Image (Width x Height = {} x {}):".format(orig_w, orig_h))
    for idx in flat_indices[:5]:
        y, x = np.unravel_index(idx, prob_map_orig.shape)
        val = prob_map_orig[y, x] * 100.0
        print(f"  Coord (X={x}, Y={y}) -> Score: {val:.2f}%")

if __name__ == "__main__":
    main()
