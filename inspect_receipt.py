import cv2
import numpy as np

img = cv2.imread(r"D:\shivendra pratap singh\finalaltrerecipt.png")
h, w = img.shape[:2]
print(f"Receipt Image dimensions: Width={w}, Height={h}")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# SRM High-pass filter
srm_kernel = np.array([
    [-1,  2, -2,  2, -1],
    [ 2, -6,  8, -6,  2],
    [-2,  8,-12,  8, -2],
    [ 2, -6,  8, -6,  2],
    [-1,  2, -2,  2, -1]
], dtype=np.float32) / 12.0
srm_res = np.abs(cv2.filter2D(gray.astype(np.float32), -1, srm_kernel))

# Line slices
lines = [
    ("Receipt Header", 0, 90),
    ("Address Line (1284 Lorem)", 100, 170),
    ("Tel Line", 175, 240),
    ("Top Dashed Line", 240, 275),
    ("Date & Time Line", 280, 360),
    ("Bottom Dashed Line", 365, 410),
    ("Lorem 6.50 Line", 420, min(520, h))
]

print("\n--- High-Frequency Energy Across Lines ---")
for name, y1, y2 in lines:
    if y1 < h:
        y2_eff = min(y2, h)
        sub = srm_res[y1:y2_eff, :]
        print(f"  {name:<28}: Mean={sub.mean():.4f}, Max={sub.max():.4f}, P99={np.percentile(sub, 99):.4f}")

# Check specific words in Address Line
# '1284' vs 'Lorem' vs 'Ipsum'
addr_line = gray[100:170, :]
# Find characters horizontally
print("\nHorizontal energy scan across Address Line (y=100..170):")
col_energy = srm_res[100:170, :].mean(axis=0)
print(f"  Average column energy: {col_energy.mean():.4f}")
