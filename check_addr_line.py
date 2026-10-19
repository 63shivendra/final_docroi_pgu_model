import cv2
import numpy as np

img = cv2.imread(r"D:\shivendra pratap singh\finalaltrerecipt.png")
h, w = img.shape[:2]
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Check characters in Address Line (y: 110..165)
addr = gray[110:165, :]
# Find bounding boxes of text characters in this line
ret, thresh = cv2.threshold(addr, 200, 255, cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

boxes = []
for c in contours:
    x, y, bw, bh = cv2.boundingRect(c)
    if bw > 3 and bh > 10:
        boxes.append((x, bw, bh))

boxes.sort(key=lambda b: b[0])
print(f"Detected {len(boxes)} character glyphs on Address line:")
for b in boxes:
    print(f"  X={b[0]:<4} W={b[1]:<3} H={b[2]:<3}")
