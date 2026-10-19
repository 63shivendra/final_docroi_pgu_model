"""
DocROI-Former: Counterfactual Minimal-Pair Generator
===================================================
Generates mathematically exact positive training pairs from authentic clean
document templates (Aadhaar, PAN, GST) via surgical character and field splicing.

Produces:
  - Clean Original: Image A (Y=0, Ground truth mask = all zeros).
  - Spliced Pair: Image B (Y=1, Exact 100% ground truth mask of altered pixels).

Design Pattern: Data Augmentor / Synthetic Factory (SOLID: Single Responsibility)
"""

from typing import Tuple
import random
import cv2
import numpy as np


class CounterfactualPairGenerator:
    """
    Surgically creates realistic minimal-pair forgeries on authentic cards.
    """

    def __init__(self, target_size: Tuple[int, int] = (384, 384)):
        self.target_size = target_size

    def generate_spliced_pair(self, clean_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Takes a clean document image [H, W, 3] and surgically creates a tampered counterfactual.
        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - tampered_img: [H, W, 3] uint8
                - ground_truth_mask: [H, W] float32 in {0.0, 1.0}
        """
        tampered = clean_img.copy()
        h, w = tampered.shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)

        # Decide forgery mode:
        # Mode 1: Small Digit / Character swap (Single character splice)
        # Mode 2: Text word / DOB line replacement
        # Mode 3: Photo box edge splicing
        mode = random.choice(["single_digit", "word_splice", "photo_splice"])

        if mode == "single_digit":
            # Realistic single digit: 12x18 to 20x25 pixels
            box_h = random.randint(12, 24)
            box_w = random.randint(10, 20)
            # Pick a text-rich zone (typically center or lower third of an ID card)
            y1 = random.randint(int(h * 0.3), int(h * 0.75) - box_h)
            x1 = random.randint(int(w * 0.25), int(w * 0.75) - box_w)
            y2, x2 = y1 + box_h, x1 + box_w

            # Copy another random patch to simulate character splicing
            src_y1 = random.randint(int(h * 0.3), int(h * 0.75) - box_h)
            src_x1 = random.randint(int(w * 0.25), int(w * 0.75) - box_w)
            patch = clean_img[src_y1:src_y1+box_h, src_x1:src_x1+box_w]

            # Slight contrast/font weight difference
            alpha = random.uniform(0.85, 1.25)
            patch_altered = np.clip(patch.astype(np.float32) * alpha, 0, 255).astype(np.uint8)

            tampered[y1:y2, x1:x2] = patch_altered
            mask[y1:y2, x1:x2] = 1.0

        elif mode == "word_splice":
            # Word-level replacement: 20x80 to 30x140 pixels
            box_h = random.randint(16, 28)
            box_w = random.randint(50, 120)
            y1 = random.randint(int(h * 0.25), int(h * 0.8) - box_h)
            x1 = random.randint(int(w * 0.15), int(w * 0.8) - box_w)
            y2, x2 = y1 + box_h, x1 + box_w

            # Inpaint / replace zone with synthetic digital text
            patch = clean_img[y1:y2, x1:x2].copy()
            # Apply slight Gaussian blur to simulate Canva text smoothing
            patch = cv2.GaussianBlur(patch, (3, 3), 0)
            tampered[y1:y2, x1:x2] = patch
            mask[y1:y2, x1:x2] = 1.0

        else: # photo_splice
            # Small edge splice on photo corner: 30x30 to 50x50
            box_h = random.randint(30, 60)
            box_w = random.randint(30, 60)
            y1 = random.randint(int(h * 0.3), int(h * 0.6) - box_h)
            x1 = random.randint(int(w * 0.05), int(w * 0.35) - box_w)
            y2, x2 = y1 + box_h, x1 + box_w

            patch = clean_img[y1:y2, x1:x2]
            # Color jitter to simulate different camera sensor
            patch_altered = cv2.convertScaleAbs(patch, alpha=1.1, beta=10)
            tampered[y1:y2, x1:x2] = patch_altered
            mask[y1:y2, x1:x2] = 1.0

        return tampered, mask
