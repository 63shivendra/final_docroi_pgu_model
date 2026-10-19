"""
DocROI-Former: Unified Multi-Dataset PyTorch Loader
===================================================
Loads and prepares samples from:
  1. Real Indian KYC documents (Aadhaar, PAN, GST, Cheques).
  2. DocTamper LMDB databases (TrainingSet, TestingSet, SCD).
  3. MIDV-500 Authentic Clean National IDs.
  4. CASIA 2.0 & Splicing benchmarks.

Seamlessly computes automated entity priors on the fly (< 1 ms).

Design Pattern: PyTorch Dataset / Data Adapter (SOLID: Single Responsibility)
"""

from typing import List, Tuple, Dict, Any, Optional
import os
import glob
import cv2
try:
    import lmdb
except ImportError:
    lmdb = None
import numpy as np

# ImageNet-1k normalization constants (required by PVTv2-B2 pretrained backbone)
# Source: torchvision.transforms.Normalize documentation
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
import torch
from torch.utils.data import Dataset

from src.dataset.entity_tokenizer import OtsuEntityTokenizer
from src.dataset.counterfactual_engine import CounterfactualPairGenerator


class UnifiedDocForensicDataset(Dataset):
    """
    High-Performance Multi-Source Document Forensic Dataset.
    Supports both file-based image directories and high-speed LMDB databases.
    """

    def __init__(
        self,
        image_paths: Optional[List[str]] = None,
        mask_paths: Optional[List[str]] = None,
        lmdb_path: Optional[str] = None,
        target_size: Tuple[int, int] = (384, 384),
        is_clean_only: bool = False,
        enable_counterfactual: bool = True,
        counterfactual_ratio: float = 0.35
    ):
        super().__init__()
        self.target_size = target_size
        self.norm_factor = float(np.sqrt(target_size[0] ** 2 + target_size[1] ** 2))
        self.is_clean_only = is_clean_only
        self.enable_counterfactual = enable_counterfactual
        self.counterfactual_ratio = counterfactual_ratio

        self.tokenizer = OtsuEntityTokenizer()
        self.cf_generator = CounterfactualPairGenerator(target_size=target_size)

        self.lmdb_path = lmdb_path
        self.env = None
        self.lmdb_keys = []

        if lmdb_path and os.path.exists(lmdb_path):
            if lmdb is None:
                raise ImportError(
                    f"LMDB dataset detected at '{lmdb_path}', but the 'lmdb' python package is not installed. "
                    "Please install it with: pip install lmdb"
                )
            env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=True, meminit=False, max_readers=126)
            with env.begin() as txn:
                # Discover image keys (DocTamper stores image as 'image-000000000' and mask as 'label-000000000')
                cursor = txn.cursor()
                for key in cursor.iternext(keys=True, values=False):
                    k_str = key.decode("ascii", errors="ignore")
                    if k_str.startswith("image-") or (not k_str.endswith("_mask") and not k_str.startswith("label-") and k_str != "num-samples"):
                        self.lmdb_keys.append(key)
            env.close()
            self.is_lmdb = True
        else:
            self.is_lmdb = False
            self.image_paths = image_paths or []
            self.mask_paths = mask_paths or []

    def _get_env(self):
        if self.env is None and self.is_lmdb:
            self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=True, meminit=False, max_readers=126)
        return self.env

    def __del__(self):
        if hasattr(self, "env") and self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass
            self.env = None

    def __len__(self) -> int:
        if self.is_lmdb:
            return len(self.lmdb_keys)
        return len(self.image_paths)

    def _load_image(self, path: str) -> np.ndarray:
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((*self.target_size, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return cv2.resize(img, self.target_size)

    def _load_mask(self, path: Optional[str]) -> np.ndarray:
        if not path or not os.path.exists(path):
            return np.zeros(self.target_size, dtype=np.float32)
        m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if m is None:
            return np.zeros(self.target_size, dtype=np.float32)
        m_resized = cv2.resize(m, self.target_size, interpolation=cv2.INTER_NEAREST)
        return (m_resized > 0.5).astype(np.float32)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.is_lmdb:
            key = self.lmdb_keys[idx]
            env = self._get_env()
            with env.begin(buffers=True) as txn:
                img_buf = txn.get(key)
                # Check DocTamper format (label-XXXXX) or standard format (key_mask)
                mask_key = key.replace(b"image-", b"label-")
                mask_buf = txn.get(mask_key)
                if mask_buf is None:
                    mask_buf = txn.get(key + b"_mask")

            img_np = cv2.imdecode(np.frombuffer(img_buf, np.uint8), cv2.IMREAD_COLOR)
            img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            img_np = cv2.resize(img_np, self.target_size)

            if mask_buf:
                m_np = cv2.imdecode(np.frombuffer(mask_buf, np.uint8), cv2.IMREAD_GRAYSCALE)
                m_np = cv2.resize(m_np, self.target_size, interpolation=cv2.INTER_NEAREST)
                mask_np = (m_np > 0.5).astype(np.float32)
            else:
                mask_np = np.zeros(self.target_size, dtype=np.float32)

            is_tampered = float(mask_np.max() > 0.5)

        else:
            img_path = self.image_paths[idx]
            img_np = self._load_image(img_path)

            if self.is_clean_only:
                # Option to synthetically splice counterfactual positive
                if self.enable_counterfactual and np.random.rand() < self.counterfactual_ratio:
                    img_np, mask_np = self.cf_generator.generate_spliced_pair(img_np)
                    is_tampered = 1.0
                else:
                    mask_np = np.zeros(self.target_size, dtype=np.float32)
                    is_tampered = 0.0
            else:
                mask_path = self.mask_paths[idx] if idx < len(self.mask_paths) else None
                mask_np = self._load_mask(mask_path)
                is_tampered = float(mask_np.max() > 0.5)

        # Automated on-the-fly entity mask generation (< 1 ms)
        entity_prior_np = self.tokenizer.extract_entity_mask(img_np)

        # Convert to PyTorch tensors
        # Normalize to ImageNet distribution: PVTv2-B2 pretrained expects mean/std normalization
        img_normalized = (img_np.astype(np.float32) / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
        img_tensor = torch.from_numpy(img_normalized.transpose(2, 0, 1)).float()
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).float()
        entity_tensor = torch.from_numpy(entity_prior_np).unsqueeze(0).float()
        label_tensor = torch.tensor([is_tampered], dtype=torch.float32)

        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "label": label_tensor,
            "entity_prior": entity_tensor
        }
