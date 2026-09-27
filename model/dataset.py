"""
Dataset loader for land-cover segmentation.

Expected folder layout (works out of the box with public datasets such as
LandCover.ai, DeepGlobe Land Cover Classification, or your own labeled tiles):

    data/
      train/
        images/  0001.png  0002.png ...
        masks/   0001.png  0002.png ...   (single-channel, pixel value = class id 0-4)
      val/
        images/ ...
        masks/  ...

If you don't have masks yet, see `prepare_masks.py` for turning RGB label
images (one solid color per class) into class-index masks automatically
using the CLASS_COLORS palette in model/unet.py.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import random


class LandCoverDataset(Dataset):
    def __init__(self, root: str, split: str = "train", img_size: int = 256, augment: bool = True):
        self.img_dir = Path(root) / split / "images"
        self.mask_dir = Path(root) / split / "masks"
        if not self.img_dir.exists():
            raise FileNotFoundError(f"Missing image dir: {self.img_dir}")
        self.files = sorted([f.name for f in self.img_dir.iterdir() if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff")])
        self.img_size = img_size
        self.augment = augment and split == "train"

    def __len__(self):
        return len(self.files)

    def _load_pair(self, fname: str):
        img = Image.open(self.img_dir / fname).convert("RGB")
        mask_path = self.mask_dir / fname
        if not mask_path.exists():
            # try same stem with .png
            mask_path = self.mask_dir / (Path(fname).stem + ".png")
        mask = Image.open(mask_path)
        return img, mask

    def __getitem__(self, idx: int):
        fname = self.files[idx]
        img, mask = self._load_pair(fname)

        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)

        if self.augment:
            if random.random() > 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)
            if random.random() > 0.5:
                img = TF.vflip(img)
                mask = TF.vflip(mask)
            angle = random.choice([0, 90, 180, 270])
            if angle:
                img = TF.rotate(img, angle)
                mask = TF.rotate(mask, angle)
            if random.random() > 0.5:
                factor = 0.8 + random.random() * 0.4  # 0.8 - 1.2
                img = TF.adjust_brightness(img, factor)

        img_t = TF.to_tensor(img)  # (3,H,W) in [0,1]
        img_t = TF.normalize(img_t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        mask_arr = np.array(mask, dtype=np.int64)
        if mask_arr.ndim == 3:
            mask_arr = mask_arr[..., 0]
        mask_t = torch.from_numpy(mask_arr)

        return img_t, mask_t


def compute_class_weights(dataset: LandCoverDataset, num_classes: int = 5) -> torch.Tensor:
    """Inverse-frequency class weights, useful for imbalanced land-cover classes
    (e.g. water/urban tiles are often much rarer than forest/agriculture)."""
    counts = np.zeros(num_classes, dtype=np.float64)
    for i in range(len(dataset)):
        _, mask = dataset[i]
        for c in range(num_classes):
            counts[c] += (mask == c).sum().item()
    counts = np.clip(counts, 1, None)
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)
