"""
Many public land-cover datasets ship masks as RGB images with one solid
color per class rather than single-channel class-index images. This script
converts a folder of such RGB label images into the single-channel index
masks that dataset.py expects (pixel value = class id 0-4).

Edit SOURCE_COLOR_MAP below to match the dataset's actual palette, then run:

    python model/prepare_masks.py --src data/train/labels_rgb --dst data/train/masks
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

# Map each class id -> the RGB color used in the *source* dataset's label images.
# CHANGE THIS to match your dataset's documentation before running.
SOURCE_COLOR_MAP = {
    0: (230, 25, 75),    # Urban
    1: (60, 180, 75),    # Forest
    2: (0, 130, 200),    # Water
    3: (255, 225, 25),   # Agriculture
    4: (170, 110, 40),   # Bare land
}


def rgb_to_index_mask(rgb_img: Image.Image) -> np.ndarray:
    arr = np.array(rgb_img.convert("RGB"))
    out = np.zeros(arr.shape[:2], dtype=np.uint8)
    for cid, color in SOURCE_COLOR_MAP.items():
        matches = np.all(arr == np.array(color), axis=-1)
        out[matches] = cid
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="Folder of RGB label images")
    p.add_argument("--dst", required=True, help="Output folder for index masks")
    args = p.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    files = sorted([f for f in src.iterdir() if f.suffix.lower() in (".png", ".jpg", ".jpeg")])
    print(f"Converting {len(files)} label images...")
    for f in files:
        idx_mask = rgb_to_index_mask(Image.open(f))
        Image.fromarray(idx_mask).save(dst / f.name)
    print("Done.")


if __name__ == "__main__":
    main()
