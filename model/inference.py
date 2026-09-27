"""
Inference wrapper used by the web app.

Two modes, chosen automatically:
  1. TRAINED MODEL  - if a checkpoint exists at CHECKPOINT_PATH, load the real
     U-Net and run proper CNN segmentation.
  2. HEURISTIC FALLBACK - if no checkpoint is present (e.g. you haven't trained
     yet, or you're just trying the demo), classify pixels using spectral
     index rules (vegetation greenness, water darkness/blueness, brightness
     for bare land / urban). This has no learning capability and is far less
     accurate than a trained CNN, but it means the app is fully functional
     out of the box, and it's a legitimate, documented baseline you can quote
     against your trained model's mIoU in a portfolio write-up.

Swap CHECKPOINT_PATH / call train.py to move from mode 2 to mode 1.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

try:
    import torch
    import torchvision.transforms.functional as TF
    from .unet import UNet, CLASS_NAMES, CLASS_COLORS
except ImportError:  # allow running standalone / relative import fallback
    import torch
    import torchvision.transforms.functional as TF
    from unet import UNet, CLASS_NAMES, CLASS_COLORS

CHECKPOINT_PATH = Path(__file__).parent.parent / "checkpoints" / "best.pt"

URBAN, FOREST, WATER, AGRICULTURE, BARE = 0, 1, 2, 3, 4


class LandCoverClassifier:
    def __init__(self, checkpoint_path: Optional[str] = None, device: Optional[str] = None, img_size: int = 256):
        self.img_size = img_size
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = Path(checkpoint_path) if checkpoint_path else CHECKPOINT_PATH
        self.model = None
        if ckpt.exists():
            self.model = UNet(in_channels=3, num_classes=5).to(self.device)
            self.model.load_state_dict(torch.load(ckpt, map_location=self.device))
            self.model.eval()
            self.mode = "cnn"
        else:
            self.mode = "heuristic"

    # ------------------------------------------------------------------ #
    def classify(self, image: Image.Image, engine: Optional[str] = None) -> np.ndarray:
        """Returns an (H, W) int array of class ids 0-4, at the original image size.
        `engine` optionally overrides the auto-detected mode ("cnn" or "heuristic"),
        e.g. to run both engines on the same image side-by-side for comparison."""
        orig_size = image.size  # (W, H)
        use_cnn = (engine == "cnn") or (engine is None and self.mode == "cnn")
        if use_cnn and self.model is not None:
            mask = self._classify_cnn_tiled(image)
        else:
            mask = self._classify_heuristic(image)
        # resize class map back to original resolution (nearest neighbor to keep labels valid)
        if mask.shape[::-1] != orig_size:
            mask_img = Image.fromarray(mask.astype(np.uint8)).resize(orig_size, Image.NEAREST)
            mask = np.array(mask_img)
        return mask

    # ------------------------------------------------------------------ #
    def _classify_cnn_single_tile(self, tile: Image.Image) -> np.ndarray:
        """Run the U-Net on one img_size x img_size tile."""
        t = TF.to_tensor(tile)
        t = TF.normalize(t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(t)
            return logits.argmax(dim=1).squeeze(0).cpu().numpy()

    def _classify_cnn_tiled(self, image: Image.Image) -> np.ndarray:
        """Sliding-window inference at native resolution instead of squashing the
        whole scene down to one 256x256 tile. A single resize loses most of the
        spatial detail on anything bigger than the model's input size, which is
        the norm for real satellite tiles (often 1000-10000+ px per side) — this
        tiles the image into img_size x img_size windows (with a small overlap
        to avoid harsh seams at tile borders, resolved by majority vote in the
        overlap region), runs the CNN on each, and stitches the full-resolution
        mask back together."""
        img = image.convert("RGB")
        w, h = img.size
        s = self.img_size
        if w <= s and h <= s:
            tile = img.resize((s, s), Image.BILINEAR)
            pred = self._classify_cnn_single_tile(tile)
            return np.array(Image.fromarray(pred.astype(np.uint8)).resize((w, h), Image.NEAREST))

        overlap = s // 8
        stride = s - overlap
        vote_counts = np.zeros((h, w, 5), dtype=np.uint16)

        y = 0
        while y < h:
            y0 = min(y, max(h - s, 0))
            x = 0
            while x < w:
                x0 = min(x, max(w - s, 0))
                tile = img.crop((x0, y0, x0 + s, y0 + s))
                if tile.size != (s, s):
                    tile = tile.resize((s, s), Image.BILINEAR)
                pred = self._classify_cnn_single_tile(tile)
                for c in range(5):
                    vote_counts[y0:y0 + s, x0:x0 + s, c] += (pred == c)
                if x0 + s >= w:
                    break
                x += stride
            if y0 + s >= h:
                break
            y += stride

        return vote_counts.argmax(axis=-1).astype(np.uint8)

    # ------------------------------------------------------------------ #
    def _classify_heuristic(self, image: Image.Image) -> np.ndarray:
        """Rule-based baseline using RGB spectral cues (no NIR available):
        - ExG (excess green) index  -> vegetation (forest vs agriculture split
          by greenness intensity + texture-free heuristic: solid saturated
          green = forest, lighter/yellow-green = agriculture)
        - Blue dominance + low brightness -> water
        - High brightness + low saturation -> bare land / urban split by hue
        """
        arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        maxc = arr.max(axis=-1)
        minc = arr.min(axis=-1)
        brightness = arr.mean(axis=-1)
        saturation = np.where(maxc > 0, (maxc - minc) / np.clip(maxc, 1e-6, None), 0)

        exg = 2 * g - r - b  # excess green vegetation index (RGB-only proxy for NDVI)
        water_score = (b - np.maximum(r, g)) + (0.5 - brightness) * 0.5  # dark + blue-dominant

        classes = np.full(arr.shape[:2], BARE, dtype=np.uint8)

        is_water = (b > r) & (b > g) & (brightness < 0.55)
        is_veg = exg > 0.05
        is_forest = is_veg & (g < 0.55) & (saturation > 0.25)
        is_agriculture = is_veg & ~is_forest
        is_urban = (~is_water) & (~is_veg) & (saturation < 0.18) & (brightness > 0.35) & (brightness < 0.85)
        is_bare = (~is_water) & (~is_veg) & (~is_urban)

        classes[is_bare] = BARE
        classes[is_urban] = URBAN
        classes[is_agriculture] = AGRICULTURE
        classes[is_forest] = FOREST
        classes[is_water] = WATER
        return classes


def stats_from_mask(mask: np.ndarray, gsd_m: Optional[float] = None) -> dict:
    """Percent cover per class + pixel counts. If gsd_m (ground sample distance,
    meters/pixel — check your imagery source's metadata) is given, also reports
    real-world area in hectares and km², not just relative percentage."""
    total = mask.size
    result = {}
    px_area_m2 = (gsd_m ** 2) if gsd_m else None
    for cid, name in enumerate(CLASS_NAMES):
        count = int((mask == cid).sum())
        entry = {
            "pixels": count,
            "percent": round(100.0 * count / total, 2) if total else 0.0,
        }
        if px_area_m2:
            area_m2 = count * px_area_m2
            entry["hectares"] = round(area_m2 / 10_000, 3)
            entry["km2"] = round(area_m2 / 1_000_000, 5)
        result[name] = entry
    return result


def colorize_mask(mask: np.ndarray) -> Image.Image:
    """Turn a class-id mask into an RGB image using CLASS_COLORS."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cid, color in CLASS_COLORS.items():
        rgb[mask == cid] = color
    return Image.fromarray(rgb)


def vegetation_index_map(image: Image.Image) -> Image.Image:
    """Proxy-NDVI heatmap from RGB alone (2*G - R - B, i.e. Excess Green),
    rendered on a brown -> yellow -> green ramp (low -> high vegetation vigor).
    With a real NIR band, swap this for true NDVI = (NIR-R)/(NIR+R)."""
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    exg = 2 * g - r - b
    norm = np.clip((exg + 1) / 2, 0, 1)  # roughly -1..1 -> 0..1

    stops = np.array([[0.55, 0.35, 0.15], [0.85, 0.75, 0.35], [0.15, 0.55, 0.15]])  # brown->yellow->green
    idx = norm * (len(stops) - 1)
    lo = np.clip(idx.astype(int), 0, len(stops) - 2)
    frac = (idx - lo)[..., None]
    rgb = stops[lo] * (1 - frac) + stops[lo + 1] * frac
    return Image.fromarray((rgb * 255).astype(np.uint8))


def confidence_map(mask: np.ndarray, classifier: "LandCoverClassifier", image: Image.Image) -> Image.Image:
    """Per-pixel confidence, grayscale (white = confident, dark = uncertain).
    CNN mode: 1 - normalized softmax entropy. Heuristic mode: distance from
    whichever decision boundary (vegetation / water / urban-vs-bare /
    brightness) the pixel sits closest to, i.e. its weakest rule margin,
    then stretched to the full 0-255 range so the map isn't uniformly dim."""
    if classifier.mode == "cnn":
        img = image.convert("RGB").resize((classifier.img_size, classifier.img_size), Image.BILINEAR)
        t = TF.to_tensor(img)
        t = TF.normalize(t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).unsqueeze(0).to(classifier.device)
        with torch.no_grad():
            probs = torch.softmax(classifier.model(t), dim=1).squeeze(0).cpu().numpy()
        eps = 1e-8
        entropy = -(probs * np.log(probs + eps)).sum(axis=0)
        conf = 1 - entropy / np.log(probs.shape[0])
        conf_img = Image.fromarray((conf * 255).astype(np.uint8)).resize(image.size, Image.BILINEAR)
    else:
        arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        maxc, minc = arr.max(-1), arr.min(-1)
        brightness = arr.mean(-1)
        saturation = np.where(maxc > 0, (maxc - minc) / np.clip(maxc, 1e-6, None), 0)

        exg = 2 * g - r - b
        veg_margin = np.abs(exg - 0.05)                     # distance from the vegetation threshold
        water_margin = np.abs(b - np.maximum(r, g))          # distance from the water rule
        urban_margin = np.abs(saturation - 0.18)             # distance from the urban/bare saturation threshold
        bright_margin = np.minimum(np.abs(brightness - 0.35), np.abs(brightness - 0.85))

        # the pixel's confidence is bounded by whichever boundary it sits closest to
        margin = np.minimum(np.minimum(veg_margin, water_margin),
                             np.minimum(urban_margin, bright_margin))

        # stretch per-image to the full 0-255 range so relative confidence is visible
        lo, hi = float(margin.min()), float(margin.max())
        norm = (margin - lo) / max(hi - lo, 1e-6)
        conf_img = Image.fromarray((norm * 255).astype(np.uint8))
    return conf_img.convert("L")


def landscape_metrics(mask: np.ndarray) -> dict:
    """Landscape-ecology style fragmentation metrics per class (FRAGSTATS-lite):
    patch count, mean patch size, largest patch's share of that class's area,
    and edge density (boundary pixels / total pixels) — indicators of habitat
    fragmentation and urban sprawl pattern, not just raw area."""
    from scipy import ndimage

    h, w = mask.shape
    result = {}
    # edges: pixels whose 4-neighborhood contains a different class
    diffs = np.zeros_like(mask, dtype=bool)
    diffs[:, :-1] |= mask[:, :-1] != mask[:, 1:]
    diffs[:-1, :] |= mask[:-1, :] != mask[1:, :]

    for cid, name in enumerate(CLASS_NAMES):
        class_mask = mask == cid
        area = int(class_mask.sum())
        if area == 0:
            result[name] = {"patch_count": 0, "mean_patch_size_px": 0, "largest_patch_percent": 0.0, "edge_density": 0.0}
            continue
        labeled, n_patches = ndimage.label(class_mask)
        sizes = ndimage.sum(class_mask, labeled, index=range(1, n_patches + 1)) if n_patches else np.array([])
        largest_pct = round(100.0 * sizes.max() / area, 2) if len(sizes) else 0.0
        edge_px = int((diffs & class_mask).sum())
        result[name] = {
            "patch_count": int(n_patches),
            "mean_patch_size_px": round(float(sizes.mean()), 1) if len(sizes) else 0.0,
            "largest_patch_percent": largest_pct,
            "edge_density": round(edge_px / area, 4),
        }
    return result


def change_detection(mask_before: np.ndarray, mask_after: np.ndarray, gsd_m: Optional[float] = None) -> dict:
    """Per-class change in % cover + a transition matrix (what turned into what),
    plus two summary indices that are the standard headline numbers in a
    land-cover-change study:
      - Urban Expansion Rate: % of the scene that was non-urban and became urban
      - Vegetation Loss Rate: % of the scene that was Forest/Agriculture and became something else
    """
    stats_before = stats_from_mask(mask_before, gsd_m)
    stats_after = stats_from_mask(mask_after, gsd_m)

    change = {}
    for name in CLASS_NAMES:
        before_pct = stats_before[name]["percent"]
        after_pct = stats_after[name]["percent"]
        change[name] = {
            "before_percent": before_pct,
            "after_percent": after_pct,
            "delta_percent": round(after_pct - before_pct, 2),
        }
        if gsd_m:
            change[name]["before_km2"] = stats_before[name]["km2"]
            change[name]["after_km2"] = stats_after[name]["km2"]

    transition = None
    urban_expansion_rate = veg_loss_rate = None
    if mask_before.shape == mask_after.shape:
        n = len(CLASS_NAMES)
        matrix = np.zeros((n, n), dtype=np.int64)
        for i in range(n):
            for j in range(n):
                matrix[i, j] = int(((mask_before == i) & (mask_after == j)).sum())
        transition = matrix.tolist()
        total = mask_before.size
        newly_urban = int(((mask_before != URBAN) & (mask_after == URBAN)).sum())
        was_vegetated = (mask_before == FOREST) | (mask_before == AGRICULTURE)
        no_longer_vegetated = int((was_vegetated & (mask_after != FOREST) & (mask_after != AGRICULTURE)).sum())
        urban_expansion_rate = round(100.0 * newly_urban / total, 2)
        veg_loss_rate = round(100.0 * no_longer_vegetated / max(int(was_vegetated.sum()), 1), 2)

    return {
        "per_class_change": change,
        "transition_matrix": transition,
        "class_names": CLASS_NAMES,
        "summary": {
            "urban_expansion_rate_percent": urban_expansion_rate,
            "vegetation_loss_rate_percent": veg_loss_rate,
        },
    }