"""
Evaluate a trained checkpoint on a held-out set: confusion matrix, per-class
precision/recall/F1/IoU, overall accuracy and mean IoU. This is the report
you actually want in a portfolio write-up or a paper — training-loop mIoU
alone doesn't tell you *which* classes the model confuses.

Usage:
    python model/evaluate.py --data-root ./data --split val --checkpoint checkpoints/best.pt
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from unet import UNet, CLASS_NAMES
from dataset import LandCoverDataset


def confusion_matrix(preds: np.ndarray, targets: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    p, t = preds.flatten(), targets.flatten()
    for i in range(num_classes):
        for j in range(num_classes):
            cm[i, j] = int(((t == i) & (p == j)).sum())
    return cm


def metrics_from_confusion(cm: np.ndarray) -> dict:
    n = cm.shape[0]
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    precision = np.divide(tp, tp + fp, out=np.zeros(n), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros(n), where=(tp + fn) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(n), where=(precision + recall) > 0)
    iou = np.divide(tp, tp + fp + fn, out=np.zeros(n), where=(tp + fp + fn) > 0)
    overall_acc = tp.sum() / max(cm.sum(), 1)

    per_class = {
        CLASS_NAMES[i]: {
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1": round(float(f1[i]), 4),
            "iou": round(float(iou[i]), 4),
            "support_px": int(cm[i].sum()),
        }
        for i in range(n)
    }
    return {
        "overall_accuracy": round(float(overall_acc), 4),
        "mean_iou": round(float(iou.mean()), 4),
        "mean_f1": round(float(f1.mean()), 4),
        "per_class": per_class,
    }


def save_confusion_heatmap(cm: np.ndarray, out_path: str):
    """Saved as a plain-text-annotated PNG using matplotlib if available;
    silently skipped if matplotlib isn't installed (metrics.json is enough)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping heatmap image (metrics.json still written).")
        return
    cm_norm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm, cmap="YlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASS_NAMES))); ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASS_NAMES))); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Ground truth")
    ax.set_title("Confusion matrix (row-normalized)")
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                     color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved confusion matrix heatmap to {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--split", type=str, default="val")
    p.add_argument("--checkpoint", type=str, default="./checkpoints/best.pt")
    p.add_argument("--img-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--out", type=str, default="./checkpoints/eval_report.json")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=3, num_classes=5).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    ds = LandCoverDataset(args.data_root, args.split, args.img_size, augment=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    cm = np.zeros((5, 5), dtype=np.int64)
    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device)
            preds = model(imgs).argmax(dim=1).cpu().numpy()
            cm += confusion_matrix(preds, masks.numpy(), num_classes=5)

    report = metrics_from_confusion(cm)
    report["confusion_matrix"] = cm.tolist()
    report["class_names"] = CLASS_NAMES

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "confusion_matrix"}, indent=2))
    print(f"\nFull report saved to {args.out}")

    save_confusion_heatmap(cm, str(Path(args.out).with_suffix(".png")))


if __name__ == "__main__":
    main()
