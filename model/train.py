"""
Train the U-Net land-cover segmentation model.

Usage:
    python model/train.py --data-root ./data --epochs 50 --batch-size 16 --lr 1e-4

Datasets you can point --data-root at (after converting to the folder layout
described in dataset.py):
    - LandCover.ai            https://landcover.ai.linuxpolska.com/
    - DeepGlobe Land Cover    https://www.kaggle.com/datasets/balraj98/deepglobe-land-cover-classification-dataset
    - EuroSAT (patch-level classification variant, needs a small head-only change)
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from unet import UNet, CLASS_NAMES
from dataset import LandCoverDataset, compute_class_weights


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, num_classes: int, eps: float = 1e-6) -> torch.Tensor:
    probs = F.softmax(logits, dim=1)
    targets_1h = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    intersection = torch.sum(probs * targets_1h, dims)
    cardinality = torch.sum(probs + targets_1h, dims)
    dice = (2.0 * intersection + eps) / (cardinality + eps)
    return 1.0 - dice.mean()


def mean_iou(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    ious = []
    for c in range(num_classes):
        pred_c = preds == c
        target_c = targets == c
        intersection = (pred_c & target_c).sum().item()
        union = (pred_c | target_c).sum().item()
        if union == 0:
            continue
        ious.append(intersection / union)
    return sum(ious) / len(ious) if ious else 0.0


def run_epoch(model, loader, optimizer, device, num_classes, class_weights, train: bool):
    model.train(mode=train)
    total_loss, total_iou, n_batches = 0.0, 0.0, 0
    ce = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)

    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        with torch.set_grad_enabled(train):
            logits = model(imgs)
            loss = ce(logits, masks) + dice_loss(logits, masks, num_classes)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        preds = logits.argmax(dim=1)
        total_iou += mean_iou(preds, masks, num_classes)
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1), total_iou / max(n_batches, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--img-size", type=int, default=256)
    p.add_argument("--num-classes", type=int, default=5)
    p.add_argument("--in-channels", type=int, default=3)
    p.add_argument("--out-dir", type=str, default="./checkpoints")
    p.add_argument("--resume", type=str, default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.out_dir, exist_ok=True)

    train_ds = LandCoverDataset(args.data_root, "train", args.img_size, augment=True)
    val_ds = LandCoverDataset(args.data_root, "val", args.img_size, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print("Computing class weights (handles land-cover class imbalance)...")
    class_weights = compute_class_weights(train_ds, args.num_classes)
    print("Class weights:", {CLASS_NAMES[i]: round(w, 3) for i, w in enumerate(class_weights.tolist())})

    model = UNet(in_channels=args.in_channels, num_classes=args.num_classes).to(device)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"Resumed from {args.resume}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_iou = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_iou = run_epoch(model, train_loader, optimizer, device, args.num_classes, class_weights, train=True)
        val_loss, val_iou = run_epoch(model, val_loader, optimizer, device, args.num_classes, class_weights, train=False)
        scheduler.step()

        print(f"Epoch {epoch:03d}/{args.epochs} | "
              f"train_loss={train_loss:.4f} train_mIoU={train_iou:.4f} | "
              f"val_loss={val_loss:.4f} val_mIoU={val_iou:.4f}")

        torch.save(model.state_dict(), Path(args.out_dir) / "last.pt")
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), Path(args.out_dir) / "best.pt")
            print(f"  -> new best model saved (val_mIoU={best_iou:.4f})")

    print(f"Training complete. Best val mIoU: {best_iou:.4f}")


if __name__ == "__main__":
    main()
