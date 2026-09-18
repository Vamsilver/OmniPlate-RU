#!/usr/bin/env python3
"""
Training Script for PlateVerifier (Binary Classifier: Plate vs Non-Plate).
Trains lightweight ConvNet on positive plate crops and hard background/vehicle negative crops.
Exports to models/plate_verifier.pt and models/plate_verifier.onnx (<300 KB).
"""

import argparse
import csv
import os
import random
import sys
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.verifier import PlateVerifierNet


class VerifierDataset(Dataset):
    def __init__(self, pos_crops: List[str], neg_crops: List[np.ndarray], is_train: bool = True):
        self.is_train = is_train
        # Balance pos and neg
        self.samples = []
        for p in pos_crops:
            self.samples.append((p, 1.0))
        for n in neg_crops:
            self.samples.append((n, 0.0))
        random.shuffle(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item, label = self.samples[idx]
        if isinstance(item, str):
            crop = cv2.imread(item)
            if crop is None:
                crop = np.zeros((36, 160, 3), dtype=np.uint8)
        else:
            crop = item.copy()

        if crop.shape[:2] != (36, 160):
            crop = cv2.resize(crop, (160, 36), interpolation=cv2.INTER_LINEAR)

        # Augmentations for train
        if self.is_train:
            if random.random() < 0.3:
                k = random.choice([3, 5])
                crop = cv2.GaussianBlur(crop, (k, k), 0)
            if random.random() < 0.35:
                alpha = random.uniform(0.7, 1.3)
                beta = random.uniform(-20, 20)
                crop = cv2.convertScaleAbs(crop, alpha=alpha, beta=beta)
            if random.random() < 0.25:
                noise = np.random.normal(0, random.uniform(3, 10), crop.shape).astype(np.float32)
                crop = np.clip(crop.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
        target = torch.tensor([label], dtype=torch.float32)
        return tensor, target


def collect_training_samples(root_dir: Path) -> Tuple[List[str], List[np.ndarray]]:
    """Collects positive plate filepaths and mines hard negative crops from non-plate images."""
    print("[*] Collecting positive plate crops...")
    manifest_path = root_dir / "dataset" / "verified_crops" / "manifest.csv"
    pos_paths = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")
        next(reader, None)
        for r in reader:
            if len(r) >= 3 and r[2] in ("type1", "type1a", "type1b", "type2"):
                p = root_dir / "dataset" / "verified_crops" / r[0]
                if p.exists():
                    pos_paths.append(str(p))
    print(f"[+] Loaded {len(pos_paths)} positive plate crops")

    print("[*] Mining negative non-plate background crops from class 'other'...")
    neg_crops = []
    meta_path = root_dir / "dataset" / "meta.csv"
    oth_images = []
    with open(meta_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for r in reader:
            if r.get("is_synthetic") == "0" and r.get("plate_type") == "other":
                # Only use true negatives without license plates
                if r.get("image", "").endswith(".jpg") and "real_other" in r["image"]:
                    num_val = int(r["image"].split("_")[-1].replace(".jpg", ""))
                    if num_val < 45 or num_val > 55:  # skip 46..54 which had real cars
                        oth_images.append(root_dir / "dataset" / r["image"])

    print(f"[+] Found {len(oth_images)} pure negative scene images.")
    for img_p in oth_images:
        if not img_p.exists():
            continue
        im = cv2.imread(str(img_p))
        if im is None:
            continue
        ih, iw = im.shape[:2]
        # Sample multiple crops with license plate aspect ratios (3.5:1 to 5.0:1 and 1.5:1 to 2.0:1)
        max_w = min(iw - 10, 400)
        if max_w < 30 or ih < 20:
            continue
        for _ in range(15):
            scale_w = random.randint(min(30, max_w), max_w)
            if random.random() < 0.6:
                scale_h = max(15, int(scale_w / random.uniform(3.5, 5.0)))
            else:
                scale_h = max(20, int(scale_w / random.uniform(1.4, 2.2)))
            if scale_h >= ih:
                scale_h = ih - 2
            x = random.randint(0, iw - scale_w)
            y = random.randint(0, ih - scale_h)
            crop = im[y:y + scale_h, x:x + scale_w]
            neg_crops.append(cv2.resize(crop, (160, 36), interpolation=cv2.INTER_LINEAR))

    # Add synthetic background noise patterns (pure gradients, random textures)
    for _ in range(1000):
        c = np.random.randint(40, 220, (36, 160, 3), dtype=np.uint8)
        neg_crops.append(c)

    print(f"[+] Total mined negative crops: {len(neg_crops)}")
    return pos_paths, neg_crops


def main():
    parser = argparse.ArgumentParser(description="Train PlateVerifier")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"[*] Training PlateVerifier on device: {device}")

    pos_paths, neg_crops = collect_training_samples(PROJECT_ROOT)

    # 90/10 train/val split
    random.seed(42)
    random.shuffle(pos_paths)
    random.shuffle(neg_crops)

    pos_val_n = max(50, int(len(pos_paths) * 0.10))
    neg_val_n = max(50, int(len(neg_crops) * 0.10))

    pos_train, pos_val = pos_paths[pos_val_n:], pos_paths[:pos_val_n]
    neg_train, neg_val = neg_crops[neg_val_n:], neg_crops[:neg_val_n]

    train_ds = VerifierDataset(pos_train, neg_train, is_train=True)
    val_ds = VerifierDataset(pos_val, neg_val, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = PlateVerifierNet(dropout=0.2).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    best_acc = 0.0
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    best_pt_path = models_dir / "plate_verifier.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for tensors, targets in train_loader:
            tensors, targets = tensors.to(device), targets.to(device)
            optimizer.zero_grad()
            preds = model(tensors)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for tensors, targets in val_loader:
                tensors, targets = tensors.to(device), targets.to(device)
                preds = model(tensors)
                probs = torch.sigmoid(preds)
                pred_binary = (probs >= 0.50).float()
                val_correct += (pred_binary == targets).sum().item()
                val_total += targets.size(0)

        val_acc = (val_correct / max(1, val_total)) * 100.0
        train_loss_avg = total_loss / max(1, len(train_loader))
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] Loss: {train_loss_avg:.4f} | Val Accuracy: {val_acc:.2f}%")

        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save({"state_dict": model.state_dict(), "best_acc": best_acc}, str(best_pt_path))

    print(f"[SUCCESS] Best validation accuracy: {best_acc:.2f}% saved to {best_pt_path}")

    # Export to ONNX
    onnx_path = models_dir / "plate_verifier.onnx"
    print(f"[*] Exporting to ONNX: {onnx_path}...")
    model.load_state_dict(torch.load(str(best_pt_path), map_location="cpu")["state_dict"])
    model.eval().cpu()
    dummy = torch.randn(1, 3, 36, 160)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    print(f"[SUCCESS] ONNX export complete: {onnx_path} ({os.path.getsize(onnx_path) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
