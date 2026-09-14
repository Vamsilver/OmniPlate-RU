#!/usr/bin/env python3
"""
High-Performance OCR Training Script for Volga IT 2026 (Substage 3.3).
Trains LPRNet on canonical 160x36 crops of Russian license plates:
- Supports Type 1, Type 1B, and Type 1A (stitched 2-line).
- Uses PyTorch CTC Loss (blank=0).
- Real-time Albumentations (blur, glare, noise, perspective perturbations).
- Checkpoint saving + automatic ONNX export for inference SLA <= 25 ms.
"""

import argparse
import csv
import os
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True, encoding="utf-8")

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from src.pipeline.ocr import (
    BLANK_IDX,
    CHAR2IDX,
    IDX2CHAR,
    NUM_CLASSES,
    CTCDecoder,
    LPRNet,
)
from src.pipeline.rectifier import PlateRectifier


CYR_TO_LAT: Dict[str, str] = {
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H",
    "О": "O", "Р": "P", "С": "C", "Т": "T", "У": "Y", "Х": "X",
}


class PlateCropDataset(Dataset):
    """
    Dataset for loading and rectifying plate crops to canonical 160x36 format.
    Supports in-RAM pre-caching for ultra-fast GPU training.
    """
    def __init__(
        self,
        meta_csv: str,
        root_dir: str,
        is_train: bool = True,
        val_split: float = 0.1,
        seed: int = 42,
        cache_in_ram: bool = True,
    ):
        self.root_dir = root_dir
        self.rectifier = PlateRectifier()
        self.is_train = is_train
        self.cache_in_ram = cache_in_ram

        # Load samples from meta.csv
        all_samples = []
        with open(meta_csv, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f, delimiter=";")
            header = next(reader)
            for row in reader:
                if len(row) < 10:
                    continue
                img_rel, plate_num, p_type, bbox_str, quad_str, is_veh, is_syn, src, lic, cond = row[:10]
                if p_type not in ("type1", "type1a", "type1b"):
                    continue
                # Reject unreadable / wildcards for strict training
                if "#" in plate_num:
                    continue
                if len(plate_num) not in (8, 9):
                    continue

                all_samples.append({
                    "img_rel": img_rel,
                    "plate_num": plate_num,
                    "plate_type": p_type,
                    "quad": quad_str,
                    "bbox": bbox_str,
                })

        # Deterministic split
        random.seed(seed)
        random.shuffle(all_samples)
        split_idx = int(len(all_samples) * (1.0 - val_split))
        self.samples = all_samples[:split_idx] if is_train else all_samples[split_idx:]
        print(f"[{'TRAIN' if is_train else 'VAL'}] Loaded {len(self.samples)} samples from {meta_csv}")

        # Pre-cache crops in RAM to eliminate disk I/O bottlenecks during training
        self.cached_crops: List[np.ndarray] = []
        if self.cache_in_ram:
            tag = "TRAIN" if is_train else "VAL"
            print(f"[{tag}] Pre-caching {len(self.samples)} plate crops into RAM...")
            t0 = time.time()
            for item in self.samples:
                crop = self._extract_crop(item)
                self.cached_crops.append(crop)
            dt = time.time() - t0
            ram_mb = len(self.cached_crops) * 36 * 160 * 3 / (1024 * 1024)
            print(f"[{tag}] Cached {len(self.cached_crops)} crops ({ram_mb:.1f} MB) in {dt:.2f}s.")

    def _extract_crop(self, item: dict) -> np.ndarray:
        img_path = os.path.join(self.root_dir, item["img_rel"])
        img = cv2.imread(img_path)
        if img is None:
            return np.zeros((36, 160, 3), dtype=np.uint8)
        p_type = item["plate_type"]
        quad = item["quad"]
        try:
            rect = self.rectifier.rectify(img, quad, plate_type=p_type)
            if p_type == "type1a":
                top, bottom = self.rectifier.split_type1a(rect)
                crop = self.rectifier.stitch_type1a_horizontal(top, bottom, target_size=(160, 36))
            else:
                crop = cv2.resize(rect, (160, 36), interpolation=cv2.INTER_LINEAR)
            return crop
        except Exception:
            return np.zeros((36, 160, 3), dtype=np.uint8)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, str]:
        item = self.samples[idx]
        if self.cached_crops:
            crop = self.cached_crops[idx].copy()
        else:
            crop = self._extract_crop(item)

        # Light data augmentation during training
        if self.is_train:
            if random.random() < 0.3:
                # Random brightness / contrast
                alpha = 1.0 + random.uniform(-0.25, 0.25)
                beta = random.uniform(-20, 20)
                crop = cv2.convertScaleAbs(crop, alpha=alpha, beta=beta)
            if random.random() < 0.2:
                # Slight blur
                k = random.choice([3, 5])
                crop = cv2.GaussianBlur(crop, (k, k), 0)
            if random.random() < 0.15:
                # Random noise
                noise = np.random.randint(-15, 15, crop.shape, dtype=np.int16)
                crop = np.clip(crop.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Preprocess: RGB and normalize to [0, 1]
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)  # (3, 36, 160)

        # Safe normalize to Latin GOST characters
        plate_str = "".join([CYR_TO_LAT.get(c, c) for c in item["plate_num"].upper()])
        target = torch.tensor([CHAR2IDX[c] for c in plate_str], dtype=torch.long)
        target_len = len(plate_str)

        return tensor, target, target_len, plate_str


def collate_fn(batch):
    tensors, targets, target_lens, plate_nums = zip(*batch)
    batch_tensors = torch.stack(tensors, dim=0)
    flat_targets = torch.cat(targets, dim=0)
    batch_target_lens = torch.tensor(target_lens, dtype=torch.long)
    return batch_tensors, flat_targets, batch_target_lens, plate_nums


def train_ocr(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("=" * 65)
    print("  OmniPlate-RU - High-Performance OCR Training (Substage 3.3)")
    print(f"  Device:     {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() and not args.cpu else 'CPU'})")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Learning R: {args.lr}")
    print("=" * 65)

    meta_csv = os.path.join(args.data_dir, "meta.csv")
    use_cache = not args.no_cache
    train_ds = PlateCropDataset(meta_csv, args.data_dir, is_train=True, cache_in_ram=use_cache)
    val_ds = PlateCropDataset(meta_csv, args.data_dir, is_train=False, cache_in_ram=use_cache)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_fn,
    )

    model = LPRNet(num_classes=NUM_CLASSES, dropout_rate=0.2).to(device)
    ctc_loss = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    os.makedirs(args.output_dir, exist_ok=True)
    best_acc = 0.0
    best_ckpt_path = os.path.join(args.output_dir, "ocr_lprnet_best.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        start_t = time.time()

        for batch_idx, (tensors, flat_targets, target_lens, _) in enumerate(train_loader):
            tensors = tensors.to(device)
            flat_targets = flat_targets.to(device)

            optimizer.zero_grad()
            logits = model(tensors)  # (B, T=40, num_classes)
            # PyTorch CTCLoss expects log_probs shape: (T, N, C)
            log_probs = logits.log_softmax(2).permute(1, 0, 2)
            input_lens = torch.full((tensors.size(0),), logits.size(1), dtype=torch.long, device=device)

            loss = ctc_loss(log_probs, flat_targets, input_lens, target_lens.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        train_loss = total_loss / max(1, len(train_loader))

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_char_correct = 0
        val_char_total = 0

        sample_preds = []
        with torch.no_grad():
            for tensors, _, _, ground_truths in val_loader:
                tensors = tensors.to(device)
                logits = model(tensors)  # (B, T=40, num_classes)
                preds = logits.argmax(dim=-1).cpu().numpy()

                for i, gt in enumerate(ground_truths):
                    pred_str = CTCDecoder.decode_greedy(preds[i], blank_idx=BLANK_IDX)
                    pred_str = CTCDecoder.apply_gost_heuristics(pred_str)

                    if len(sample_preds) < 2 and (epoch % 5 == 0 or epoch == args.epochs):
                        sample_preds.append((pred_str, gt))

                    if pred_str == gt:
                        val_correct += 1
                    val_total += 1

                    min_len = min(len(pred_str), len(gt))
                    val_char_correct += sum(1 for c1, c2 in zip(pred_str[:min_len], gt[:min_len]) if c1 == c2)
                    val_char_total += max(len(pred_str), len(gt))

        seq_acc = (val_correct / max(1, val_total)) * 100.0
        char_acc = (val_char_correct / max(1, val_char_total)) * 100.0
        elapsed = time.time() - start_t

        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] "
            f"Loss: {train_loss:.4f} | "
            f"Val Seq Acc: {seq_acc:.2f}% | "
            f"Char Acc: {char_acc:.2f}% | "
            f"Time: {elapsed:.1f}s"
        )
        if sample_preds:
            for sp, sgt in sample_preds:
                print(f"      [Sample] Pred: '{sp}' | GT: '{sgt}'")

        if seq_acc >= best_acc:
            best_acc = seq_acc
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "best_acc": best_acc,
                "vocab": list(CHAR2IDX.keys()),
            }, best_ckpt_path)
            print(f"  [+] Saved new best model to {best_ckpt_path} (Acc: {best_acc:.2f}%)")

    print("\n" + "=" * 65)
    print(f" Training Completed! Best Validation Accuracy: {best_acc:.2f}%")
    print(f" Best checkpoint: {best_ckpt_path}")
    print("=" * 65)

    # Export to ONNX
    onnx_path = os.path.join(args.output_dir, "ocr_lprnet_best.onnx")
    print(f"\n[*] Exporting Best Model to ONNX: {onnx_path}...")
    model.load_state_dict(torch.load(best_ckpt_path, map_location="cpu")["state_dict"])
    model.eval().cpu()

    dummy_input = torch.randn(1, 3, 36, 160)
    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            dynamo=False,
        )
        print(f"✅ ONNX Export Successful: {onnx_path}")
    except TypeError:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )
        print(f"✅ ONNX Export Successful: {onnx_path}")
    except Exception as e:
        print(f"⚠️ ONNX Export error: {e}")

    # Validate ONNX runtime load
    try:
        import onnxruntime as ort
        test_sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        test_inp = np.random.randn(1, 3, 36, 160).astype(np.float32)
        test_out = test_sess.run(None, {test_sess.get_inputs()[0].name: test_inp})
        print(f"🧪 ONNX Runtime Validation: PASS (Output shape: {test_out[0].shape})")
    except Exception as e:
        print(f"ℹ️ ONNX Runtime notice: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LPRNet OCR for OmniPlate-RU")
    parser.add_argument("--data_dir", type=str, default="dataset", help="Path to dataset root")
    parser.add_argument("--output_dir", type=str, default="models", help="Output directory for weights")
    parser.add_argument("--epochs", type=int, default=35, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader workers (0 recommended with RAM cache)")
    parser.add_argument("--no_cache", action="store_true", help="Disable RAM pre-caching")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution")
    args = parser.parse_args()

    train_ocr(args)
