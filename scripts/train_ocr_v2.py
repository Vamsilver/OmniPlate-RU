#!/usr/bin/env python3
"""
scripts/train_ocr_v2.py
OmniPlate-RU OCR Training with Dilated Horizontal Receptive Field (Stage 4.5 LPRNet-v2).
Expands horizontal receptive field from 37px (23%) to 61px (38%) using horizontal dilation
(d=2 in block3, d=3 in block4) with 0 additional parameters and 0 additional FLOPs.
Warm-starts from models/ocr_lprnet_best.pt for rapid, stable convergence.
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

        synth_samples = []
        real_meta_samples = []
        with open(meta_csv, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f, delimiter=";")
            header = next(reader)
            for row in reader:
                if len(row) < 10:
                    continue
                img_rel, plate_num, p_type, bbox_str, quad_str, is_veh, is_syn = row[:7]
                if p_type not in ("type1", "type1a", "type1b"):
                    continue
                item_dict = {
                    "img_rel": img_rel,
                    "plate_num": plate_num,
                    "plate_type": p_type,
                    "quad": quad_str,
                    "bbox": bbox_str,
                    "is_pre_cropped": False,
                }
                if is_syn == "1":
                    synth_samples.append(item_dict)
                else:
                    real_meta_samples.append(item_dict)

        manifest_path = os.path.join(self.root_dir, "verified_crops", "manifest.csv")
        real_curated = []
        synth_hard = []
        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as mf:
                m_reader = csv.reader(mf, delimiter=";")
                next(m_reader, None)
                for m_row in m_reader:
                    if len(m_row) >= 3:
                        c_file, p_num, p_type = m_row[0], m_row[1], m_row[2]
                        src_val = m_row[4] if len(m_row) >= 5 else ""
                        c_full_p = os.path.join(self.root_dir, "verified_crops", c_file)
                        if os.path.exists(c_full_p):
                            sample_item = {
                                "img_rel": os.path.join("verified_crops", c_file),
                                "plate_num": p_num,
                                "plate_type": p_type,
                                "quad": "",
                                "bbox": "",
                                "is_pre_cropped": True,
                            }
                            if "synth" in src_val.lower():
                                synth_hard.append(sample_item)
                            else:
                                real_curated.append(sample_item)

        t1a_lines_manifest = os.path.join(self.root_dir, "type1a_lines", "manifest.csv")
        t1a_lines_real = []
        t1a_lines_synth = []
        if os.path.exists(t1a_lines_manifest):
            with open(t1a_lines_manifest, "r", encoding="utf-8") as lf:
                l_reader = csv.reader(lf, delimiter=";")
                next(l_reader, None)
                for l_row in l_reader:
                    if len(l_row) >= 6:
                        c_file, p_num, p_type, conf, src_val, is_syn = l_row[:6]
                        c_full_p = os.path.join(self.root_dir, c_file)
                        if os.path.exists(c_full_p):
                            line_item = {
                                "img_rel": c_file,
                                "plate_num": p_num,
                                "plate_type": p_type,
                                "quad": "",
                                "bbox": "",
                                "is_pre_cropped": True,
                            }
                            if is_syn == "1":
                                t1a_lines_synth.append(line_item)
                            else:
                                t1a_lines_real.append(line_item)

        random.seed(seed)
        random.shuffle(synth_samples)
        synth_val_idx = max(1, int(len(synth_samples) * val_split))
        synth_train = synth_samples[synth_val_idx:]
        synth_val = synth_samples[:synth_val_idx]

        random.shuffle(real_meta_samples)
        real_meta_val_idx = max(1, int(len(real_meta_samples) * val_split))
        real_meta_train = real_meta_samples[real_meta_val_idx:]
        real_meta_val = real_meta_samples[:real_meta_val_idx]

        random.shuffle(synth_hard)
        hard_val_idx = max(1, int(len(synth_hard) * val_split))
        hard_train = synth_hard[hard_val_idx:]
        hard_val = synth_hard[:hard_val_idx]

        random.shuffle(real_curated)
        real_val_idx = max(1, int(len(real_curated) * val_split))
        real_train = real_curated[real_val_idx:]
        real_val = real_curated[:real_val_idx]

        random.shuffle(t1a_lines_synth)
        t1a_synth_val_idx = max(1, int(len(t1a_lines_synth) * val_split))
        t1a_synth_train = t1a_lines_synth[t1a_synth_val_idx:]
        t1a_synth_val = t1a_lines_synth[:t1a_synth_val_idx]

        random.shuffle(t1a_lines_real)
        t1a_real_val_idx = max(1, int(len(t1a_lines_real) * val_split))
        t1a_real_train = t1a_lines_real[t1a_real_val_idx:]
        t1a_real_val = t1a_lines_real[:t1a_real_val_idx]

        if is_train:
            oversampled_real = []
            for _ in range(4):
                oversampled_real.extend(real_train)
                oversampled_real.extend(real_meta_train)
            extra_1a = [s for s in (real_train + real_meta_train + hard_train) if s["plate_type"] == "type1a"]
            for _ in range(4):
                oversampled_real.extend(extra_1a)
            extra_t1_real = [s for s in (real_train + real_meta_train) if s["plate_type"] == "type1" and "#" not in s["plate_num"]]
            for _ in range(6):
                oversampled_real.extend(extra_t1_real)
            oversampled_1a_lines = []
            for _ in range(4):
                oversampled_1a_lines.extend(t1a_real_train)

            self.samples = (
                synth_train
                + hard_train
                + oversampled_real
                + t1a_synth_train
                + oversampled_1a_lines
            )
            random.shuffle(self.samples)
            print(f"[TRAIN] Total: {len(self.samples)} plate samples cached.")
        else:
            self.samples = (
                synth_val
                + hard_val
                + real_val
                + real_meta_val
                + t1a_synth_val
                + t1a_real_val
            )
            print(f"[VAL] Total: {len(self.samples)} plate samples cached.")

        self.cached_crops: List[np.ndarray] = []
        if self.cache_in_ram:
            tag = "TRAIN" if is_train else "VAL"
            t0 = time.time()
            crop_dict = {}
            for item in self.samples:
                key = (item["img_rel"], item.get("quad", ""), item.get("bbox", ""))
                if key not in crop_dict:
                    crop_dict[key] = self._extract_crop(item)
                self.cached_crops.append(crop_dict[key])
            dt = time.time() - t0
            ram_mb = len(crop_dict) * 36 * 160 * 3 / (1024 * 1024)
            print(f"[{tag}] Cached {len(crop_dict)} unique crops ({len(self.cached_crops)} total samples, {ram_mb:.1f} MB) in {dt:.2f}s.")

    def _extract_crop(self, item: dict) -> np.ndarray:
        img_path = os.path.join(self.root_dir, item["img_rel"])
        img = cv2.imread(img_path)
        if img is None:
            return np.zeros((36, 160, 3), dtype=np.uint8)

        if item.get("is_pre_cropped", False):
            if img.shape[:2] != (36, 160):
                return cv2.resize(img, (160, 36), interpolation=cv2.INTER_LINEAR)
            return img

        p_type = item["plate_type"]
        quad = item.get("quad", "")
        bbox = item.get("bbox", "")
        try:
            if quad and len([v for v in quad.split(",") if v.strip()]) == 8:
                rect = self.rectifier.rectify(img, quad, plate_type=p_type, margin=(0.012, 0.006))
            elif bbox and len([v for v in bbox.split(",") if v.strip()]) == 4:
                rect = self.rectifier.rectify_bbox(img, bbox, plate_type=p_type)
            else:
                rect = np.zeros((36, 160, 3), dtype=np.uint8)

            if p_type == "type1a" and rect.shape[:2] == (96, 160):
                top_l, bot_l = self.rectifier.split_type1a(rect)
                return self.rectifier.stitch_type1a_horizontal(top_l, bot_l, target_size=(160, 36))
            return rect
        except Exception:
            return np.zeros((36, 160, 3), dtype=np.uint8)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, str, str]:
        item = self.samples[idx]
        if self.cached_crops:
            crop = self.cached_crops[idx].copy()
        else:
            crop = self._extract_crop(item)

        if self.is_train:
            h, w = crop.shape[:2]
            if random.random() < 0.40:
                k = random.choice([3, 5, 7])
                crop = cv2.GaussianBlur(crop, (k, k), 0)
            if random.random() < 0.30:
                alpha = random.uniform(0.75, 1.30)
                beta = random.uniform(-20, 20)
                crop = cv2.convertScaleAbs(crop, alpha=alpha, beta=beta)
            if random.random() < 0.20:
                noise = np.random.normal(0, random.uniform(4, 12), crop.shape).astype(np.float32)
                crop = np.clip(crop.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # Normalize to (3, 36, 160) [0, 1]
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb.transpose((2, 0, 1))).float() / 255.0

        raw_plate = item["plate_num"].strip().upper()
        plate_str = "".join(CYR_TO_LAT.get(c, c) for c in raw_plate if c in CYR_TO_LAT or c.isalnum() or c == "#")
        plate_str = "".join(c for c in plate_str if c in CHAR2IDX)

        target = torch.tensor([CHAR2IDX[c] for c in plate_str], dtype=torch.long)
        target_len = len(plate_str)
        return tensor, target, target_len, plate_str, item["plate_type"]


def collate_fn(batch):
    tensors, targets, target_lens, plate_nums, plate_types = zip(*batch)
    batch_tensors = torch.stack(tensors, dim=0)
    flat_targets = torch.cat(targets, dim=0)
    batch_target_lens = torch.tensor(target_lens, dtype=torch.long)
    return batch_tensors, flat_targets, batch_target_lens, plate_nums, plate_types


def train_ocr_v2(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("=" * 65)
    print("  OmniPlate-RU - LPRNet-v2 (Dilated Horizontal Context Training)")
    print(f"  Device:     {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() and not args.cpu else 'CPU'})")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Learning R: {args.lr}")
    print(f"  Dilation:   Stage 3 d=2, Stage 4 d=3 (RF: 37px -> 61px, +65%)")
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

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Initialize Dilated LPRNet-v2
    model = LPRNet(num_classes=NUM_CLASSES, dropout_rate=0.2, dilated=True).to(device)

    # Warm-start from best existing checkpoint if available
    init_ckpt = os.path.join(args.output_dir, "ocr_lprnet_best.pt")
    if os.path.exists(init_ckpt):
        try:
            ckpt = torch.load(init_ckpt, map_location=device)
            st = ckpt.get("state_dict", ckpt)
            model.load_state_dict(st, strict=True)
            print(f"[+] Warm-start: Loaded pre-trained weights from {init_ckpt} successfully!")
        except Exception as e:
            print(f"[!] Warning: Could not warm-start from checkpoint: {e}")

    ctc_loss = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=0.20,
        anneal_strategy="cos",
        div_factor=8.0,
        final_div_factor=500.0,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    best_acc = 0.0
    best_ckpt_path = os.path.join(args.output_dir, "ocr_lprnet_v2.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        start_t = time.time()

        for batch_idx, (tensors, flat_targets, target_lens, _, _) in enumerate(train_loader):
            tensors = tensors.to(device)
            flat_targets = flat_targets.to(device)

            optimizer.zero_grad()
            logits = model(tensors)  # (B, T=40, num_classes)
            log_probs = logits.log_softmax(2).permute(1, 0, 2)  # (T, B, C)
            input_lens = torch.full((tensors.size(0),), logits.size(1), dtype=torch.long, device=device)

            loss = ctc_loss(log_probs, flat_targets, input_lens, target_lens.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        train_loss = total_loss / max(1, len(train_loader))

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_char_correct = 0
        val_char_total = 0
        by_type_correct = {"type1": 0, "type1a": 0, "type1b": 0, "type2": 0}
        by_type_total = {"type1": 0, "type1a": 0, "type1b": 0, "type2": 0}

        with torch.no_grad():
            for tensors, _, _, ground_truths, plate_types in val_loader:
                tensors = tensors.to(device)
                logits = model(tensors)  # (B, T=40, num_classes)
                preds = logits.argmax(dim=-1).cpu().numpy()

                for i, gt in enumerate(ground_truths):
                    pt = plate_types[i]
                    pred_str = CTCDecoder.decode_greedy(preds[i], blank_idx=BLANK_IDX)
                    pred_str = CTCDecoder.apply_gost_heuristics(pred_str, plate_type=pt)

                    if pred_str == gt:
                        val_correct += 1
                        if pt in by_type_correct:
                            by_type_correct[pt] += 1
                    val_total += 1
                    if pt in by_type_total:
                        by_type_total[pt] += 1

                    min_len = min(len(pred_str), len(gt))
                    val_char_correct += sum(1 for c1, c2 in zip(pred_str[:min_len], gt[:min_len]) if c1 == c2)
                    val_char_total += max(len(pred_str), len(gt))

        seq_acc = (val_correct / max(1, val_total)) * 100.0
        char_acc = (val_char_correct / max(1, val_char_total)) * 100.0
        acc_t1 = (by_type_correct["type1"] / max(1, by_type_total["type1"])) * 100.0
        acc_1a = (by_type_correct["type1a"] / max(1, by_type_total["type1a"])) * 100.0
        acc_1b = (by_type_correct["type1b"] / max(1, by_type_total["type1b"])) * 100.0
        elapsed = time.time() - start_t

        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] "
            f"Loss: {train_loss:.4f} | "
            f"Val: {seq_acc:.2f}% (T1:{acc_t1:.1f}% 1A:{acc_1a:.1f}% 1B:{acc_1b:.1f}%) | "
            f"Char: {char_acc:.1f}% | "
            f"Time: {elapsed:.1f}s"
        )

        if seq_acc >= best_acc:
            best_acc = seq_acc
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "best_acc": best_acc,
                "vocab": list(CHAR2IDX.keys()),
                "dilated": True,
            }, best_ckpt_path)
            print(f"  [+] Saved new best model to {best_ckpt_path} (Acc: {best_acc:.2f}%)")

    print("\n" + "=" * 65)
    print(f" Training Completed! Best Validation Accuracy: {best_acc:.2f}%")
    print(f" Best checkpoint: {best_ckpt_path}")
    print("=" * 65)

    # Export to ONNX
    onnx_path = os.path.join(args.output_dir, "ocr_lprnet_v2.onnx")
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
        print(f"[SUCCESS] ONNX Export Successful: {onnx_path}")
    except Exception as e:
        print(f"[ERROR] ONNX Export error: {e}")

    # Validate ONNX runtime load
    try:
        import onnxruntime as ort
        test_sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        test_inp = np.random.randn(1, 3, 36, 160).astype(np.float32)
        test_out = test_sess.run(None, {test_sess.get_inputs()[0].name: test_inp})
        print(f"[PASS] ONNX Runtime Validation: PASS (Output shape: {test_out[0].shape})")
    except Exception as e:
        print(f"[INFO] ONNX Runtime notice: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Dilated LPRNet-v2 OCR")
    parser.add_argument("--data_dir", type=str, default="dataset", help="Path to dataset root")
    parser.add_argument("--output_dir", type=str, default="models", help="Output directory for weights")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=4e-4, help="Initial learning rate")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--no_cache", action="store_true", help="Disable RAM pre-caching")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution")
    args = parser.parse_args()

    train_ocr_v2(args)
