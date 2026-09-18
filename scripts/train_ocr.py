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

        # 1. Load samples from meta.csv (synthetic and all real road scenes)
        synth_samples = []
        real_meta_samples = []
        with open(meta_csv, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f, delimiter=";")
            header = next(reader)
            for row in reader:
                if len(row) < 10:
                    continue
                img_rel, plate_num, p_type, bbox_str, quad_str, is_veh, is_syn, src, lic, cond = row[:10]
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
        print(f"[+] Loaded from meta.csv: {len(synth_samples)} synthetic and {len(real_meta_samples)} real plate scenes")

        # 2. Ingest verified real crops and targeted hard synth crops from dataset/verified_crops/manifest.csv
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
            print(f"[+] Loaded {len(real_curated)} real plate crops and {len(synth_hard)} hard synth crops from {manifest_path}")

        # Deterministic split
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

        if is_train:
            # Oversample real road samples so model deeply learns real camera artifacts, fonts, and angles
            oversampled_real = []
            for _ in range(4):
                oversampled_real.extend(real_train)
                oversampled_real.extend(real_meta_train)
            # Targeted 1A boost to balance square plates with Type 1
            extra_1a = [s for s in (real_train + real_meta_train + hard_train) if s["plate_type"] == "type1a"]
            for _ in range(4):
                oversampled_real.extend(extra_1a)
            # Targeted Type 1 boost on clean real road plates to reinforce real camera fonts
            extra_t1_real = [s for s in (real_train + real_meta_train) if s["plate_type"] == "type1" and "#" not in s["plate_num"]]
            for _ in range(3):
                oversampled_real.extend(extra_t1_real)
            self.samples = synth_train + hard_train + oversampled_real
            random.shuffle(self.samples)
            print(
                f"[TRAIN] Total: {len(self.samples)} ({len(synth_train)} synth + {len(hard_train)} hard synth + "
                f"{len(oversampled_real)} oversampled real/1A [{len(real_train) + len(real_meta_train)} unique real])"
            )
        else:
            self.samples = synth_val + hard_val + real_val + real_meta_val
            print(
                f"[VAL] Total: {len(self.samples)} ({len(synth_val)} synth + "
                f"{len(hard_val)} hard synth + {len(real_val) + len(real_meta_val)} real [{len(real_val)} curated + {len(real_meta_val)} meta])"
            )

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

        if item.get("is_pre_cropped", False):
            if img.shape[:2] != (36, 160):
                return cv2.resize(img, (160, 36), interpolation=cv2.INTER_LINEAR)
            return img

        p_type = item["plate_type"]
        quad = item.get("quad", "")
        bbox = item.get("bbox", "")
        try:
            if quad and len([v for v in quad.split(",") if v.strip()]) == 8:
                rect = self.rectifier.rectify(img, quad, plate_type=p_type)
            elif bbox and len([v for v in bbox.split(",") if v.strip()]) == 4:
                rect = self.rectifier.rectify_bbox(img, bbox, plate_type=p_type)
            else:
                rect = cv2.resize(img, (160, 36), interpolation=cv2.INTER_LINEAR)

            if p_type == "type1a":
                top, bottom = self.rectifier.split_type1a(rect, adaptive_seam=True, vertical_margin=0)
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

        # Heavy photorealistic data augmentation during training
        if self.is_train:
            h, w = crop.shape[:2]

            # 1. Angled Motion blur or Gaussian blur (camera shake, high-speed vehicle movement)
            if random.random() < 0.30:
                k = random.choice([3, 5, 7, 9])
                angle = random.uniform(-25.0, 25.0)
                kernel = np.zeros((k, k), dtype=np.float32)
                kernel[k // 2, :] = 1.0
                M_rot = cv2.getRotationMatrix2D((k / 2.0 - 0.5, k / 2.0 - 0.5), angle, 1.0)
                k_rot = cv2.warpAffine(kernel, M_rot, (k, k))
                k_sum = float(np.sum(k_rot))
                kernel_final = k_rot / k_sum if k_sum > 0 else kernel / k
                crop = cv2.filter2D(crop, -1, kernel_final)
            elif random.random() < 0.20:
                k = random.choice([3, 5])
                crop = cv2.GaussianBlur(crop, (k, k), 0)

            # 2. Lighting, Headlight Lens Flare & Contrast Jitter
            if random.random() < 0.25:
                # Direct sunlight / headlight glare flare
                cx = random.randint(0, w)
                cy = random.randint(0, h)
                rx = random.randint(15, 45)
                ry = random.randint(8, 20)
                flare_layer = np.zeros((h, w), dtype=np.uint8)
                cv2.ellipse(flare_layer, (cx, cy), (rx, ry), random.randint(-30, 30), 0, 360, 255, -1)
                flare_blur = cv2.GaussianBlur(flare_layer, (15, 15), 0)
                flare_intensity = random.uniform(0.18, 0.45)
                crop = np.clip(crop.astype(np.float32) + (flare_blur[:, :, None] * flare_intensity), 0, 255).astype(np.uint8)

            if random.random() < 0.35:
                alpha = random.uniform(0.70, 1.35)
                beta = random.uniform(-25, 25)
                crop = cv2.convertScaleAbs(crop, alpha=alpha, beta=beta)

            # 3. HSV hue / saturation shift (sodium street lamps, sunsets, shadows)
            if random.random() < 0.30:
                hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)
                hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-8, 8)) % 180
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.75, 1.25), 0, 255)
                crop = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

            # 4. Sensor noise (Gaussian + salt/pepper)
            if random.random() < 0.25:
                noise = np.random.normal(0, random.uniform(4, 16), crop.shape).astype(np.float32)
                crop = np.clip(crop.astype(np.float32) + noise, 0, 255).astype(np.uint8)

            # 5. Road grime scratches & bolt holes
            if random.random() < 0.30:
                for _ in range(random.randint(1, 2)):
                    pt1 = (random.randint(0, w), random.randint(0, h))
                    pt2 = (max(0, min(w - 1, pt1[0] + random.randint(-25, 25))), max(0, min(h - 1, pt1[1] + random.randint(-6, 6))))
                    color = (random.randint(20, 65), random.randint(20, 65), random.randint(20, 65))
                    cv2.line(crop, pt1, pt2, color, thickness=1)

            if random.random() < 0.30:
                num_spots = random.randint(1, 3)
                for _ in range(num_spots):
                    cx = random.randint(10, w - 10)
                    cy = random.randint(5, h - 5)
                    rx = random.randint(2, 5)
                    ry = random.randint(2, 4)
                    color = (random.randint(25, 65), random.randint(25, 65), random.randint(25, 65))
                    cv2.ellipse(crop, (cx, cy), (rx, ry), random.randint(0, 180), 0, 360, color, -1)

            # 6. Perspective perturbation (small yaw/pitch warping)
            if random.random() < 0.25:
                dx = random.uniform(-3, 3)
                dy = random.uniform(-2, 2)
                pts1 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
                pts2 = np.float32([[dx, dy], [w - dx, -dy], [w + dx, h + dy], [-dx, h - dy]])
                M = cv2.getPerspectiveTransform(pts1, pts2)
                crop = cv2.warpPerspective(crop, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

            # 7. Low-Resolution & Distance Downscaling (simulating distant traffic cameras)
            if random.random() < 0.40:
                scale_h = random.randint(12, 24)
                scale_w = max(40, int(w * (scale_h / float(h))))
                interp = random.choice([cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LINEAR])
                small = cv2.resize(crop, (scale_w, scale_h), interpolation=interp)
                crop = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

            # 8. JPEG Compression Artifacts (road CCTV / dashcam codecs)
            if random.random() < 0.35:
                quality = random.randint(20, 65)
                success, enc = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, quality])
                if success:
                    crop = cv2.imdecode(enc, cv2.IMREAD_COLOR)

        # Preprocess: RGB and normalize to [0, 1]
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)  # (3, 36, 160)

        # Safe normalize to Latin GOST characters
        plate_str = "".join([CYR_TO_LAT.get(c, c) for c in item["plate_num"].upper()])
        target = torch.tensor([CHAR2IDX[c] for c in plate_str], dtype=torch.long)
        target_len = len(plate_str)

        return tensor, target, target_len, plate_str, item["plate_type"]


def collate_fn(batch):
    tensors, targets, target_lens, plate_nums, plate_types = zip(*batch)
    batch_tensors = torch.stack(tensors, dim=0)
    flat_targets = torch.cat(targets, dim=0)
    batch_target_lens = torch.tensor(target_lens, dtype=torch.long)
    return batch_tensors, flat_targets, batch_target_lens, plate_nums, plate_types


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

    # Deterministic seeding for reproducible convergence
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    model = LPRNet(num_classes=NUM_CLASSES, dropout_rate=0.2).to(device)
    ctc_loss = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=0.25,
        anneal_strategy="cos",
        div_factor=10.0,
        final_div_factor=1000.0,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    best_acc = 0.0
    best_ckpt_path = os.path.join(args.output_dir, "ocr_lprnet_best.pt")

    if args.resume and os.path.exists(best_ckpt_path):
        try:
            ckpt = torch.load(best_ckpt_path, map_location=device)
            model.load_state_dict(ckpt["state_dict"])
            prev_acc = float(ckpt.get("best_acc", 0.0))
            best_acc = 0.0
            print(f"[*] Loaded pre-trained weights from {best_ckpt_path} (previous best acc: {prev_acc:.2f}%) for fine-tuning!")
        except Exception as e:
            print(f"[!] Warning: Could not resume from checkpoint: {e}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        start_t = time.time()

        for batch_idx, (tensors, flat_targets, target_lens, _, _) in enumerate(train_loader):
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

        sample_preds = []
        with torch.no_grad():
            for tensors, _, _, ground_truths, plate_types in val_loader:
                tensors = tensors.to(device)
                logits = model(tensors)  # (B, T=40, num_classes)
                preds = logits.argmax(dim=-1).cpu().numpy()

                for i, gt in enumerate(ground_truths):
                    pt = plate_types[i]
                    pred_str = CTCDecoder.decode_greedy(preds[i], blank_idx=BLANK_IDX)
                    pred_str = CTCDecoder.apply_gost_heuristics(pred_str, plate_type=pt)

                    if len(sample_preds) < 2 and (epoch % 5 == 0 or epoch == args.epochs):
                        sample_preds.append((pred_str, gt))

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
        acc_t2 = (by_type_correct["type2"] / max(1, by_type_total["type2"])) * 100.0
        elapsed = time.time() - start_t

        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] "
            f"Loss: {train_loss:.4f} | "
            f"Val: {seq_acc:.1f}% (T1:{acc_t1:.0f}% 1A:{acc_1a:.0f}% 1B:{acc_1b:.0f}% T2:{acc_t2:.0f}%) | "
            f"Char: {char_acc:.1f}% | "
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
        print(f"[SUCCESS] ONNX Export Successful: {onnx_path}")
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
        print(f"[SUCCESS] ONNX Export Successful: {onnx_path}")
        # Mirror to ocr_lprnet.onnx for compatibility
        compat_onnx = os.path.join(args.output_dir, "ocr_lprnet.onnx")
        import shutil
        shutil.copyfile(onnx_path, compat_onnx)
        print(f"[SUCCESS] Mirrored ONNX to {compat_onnx}")
    except Exception as e:
        print(f"[ERROR] ONNX Export error: {e}")

    # Mirror to ocr_lprnet.onnx for backwards compatibility
    compat_onnx = os.path.join(args.output_dir, "ocr_lprnet.onnx")
    if os.path.exists(onnx_path):
        import shutil
        shutil.copyfile(onnx_path, compat_onnx)
        print(f"[SUCCESS] Mirrored ONNX to {compat_onnx}")

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
    parser = argparse.ArgumentParser(description="Train LPRNet OCR for OmniPlate-RU")
    parser.add_argument("--data_dir", type=str, default="dataset", help="Path to dataset root")
    parser.add_argument("--output_dir", type=str, default="models", help="Output directory for weights")
    parser.add_argument("--epochs", type=int, default=35, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader workers (0 recommended with RAM cache)")
    parser.add_argument("--no_cache", action="store_true", help="Disable RAM pre-caching")
    parser.add_argument("--resume", action="store_true", help="Resume from existing best model checkpoint")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution")
    args = parser.parse_args()

    train_ocr(args)
