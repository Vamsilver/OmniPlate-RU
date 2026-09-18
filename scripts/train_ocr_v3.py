#!/usr/bin/env python3
"""
scripts/train_ocr_v3.py
OmniPlate-RU OCR Training: LPRNet-v3 (Multi-Scale 1D-ASPP + ECA-Net Channel Attention).
Architecture features:
1. Block 4 equipped with 1D Atrous Spatial Pyramid Pooling (ASPP) with d=(1,1), d=(1,2), d=(1,4).
2. Efficient Channel Attention (ECA-Net 1D) on multi-scale fused features (448 channels).
3. Warm-starts backbone weights from models/ocr_lprnet_best.pt for rapid convergence.
4. Fully isolated: outputs to models/ocr_lprnet_v3.pt and models/ocr_lprnet_v3.onnx.
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
)
from src.pipeline.rectifier import PlateRectifier

CYR_TO_LAT: Dict[str, str] = {
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H",
    "О": "O", "Р": "P", "С": "C", "Т": "T", "У": "Y", "Х": "X",
}


# ---------------------------------------------------------------------------
# Architecture: LPRNet-v3 (1D-ASPP + ECA-Net Attention)
# ---------------------------------------------------------------------------

class SmallBasicBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, dilation_w: int = 1):
        super().__init__()
        pad_w = dilation_w
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c // 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_c // 4,
                out_c // 4,
                kernel_size=(3, 1),
                padding=(1, 0),
                bias=False,
            ),
            nn.BatchNorm2d(out_c // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_c // 4,
                out_c // 4,
                kernel_size=(1, 3),
                padding=(0, pad_w),
                dilation=(1, dilation_w),
                bias=False,
            ),
            nn.BatchNorm2d(out_c // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c // 4, out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c),
        )
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_c),
            )
            if in_c != out_c
            else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x) + self.shortcut(x))


class ASPPBlock1D(nn.Module):
    """
    1D Atrous Spatial Pyramid Pooling Block:
    Evaluates fine strokes (d=1), character bodies (d=2), and multi-char context (d=4).
    """
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        mid_c = out_c // 4
        # Branch 1: Identity 1x1 projection
        self.b1 = nn.Sequential(
            nn.Conv2d(in_c, mid_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_c),
            nn.ReLU(inplace=True),
        )
        # Branch 2: Local stroke d=1
        self.b2 = nn.Sequential(
            nn.Conv2d(in_c, mid_c, kernel_size=(1, 3), padding=(0, 1), dilation=(1, 1), bias=False),
            nn.BatchNorm2d(mid_c),
            nn.ReLU(inplace=True),
        )
        # Branch 3: Standard character geometry d=2
        self.b3 = nn.Sequential(
            nn.Conv2d(in_c, mid_c, kernel_size=(1, 3), padding=(0, 2), dilation=(1, 2), bias=False),
            nn.BatchNorm2d(mid_c),
            nn.ReLU(inplace=True),
        )
        # Branch 4: Extended multi-char span d=4
        self.b4 = nn.Sequential(
            nn.Conv2d(in_c, mid_c, kernel_size=(1, 3), padding=(0, 4), dilation=(1, 4), bias=False),
            nn.BatchNorm2d(mid_c),
            nn.ReLU(inplace=True),
        )
        # Projection and residual shortcut
        self.proj = nn.Sequential(
            nn.Conv2d(mid_c * 4, out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c),
        )
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_c),
            )
            if in_c != out_c
            else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        cat = torch.cat([y1, y2, y3, y4], dim=1)
        return self.relu(self.proj(cat) + self.shortcut(x))


class ECANet1D(nn.Module):
    """
    Efficient Channel Attention (ECA-Net) for 1D feature sequences:
    Captures local cross-channel interaction without dimensionality reduction.
    Parameter overhead: exactly 3 scalar weights.
    """
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H=1, W=40)
        w = self.avg_pool(x).squeeze(-1).permute(0, 2, 1)  # (B, 1, C)
        w = self.conv(w).permute(0, 2, 1).unsqueeze(-1)    # (B, C, 1, 1)
        return x * self.sigmoid(w)


class LPRNetV3(nn.Module):
    """
    LPRNet-v3: Multi-Scale 1D-ASPP + ECA-Net Attention Backbone.
    Input: (B, 3, 36, 160)
    Output: (B, T=40, num_classes)
    """
    def __init__(self, num_classes: int = NUM_CLASSES, dropout_rate: float = 0.2):
        super().__init__()
        self.num_classes = num_classes

        # Backbone Stem: 36x160 -> 18x80
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=(2, 2), padding=1),  # (B, 64, 18, 80)
        )

        # Stage 1: 18x80 -> 18x80
        self.block1 = SmallBasicBlock(64, 64, dilation_w=1)

        # Downsample 1: 18x80 -> 9x40
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=(2, 2), padding=1)

        # Stage 2: 9x40 -> 9x40
        self.block2 = SmallBasicBlock(64, 128, dilation_w=1)

        # Stage 3: 9x40 -> 9x40 (Dilated d=2)
        self.block3 = SmallBasicBlock(128, 256, dilation_w=2)

        # Downsample 2: (Height down to 5, width preserved at 40)
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))  # 5x40

        # Stage 4: 1D-ASPP Block (Multi-scale receptive field)
        self.block4 = ASPPBlock1D(256, 256)

        # Downsample 3: collapse height to 1
        self.pool3 = nn.AdaptiveAvgPool2d((1, 40))  # (B, 256, 1, 40)

        # Global multi-scale projection
        self.global_pool_stem = nn.AdaptiveAvgPool2d((1, 40))
        self.global_pool_b2 = nn.AdaptiveAvgPool2d((1, 40))

        # Total fused channels = 64 + 128 + 256 = 448
        fused_channels = 64 + 128 + 256
        self.eca = ECANet1D(fused_channels, kernel_size=3)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Conv2d(fused_channels, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(256, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f_stem = self.stem(x)         # (B, 64, 18, 80)
        f_b1 = self.block1(f_stem)    # (B, 64, 18, 80)
        f_p1 = self.pool1(f_b1)       # (B, 64, 9, 40)

        f_b2 = self.block2(f_p1)      # (B, 128, 9, 40)
        f_b3 = self.block3(f_b2)      # (B, 256, 9, 40)
        f_p2 = self.pool2(f_b3)       # (B, 256, 5, 40)
        f_b4 = self.block4(f_p2)      # (B, 256, 5, 40)
        f_p3 = self.pool3(f_b4)       # (B, 256, 1, 40)

        f_stem_proj = self.global_pool_stem(f_stem)  # (B, 64, 1, 40)
        f_b2_proj = self.global_pool_b2(f_b2)        # (B, 128, 1, 40)

        fused = torch.cat([f_stem_proj, f_b2_proj, f_p3], dim=1)  # (B, 448, 1, 40)
        fused = self.eca(fused)                                    # Channel Attention

        out = self.classifier(fused)  # (B, num_classes, 1, 40)
        out = out.squeeze(2)          # (B, num_classes, 40)
        out = out.permute(0, 2, 1)    # (B, 40, num_classes)
        return out


# ---------------------------------------------------------------------------
# Dataset & In-RAM Pre-caching
# ---------------------------------------------------------------------------

class PlateCropDataset(Dataset):
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
                        if p_type not in ("type1", "type1a", "type1b", "type2"):
                            continue
                        c_path = os.path.join("verified_crops", c_file)
                        item_dict = {
                            "img_rel": c_path,
                            "plate_num": p_num,
                            "plate_type": p_type,
                            "is_pre_cropped": True,
                        }
                        if c_file.startswith("crop_ref_") or c_file.startswith("crop_1b_") or "real" in c_file:
                            real_curated.append(item_dict)
                        else:
                            synth_hard.append(item_dict)

        random.seed(seed)
        all_synth = synth_samples + synth_hard
        random.shuffle(all_synth)
        n_val_synth = int(len(all_synth) * val_split)
        val_synth = all_synth[:n_val_synth]
        train_synth = all_synth[n_val_synth:]

        all_real = real_meta_samples + real_curated
        random.shuffle(all_real)
        n_val_real = int(len(all_real) * val_split)
        val_real = all_real[:n_val_real]
        train_real = all_real[n_val_real:]

        if self.is_train:
            self.samples = train_synth + (train_real * 8)
            random.shuffle(self.samples)
        else:
            self.samples = val_synth + val_real

        self.ram_cache: Dict[int, np.ndarray] = {}
        if self.cache_in_ram:
            self._preload_ram()

    def _preload_ram(self):
        t0 = time.time()
        for idx in range(len(self.samples)):
            item = self.samples[idx]
            img_path = os.path.join(self.root_dir, item["img_rel"])
            if not os.path.exists(img_path):
                continue
            img = cv2.imread(img_path)
            if img is None:
                continue

            if item["is_pre_cropped"]:
                crop = img
            else:
                try:
                    quad = [float(x) for x in item["quad"].split(",")]
                    pts = np.array(quad, dtype=np.float32).reshape(4, 2)
                    crop = self.rectifier.rectify(img, pts, plate_type=item["plate_type"])
                except Exception:
                    continue

            if crop is None or crop.size == 0:
                continue

            if item["plate_type"] == "type1a" and not item["is_pre_cropped"]:
                if crop.shape[:2] == (96, 160):
                    top_l, bot_l = self.rectifier.split_type1a(crop)
                    crop = self.rectifier.stitch_type1a_horizontal(top_l, bot_l, target_size=(160, 36))

            h, w = crop.shape[:2]
            if (w, h) != (160, 36):
                crop = cv2.resize(crop, (160, 36), interpolation=cv2.INTER_LINEAR)

            self.ram_cache[idx] = crop

        dt = time.time() - t0
        mem_mb = sum(c.nbytes for c in self.ram_cache.values()) / (1024 * 1024)
        print(f"[{'Train' if self.is_train else 'Val'}] Cached {len(self.ram_cache)}/{len(self.samples)} crops in RAM ({mem_mb:.1f} MB, {dt:.2f}s)")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Optional[Tuple[torch.Tensor, str, str]]:
        crop = self.ram_cache.get(idx, None)
        if crop is None:
            # Generate dummy image on failure
            crop = np.zeros((36, 160, 3), dtype=np.uint8)

        text = self.samples[idx]["plate_num"].upper().strip()
        norm_text = []
        for ch in text:
            ch = CYR_TO_LAT.get(ch, ch)
            if ch in CHAR2IDX and ch != "-":
                norm_text.append(ch)
        clean_text = "".join(norm_text)
        if not clean_text:
            clean_text = "A123BC77"

        # Augmentation on train
        if self.is_train:
            # Random brightness / contrast
            if random.random() < 0.35:
                alpha = random.uniform(0.7, 1.3)
                beta = random.uniform(-20, 20)
                crop = np.clip(alpha * crop + beta, 0, 255).astype(np.uint8)
            # Random horizontal blur
            if random.random() < 0.25:
                k = random.choice([3, 5])
                crop = cv2.blur(crop, (k, 1))

        tensor = crop.astype(np.float32) / 255.0
        tensor = tensor.transpose(2, 0, 1)  # (3, 36, 160)
        return torch.from_numpy(tensor), clean_text, self.samples[idx]["plate_type"]


def collate_fn(batch):
    tensors, texts, plate_types = zip(*batch)
    tensors = torch.stack(tensors, dim=0)

    target_lens = []
    flat_targets = []
    for t in texts:
        idxs = [CHAR2IDX[c] for c in t if c in CHAR2IDX]
        target_lens.append(len(idxs))
        flat_targets.extend(idxs)

    flat_targets = torch.tensor(flat_targets, dtype=torch.long)
    target_lens = torch.tensor(target_lens, dtype=torch.long)
    return tensors, flat_targets, target_lens, texts, plate_types


# ---------------------------------------------------------------------------
# Training Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train LPRNet-v3 (1D-ASPP + ECA-Net Attention)")
    parser.add_argument("--epochs", type=int, default=15, help="Number of fine-tuning epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Max learning rate for OneCycleLR")
    parser.add_argument("--workers", type=int, default=0, help="Dataloader workers")
    parser.add_argument("--output-dir", type=str, default="models", help="Output directory for checkpoints")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Active training device: {device}")

    meta_csv = os.path.join(ROOT_DIR, "dataset", "meta.csv")
    dataset_dir = os.path.join(ROOT_DIR, "dataset")

    print("[+] Initializing Train Dataset...")
    train_ds = PlateCropDataset(meta_csv, dataset_dir, is_train=True, cache_in_ram=True)
    print("[+] Initializing Val Dataset...")
    val_ds = PlateCropDataset(meta_csv, dataset_dir, is_train=False, cache_in_ram=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
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

    # Initialize LPRNet-v3 (1D-ASPP + ECA-Net)
    model = LPRNetV3(num_classes=NUM_CLASSES, dropout_rate=0.2).to(device)

    # Smart warm-start from best existing checkpoint
    init_ckpt = os.path.join(args.output_dir, "ocr_lprnet_best.pt")
    if os.path.exists(init_ckpt):
        try:
            ckpt = torch.load(init_ckpt, map_location=device)
            st = ckpt.get("state_dict", ckpt)
            model_st = model.state_dict()
            loaded = 0
            for k, v in st.items():
                if k in model_st and model_st[k].shape == v.shape:
                    model_st[k] = v
                    loaded += 1
            model.load_state_dict(model_st)
            print(f"[+] Smart Warm-Start: Loaded {loaded}/{len(model_st)} parameter tensors from {init_ckpt} successfully!")
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
    best_ckpt_path = os.path.join(args.output_dir, "ocr_lprnet_v3.pt")

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
        acc_t1a = (by_type_correct["type1a"] / max(1, by_type_total["type1a"])) * 100.0
        acc_t1b = (by_type_correct["type1b"] / max(1, by_type_total["type1b"])) * 100.0
        acc_t2 = (by_type_correct["type2"] / max(1, by_type_total["type2"])) * 100.0

        elapsed = time.time() - start_t
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({elapsed:.1f}s) | "
              f"Loss: {train_loss:.4f} | "
              f"Val SeqAcc: {seq_acc:.2f}% (CharAcc: {char_acc:.2f}%) | "
              f"T1: {acc_t1:.1f}% | 1A: {acc_t1a:.1f}% | 1B: {acc_t1b:.1f}% | T2: {acc_t2:.1f}%")

        if seq_acc >= best_acc:
            best_acc = seq_acc
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "seq_acc": seq_acc,
                "char_acc": char_acc,
                "arch": "LPRNetV3_1D_ASPP_ECA",
            }, best_ckpt_path)
            print(f"  [*] Saved new best LPRNet-v3 checkpoint: {best_ckpt_path} (SeqAcc: {seq_acc:.2f}%)")

    # Export ONNX
    print("\n[+] Exporting Best LPRNet-v3 to ONNX...")
    model.eval()
    if os.path.exists(best_ckpt_path):
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])

    onnx_path = os.path.join(args.output_dir, "ocr_lprnet_v3.onnx")
    dummy_input = torch.randn(1, 3, 36, 160, device=device)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=17,
        dynamo=False,
    )
    # Remove any stray .data file if created
    data_file = onnx_path + ".data"
    if os.path.exists(data_file):
        try:
            os.remove(data_file)
        except Exception:
            pass
    print(f"[+] LPRNet-v3 ONNX model saved to {onnx_path} ({os.path.getsize(onnx_path) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
