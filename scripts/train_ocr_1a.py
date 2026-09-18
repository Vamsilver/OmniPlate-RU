#!/usr/bin/env python3
"""
OmniPlate-RU — High-Speed LPRNet-2D Training Script for Type 1A Square Plates.
Volga IT 2026: Automatic Vehicle License Plate Recognition.

Trains LPRNet-2D on native canonical (96, 160, 3) crops without Split & Stitch seams:
- Dual CTC Loss (Top line: 40 steps, Bottom line: 40 steps, Total: 80 steps).
- In-RAM pre-caching for ultra-fast GPU training on RTX 5080.
- Automatic ONNX export to models/ocr_lprnet_1a.onnx.
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
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.pipeline.ocr import (
    BLANK_IDX,
    CHAR2IDX,
    IDX2CHAR,
    NUM_CLASSES,
    SmallBasicBlock,
)
from src.pipeline.rectifier import PlateRectifier


class LPRNet2D(nn.Module):
    """
    Native 2D License Plate Recognition Network for Type 1A square plates.
    Accepts canonical (B, 3, 96, 160) input crops without Split & Stitch seams.
    Outputs CTC logits (B, seq_len=80, num_classes) where:
      - Steps 0..39: Top line (Series letter + 3 digits, e.g. 'A123')
      - Steps 40..79: Bottom line (2 Series letters + 2-3 digits region, e.g. 'BC77' / 'BC716')
    """
    def __init__(self, num_classes: int = NUM_CLASSES, dropout_rate: float = 0.2):
        super().__init__()
        self.num_classes = num_classes

        # Stem: 96x160 -> 48x80
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=(2, 2), padding=1),
        )
        self.block1 = SmallBasicBlock(64, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=(2, 2), padding=1)  # 24x40
        self.block2 = SmallBasicBlock(64, 128)
        self.block3 = SmallBasicBlock(128, 256)
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))  # 12x40
        self.block4 = SmallBasicBlock(256, 256)

        # Spatial pooling: preserves 2 vertical bins (row 0 = top line, row 1 = bottom line)
        self.pool3 = nn.AdaptiveAvgPool2d((2, 40))
        self.global_pool_stem = nn.AdaptiveAvgPool2d((2, 40))
        self.global_pool_b2 = nn.AdaptiveAvgPool2d((2, 40))

        fused_channels = 64 + 128 + 256

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Conv2d(fused_channels, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(256, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f_stem = self.stem(x)
        f_b1 = self.block1(f_stem)
        f_p1 = self.pool1(f_b1)
        f_b2 = self.block2(f_p1)
        f_b3 = self.block3(f_b2)
        f_p2 = self.pool2(f_b3)
        f_b4 = self.block4(f_p2)
        f_p3 = self.pool3(f_b4)

        f1 = self.global_pool_stem(f_stem)
        f2 = self.global_pool_b2(f_b2)
        f_fused = torch.cat([f1, f2, f_p3], dim=1)  # (B, 448, 2, 40)

        logits = self.classifier(f_fused)  # (B, num_classes, 2, 40)
        r0 = logits[:, :, 0, :]            # (B, num_classes, 40)
        r1 = logits[:, :, 1, :]            # (B, num_classes, 40)
        unrolled = torch.cat([r0, r1], dim=2)  # (B, num_classes, 80)
        return unrolled.permute(0, 2, 1)      # (B, 80, num_classes)


class Type1ACropDataset(Dataset):
    """
    In-RAM pre-cached dataset for Type 1A square plates (96x160 px).
    Loads both procedural synthetic and real road photos.
    """
    def __init__(
        self,
        meta_csv: str,
        root_dir: str,
        is_train: bool = True,
        val_split: float = 0.12,
        seed: int = 42,
    ):
        self.root_dir = root_dir
        self.rectifier = PlateRectifier()
        self.is_train = is_train

        synth_samples = []
        real_samples = []

        with open(meta_csv, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                if row.get("plate_type") != "type1a":
                    continue
                p_num = row.get("plate_num", "").strip()
                if len(p_num) < 8 or len(p_num) > 9:
                    continue

                item = {
                    "image": row["image"],
                    "plate_num": p_num,
                    "quad": row.get("quad", ""),
                    "bbox": row.get("bbox", ""),
                    "is_synth": row.get("is_synthetic") == "1",
                }
                if item["is_synth"]:
                    synth_samples.append(item)
                else:
                    real_samples.append(item)

        # Deterministic split
        random.seed(seed)
        random.shuffle(synth_samples)
        random.shuffle(real_samples)

        s_val_n = max(1, int(len(synth_samples) * val_split))
        r_val_n = max(1, int(len(real_samples) * val_split))

        synth_train = synth_samples[s_val_n:]
        synth_val = synth_samples[:s_val_n]

        real_train = real_samples[r_val_n:]
        real_val = real_samples[:r_val_n]

        if is_train:
            # 6x oversampling of real road samples to anchor real camera textures & wear
            oversampled_real = []
            for _ in range(6):
                oversampled_real.extend(real_train)
            self.items = synth_train + oversampled_real
            random.shuffle(self.items)
            print(f"[TRAIN 1A] Loaded {len(self.items)} samples ({len(synth_train)} synth + {len(oversampled_real)} oversampled real [{len(real_train)} unique])")
        else:
            self.items = synth_val + real_val
            print(f"[VAL 1A] Loaded {len(self.items)} samples ({len(synth_val)} synth + {len(real_val)} real)")

        # Pre-cache all crops in RAM (96x160x3 is only ~45 KB per sample)
        print(f"[*] Pre-caching {len(self.items)} Type 1A crops into RAM...")
        t0 = time.time()
        self.cached_crops = []
        for it in self.items:
            crop = self._load_and_rectify(it)
            self.cached_crops.append(crop)
        mb = len(self.cached_crops) * 96 * 160 * 3 / (1024 * 1024)
        print(f"[*] Pre-caching complete ({mb:.1f} MB in {time.time() - t0:.2f}s)")

    def _load_and_rectify(self, item: dict) -> np.ndarray:
        im_path = os.path.join(self.root_dir, item["image"])
        im = cv2.imread(im_path)
        if im is None:
            return np.zeros((96, 160, 3), dtype=np.uint8)

        quad = item["quad"]
        bbox = item["bbox"]
        try:
            if quad and len([v for v in quad.split(",") if v.strip()]) == 8:
                rect = self.rectifier.rectify(im, quad, plate_type="type1a")
            elif bbox and len([v for v in bbox.split(",") if v.strip()]) == 4:
                rect = self.rectifier.rectify_bbox(im, bbox, plate_type="type1a")
            else:
                rect = cv2.resize(im, (160, 96), interpolation=cv2.INTER_LINEAR)
            if rect.shape[:2] != (96, 160):
                rect = cv2.resize(rect, (160, 96), interpolation=cv2.INTER_LINEAR)
            return rect
        except Exception:
            return np.zeros((96, 160, 3), dtype=np.uint8)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        crop = self.cached_crops[idx].copy()
        item = self.items[idx]
        p_num = item["plate_num"]

        # Light training augmentations
        if self.is_train:
            # 1. Random contrast / brightness
            if random.random() < 0.5:
                alpha = random.uniform(0.75, 1.25)
                beta = random.uniform(-20, 20)
                crop = np.clip(crop.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
            # 2. Random slight blur
            if random.random() < 0.3:
                k = random.choice([3, 5])
                crop = cv2.GaussianBlur(crop, (k, k), 0)
            # 3. Additive sensor noise
            if random.random() < 0.3:
                noise = np.random.normal(0, 10, crop.shape).astype(np.float32)
                crop = np.clip(crop.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # Normalize BGR -> RGB float tensor (3, 96, 160)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)

        # Labels: top line = p_num[:4], bottom line = p_num[4:]
        top_str = p_num[:4]
        bot_str = p_num[4:]
        top_tokens = [CHAR2IDX.get(c, CHAR2IDX["#"]) for c in top_str]
        bot_tokens = [CHAR2IDX.get(c, CHAR2IDX["#"]) for c in bot_str]

        return tensor, top_tokens, bot_tokens, p_num


def collate_fn(batch):
    tensors, top_toks, bot_toks, p_nums = zip(*batch)
    tensors = torch.stack(tensors, dim=0)

    flat_top = []
    top_lens = []
    for t in top_toks:
        flat_top.extend(t)
        top_lens.append(len(t))

    flat_bot = []
    bot_lens = []
    for b in bot_toks:
        flat_bot.extend(b)
        bot_lens.append(len(b))

    flat_top_t = torch.tensor(flat_top, dtype=torch.long)
    top_lens_t = torch.tensor(top_lens, dtype=torch.long)
    flat_bot_t = torch.tensor(flat_bot, dtype=torch.long)
    bot_lens_t = torch.tensor(bot_lens, dtype=torch.long)

    return tensors, flat_top_t, top_lens_t, flat_bot_t, bot_lens_t, list(p_nums)


def decode_type1a_greedy(logits: np.ndarray) -> str:
    """Decodes 80-step unrolled logits into Type 1A string."""
    top_logits = logits[:40]
    top_preds = np.argmax(top_logits, axis=1)
    top_chars = []
    prev = -1
    for p in top_preds:
        if p != 0 and p != prev:
            top_chars.append(IDX2CHAR.get(p, ""))
        prev = p

    bot_logits = logits[40:]
    bot_preds = np.argmax(bot_logits, axis=1)
    bot_chars = []
    prev = -1
    for p in bot_preds:
        if p != 0 and p != prev:
            bot_chars.append(IDX2CHAR.get(p, ""))
        prev = p

    return "".join(top_chars) + "".join(bot_chars)


def levenshtein(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            ins = prev[j + 1] + 1
            dels = curr[j] + 1
            sub = prev[j] + (c1 != c2)
            curr.append(min(ins, dels, sub))
        prev = curr
    return prev[-1]


def evaluate(model: nn.Module, val_loader: DataLoader, device: str) -> Tuple[float, float, float]:
    model.eval()
    total_samples = 0
    correct_samples = 0
    total_chars = 0
    char_errors = 0

    with torch.no_grad():
        for tensors, _, _, _, _, gt_nums in val_loader:
            tensors = tensors.to(device)
            out = model(tensors)  # (B, 80, C)
            out_np = out.cpu().numpy()

            for i in range(len(gt_nums)):
                gt = gt_nums[i]
                pred = decode_type1a_greedy(out_np[i])

                if pred == gt:
                    correct_samples += 1

                ed = levenshtein(pred, gt)
                char_errors += ed
                total_chars += max(len(pred), len(gt))
                total_samples += 1

    seq_acc = (correct_samples / total_samples) * 100.0 if total_samples > 0 else 0.0
    cer = (char_errors / total_chars) * 100.0 if total_chars > 0 else 0.0
    return seq_acc, cer, correct_samples



def train(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print("=" * 65)
    print(f"🚀 Training LPRNet-2D on Type 1A Square Plates")
    print(f"   Device: {device}")
    print(f"   Epochs: {args.epochs} | Batch size: {args.batch_size} | LR: {args.lr}")
    print("=" * 65)

    meta_csv = os.path.join(args.data_dir, "meta.csv")
    train_dataset = Type1ACropDataset(meta_csv, args.data_dir, is_train=True)
    val_dataset = Type1ACropDataset(meta_csv, args.data_dir, is_train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = LPRNet2D(num_classes=NUM_CLASSES, dropout_rate=0.15).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    ctc = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)

    best_acc = 0.0
    best_cer = 100.0
    best_weights_path = os.path.join(args.output_dir, "ocr_lprnet_1a_best.pt")
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n[*] Starting training loop...")
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for tensors, flat_top, top_lens, flat_bot, bot_lens, _ in train_loader:
            tensors = tensors.to(device)
            flat_top = flat_top.to(device)
            top_lens = top_lens.to(device)
            flat_bot = flat_bot.to(device)
            bot_lens = bot_lens.to(device)

            bs = tensors.size(0)
            optimizer.zero_grad()

            logits = model(tensors)  # (B, 80, C)
            log_probs = logits.log_softmax(2).permute(1, 0, 2)  # (80, B, C)

            # Dual CTC Loss: Top line on steps 0..40, Bottom line on steps 40..80
            in_lens_top = torch.full((bs,), 40, dtype=torch.long, device=device)
            loss_top = ctc(log_probs[:40], flat_top, in_lens_top, top_lens)

            in_lens_bot = torch.full((bs,), 40, dtype=torch.long, device=device)
            loss_bot = ctc(log_probs[40:], flat_bot, in_lens_bot, bot_lens)

            loss = loss_top + loss_bot
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(1, num_batches)

        # Validation every 2 epochs and on final epoch
        if epoch % 2 == 0 or epoch == args.epochs:
            acc, cer, n_ok = evaluate(model, val_loader, device)
            is_best = acc > best_acc or (acc == best_acc and cer < best_cer)
            mark = " 🏆 BEST" if is_best else ""
            if is_best:
                best_acc = acc
                best_cer = cer
                torch.save({"state_dict": model.state_dict(), "acc": acc, "cer": cer}, best_weights_path)

            print(
                f"Epoch [{epoch:02d}/{args.epochs:02d}] | Loss: {avg_loss:6.3f} | "
                f"Val Acc: {acc:5.2f}% | CER: {cer:4.2f}% ({n_ok}/{len(val_dataset)}){mark}"
            )
        else:
            print(f"Epoch [{epoch:02d}/{args.epochs:02d}] | Loss: {avg_loss:6.3f}")

    total_time = time.time() - t_start
    print("\n" + "=" * 65)
    print(f"✅ Training completed in {total_time:.1f}s!")
    print(f"   Best Validation Seq Acc: {best_acc:.2f}% | Best CER: {best_cer:.2f}%")
    print(f"   Saved best checkpoint: {best_weights_path}")
    print("=" * 65)

    # Export to ONNX
    onnx_path = os.path.join(args.output_dir, "ocr_lprnet_1a.onnx")
    print(f"\n[*] Exporting Best Model to ONNX: {onnx_path}...")
    model.load_state_dict(torch.load(best_weights_path, map_location="cpu")["state_dict"])
    model.eval().cpu()

    dummy_input = torch.randn(1, 3, 96, 160)
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
    except Exception:
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

    print(f"✅ ONNX Export Successful: {onnx_path} ({os.path.getsize(onnx_path)/1024:.1f} KB)")

    # Validate ONNX inference
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    out = sess.run(None, {"input": np.random.randn(1, 3, 96, 160).astype(np.float32)})
    print(f"[PASS] ONNX Runtime Verification PASS: Output shape = {out[0].shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LPRNet-2D on Type 1A Russian Plates")
    parser.add_argument("--data_dir", type=str, default="dataset", help="Path to dataset directory")
    parser.add_argument("--output_dir", type=str, default="models", help="Output directory for weights")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--cpu", action="store_true", help="Force CPU training")
    args = parser.parse_args()

    train(args)
