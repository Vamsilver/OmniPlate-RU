#!/usr/bin/env python3
"""
OmniPlate-RU — Isolated MoE Fine-Tuning: LPRNet-2D for Type 1A Square Plates.
Volga IT 2026 — Iteration: 1A Specialization.

Key improvements over v1:
- Integrated 2888 verified_crops from manifest.csv (36×160 stitched → unstitch to 96×160)
- 8x oversampling of real road scenes (vs 6x previously)
- OneCycleLR scheduler instead of CosineAnnealing
- Heavy augmentations: Motion Blur, Defocus Blur, Perspective Warp, CLAHE, Salt-and-Pepper
- Warm-start from best existing 1A checkpoint (ocr_lprnet_1a_best.pt, 96.01% val acc)
- Auto-log to train_ocr_1a.log for monitoring
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


# ---------------------------------------------------------------------------
# Architecture: LPRNet-2D (unchanged — mirrors scripts/train_ocr_1a.py v1)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Augmentation helpers
# ---------------------------------------------------------------------------

def apply_motion_blur(img: np.ndarray, max_kernel: int = 9, max_angle: float = 35.0) -> np.ndarray:
    """Applies random directional motion blur."""
    k = random.choice(range(3, max_kernel + 1, 2))
    angle = random.uniform(-max_angle, max_angle)
    M = cv2.getRotationMatrix2D((k // 2, k // 2), angle, 1)
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0
    kernel = cv2.warpAffine(kernel, M, (k, k))
    s = kernel.sum()
    kernel = kernel / s if s > 0 else kernel
    return cv2.filter2D(img, -1, kernel)


def apply_defocus_blur(img: np.ndarray, max_radius: int = 5) -> np.ndarray:
    """Applies defocus (disk) blur."""
    r = random.choice(range(1, max_radius + 1))
    k = 2 * r + 1
    kernel = np.zeros((k, k), dtype=np.float32)
    cv2.circle(kernel, (r, r), r, 1.0, -1)
    s = kernel.sum()
    kernel = kernel / s if s > 0 else kernel
    return cv2.filter2D(img, -1, kernel)


def apply_perspective_jitter(img: np.ndarray, strength: float = 0.04) -> np.ndarray:
    """Applies random perspective distortion."""
    h, w = img.shape[:2]
    dx = w * strength
    dy = h * strength
    pts1 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    pts2 = pts1 + np.random.uniform(-dx, dx, pts1.shape).astype(np.float32)
    pts2[:, 0] = np.clip(pts2[:, 0], 0, w - 1)
    pts2[:, 1] = np.clip(pts2[:, 1], 0, h - 1)
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def apply_clahe(img: np.ndarray) -> np.ndarray:
    """Applies adaptive CLAHE on dark crops."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 8))
    l_eq = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2BGR)


def apply_heavy_augmentations(crop: np.ndarray) -> np.ndarray:
    """Full set of heavy augmentations for Type 1A fine-tuning."""
    # 1. Perspective jitter (always-on with prob 0.45)
    if random.random() < 0.45:
        crop = apply_perspective_jitter(crop, strength=random.uniform(0.02, 0.05))

    # 2. Brightness & contrast
    if random.random() < 0.65:
        alpha = random.uniform(0.65, 1.35)
        beta = random.uniform(-25, 25)
        crop = np.clip(crop.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    # 3. Blur family (motion or defocus or gaussian)
    r = random.random()
    if r < 0.25:
        crop = apply_motion_blur(crop, max_kernel=9, max_angle=30.0)
    elif r < 0.42:
        crop = apply_defocus_blur(crop, max_radius=3)
    elif r < 0.55:
        k = random.choice([3, 5])
        crop = cv2.GaussianBlur(crop, (k, k), 0)

    # 4. Sensor noise
    if random.random() < 0.40:
        std = random.uniform(5, 18)
        noise = np.random.normal(0, std, crop.shape).astype(np.float32)
        crop = np.clip(crop.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # 5. CLAHE on dark crops
    mean_l = float(np.mean(cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)[:, :, 0]))
    if random.random() < 0.25 and mean_l < 110:
        crop = apply_clahe(crop)

    # 6. Salt-and-pepper (simulates dirty lens / camera artifact)
    if random.random() < 0.15:
        mask = np.random.rand(*crop.shape[:2])
        crop[mask < 0.005] = 0
        crop[mask > 0.995] = 255

    return crop


def unstitch_36x160_to_96x160(stitched: np.ndarray) -> np.ndarray:
    """
    Reconstructs a 96×160 full Type 1A plate from a 36×160 stitched crop.
    The stitched format stores: top_line (left 80 cols) | bot_line (right 80 cols).
    Each half is 36×80 (compressed from original 48×160 → resized to 36×80).

    Reconstruction:
      1. top_shrunk = stitched[:, :80]   → resize to 48×160 (top line)
      2. bot_shrunk = stitched[:, 80:]   → resize to 48×160 (bot line)
      3. stacked = np.vstack([top, bot]) → 96×160 (canonical input for LPRNet2D)
    """
    if stitched.shape[:2] != (36, 160):
        stitched = cv2.resize(stitched, (160, 36), interpolation=cv2.INTER_LINEAR)

    top_shrunk = stitched[:, :80]   # (36, 80)
    bot_shrunk = stitched[:, 80:]   # (36, 80)

    # Restore each line to its full 48×160 canonical height
    top_line = cv2.resize(top_shrunk, (160, 48), interpolation=cv2.INTER_LINEAR)
    bot_line = cv2.resize(bot_shrunk, (160, 48), interpolation=cv2.INTER_LINEAR)

    return np.vstack([top_line, bot_line])  # (96, 160, 3)


# ---------------------------------------------------------------------------
# Dataset: Isolated Type 1A — meta.csv scenes + manifest.csv hard crops
# ---------------------------------------------------------------------------

class Type1ADatasetMoE(Dataset):
    """
    Isolated MoE Type 1A training dataset:
    1. Real road scenes from meta.csv (318 entries) with homography rectification → 8x oversample.
    2. Synth scenes from meta.csv (2000 GOST-rendered entries).
    3. Hard crops from dataset/verified_crops/manifest.csv (2888 entries, 36×160 stitched → unstitch).
    All samples are pre-cached in RAM as 96×160×3 BGR arrays.
    """

    REAL_OVERSAMPLE = 8  # MoE specialization: heavy real-domain anchoring

    def __init__(
        self,
        meta_csv: str,
        root_dir: str,
        manifest_csv: Optional[str] = None,
        is_train: bool = True,
        val_split: float = 0.12,
        seed: int = 42,
    ):
        self.root_dir = root_dir
        self.crops_dir = os.path.join(root_dir, "verified_crops")
        self.rectifier = PlateRectifier()
        self.is_train = is_train

        # ---------- A. Load from meta.csv ----------
        synth_meta: List[Dict] = []
        real_meta: List[Dict] = []
        with open(meta_csv, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                if row.get("plate_type") != "type1a":
                    continue
                p_num = row.get("plate_num", "").strip()
                if len(p_num) < 8 or len(p_num) > 9:
                    continue
                item = {
                    "source": "meta",
                    "image": row["image"],
                    "plate_num": p_num,
                    "quad": row.get("quad", ""),
                    "bbox": row.get("bbox", ""),
                    "is_synth": row.get("is_synthetic") == "1",
                }
                if item["is_synth"]:
                    synth_meta.append(item)
                else:
                    real_meta.append(item)

        # ---------- B. Load from manifest.csv (verified_crops) ----------
        synth_crops: List[Dict] = []
        real_crops: List[Dict] = []
        if manifest_csv and os.path.exists(manifest_csv):
            REAL_SOURCES = {"curated_real_1a", "user_audit_corrected", "roboflow_1a_verified"}
            with open(manifest_csv, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.reader(f, delimiter=";")
                next(reader, None)
                for row in reader:
                    if len(row) < 3 or row[2] != "type1a":
                        continue
                    c_file, p_num = row[0], row[1].strip()
                    src = row[4].strip() if len(row) > 4 else ""
                    if len(p_num) < 8 or len(p_num) > 9:
                        continue
                    fpath = os.path.join(self.crops_dir, c_file)
                    if not os.path.exists(fpath):
                        continue
                    item = {
                        "source": "manifest",
                        "crop_file": c_file,
                        "plate_num": p_num,
                        "src_tag": src,
                    }
                    if src in REAL_SOURCES:
                        real_crops.append(item)
                    else:
                        synth_crops.append(item)

        # ---------- C. Train/Val split (deterministic) ----------
        random.seed(seed)
        for lst in [synth_meta, real_meta, synth_crops, real_crops]:
            random.shuffle(lst)

        def split_list(lst, frac):
            n = max(1, int(len(lst) * frac))
            return lst[n:], lst[:n]  # train, val

        sm_tr, sm_val = split_list(synth_meta, val_split)
        rm_tr, rm_val = split_list(real_meta, val_split)
        sc_tr, sc_val = split_list(synth_crops, val_split)
        rc_tr, rc_val = split_list(real_crops, val_split)

        if is_train:
            # 8x oversampling of ALL real-domain items
            real_all_train = rm_tr + rc_tr
            oversampled_real = real_all_train * self.REAL_OVERSAMPLE
            self.items = sm_tr + sc_tr + oversampled_real
            random.shuffle(self.items)
            print(
                f"[TRAIN 1A MoE] {len(self.items)} samples total:\n"
                f"   synth_meta={len(sm_tr)} | synth_crops={len(sc_tr)}\n"
                f"   real_meta={len(rm_tr)} | real_crops={len(rc_tr)}\n"
                f"   oversampled_real (×{self.REAL_OVERSAMPLE}) = {len(oversampled_real)}"
            )
        else:
            self.items = sm_val + sc_val + rm_val + rc_val
            print(
                f"[VAL 1A MoE] {len(self.items)} samples: "
                f"synth={len(sm_val)+len(sc_val)}, real={len(rm_val)+len(rc_val)}"
            )

        # ---------- D. Pre-cache into RAM ----------
        print(f"[*] Pre-caching {len(self.items)} Type 1A crops into RAM...")
        t0 = time.time()
        self.cached_crops: List[np.ndarray] = []
        skipped = 0
        for it in self.items:
            crop = self._load_crop(it)
            self.cached_crops.append(crop)
            if crop is None or crop.sum() == 0:
                skipped += 1
        mb = len(self.cached_crops) * 96 * 160 * 3 / (1024 * 1024)
        print(f"[*] Cache complete: {mb:.1f} MB in {time.time()-t0:.2f}s (skipped {skipped} empty)")

    def _load_crop(self, item: dict) -> np.ndarray:
        """Loads and normalizes a single item to 96×160 BGR."""
        fallback = np.zeros((96, 160, 3), dtype=np.uint8)

        try:
            if item["source"] == "manifest":
                fpath = os.path.join(self.crops_dir, item["crop_file"])
                img = cv2.imread(fpath)
                if img is None:
                    return fallback
                # Unstitch 36×160 → 96×160
                if img.shape[:2] == (36, 160):
                    img = unstitch_36x160_to_96x160(img)
                elif img.shape[:2] != (96, 160):
                    img = cv2.resize(img, (160, 96), interpolation=cv2.INTER_LINEAR)
                return img

            else:  # "meta" source (full-scene)
                im_path = os.path.join(self.root_dir, item["image"])
                im = cv2.imread(im_path)
                if im is None:
                    return fallback
                quad = item.get("quad", "")
                bbox = item.get("bbox", "")
                if quad and len([v for v in quad.split(",") if v.strip()]) == 8:
                    rect = self.rectifier.rectify(im, quad, plate_type="type1a",
                                                   margin=(0.020, 0.015), refine_corners=True)
                elif bbox and len([v for v in bbox.split(",") if v.strip()]) == 4:
                    rect = self.rectifier.rectify_bbox(im, bbox, plate_type="type1a")
                else:
                    rect = cv2.resize(im, (160, 96), interpolation=cv2.INTER_LINEAR)
                if rect.shape[:2] != (96, 160):
                    rect = cv2.resize(rect, (160, 96), interpolation=cv2.INTER_LINEAR)
                return rect
        except Exception:
            return fallback

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        crop = self.cached_crops[idx].copy()
        p_num = self.items[idx]["plate_num"]

        if self.is_train:
            crop = apply_heavy_augmentations(crop)

        # Normalize BGR -> RGB float tensor (3, 96, 160)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)

        # Labels: top line = p_num[:4], bottom line = p_num[4:]
        top_str = p_num[:4]
        bot_str = p_num[4:]
        top_tokens = [CHAR2IDX.get(c, CHAR2IDX["#"]) for c in top_str]
        bot_tokens = [CHAR2IDX.get(c, CHAR2IDX["#"]) for c in bot_str]

        return tensor, top_tokens, bot_tokens, p_num


# ---------------------------------------------------------------------------
# Collate function (unchanged)
# ---------------------------------------------------------------------------

def collate_fn(batch):
    tensors, top_toks, bot_toks, p_nums = zip(*batch)
    tensors = torch.stack(tensors, dim=0)

    flat_top, top_lens = [], []
    for t in top_toks:
        flat_top.extend(t)
        top_lens.append(len(t))

    flat_bot, bot_lens = [], []
    for b in bot_toks:
        flat_bot.extend(b)
        bot_lens.append(len(b))

    return (
        tensors,
        torch.tensor(flat_top, dtype=torch.long),
        torch.tensor(top_lens, dtype=torch.long),
        torch.tensor(flat_bot, dtype=torch.long),
        torch.tensor(bot_lens, dtype=torch.long),
        list(p_nums),
    )


# ---------------------------------------------------------------------------
# Evaluation (unchanged)
# ---------------------------------------------------------------------------

def decode_type1a_greedy(logits: np.ndarray) -> str:
    """Decodes 80-step unrolled logits into Type 1A string."""
    result_chars = []
    for segment in [logits[:40], logits[40:]]:
        preds = np.argmax(segment, axis=1)
        prev = -1
        for p in preds:
            if p != 0 and p != prev:
                result_chars.append(IDX2CHAR.get(int(p), ""))
            prev = p
    return "".join(result_chars)


def levenshtein(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
        prev = curr
    return prev[-1]


def evaluate(model: nn.Module, val_loader: DataLoader, device: str) -> Tuple[float, float, int]:
    model.eval()
    total, correct, total_chars, char_errors = 0, 0, 0, 0

    with torch.no_grad():
        for tensors, _, _, _, _, gt_nums in val_loader:
            tensors = tensors.to(device)
            out = model(tensors).cpu().numpy()
            for i, gt in enumerate(gt_nums):
                pred = decode_type1a_greedy(out[i])
                if pred == gt:
                    correct += 1
                char_errors += levenshtein(pred, gt)
                total_chars += max(len(pred), len(gt))
                total += 1

    seq_acc = (correct / total) * 100.0 if total > 0 else 0.0
    cer = (char_errors / total_chars) * 100.0 if total_chars > 0 else 0.0
    return seq_acc, cer, correct


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    # Tee stdout to log file
    log_path = os.path.join(ROOT_DIR, "train_ocr_1a.log")
    log_fh = open(log_path, "w", encoding="utf-8", buffering=1)

    class Tee:
        def __init__(self, *streams): self.streams = streams
        def write(self, data):
            for s in self.streams:
                s.write(data)
        def flush(self):
            for s in self.streams:
                s.flush()

    sys.stdout = Tee(sys.__stdout__, log_fh)

    print("=" * 70)
    print("🚀 LPRNet-2D Isolated MoE Fine-Tuning: Type 1A Square Plates")
    print(f"   Baseline: 71.7% SeqAcc / CER 13.25% on 318 real scenes (benchmark)")
    print(f"   Device: {device} | Epochs: {args.epochs} | Batch: {args.batch_size} | LR: {args.lr}")
    print(f"   Real oversample: ×{Type1ADatasetMoE.REAL_OVERSAMPLE} | Scheduler: OneCycleLR")
    print("=" * 70)

    meta_csv = os.path.join(args.data_dir, "meta.csv")
    manifest_csv = os.path.join(args.data_dir, "verified_crops", "manifest.csv")

    train_ds = Type1ADatasetMoE(meta_csv, args.data_dir, manifest_csv=manifest_csv, is_train=True)
    val_ds   = Type1ADatasetMoE(meta_csv, args.data_dir, manifest_csv=manifest_csv, is_train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=0,  # single-process: data already in RAM
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    model = LPRNet2D(num_classes=NUM_CLASSES, dropout_rate=0.15).to(device)

    # Warm-start from best existing 1A checkpoint
    ckpt_1a = os.path.join(args.output_dir, "ocr_lprnet_1a_best.pt")
    if os.path.exists(ckpt_1a):
        try:
            ckpt = torch.load(ckpt_1a, map_location=device)
            sd = ckpt.get("state_dict", ckpt)
            model.load_state_dict(sd, strict=True)
            print(f"[+] Warm-Start from {ckpt_1a} (acc={ckpt.get('acc', '?'):.2f}%)")
        except Exception as e:
            print(f"[!] Strict load failed, trying partial warm-start: {e}")
            try:
                ckpt = torch.load(ckpt_1a, map_location=device)
                sd = ckpt.get("state_dict", ckpt)
                m_sd = model.state_dict()
                loaded = sum(1 for k, v in sd.items() if k in m_sd and m_sd[k].shape == v.shape and not m_sd.update({k: v}))
                model.load_state_dict(m_sd)
                print(f"[+] Partial warm-start: {loaded}/{len(m_sd)} tensors loaded")
            except Exception as e2:
                print(f"[!] Warm-start skipped entirely: {e2}")
    else:
        print(f"[!] No 1A checkpoint found at {ckpt_1a} — training from scratch")

    # OneCycleLR: peak lr at cycle top, rapid cosine decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr / 10, weight_decay=5e-5)
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.15,          # warm-up 15% of total steps
        anneal_strategy="cos",
        div_factor=10.0,         # start lr = max_lr / 10
        final_div_factor=500.0,  # end lr = max_lr / 5000
    )
    ctc = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)

    best_acc, best_cer = 0.0, 100.0
    best_weights_path = os.path.join(args.output_dir, "ocr_lprnet_1a_best.pt")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n[*] Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")
    print(f"[*] Steps/epoch: {steps_per_epoch} | Starting training loop...\n")
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, num_batches = 0.0, 0

        for tensors, flat_top, top_lens, flat_bot, bot_lens, _ in train_loader:
            tensors  = tensors.to(device)
            flat_top = flat_top.to(device)
            top_lens = top_lens.to(device)
            flat_bot = flat_bot.to(device)
            bot_lens = bot_lens.to(device)

            bs = tensors.size(0)
            optimizer.zero_grad()

            logits   = model(tensors)                        # (B, 80, C)
            log_prob = logits.log_softmax(2).permute(1, 0, 2)  # (80, B, C)

            in_top = torch.full((bs,), 40, dtype=torch.long, device=device)
            in_bot = torch.full((bs,), 40, dtype=torch.long, device=device)
            loss_top = ctc(log_prob[:40], flat_top, in_top, top_lens)
            loss_bot = ctc(log_prob[40:], flat_bot, in_bot, bot_lens)

            loss = loss_top + loss_bot
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        cur_lr   = scheduler.get_last_lr()[0]

        # Validate every epoch
        acc, cer, n_ok = evaluate(model, val_loader, device)
        is_best = acc > best_acc or (acc == best_acc and cer < best_cer)
        mark = " 🏆 BEST" if is_best else ""
        if is_best:
            best_acc = acc
            best_cer = cer
            torch.save({"state_dict": model.state_dict(), "acc": acc, "cer": cer}, best_weights_path)

        elapsed = time.time() - t_start
        eta = (elapsed / epoch) * (args.epochs - epoch) if epoch > 0 else 0
        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] | Loss: {avg_loss:6.4f} | LR: {cur_lr:.2e} | "
            f"Val Acc: {acc:5.2f}% | CER: {cer:4.2f}% ({n_ok}/{len(val_ds)}) | "
            f"ETA: {eta/60:.1f}m{mark}"
        )

    total_time = time.time() - t_start
    print("\n" + "=" * 70)
    print(f"✅ Training completed in {total_time:.1f}s ({total_time/60:.1f}m)!")
    print(f"   Best Validation Seq Acc: {best_acc:.2f}% | Best CER: {best_cer:.2f}%")
    print(f"   Saved best checkpoint: {best_weights_path}")
    print("=" * 70)

    # ---------- ONNX Export ----------
    onnx_path = os.path.join(args.output_dir, "ocr_lprnet_1a.onnx")
    print(f"\n[*] Exporting best model to ONNX: {onnx_path} ...")
    ckpt_best = torch.load(best_weights_path, map_location="cpu")
    model.load_state_dict(ckpt_best["state_dict"])
    model.eval().cpu()

    dummy = torch.randn(1, 3, 96, 160)
    try:
        torch.onnx.export(
            model, dummy, onnx_path,
            export_params=True, opset_version=14, do_constant_folding=True,
            input_names=["input"], output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            dynamo=False,
        )
    except TypeError:
        torch.onnx.export(
            model, dummy, onnx_path,
            export_params=True, opset_version=14, do_constant_folding=True,
            input_names=["input"], output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )

    sz_kb = os.path.getsize(onnx_path) / 1024
    print(f"✅ ONNX Export: {onnx_path} ({sz_kb:.1f} KB)")

    # Validate ONNX
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    out = sess.run(None, {"input": np.random.randn(1, 3, 96, 160).astype(np.float32)})
    print(f"[PASS] ONNX Runtime Verification: output shape = {out[0].shape}")

    sys.stdout = sys.__stdout__
    log_fh.close()
    print(f"[*] Training log saved to: {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LPRNet-2D Isolated MoE Fine-Tuning for Type 1A")
    parser.add_argument("--data_dir",    type=str, default="dataset",  help="Dataset root dir")
    parser.add_argument("--output_dir",  type=str, default="models",   help="Weights output dir")
    parser.add_argument("--epochs",      type=int, default=18,         help="Training epochs")
    parser.add_argument("--batch_size",  type=int, default=64,         help="Batch size")
    parser.add_argument("--lr",          type=float, default=5e-4,     help="OneCycleLR max LR")
    parser.add_argument("--cpu",         action="store_true",          help="Force CPU")
    args = parser.parse_args()
    train(args)
