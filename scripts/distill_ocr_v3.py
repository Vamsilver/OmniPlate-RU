#!/usr/bin/env python3
"""
scripts/distill_ocr_v3.py
Knowledge Distillation for LPRNet-v3 (Teacher-Student):
- Teacher: Pre-trained LPRNet-v2 (models/ocr_lprnet_best.pt, 96.5% accuracy)
- Student: LPRNet-v3 with 1D-ASPP + ECA-Net Attention
Transfers fine-grained soft probability distributions across all 40 time steps,
enabling LPRNet-v3 to surpass the teacher while maintaining sub-2ms latency.
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
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from src.pipeline.ocr import (
    BLANK_IDX,
    CHAR2IDX,
    NUM_CLASSES,
    CTCDecoder,
    LPRNet,
)
from scripts.train_ocr_v3 import LPRNetV3, PlateCropDataset, collate_fn


def main():
    parser = argparse.ArgumentParser(description="Distill Knowledge from LPRNet-v2 into LPRNet-v3")
    parser.add_argument("--epochs", type=int, default=15, help="Distillation epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--temperature", type=float, default=2.0, help="Distillation softmax temperature")
    parser.add_argument("--alpha", type=float, default=0.4, help="Weight for KL-divergence loss (0.0 to 1.0)")
    parser.add_argument("--teacher-ckpt", type=str, default="models/ocr_lprnet_best.pt")
    parser.add_argument("--output-dir", type=str, default="models")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Distillation active on device: {device}")

    # 1. Load Teacher (LPRNet-v2)
    teacher_path = os.path.join(ROOT_DIR, args.teacher_ckpt) if not os.path.isabs(args.teacher_ckpt) else args.teacher_ckpt
    if not os.path.exists(teacher_path):
        raise FileNotFoundError(f"Teacher checkpoint not found at: {teacher_path}")

    print(f"[+] Loading Teacher model from {teacher_path}...")
    teacher = LPRNet(num_classes=NUM_CLASSES, dropout_rate=0.0, dilated=True).to(device)
    t_ckpt = torch.load(teacher_path, map_location=device)
    teacher.load_state_dict(t_ckpt.get("state_dict", t_ckpt))
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    print("[+] Teacher loaded and frozen successfully.")

    # 2. Initialize Student (LPRNet-v3)
    student = LPRNetV3(num_classes=NUM_CLASSES, dropout_rate=0.2).to(device)
    student_ckpt_init = os.path.join(ROOT_DIR, "models", "ocr_lprnet_v3.pt")
    if os.path.exists(student_ckpt_init):
        s_ckpt = torch.load(student_ckpt_init, map_location=device)
        student.load_state_dict(s_ckpt.get("state_dict", s_ckpt))
        print(f"[+] Initialized Student from {student_ckpt_init}")
    else:
        # Smart warm-start from teacher
        t_st = teacher.state_dict()
        s_st = student.state_dict()
        for k, v in t_st.items():
            if k in s_st and s_st[k].shape == v.shape:
                s_st[k] = v
        student.load_state_dict(s_st)
        print("[+] Initialized Student with Teacher backbone weights.")

    # 3. Datasets
    meta_csv = os.path.join(ROOT_DIR, "dataset", "meta.csv")
    dataset_dir = os.path.join(ROOT_DIR, "dataset")

    print("[+] Preparing pre-cached training dataset...")
    train_ds = PlateCropDataset(meta_csv, dataset_dir, is_train=True, cache_in_ram=True)
    val_ds = PlateCropDataset(meta_csv, dataset_dir, is_train=False, cache_in_ram=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # 4. Optimization & Losses
    ctc_loss = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)
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

    best_acc = 0.0
    best_student_pt = os.path.join(ROOT_DIR, args.output_dir, "ocr_lprnet_v3.pt")
    best_student_onnx = os.path.join(ROOT_DIR, args.output_dir, "ocr_lprnet_v3.onnx")

    T = args.temperature
    alpha = args.alpha

    print(f"\n[+] Starting Knowledge Distillation (T={T}, alpha={alpha}, epochs={args.epochs})...")
    for epoch in range(1, args.epochs + 1):
        student.train()
        total_train_loss = 0.0
        total_ctc_loss = 0.0
        total_kd_loss = 0.0
        start_t = time.time()

        for tensors, flat_targets, target_lens, _, _ in train_loader:
            tensors = tensors.to(device)
            flat_targets = flat_targets.to(device)

            optimizer.zero_grad()

            # Forward Teacher (no grad)
            with torch.no_grad():
                t_logits = teacher(tensors)  # (B, 40, num_classes)
                p_teacher = F.softmax(t_logits / T, dim=-1)

            # Forward Student
            s_logits = student(tensors)      # (B, 40, num_classes)
            log_p_student = F.log_softmax(s_logits / T, dim=-1)

            # 1. CTC Loss
            log_probs_s = s_logits.log_softmax(2).permute(1, 0, 2)  # (T, B, C)
            input_lens = torch.full((tensors.size(0),), s_logits.size(1), dtype=torch.long, device=device)
            loss_c = ctc_loss(log_probs_s, flat_targets, input_lens, target_lens.to(device))

            # 2. KL-Divergence Distillation Loss
            loss_k = F.kl_div(log_p_student, p_teacher, reduction="batchmean") * (T * T)

            # Combined Loss
            loss = (1.0 - alpha) * loss_c + alpha * loss_k

            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()
            total_ctc_loss += loss_c.item()
            total_kd_loss += loss_k.item()

        n_batches = max(1, len(train_loader))
        train_loss = total_train_loss / n_batches
        avg_ctc = total_ctc_loss / n_batches
        avg_kd = total_kd_loss / n_batches

        # Validation
        student.eval()
        val_correct = 0
        val_total = 0
        val_char_correct = 0
        val_char_total = 0
        by_type_correct = {"type1": 0, "type1a": 0, "type1b": 0, "type2": 0}
        by_type_total = {"type1": 0, "type1a": 0, "type1b": 0, "type2": 0}

        with torch.no_grad():
            for tensors, _, _, ground_truths, plate_types in val_loader:
                tensors = tensors.to(device)
                s_logits = student(tensors)
                preds = s_logits.argmax(dim=-1).cpu().numpy()

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
        print(f"Distill Epoch [{epoch:02d}/{args.epochs:02d}] ({elapsed:.1f}s) | "
              f"Loss: {train_loss:.4f} (CTC: {avg_ctc:.4f}, KD: {avg_kd:.4f}) | "
              f"Val SeqAcc: {seq_acc:.2f}% (Char: {char_acc:.2f}%) | "
              f"T1: {acc_t1:.1f}% | 1A: {acc_t1a:.1f}% | 1B: {acc_t1b:.1f}% | T2: {acc_t2:.1f}%")

        if seq_acc >= best_acc:
            best_acc = seq_acc
            torch.save({
                "epoch": epoch,
                "state_dict": student.state_dict(),
                "seq_acc": seq_acc,
                "char_acc": char_acc,
                "arch": "LPRNetV3_Distilled",
            }, best_student_pt)
            print(f"  [*] Saved new best distilled student checkpoint: {best_student_pt} (SeqAcc: {seq_acc:.2f}%)")

    # Export to monolithic ONNX
    print("\n[+] Exporting Best Distilled LPRNet-v3 to ONNX...")
    student.eval()
    if os.path.exists(best_student_pt):
        ckpt = torch.load(best_student_pt, map_location=device)
        student.load_state_dict(ckpt["state_dict"])

    dummy_input = torch.randn(1, 3, 36, 160, device=device)
    torch.onnx.export(
        student,
        dummy_input,
        best_student_onnx,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=17,
        dynamo=False,
    )
    print(f"[+] Distilled LPRNet-v3 ONNX saved to {best_student_onnx} ({os.path.getsize(best_student_onnx) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
