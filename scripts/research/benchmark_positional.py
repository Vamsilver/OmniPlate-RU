#!/usr/bin/env python3
"""
scripts/research/benchmark_positional.py
=========================================
Вектор 3 — Positional Multi-Head Classifier (PositionalPlateNet) Benchmark.

Методология:
  1. Загрузка датасета через PlateCropDataset (train/val split, val_split=0.15).
  2. Обёртка PositionalDataset: Latin→Cyrillic конвертация + encode_plate_positional().
  3. Минимальное обучение: 15 эпох, AdamW lr=1e-3, OneCycleLR, batch=64, GPU.
  4. Best checkpoint по val loss → eval на val-сплите.
  5. Метрики: SeqAcc (exact match), CER, параметры модели, ONNX size, GPU/CPU latency.
  6. Сохранение: test_output/positional_benchmark_report.json + models/positional_net.onnx.

Сравнение с парадигмами:
  Solo v2:  SeqAcc 95.30%, CER 1.230%  (moe_benchmark_report.json)
  Solo v3:  SeqAcc 95.68%, CER 1.319%  (moe_benchmark_report.json)
  MoE:      SeqAcc 95.93%, CER 1.211%  (moe_benchmark_report.json)
  Micro-STN: SeqAcc 95.33% clean / collapses at Roll>6° (stn_robustness_report.json)
  Positional: TBD (этот скрипт)
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True, encoding="utf-8")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT_DIR = r"D:\AIProjects\VolgaIT"
sys.path.insert(0, ROOT_DIR)

from scripts.train_ocr_v3 import PlateCropDataset
from src.pipeline.positional_net import (
    LETTER_VOCAB,
    DIGIT_VOCAB,
    REGION_FIRST_VOCAB,
    REGION_VOCAB,
    POSITION_SPEC,
    PositionalCELoss,
    PositionalPlateNet,
    decode_positional,
    encode_plate_positional,
    export_positional_onnx,
)

# ─── Конвертация Latin → Cyrillic (обратный CYR_TO_LAT из train_ocr_v3.py) ──────
LAT_TO_CYR: Dict[str, str] = {
    "A": "А", "B": "В", "E": "Е", "K": "К", "M": "М",
    "H": "Н", "O": "О", "P": "Р", "C": "С", "T": "Т",
    "Y": "У", "X": "Х",
}


def lat_to_cyr(text: str) -> str:
    """Конвертирует строку из латинской транслитерации в кириллицу для ГОСТ-номеров."""
    return "".join(LAT_TO_CYR.get(ch, ch) for ch in text.upper())


def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            current_row.append(min(
                previous_row[j + 1] + 1,
                current_row[j] + 1,
                previous_row[j] + (c1 != c2),
            ))
        previous_row = current_row
    return previous_row[-1]


# ─── Positional Dataset Wrapper ───────────────────────────────────────────────

class PositionalDataset(Dataset):
    """
    Обёртка над PlateCropDataset для PositionalPlateNet.

    - Конвертирует clean_text из латинской транслитерации в кириллицу.
    - Вызывает encode_plate_positional() для получения (9,) таргета.
    - Фильтрует примеры с None-таргетом (нераспознанные форматы).
    - При инициализации выполняется предфильтрация индексов.
    """

    def __init__(self, base_dataset: PlateCropDataset):
        self.base = base_dataset
        # Предфильтрация: оставляем только те индексы, для которых энкодер не вернул None
        self.valid_indices: List[int] = []
        n_skipped = 0
        for i in range(len(base_dataset)):
            try:
                item = base_dataset[i]
                if item is None:
                    n_skipped += 1
                    continue
                _, clean_text, _ = item
                cyr_text = lat_to_cyr(clean_text)
                target = encode_plate_positional(cyr_text)
                if target is not None:
                    self.valid_indices.append(i)
                else:
                    n_skipped += 1
            except Exception:
                n_skipped += 1
        print(f"  PositionalDataset: {len(self.valid_indices)} valid / {len(base_dataset)} total "
              f"({n_skipped} skipped, non-encodable plates)")

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        real_idx = self.valid_indices[idx]
        tensor, clean_text, plate_type = self.base[real_idx]
        cyr_text = lat_to_cyr(clean_text)
        target = encode_plate_positional(cyr_text)
        return tensor, target, cyr_text


def collate_positional(batch):
    tensors, targets, texts = zip(*batch)
    tensors = torch.stack(tensors, dim=0)
    targets = torch.stack(targets, dim=0)
    return tensors, targets, list(texts)


# ─── Training ─────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: PositionalPlateNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    criterion: PositionalCELoss,
    device: torch.device,
    epoch: int,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch_idx, (images, targets, _) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        n_batches += 1

        if (batch_idx + 1) % 50 == 0:
            print(f"    Epoch {epoch:02d} | Batch {batch_idx+1}/{len(loader)} | "
                  f"Loss {loss.item():.4f}")

    return total_loss / max(1, n_batches)


@torch.no_grad()
def evaluate(
    model: PositionalPlateNet,
    loader: DataLoader,
    criterion: PositionalCELoss,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    exact_matches = 0
    total_samples = 0
    total_edit_dist = 0
    total_gt_chars = 0

    for images, targets, gt_texts in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)
        total_loss += loss.item()
        n_batches += 1

        # Decode predictions
        pred_strs = decode_positional(logits)

        for pred, gt in zip(pred_strs, gt_texts):
            total_samples += 1
            # Для сравнения конвертируем gt обратно в латиницу (ASCII-совместимо)
            # Нет — decode_positional возвращает кириллицу, gt уже кириллица
            if pred == gt:
                exact_matches += 1
            dist = levenshtein_distance(pred, gt)
            total_edit_dist += dist
            total_gt_chars += len(gt)

    seq_acc = (exact_matches / max(1, total_samples)) * 100.0
    cer = (total_edit_dist / max(1, total_gt_chars)) * 100.0
    avg_loss = total_loss / max(1, n_batches)

    return {
        "val_loss": round(avg_loss, 4),
        "seq_acc_pct": round(seq_acc, 2),
        "cer_pct": round(cer, 3),
        "exact_matches": exact_matches,
        "total_samples": total_samples,
    }


# ─── Latency Benchmark ────────────────────────────────────────────────────────

@torch.no_grad()
def benchmark_pytorch_latency(
    model: PositionalPlateNet,
    device: torch.device,
    num_iters: int = 200,
    warmup: int = 30,
) -> Dict[str, float]:
    """Измерение latency PyTorch-инференса (batch=1)."""
    model.eval()
    dummy = torch.randn(1, 3, 36, 160, device=device)

    # Warmup
    for _ in range(warmup):
        _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()

    latencies = []
    for _ in range(num_iters):
        t0 = time.perf_counter()
        _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000.0)

    import numpy as np
    latencies = sorted(latencies)
    return {
        "mean_ms": round(float(np.mean(latencies)), 3),
        "p50_ms":  round(float(np.percentile(latencies, 50)), 3),
        "p95_ms":  round(float(np.percentile(latencies, 95)), 3),
        "min_ms":  round(float(np.min(latencies)), 3),
        "max_ms":  round(float(np.max(latencies)), 3),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    import numpy as np

    parser = argparse.ArgumentParser(description="PositionalPlateNet Benchmark (Vector 3)")
    parser.add_argument("--device",   type=str, default="cuda",  help="cuda or cpu")
    parser.add_argument("--epochs",   type=int, default=15,      help="Training epochs")
    parser.add_argument("--lr",       type=float, default=1e-3,  help="Max learning rate (OneCycleLR)")
    parser.add_argument("--batch",    type=int, default=64,      help="Batch size")
    parser.add_argument("--val_split",type=float, default=0.15,  help="Val split fraction")
    parser.add_argument("--output_json", type=str,
                        default=os.path.join(ROOT_DIR, "test_output", "positional_benchmark_report.json"))
    parser.add_argument("--onnx_path", type=str,
                        default=os.path.join(ROOT_DIR, "models", "positional_net.onnx"))
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print("=" * 80)
    print("   POSITIONAL PLATE NET BENCHMARK — Vector 3 (Non-CTC Positional Classifier)")
    print("=" * 80)
    print(f"[*] Device: {device} | Epochs: {args.epochs} | LR: {args.lr} | Batch: {args.batch}")

    meta_csv   = os.path.join(ROOT_DIR, "dataset", "meta.csv")
    dataset_dir = os.path.join(ROOT_DIR, "dataset")

    # ── 1. Загрузка датасета ──────────────────────────────────────────────────
    print("\n[+] Loading train split (PlateCropDataset)...")
    base_train = PlateCropDataset(
        meta_csv=meta_csv,
        root_dir=dataset_dir,
        is_train=True,
        val_split=args.val_split,
        cache_in_ram=True,
    )

    print("[+] Loading val split (PlateCropDataset)...")
    base_val = PlateCropDataset(
        meta_csv=meta_csv,
        root_dir=dataset_dir,
        is_train=False,
        val_split=args.val_split,
        cache_in_ram=True,
    )

    # ── 2. Обёртка с фильтрацией ─────────────────────────────────────────────
    print("\n[+] Building PositionalDataset (Latin→Cyrillic + encode filter)...")
    train_ds = PositionalDataset(base_train)
    val_ds   = PositionalDataset(base_val)
    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_positional,
        pin_memory=(device.type == "cuda"),
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_positional,
        pin_memory=(device.type == "cuda"),
        persistent_workers=True,
    )

    # ── 3. Модель и оптимизатор ───────────────────────────────────────────────
    print("\n[+] Initializing PositionalPlateNet...")
    model = PositionalPlateNet(fused_channels=448, dropout_rate=0.3)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_params_m = round(n_params / 1e6, 3)
    print(f"  Parameters: {n_params:,} ({n_params_m}M)")

    criterion = PositionalCELoss(ignore_index=0, reduction="mean")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        steps_per_epoch=len(train_loader),
        epochs=args.epochs,
        pct_start=0.15,
        anneal_strategy="cos",
    )

    # ── 4. Обучение ───────────────────────────────────────────────────────────
    print(f"\n[+] Training for {args.epochs} epochs...")
    print("-" * 80)

    best_val_loss = float("inf")
    best_model_state = None
    epoch_history = []

    t_train_start = time.time()

    for epoch in range(1, args.epochs + 1):
        t_ep = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler,
                                     criterion, device, epoch)
        val_metrics = evaluate(model, val_loader, criterion, device)

        ep_time = time.time() - t_ep
        print(f"  Epoch {epoch:02d}/{args.epochs} | "
              f"TrainLoss {train_loss:.4f} | "
              f"ValLoss {val_metrics['val_loss']:.4f} | "
              f"SeqAcc {val_metrics['seq_acc_pct']:.2f}% | "
              f"CER {val_metrics['cer_pct']:.3f}% | "
              f"{ep_time:.1f}s")

        epoch_history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            **val_metrics,
        })

        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"    ★ New best val_loss: {best_val_loss:.4f}")

    total_train_time = time.time() - t_train_start
    print(f"\n[+] Training complete in {total_train_time:.1f}s")

    # ── 5. Eval с лучшим checkpoint ───────────────────────────────────────────
    print("\n[+] Loading best checkpoint for final evaluation...")
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    model = model.to(device)

    final_metrics = evaluate(model, val_loader, criterion, device)
    best_epoch_info = min(epoch_history, key=lambda x: x["val_loss"])

    print("\n" + "=" * 80)
    print("   FINAL EVALUATION RESULTS (Best Checkpoint)")
    print("=" * 80)
    print(f"  Sequence Accuracy:  {final_metrics['seq_acc_pct']:.2f}%")
    print(f"  CER:                {final_metrics['cer_pct']:.3f}%")
    print(f"  Exact Matches:      {final_metrics['exact_matches']} / {final_metrics['total_samples']}")
    print(f"  Val Loss:           {final_metrics['val_loss']:.4f}")
    print(f"  Best Epoch:         {best_epoch_info['epoch']}")
    print("=" * 80)

    # ── 6. Latency Benchmark ──────────────────────────────────────────────────
    print(f"\n[+] Benchmarking GPU latency ({device})...")
    lat_gpu = benchmark_pytorch_latency(model, device=device, num_iters=200)
    print(f"  GPU Mean: {lat_gpu['mean_ms']} ms | P50: {lat_gpu['p50_ms']} ms | P95: {lat_gpu['p95_ms']} ms")

    print(f"\n[+] Benchmarking CPU latency...")
    model_cpu = model.cpu()
    lat_cpu = benchmark_pytorch_latency(model_cpu, device=torch.device("cpu"), num_iters=100)
    model = model_cpu.to(device)  # restore to original device
    print(f"  CPU Mean: {lat_cpu['mean_ms']} ms | P50: {lat_cpu['p50_ms']} ms | P95: {lat_cpu['p95_ms']} ms")

    # ── 7. ONNX Export ────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.onnx_path), exist_ok=True)
    print(f"\n[+] Exporting ONNX → {args.onnx_path}")
    try:
        export_positional_onnx(model, args.onnx_path, opset_version=17)
        onnx_size_kb = os.path.getsize(args.onnx_path) / 1024
        print(f"  ONNX size: {onnx_size_kb:.1f} KB")
    except Exception as e:
        print(f"  [WARN] ONNX export failed: {e}")
        onnx_size_kb = 0.0

    # ── 8. Сохранение JSON Report ─────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "paradigm": "Positional Non-CTC Multi-Head Classifier (Vector 3)",
        "architecture": {
            "model": "PositionalPlateNet",
            "loss": "PositionalCELoss (sum of 9 CrossEntropyLoss per position)",
            "decoder": "argmax per head → concat (no CTC, no blank token)",
            "fused_channels": 448,
            "heads": 9,
            "parameters": n_params,
            "parameters_m": n_params_m,
            "input_shape": [1, 3, 36, 160],
            "output": "List[9 Tensors of shape (B, num_classes_i)]",
        },
        "training": {
            "epochs": args.epochs,
            "max_lr": args.lr,
            "batch_size": args.batch,
            "optimizer": "AdamW (weight_decay=1e-4)",
            "scheduler": "OneCycleLR (pct_start=0.15, cos annealing)",
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "total_train_time_sec": round(total_train_time, 1),
            "best_epoch": best_epoch_info["epoch"],
            "best_val_loss": best_epoch_info["val_loss"],
        },
        "final_metrics": final_metrics,
        "epoch_history": epoch_history,
        "latency_gpu": lat_gpu,
        "latency_cpu": lat_cpu,
        "onnx_size_kb": round(onnx_size_kb, 1),
        "comparison_context": {
            "note": "Benchmarked on same val split (val_split=0.15) as MoE/Solo benchmarks",
            "solo_v2_seq_acc_pct": 95.30,
            "solo_v2_cer_pct": 1.230,
            "solo_v3_seq_acc_pct": 95.68,
            "solo_v3_cer_pct": 1.319,
            "moe_seq_acc_pct": 95.93,
            "moe_cer_pct": 1.211,
        },
    }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[+] Report saved → {args.output_json}")

    # ── 9. Сводная таблица ────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print(f"{'Paradigm':<22} | {'SeqAcc':>8} | {'CER':>8} | {'Params':>8} | {'ONNX/Size':>10} | Note")
    print("-" * 90)
    print(f"{'Solo LPRNet-v2':<22} | {'95.30%':>8} | {'1.230%':>8} | {'~1.1M':>8} | {'1130 KB':>10} | Baseline Type1B")
    print(f"{'Solo LPRNet-v3':<22} | {'95.68%':>8} | {'1.319%':>8} | {'~1.8M':>8} | {'1808 KB':>10} | Best Type1")
    print(f"{'MoE Routing':<22} | {'95.93%':>8} | {'1.211%':>8} | {'v2+v3':>8} | {'N/A':>10} | Current SOTA")
    print(f"{'Micro-STN':<22} | {'95.33%':>8} | {'~1.2%':>8} | {'+70K':>8} | {'38.8 KB':>10} | Fails Roll>6°")
    stn_note = f"After {args.epochs}ep training"
    print(f"{'Positional (ours)':<22} | {final_metrics['seq_acc_pct']:>7.2f}% | "
          f"{final_metrics['cer_pct']:>7.3f}% | {f'{n_params_m}M':>8} | "
          f"{f'{onnx_size_kb:.0f} KB':>10} | {stn_note}")
    print("=" * 90)

    print("\n[✓] Benchmark complete.")


if __name__ == "__main__":
    main()
