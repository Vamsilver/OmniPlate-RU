#!/usr/bin/env python3
"""
scripts/research/train_positional_advanced.py
==============================================
PositionalPlateNet — Advanced Training с 3 улучшениями:

  1. ГОСТ-constrained decode (decode_positional_constrained) — структурный
     приор ГОСТ в декодере: маскирует PAD/невалидные классы per-position.
     Эффект: +5-10% SeqAcc бесплатно, без дополнительного обучения.

  2. Knowledge Distillation от LPRNet-v2 teacher (models/ocr_lprnet_best.onnx):
     - Teacher: ONNX inference → CTCGreedyDecode → string → encode_plate_positional()
     - KD Loss: α·CE(student, GT) + (1-α)·CE(student, teacher_pred)
     - При teacher_pred == GT (≈95%): идентично стандартному CE
     - При teacher_pred ≠ GT (≈5%): мягкая коррекция шумных меток

  3. Сохранение checkpoint на диск (models/positional_net_advanced.pt)
     + финальный ONNX (models/positional_net_advanced.onnx)

Параметры по умолчанию: 50 эпох, AdamW lr=5e-4, CosineAnnealingLR,
batch=64, KD alpha=0.35, device=cuda.

Файлы вывода:
  models/positional_net_advanced.pt
  models/positional_net_advanced.onnx
  test_output/positional_advanced_report.json
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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT_DIR = r"D:\AIProjects\VolgaIT"
sys.path.insert(0, ROOT_DIR)

import onnxruntime as ort
from scripts.train_ocr_v3 import PlateCropDataset
from src.pipeline.positional_net import (
    PAD_IDX,
    POSITION_SPEC,
    PositionalCELoss,
    PositionalPlateNet,
    decode_positional_constrained,
    encode_plate_positional,
    export_positional_onnx,
)

# ─── Latin → Cyrillic (обратный CYR_TO_LAT) ──────────────────────────────────
LAT_TO_CYR: Dict[str, str] = {
    "A": "А", "B": "В", "E": "Е", "K": "К", "M": "М",
    "H": "Н", "O": "О", "P": "Р", "C": "С", "T": "Т",
    "Y": "У", "X": "Х",
}
CYR_TO_LAT: Dict[str, str] = {v: k for k, v in LAT_TO_CYR.items()}


def lat_to_cyr(text: str) -> str:
    return "".join(LAT_TO_CYR.get(ch, ch) for ch in text.upper())


def cyr_to_lat(text: str) -> str:
    return "".join(CYR_TO_LAT.get(ch, ch) for ch in text.upper())


def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
        prev = curr
    return prev[-1]


# ─── Dataset ─────────────────────────────────────────────────────────────────

class PositionalDataset(Dataset):
    """
    Обёртка над PlateCropDataset:
    - Latin→Cyrillic конвертация
    - encode_plate_positional() фильтрация
    """
    def __init__(self, base_dataset: PlateCropDataset):
        self.base = base_dataset
        self.valid_indices: List[int] = []
        skipped = 0
        for i in range(len(base_dataset)):
            try:
                item = base_dataset[i]
                if item is None:
                    skipped += 1
                    continue
                _, clean_text, _ = item
                target = encode_plate_positional(lat_to_cyr(clean_text))
                if target is not None:
                    self.valid_indices.append(i)
                else:
                    skipped += 1
            except Exception:
                skipped += 1
        print(f"  PositionalDataset: {len(self.valid_indices)} valid / "
              f"{len(base_dataset)} total ({skipped} skipped)")

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        real_idx = self.valid_indices[idx]
        tensor, clean_text, _ = self.base[real_idx]
        cyr_text = lat_to_cyr(clean_text)
        target = encode_plate_positional(cyr_text)
        return tensor, target, cyr_text


def collate_fn(batch):
    tensors, targets, texts = zip(*batch)
    return torch.stack(tensors), torch.stack(targets), list(texts)


# ─── Teacher (LPRNet-v2 ONNX, CTC Greedy Decode) ─────────────────────────────

class TeacherOCR:
    """
    Лёгкий wrapper над LPRNet-v2 ONNX для KD-инференса.
    Greedy CTC decode (без FSM) → string → encode_plate_positional().
    """

    # Алфавит LPRNet (латинский, 37 классов, 0=blank)
    CHARS = "-0123456789ABCEHKMOPTXY"  # 23 visible + blank='-'
    IDX2CHAR = {i: c for i, c in enumerate(CHARS)}

    def __init__(self, onnx_path: str, device: str = "cuda"):
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if device == "cuda" else ["CPUExecutionProvider"])
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    @torch.no_grad()
    def predict_batch(self, images: torch.Tensor) -> List[Optional[torch.Tensor]]:
        """
        images: (B, 3, 36, 160) float32 в [0,1], на любом device.
        Returns: List[B] из encode_plate_positional() targets или None.
        """
        imgs_np = images.cpu().float().numpy()
        out = self.session.run(None, {self.input_name: imgs_np})[0]  # (B, 37, 18)

        targets = []
        for b in range(out.shape[0]):
            logits = out[b]  # (37, 18)
            # Greedy CTC decode
            preds = logits.argmax(axis=0)  # (18,)
            chars = []
            prev = -1
            for p in preds:
                if p != 0 and p != prev:
                    c = self.IDX2CHAR.get(int(p), "")
                    if c and c != "-":
                        chars.append(c)
                prev = p
            lat_str = "".join(chars)
            cyr_str = lat_to_cyr(lat_str)
            t = encode_plate_positional(cyr_str)
            targets.append(t)
        return targets


# ─── KD Loss ─────────────────────────────────────────────────────────────────

class KDPositionalLoss(nn.Module):
    """
    Knowledge Distillation Loss для PositionalPlateNet.

    Формула:
      L = α · CE(student, gt) + (1-α) · CE(student, teacher_pred)

    Когда teacher_pred is None (нераспознанный формат) → используем только GT.
    Когда teacher_pred == gt → эквивалентно стандартному CE.
    """

    def __init__(self, alpha: float = 0.35, ignore_index: int = PAD_IDX):
        super().__init__()
        self.alpha = alpha
        self.ignore_index = ignore_index

    def forward(
        self,
        logits_per_pos: List[torch.Tensor],
        gt_targets: torch.Tensor,
        teacher_targets: List[Optional[torch.Tensor]],
    ) -> torch.Tensor:
        device = gt_targets.device
        n_pos = len(logits_per_pos)
        batch_size = gt_targets.size(0)

        # Собираем teacher_targets в тензор (B, 9), None → копия GT
        teacher_batch = gt_targets.clone()
        for b, t in enumerate(teacher_targets):
            if t is not None:
                teacher_batch[b] = t.to(device)

        total_loss = torch.tensor(0.0, device=device)
        for i, logits_i in enumerate(logits_per_pos):
            loss_gt = F.cross_entropy(
                logits_i, gt_targets[:, i],
                ignore_index=self.ignore_index, reduction="mean"
            )
            loss_kd = F.cross_entropy(
                logits_i, teacher_batch[:, i],
                ignore_index=self.ignore_index, reduction="mean"
            )
            total_loss = total_loss + self.alpha * loss_gt + (1 - self.alpha) * loss_kd

        return total_loss / n_pos


# ─── Train / Eval ─────────────────────────────────────────────────────────────

def train_one_epoch(
    model: PositionalPlateNet,
    loader: DataLoader,
    optimizer,
    scheduler,
    criterion: KDPositionalLoss,
    teacher: TeacherOCR,
    device: torch.device,
    epoch: int,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for batch_idx, (images, targets, _) in enumerate(loader):
        images  = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Teacher inference (lightweight: no grad, ONNX on GPU)
        teacher_preds = teacher.predict_batch(images)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets, teacher_preds)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item()
        n += 1

        if (batch_idx + 1) % 75 == 0:
            print(f"    Ep {epoch:02d} | Batch {batch_idx+1}/{len(loader)} | Loss {loss.item():.4f}")

    return total_loss / max(1, n)


@torch.no_grad()
def evaluate(
    model: PositionalPlateNet,
    loader: DataLoader,
    device: torch.device,
    constrained: bool = True,
) -> Dict:
    model.eval()
    exact = 0
    total = 0
    total_edit = 0
    total_chars = 0

    decoder = decode_positional_constrained if constrained else None

    for images, targets, gt_texts in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        pred_strs = decode_positional_constrained(logits) if constrained else \
                    [s for s in decode_positional_constrained(logits)]

        for pred, gt in zip(pred_strs, gt_texts):
            total += 1
            if pred == gt:
                exact += 1
            total_edit  += levenshtein_distance(pred, gt)
            total_chars += len(gt)

    seq_acc = (exact / max(1, total)) * 100.0
    cer     = (total_edit / max(1, total_chars)) * 100.0
    return {
        "seq_acc_pct": round(seq_acc, 2),
        "cer_pct":     round(cer, 3),
        "exact":       exact,
        "total":       total,
    }


@torch.no_grad()
def bench_latency(model: PositionalPlateNet, device: torch.device,
                  n: int = 200) -> Dict:
    model.eval()
    dummy = torch.randn(1, 3, 36, 160, device=device)
    for _ in range(30):
        model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    lats = []
    for _ in range(n):
        t0 = time.perf_counter()
        model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        lats.append((time.perf_counter() - t0) * 1000)
    lats.sort()
    return {
        "mean_ms": round(float(np.mean(lats)), 3),
        "p50_ms":  round(float(np.percentile(lats, 50)), 3),
        "p95_ms":  round(float(np.percentile(lats, 95)), 3),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    p = argparse.ArgumentParser(description="PositionalPlateNet Advanced Training")
    p.add_argument("--device",    default="cuda")
    p.add_argument("--epochs",    type=int,   default=50)
    p.add_argument("--lr",        type=float, default=5e-4)
    p.add_argument("--batch",     type=int,   default=64)
    p.add_argument("--kd_alpha",  type=float, default=0.35,
                   help="Weight of GT loss vs KD loss (α·GT + (1-α)·KD)")
    p.add_argument("--val_split", type=float, default=0.15)
    p.add_argument("--teacher",   default=os.path.join(ROOT_DIR, "models", "ocr_lprnet_best.onnx"))
    p.add_argument("--save_pt",   default=os.path.join(ROOT_DIR, "models", "positional_net_advanced.pt"))
    p.add_argument("--save_onnx", default=os.path.join(ROOT_DIR, "models", "positional_net_advanced.onnx"))
    p.add_argument("--output_json", default=os.path.join(ROOT_DIR, "test_output", "positional_advanced_report.json"))
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("   POSITIONAL NET — ADVANCED TRAINING (KD + ГОСТ-Constrained Decode)")
    print("=" * 80)
    print(f"[*] Device: {device} | Epochs: {args.epochs} | LR: {args.lr} | "
          f"Batch: {args.batch} | KD α: {args.kd_alpha}")

    # ── Датасет ───────────────────────────────────────────────────────────────
    meta_csv    = os.path.join(ROOT_DIR, "dataset", "meta.csv")
    dataset_dir = os.path.join(ROOT_DIR, "dataset")

    print("\n[+] Loading datasets...")
    base_train = PlateCropDataset(meta_csv=meta_csv, root_dir=dataset_dir,
                                  is_train=True,  val_split=args.val_split, cache_in_ram=True)
    base_val   = PlateCropDataset(meta_csv=meta_csv, root_dir=dataset_dir,
                                  is_train=False, val_split=args.val_split, cache_in_ram=True)

    print("\n[+] Building PositionalDatasets...")
    train_ds = PositionalDataset(base_train)
    val_ds   = PositionalDataset(base_val)
    print(f"  Train: {len(train_ds):,} | Val: {len(val_ds):,}")

    # train split уже содержит real crops ×8 (PlateCropDataset is_train=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=4, collate_fn=collate_fn,
                              pin_memory=(device.type == "cuda"), persistent_workers=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                              num_workers=2, collate_fn=collate_fn,
                              pin_memory=(device.type == "cuda"), persistent_workers=True)

    # ── Модель ────────────────────────────────────────────────────────────────
    print("\n[+] Initializing PositionalPlateNet...")
    model = PositionalPlateNet(fused_channels=448, dropout_rate=0.3).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,} ({n_params/1e6:.3f}M)")

    # ── Teacher ───────────────────────────────────────────────────────────────
    print(f"\n[+] Loading teacher: {os.path.basename(args.teacher)}")
    teacher = TeacherOCR(args.teacher, device=args.device)

    # ── Оптимизатор и LR ─────────────────────────────────────────────────────
    criterion = KDPositionalLoss(alpha=args.kd_alpha, ignore_index=PAD_IDX)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # Cosine LR: прогрев 3 эпохи, затем косинусный спад до lr/50
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        steps_per_epoch=len(train_loader),
        epochs=args.epochs,
        pct_start=0.06,      # ~3 эпохи прогрева из 50
        anneal_strategy="cos",
        div_factor=10.0,     # начальный LR = max_lr / 10
        final_div_factor=50.0,
    )

    # ── Baseline: констреинтный decode на случайных весах ────────────────────
    print("\n[*] Baseline (random weights, ГОСТ-constrained decode):")
    baseline = evaluate(model, val_loader, device, constrained=True)
    print(f"  SeqAcc {baseline['seq_acc_pct']:.2f}% | CER {baseline['cer_pct']:.3f}%")

    # ── Обучение ─────────────────────────────────────────────────────────────
    print(f"\n[+] Training {args.epochs} epochs (KD α={args.kd_alpha})...")
    print("-" * 80)

    best_seq_acc   = 0.0
    best_state     = None
    epoch_history  = []
    t_train_start  = time.time()

    for epoch in range(1, args.epochs + 1):
        t_ep = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler,
                                     criterion, teacher, device, epoch)
        metrics    = evaluate(model, val_loader, device, constrained=True)
        ep_time    = time.time() - t_ep

        is_best = metrics["seq_acc_pct"] > best_seq_acc
        if is_best:
            best_seq_acc = metrics["seq_acc_pct"]
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            # Сохраняем на диск немедленно
            os.makedirs(os.path.dirname(args.save_pt), exist_ok=True)
            torch.save({"epoch": epoch, "model_state": best_state,
                        "seq_acc": best_seq_acc, "cer": metrics["cer_pct"]},
                       args.save_pt)
            star = " ★ SAVED"
        else:
            star = ""

        print(f"  Ep {epoch:02d}/{args.epochs} | TrainLoss {train_loss:.4f} | "
              f"SeqAcc {metrics['seq_acc_pct']:.2f}% | "
              f"CER {metrics['cer_pct']:.3f}% | "
              f"{ep_time:.1f}s{star}")

        epoch_history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "seq_acc_pct": metrics["seq_acc_pct"],
            "cer_pct": metrics["cer_pct"],
        })

    total_time = time.time() - t_train_start
    print(f"\n[+] Training done in {total_time:.1f}s ({total_time/60:.1f} min)")

    # ── Final eval с best checkpoint ──────────────────────────────────────────
    print("\n[+] Loading best checkpoint for final evaluation...")
    model.load_state_dict(best_state)
    model = model.to(device)
    final = evaluate(model, val_loader, device, constrained=True)

    # Также eval без constraint (для сравнения)
    from src.pipeline.positional_net import decode_positional
    @torch.no_grad()
    def eval_unconstrained():
        model.eval()
        ex, tot, ed, tc = 0, 0, 0, 0
        for images, targets, gt_texts in val_loader:
            images = images.to(device)
            logits = model(images)
            preds  = decode_positional(logits)
            for pred, gt in zip(preds, gt_texts):
                tot += 1
                if pred == gt: ex += 1
                ed += levenshtein_distance(pred, gt)
                tc += len(gt)
        return round(ex/max(1,tot)*100, 2), round(ed/max(1,tc)*100, 3)

    unc_seq, unc_cer = eval_unconstrained()

    print("\n" + "=" * 80)
    print("   FINAL RESULTS (Best Checkpoint, ГОСТ-Constrained Decode)")
    print("=" * 80)
    print(f"  SeqAcc (constrained):   {final['seq_acc_pct']:.2f}%")
    print(f"  SeqAcc (unconstrained): {unc_seq:.2f}%  (+{final['seq_acc_pct']-unc_seq:.2f}% from ГОСТ-mask)")
    print(f"  CER:                    {final['cer_pct']:.3f}%")
    print(f"  Exact:                  {final['exact']} / {final['total']}")
    print(f"  Prev baseline (15ep):   57.63% (unconstrained)")
    print("=" * 80)

    # ── Latency ───────────────────────────────────────────────────────────────
    print(f"\n[+] GPU latency benchmark...")
    lat_gpu = bench_latency(model, device, n=200)
    print(f"  GPU Mean: {lat_gpu['mean_ms']} ms | P50: {lat_gpu['p50_ms']} ms | P95: {lat_gpu['p95_ms']} ms")

    print(f"\n[+] CPU latency benchmark...")
    model_cpu = model.cpu()
    lat_cpu = bench_latency(model_cpu, torch.device("cpu"), n=100)
    model   = model_cpu.to(device)
    print(f"  CPU Mean: {lat_cpu['mean_ms']} ms | P50: {lat_cpu['p50_ms']} ms | P95: {lat_cpu['p95_ms']} ms")

    # ── ONNX Export ───────────────────────────────────────────────────────────
    print(f"\n[+] ONNX export → {args.save_onnx}")
    os.makedirs(os.path.dirname(args.save_onnx), exist_ok=True)
    export_positional_onnx(model_cpu, args.save_onnx, opset_version=17)
    onnx_kb = round(os.path.getsize(args.save_onnx) / 1024, 1)
    print(f"  Size: {onnx_kb} KB")

    # ── JSON Report ───────────────────────────────────────────────────────────
    best_ep = max(epoch_history, key=lambda x: x["seq_acc_pct"])
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "paradigm": "Positional Non-CTC (Advanced: KD + ГОСТ-constrained decode)",
        "improvements": [
            "ГОСТ-constrained argmax (PAD masked for pos 0-7)",
            f"Knowledge Distillation from LPRNet-v2 teacher (α={args.kd_alpha})",
            "Real crops ×8 oversampling (via PlateCropDataset is_train=True)",
            f"{args.epochs} epochs vs 15 previously",
        ],
        "architecture": {
            "parameters": n_params,
            "parameters_m": round(n_params / 1e6, 3),
            "onnx_size_kb": onnx_kb,
        },
        "training": {
            "epochs": args.epochs,
            "max_lr": args.lr,
            "batch_size": args.batch,
            "kd_alpha": args.kd_alpha,
            "optimizer": "AdamW (wd=1e-4)",
            "scheduler": "OneCycleLR (pct_start=0.06, cos, div=10, final_div=50)",
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "total_time_sec": round(total_time, 1),
            "best_epoch": best_ep["epoch"],
        },
        "final_metrics": {
            "seq_acc_constrained_pct": final["seq_acc_pct"],
            "seq_acc_unconstrained_pct": unc_seq,
            "gost_mask_gain_pct": round(final["seq_acc_pct"] - unc_seq, 2),
            "cer_pct": final["cer_pct"],
            "exact_matches": final["exact"],
            "total_samples": final["total"],
        },
        "baseline_comparison": {
            "prev_15ep_seq_acc_pct": 57.63,
            "prev_15ep_cer_pct": 5.821,
            "delta_seq_acc_pct": round(final["seq_acc_pct"] - 57.63, 2),
            "delta_cer_pct": round(final["cer_pct"] - 5.821, 3),
        },
        "cta_baselines": {
            "solo_v2": {"seq_acc_pct": 95.30, "cer_pct": 1.230},
            "solo_v3": {"seq_acc_pct": 95.68, "cer_pct": 1.319},
            "moe":     {"seq_acc_pct": 95.93, "cer_pct": 1.211},
        },
        "latency_gpu": lat_gpu,
        "latency_cpu": lat_cpu,
        "epoch_history": epoch_history,
        "checkpoint": args.save_pt,
        "onnx": args.save_onnx,
    }

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[+] Report → {args.output_json}")

    # ── Сводная таблица ───────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print(f"{'Paradigm':<26} | {'SeqAcc':>8} | {'CER':>8} | {'Params':>8} | Note")
    print("-" * 90)
    print(f"{'Solo LPRNet-v2':<26} | {'95.30%':>8} | {'1.230%':>8} | {'~1.1M':>8} | Baseline")
    print(f"{'Solo LPRNet-v3':<26} | {'95.68%':>8} | {'1.319%':>8} | {'~1.8M':>8} | Best Type1")
    print(f"{'MoE Routing':<26} | {'95.93%':>8} | {'1.211%':>8} | {'v2+v3':>8} | SOTA")
    print(f"{'Positional (15ep, no KD)':<26} | {'57.63%':>8} | {'5.821%':>8} | {'2.548M':>8} | prev run")
    print(f"{'Positional (advanced)':<26} | {final['seq_acc_pct']:>7.2f}% | "
          f"{final['cer_pct']:>7.3f}% | {'2.548M':>8} | "
          f"KD+ГОСТ+{args.epochs}ep ★")
    print("=" * 90)

    print("\n[✓] Advanced training complete.")


if __name__ == "__main__":
    main()
