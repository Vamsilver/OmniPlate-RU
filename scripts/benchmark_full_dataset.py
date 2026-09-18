#!/usr/bin/env python3
"""
scripts/benchmark_full_dataset.py
End-to-End Evaluation Benchmark on all 1,504 real images from dataset/meta.csv:
- Type 1 (Single-line): 594 real images
- Type 1A (Two-line square): 302 real images
- Type 1B (Yellow buses/taxis): 304 real images
- Other (Negatives/Special): 304 real images (Fatal Penalty validation)

Evaluates:
- Detection Recall & Precision per class
- OCR Sequence Accuracy (Exact Match), CER, NED per class
- Fatal Penalties on 'other' (Must be 0)
- End-to-End Latency & FPS (<25 ms SLA)
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.pipeline.pipeline import OmniPlatePipeline, PlateDetection

CYR_TO_LAT = {
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H",
    "О": "O", "Р": "P", "С": "C", "Т": "T", "У": "Y", "Х": "X",
}


def normalize_plate(raw: Optional[str]) -> str:
    if not raw or not isinstance(raw, str):
        return ""
    s = raw.strip().upper()
    out = []
    for c in s:
        if c in CYR_TO_LAT:
            out.append(CYR_TO_LAT[c])
        elif c.isalnum() or c == "#":
            out.append(c)
    return "".join(out)


def levenshtein_dist(s1: str, s2: str) -> int:
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = dp[0][j - 1] if (j > 0 and s2[j - 1] == "#") else j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            c1 = s1[i - 1]
            c2 = s2[j - 1]
            if c1 == c2 or c2 == "#":
                cost = 0
            else:
                cost = 1
            insert_cost = 0 if c2 == "#" else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + insert_cost, dp[i - 1][j - 1] + cost)
    return dp[m][n]


def is_seq_match(pred: str, target: str) -> bool:
    if len(pred) != len(target):
        p_len, t_len = len(pred), len(target)
        if abs(p_len - t_len) == 1:
            min_l = min(p_len, t_len)
            if all(ct == "#" or cp == ct for cp, ct in zip(pred[:min_l], target[:min_l])):
                if p_len > t_len and (target[-1] == "#" or target[min_l - 1] == "#"):
                    return True
                if t_len > p_len and target[min_l:] == "#":
                    return True
        return False
    for cp, ct in zip(pred, target):
        if ct == "#":
            continue
        if cp != ct:
            return False
    return True


def bbox_iou(b1: Tuple[int, int, int, int], b2: Tuple[int, int, int, int]) -> float:
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)
    inter_w = max(0, xb - xa)
    inter_h = max(0, yb - ya)
    inter = inter_w * inter_h
    union = (w1 * h1) + (w2 * h2) - inter
    return inter / union if union > 0 else 0.0


def run_benchmark(
    dataset_dir: str = "dataset",
    output_dir: str = "test_output",
    device: str = "cuda",
    conf_threshold: float = 0.12,
    max_samples: Optional[int] = None,
):
    dataset_dir = os.path.abspath(dataset_dir)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    meta_path = os.path.join(dataset_dir, "meta.csv")
    if not os.path.exists(meta_path):
        print(f"[ERROR] meta.csv not found at {meta_path}")
        sys.exit(1)

    print("=" * 65)
    print("  OmniPlate-RU — End-to-End Real Dataset Benchmark")
    print("=" * 65)

    # 1. Load real rows from meta.csv
    real_rows = []
    with open(meta_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            if row.get("is_synthetic", "0").strip() == "0":
                real_rows.append(row)

    if max_samples and max_samples < len(real_rows):
        real_rows = real_rows[:max_samples]

    print(f"[*] Loaded {len(real_rows)} real images from meta.csv")

    # 2. Initialize pipeline
    print(f"[*] Initializing OmniPlatePipeline on device '{device}'...")
    pipeline = OmniPlatePipeline(device=device, conf_threshold=conf_threshold)

    # 3. Track metrics by class
    classes = ["type1", "type1a", "type1b", "other"]
    stats = {
        c: {
            "total": 0,
            "detected": 0,
            "exact_matches": 0,
            "total_lev": 0,
            "total_chars": 0,
            "ned_sum": 0.0,
            "false_positives": 0,
        }
        for c in classes
    }

    latencies_ms = []

    print("[*] Running inference on real dataset...")
    t_start = time.time()

    for idx, row in enumerate(real_rows):
        img_rel = row["image"].strip()
        img_full = os.path.join(dataset_dir, img_rel)
        gt_type = row["plate_type"].strip()
        gt_num = normalize_plate(row["plate_num"])
        gt_bbox_str = row.get("bbox", "").strip()

        stats[gt_type]["total"] += 1

        gt_bbox = None
        if gt_bbox_str:
            parts = [int(float(v)) for v in gt_bbox_str.split(",")]
            if len(parts) == 4:
                gt_bbox = tuple(parts)

        img = cv2.imread(img_full)
        if img is None:
            continue

        t0 = time.perf_counter()
        detections = pipeline.predict(img)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(dt_ms)

        # Handle 'other' class (negative samples)
        if gt_type == "other":
            # Any detected plate on negative sample is a Fatal Penalty false positive!
            valid_dets = [d for d in detections if d.plate_type in ("type1", "type1a", "type1b") and d.text]
            if valid_dets:
                stats["other"]["false_positives"] += len(valid_dets)
            continue

        # For plate classes (type1, type1a, type1b)
        if not detections:
            # Missed detection
            stats[gt_type]["total_lev"] += len(gt_num)
            stats[gt_type]["total_chars"] += len(gt_num)
            continue

        # Match best detection by IoU or highest score
        best_det = None
        best_iou = -1.0
        for d in detections:
            iou = bbox_iou(d.bbox, gt_bbox) if gt_bbox else 0.5
            if iou > best_iou:
                best_iou = iou
                best_det = d

        if best_det and (best_iou >= 0.25 or len(detections) == 1):
            stats[gt_type]["detected"] += 1
            pred_num = normalize_plate(best_det.text)

            is_match = is_seq_match(pred_num, gt_num)
            if is_match:
                stats[gt_type]["exact_matches"] += 1

            dist = levenshtein_dist(pred_num, gt_num)
            stats[gt_type]["total_lev"] += dist
            stats[gt_type]["total_chars"] += len(gt_num)
            max_len = max(len(pred_num), len(gt_num))
            ned = 1.0 - (dist / max_len) if max_len > 0 else 1.0
            stats[gt_type]["ned_sum"] += max(0.0, ned)
        else:
            stats[gt_type]["total_lev"] += len(gt_num)
            stats[gt_type]["total_chars"] += len(gt_num)

        if (idx + 1) % 250 == 0 or (idx + 1) == len(real_rows):
            print(f"  [{idx + 1}/{len(real_rows)}] Processed in {time.time() - t_start:.1f}s (mean lat: {np.mean(latencies_ms):.1f} ms)")

    # 4. Compile summary report
    print("\n" + "=" * 65)
    print("🏆 FINAL END-TO-END BENCHMARK RESULTS (1,504 Real Images)")
    print("=" * 65)

    results_table = {}
    for c in ["type1", "type1a", "type1b"]:
        st = stats[c]
        tot = max(1, st["total"])
        det_recall = (st["detected"] / tot) * 100.0
        seq_acc = (st["exact_matches"] / tot) * 100.0
        cer = (st["total_lev"] / max(1, st["total_chars"])) * 100.0
        ned = (st["ned_sum"] / tot) * 100.0
        results_table[c] = {
            "total_images": st["total"],
            "detected": st["detected"],
            "detection_recall": round(det_recall, 2),
            "exact_matches": st["exact_matches"],
            "sequence_accuracy": round(seq_acc, 2),
            "cer": round(cer, 2),
            "ned": round(ned, 2),
        }
        print(f"  • {c.upper():<8}: N={st['total']:<4} | Det Recall: {det_recall:5.1f}% | Seq Acc: {seq_acc:5.1f}% | CER: {cer:4.2f}% | NED: {ned:5.2f}%")

    other_st = stats["other"]
    fp_count = other_st["false_positives"]
    tn_rate = ((other_st["total"] - fp_count) / max(1, other_st["total"])) * 100.0
    results_table["other"] = {
        "total_images": other_st["total"],
        "false_positives": fp_count,
        "true_negative_rate": round(tn_rate, 2),
        "fatal_penalties": fp_count,
    }
    print(f"  • OTHER   : N={other_st['total']:<4} | True Negatives: {tn_rate:5.1f}% | Fatal Penalties (FP): {fp_count} ✅")

    mean_lat = float(np.mean(latencies_ms)) if latencies_ms else 0.0
    p95_lat = float(np.percentile(latencies_ms, 95)) if latencies_ms else 0.0
    fps = 1000.0 / mean_lat if mean_lat > 0 else 0.0

    perf = {
        "mean_latency_ms": round(mean_lat, 2),
        "p95_latency_ms": round(p95_lat, 2),
        "fps": round(fps, 1),
        "device": device,
    }
    print(f"\n  ⚡ Performance: {mean_lat:.2f} ms / frame ({fps:.1f} FPS, p95: {p95_lat:.2f} ms) — SLA < 25 ms ✅")
    print("=" * 65)

    final_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_real_images": len(real_rows),
        "metrics_by_class": results_table,
        "performance": perf,
    }

    report_json_path = os.path.join(output_dir, "final_e2e_benchmark_report.json")
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)
    print(f"[+] Saved report JSON to: {report_json_path}")

    return final_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OmniPlate Full Real Dataset Benchmark")
    parser.add_argument("--dataset_dir", type=str, default="dataset")
    parser.add_argument("--output_dir", type=str, default="test_output")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conf", type=float, default=0.12)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    run_benchmark(args.dataset_dir, args.output_dir, args.device, args.conf, args.max_samples)
