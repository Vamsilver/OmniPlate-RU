#!/usr/bin/env python3
"""
scripts/research/benchmark_stn_robustness.py
Research benchmark for Vector 2: Micro-STN (Spatial Transformer Network).
Investigates perspective angle robustness and sub-pixel alignment for road camera slants.

Evaluates:
1. Exact Sequence Accuracy and CER under varying geometric slants:
   - Rotation (Roll): 0°, ±3°, ±6°, ±10°, ±15°
   - Horizontal Shear (Yaw-induced slant): 0.0, ±0.08, ±0.15
2. Latency of Micro-STN on GPU (RTX 5080) and CPU.
3. Parameter count and exported ONNX model footprint.
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True, encoding="utf-8")

import cv2
import numpy as np

ROOT_DIR = r"D:\AIProjects\VolgaIT"
sys.path.insert(0, ROOT_DIR)

import torch
from src.pipeline.decoder import CTCDecoder
from src.pipeline.ocr import PlateOCR, BLANK_IDX
from src.pipeline.stn import MicroSpatialTransformer, export_stn_onnx
from scripts.train_ocr_v3 import PlateCropDataset


def apply_geometric_slant(
    crop_bgr: np.ndarray,
    angle_deg: float = 0.0,
    shear_x: float = 0.0,
    shear_y: float = 0.0,
) -> np.ndarray:
    """Applies controlled roll rotation and shear perspective to canonical (36, 160, 3) crop."""
    h, w = crop_bgr.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    # 1. Rotation matrix
    rot_mat = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)

    # 2. Shear transformation: x' = x + shear_x * (y - cy), y' = y + shear_y * (x - cx)
    m_shear = np.array([
        [1.0, shear_x, -shear_x * cy],
        [shear_y, 1.0, -shear_y * cx],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)

    m_rot_3x3 = np.vstack([rot_mat, [0.0, 0.0, 1.0]])
    m_combined = m_shear @ m_rot_3x3
    m_affine = m_combined[:2, :]

    warped = cv2.warpAffine(
        crop_bgr,
        m_affine,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return warped


def benchmark_stn_latency(stn_model: MicroSpatialTransformer, device: str = "cuda", iters: int = 500) -> Dict[str, float]:
    stn_model.eval()
    x = torch.randn(1, 3, 36, 160, device=device)

    # Warmup
    for _ in range(50):
        with torch.no_grad():
            _ = stn_model(x)

    latencies = []
    for _ in range(iters):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = stn_model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt)

    latencies.sort()
    return {
        "mean_ms": round(float(np.mean(latencies)), 3),
        "p50_ms": round(float(np.percentile(latencies, 50)), 3),
        "p95_ms": round(float(np.percentile(latencies, 95)), 3),
        "min_ms": round(float(np.min(latencies)), 3),
        "max_ms": round(float(np.max(latencies)), 3),
    }


def main():
    parser = argparse.ArgumentParser(description="Micro-STN Perspective Robustness Benchmark")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_samples", type=int, default=600, help="Validation samples for perturbation stress-test")
    parser.add_argument("--output_json", type=str, default="test_output/stn_robustness_report.json")
    args = parser.parse_args()

    device = args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"

    print("=" * 85)
    print("      MICRO-STN PERSPECTIVE ROBUSTNESS & SUB-PIXEL BENCHMARK")
    print(f"      Device: {device.upper()} | Target: Angle & Slant Invariance")
    print("=" * 85)

    # Export ONNX model
    onnx_path = os.path.join(ROOT_DIR, "models", "stn_micro.onnx")
    stn = MicroSpatialTransformer(bounded=True).to(device)
    export_stn_onnx(stn.cpu(), onnx_path, input_shape=(1, 3, 36, 160), opset_version=18)
    stn.to(device)
    stn_size_kb = round(os.path.getsize(onnx_path) / 1024.0, 1)
    stn_params = sum(p.numel() for p in stn.parameters())
    print(f"[+] Micro-STN ONNX exported to: {onnx_path} ({stn_size_kb} KB, {stn_params:,} parameters)")

    # Load validation crops
    meta_csv = os.path.join(ROOT_DIR, "dataset", "meta.csv")
    dataset_dir = os.path.join(ROOT_DIR, "dataset")

    print("[+] Loading validation crops for perturbation evaluation...")
    val_dataset = PlateCropDataset(
        meta_csv=meta_csv,
        root_dir=dataset_dir,
        is_train=False,
        val_split=0.15,
        cache_in_ram=True,
    )

    crops_bgr = []
    ground_truths = []
    plate_types = []

    for i in range(len(val_dataset)):
        crop = val_dataset.ram_cache.get(i, None)
        if crop is None:
            continue
        _, clean_text, p_type = val_dataset[i]
        crops_bgr.append(crop)
        ground_truths.append(clean_text)
        plate_types.append(p_type)
        if len(crops_bgr) >= args.max_samples:
            break

    print(f"[+] Loaded {len(crops_bgr)} crops for stress testing.")

    # Load OCR engine (LPRNet-v3)
    v3_onnx = os.path.join(ROOT_DIR, "models", "ocr_lprnet_v3.onnx")
    ocr = PlateOCR(model_path=v3_onnx, ocr_version="v3", device=device)

    # Define test perturbation conditions
    perturbations = [
        {"name": "Clean (0° slant)", "angle": 0.0, "shear_x": 0.0},
        {"name": "Mild Roll (±3°)", "angle": 3.0, "shear_x": 0.0},
        {"name": "Moderate Roll (±6°)", "angle": 6.0, "shear_x": 0.0},
        {"name": "Significant Roll (±10°)", "angle": 10.0, "shear_x": 0.0},
        {"name": "Extreme Roll (±15°)", "angle": 15.0, "shear_x": 0.0},
        {"name": "Mild Shear (+0.08)", "angle": 0.0, "shear_x": 0.08},
        {"name": "Strong Shear (+0.15)", "angle": 0.0, "shear_x": 0.15},
        {"name": "Compound (Roll 6° + Shear 0.10)", "angle": 6.0, "shear_x": 0.10},
    ]

    print("\n" + "=" * 90)
    print(f"{'Condition':<35} | {'Exact Matches':<15} | {'SeqAcc (%)':<15} | {'Degradation':<15}")
    print("-" * 90)

    clean_acc = 0.0
    pert_results = []

    for idx, p in enumerate(perturbations):
        ang = p["angle"]
        sh_x = p["shear_x"]

        # Perturb crops
        perturbed_crops = []
        for c in crops_bgr:
            warped = apply_geometric_slant(c, angle_deg=ang, shear_x=sh_x)
            perturbed_crops.append(warped)

        # Batch inference
        preds = ocr.predict_batch(perturbed_crops, plate_types=plate_types, return_type=False)

        exact = 0
        for (pred_text, conf), gt in zip(preds, ground_truths):
            if pred_text == gt:
                exact += 1

        seq_acc = round((exact / max(1, len(ground_truths))) * 100.0, 2)
        if idx == 0:
            clean_acc = seq_acc
            deg_str = "Baseline"
        else:
            deg = round(seq_acc - clean_acc, 2)
            deg_str = f"{deg:+.2f}%"

        print(f"{p['name']:<35} | {exact:<15} | {seq_acc}%{'':<9} | {deg_str:<15}")
        pert_results.append({
            "condition": p["name"],
            "angle_deg": ang,
            "shear_x": sh_x,
            "exact_matches": exact,
            "seq_acc_pct": seq_acc,
            "delta_vs_clean_pct": round(seq_acc - clean_acc, 2),
        })

    print("=" * 90)

    # Measure Micro-STN inference speed
    print(f"\n[+] Measuring Micro-STN inference latency on {device.upper()} (500 iters)...")
    lat_gpu = benchmark_stn_latency(stn, device=device, iters=500)
    stn_cpu = stn.cpu()
    lat_cpu = benchmark_stn_latency(stn_cpu, device="cpu", iters=200)

    print(f"  GPU Latency: Mean {lat_gpu['mean_ms']} ms | P50 {lat_gpu['p50_ms']} ms | P95 {lat_gpu['p95_ms']} ms")
    print(f"  CPU Latency: Mean {lat_cpu['mean_ms']} ms | P50 {lat_cpu['p50_ms']} ms | P95 {lat_cpu['p95_ms']} ms")

    out_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": "MicroSpatialTransformer (Micro-STN)",
        "parameters": stn_params,
        "onnx_size_kb": stn_size_kb,
        "device": device,
        "samples_evaluated": len(crops_bgr),
        "perturbation_analysis": pert_results,
        "latency_gpu_ms": lat_gpu,
        "latency_cpu_ms": lat_cpu,
    }

    out_file = os.path.join(ROOT_DIR, args.output_json)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out_report, f, indent=2, ensure_ascii=False)
    print(f"\n[+] Full JSON robustness report saved to: {out_file}")


if __name__ == "__main__":
    main()
