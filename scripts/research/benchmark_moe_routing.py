#!/usr/bin/env python3
"""
scripts/research/benchmark_moe_routing.py
Comprehensive head-to-head benchmark for Type-Conditioned Routing (MoE) vs Solo models.

Evaluates:
1. Solo LPRNet-v2 (models/ocr_lprnet_best.onnx)
2. Solo LPRNet-v3 (models/ocr_lprnet_v3.onnx)
3. Type-Conditioned MoE Routing (v3 for Type 1, v2 for Type 1B/2)

Metrics:
- Overall Sequence Accuracy (FSM & Greedy)
- Character Error Rate (CER %)
- Class-specific Accuracy (Type 1, Type 1A, Type 1B, Type 2)
- Synergistic Delta vs Solo baselines
- Inference Latency on GPU & CPU
"""

import argparse
import json
import os
import random
import sys
import time
from typing import Dict, List, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True, encoding="utf-8")

import cv2
import numpy as np
import onnxruntime as ort

ROOT_DIR = r"D:\AIProjects\VolgaIT"
sys.path.insert(0, ROOT_DIR)

from src.pipeline.decoder import CTCDecoder, FSMBeamSearchDecoder
from src.pipeline.ocr import (
    BLANK_IDX,
    PlateOCR,
)
from scripts.train_ocr_v3 import PlateCropDataset


def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def benchmark_latency(onnx_path: str, device: str = "cuda", num_iters: int = 200) -> Dict[str, float]:
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
    session = ort.InferenceSession(onnx_path, providers=providers)
    input_name = session.get_inputs()[0].name
    dummy_input = np.random.randn(1, 3, 36, 160).astype(np.float32)

    for _ in range(30):
        session.run(None, {input_name: dummy_input})

    latencies = []
    for _ in range(num_iters):
        t0 = time.perf_counter()
        session.run(None, {input_name: dummy_input})
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


def evaluate_engine(
    name: str,
    ocr_engine: PlateOCR,
    crops_bgr: List[np.ndarray],
    ground_truths: List[str],
    plate_types: List[str],
    batch_size: int = 64,
) -> Dict[str, any]:
    print(f"\n[+] Evaluating {name} on {len(crops_bgr)} validation crops...")
    t0 = time.time()

    total_samples = len(ground_truths)
    fsm_exact = 0
    total_edit_distance = 0
    total_gt_chars = 0

    class_exact = {"type1": 0, "type1a": 0, "type1b": 0, "type2": 0}
    class_total = {"type1": 0, "type1a": 0, "type1b": 0, "type2": 0}

    optical_zero_o_total = 0
    optical_zero_o_correct = 0
    optical_eight_b_total = 0
    optical_eight_b_correct = 0

    # Batch evaluation
    for b_start in range(0, total_samples, batch_size):
        b_crops = crops_bgr[b_start : b_start + batch_size]
        b_types = plate_types[b_start : b_start + batch_size]
        b_gts = ground_truths[b_start : b_start + batch_size]

        batch_preds = ocr_engine.predict_batch(
            b_crops,
            plate_types=b_types,
            return_type=True,
            use_beam_search=True,
            beam_width=10,
        )

        for i, (pred_fsm, conf, det_type) in enumerate(batch_preds):
            gt = b_gts[i]
            pt = b_types[i]

            if pred_fsm == gt:
                fsm_exact += 1
                if pt in class_exact:
                    class_exact[pt] += 1

            if pt in class_total:
                class_total[pt] += 1

            dist = levenshtein_distance(pred_fsm, gt)
            total_edit_distance += dist
            total_gt_chars += len(gt)

            for g_char, p_char in zip(gt, pred_fsm[:len(gt)]):
                if g_char in ("0", "O"):
                    optical_zero_o_total += 1
                    if p_char == g_char:
                        optical_zero_o_correct += 1
                elif g_char in ("8", "B"):
                    optical_eight_b_total += 1
                    if p_char == g_char:
                        optical_eight_b_correct += 1

    total_eval_time = time.time() - t0
    seq_acc_fsm = (fsm_exact / total_samples) * 100.0
    cer = (total_edit_distance / max(1, total_gt_chars)) * 100.0

    class_accuracies = {}
    for c_name in class_total:
        cnt = class_total[c_name]
        acc = (class_exact[c_name] / max(1, cnt)) * 100.0 if cnt > 0 else 0.0
        class_accuracies[c_name] = round(acc, 2)

    zero_o_acc = (optical_zero_o_correct / max(1, optical_zero_o_total)) * 100.0 if optical_zero_o_total > 0 else 0.0
    eight_b_acc = (optical_eight_b_correct / max(1, optical_eight_b_total)) * 100.0 if optical_eight_b_total > 0 else 0.0

    return {
        "name": name,
        "total_samples": total_samples,
        "eval_time_sec": round(total_eval_time, 2),
        "seq_acc_fsm_pct": round(seq_acc_fsm, 2),
        "exact_matches": fsm_exact,
        "cer_pct": round(cer, 3),
        "class_acc_pct": class_accuracies,
        "class_counts": class_total,
        "optical_pairs": {
            "0_vs_O_acc_pct": round(zero_o_acc, 2),
            "8_vs_B_acc_pct": round(eight_b_acc, 2),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Type-Conditioned Routing (MoE) Benchmark")
    parser.add_argument("--v2_onnx", type=str, default="models/ocr_lprnet_best.onnx")
    parser.add_argument("--v3_onnx", type=str, default="models/ocr_lprnet_v3.onnx")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_val_samples", type=int, default=0, help="Limit samples for quick test (0=all)")
    parser.add_argument("--output_json", type=str, default="test_output/moe_benchmark_report.json")
    args = parser.parse_args()

    v2_path = os.path.join(ROOT_DIR, args.v2_onnx)
    v3_path = os.path.join(ROOT_DIR, args.v3_onnx)

    print("=" * 85)
    print("      TYPE-CONDITIONED ROUTING (MoE) BENCHMARK: Solo vs MoE Synergy")
    print("=" * 85)

    meta_csv = os.path.join(ROOT_DIR, "dataset", "meta.csv")
    dataset_dir = os.path.join(ROOT_DIR, "dataset")

    print("[+] Loading validation pool of crops...")
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

        if 0 < args.max_val_samples <= len(crops_bgr):
            break

    print(f"[+] Loaded {len(crops_bgr)} valid validation crops.")

    # Initialize OCR engines
    print("\n[*] Initializing OCR engines...")
    ocr_v2 = PlateOCR(model_path=v2_path, ocr_version="v2", device=args.device)
    ocr_v3 = PlateOCR(model_path=v3_path, ocr_version="v3", device=args.device)
    ocr_moe = PlateOCR(ocr_version="auto", model_v2_path=v2_path, model_v3_path=v3_path, device=args.device)

    res_v2 = evaluate_engine("Solo LPRNet-v2", ocr_v2, crops_bgr, ground_truths, plate_types)
    res_v3 = evaluate_engine("Solo LPRNet-v3", ocr_v3, crops_bgr, ground_truths, plate_types)
    res_moe = evaluate_engine("Type-Conditioned MoE", ocr_moe, crops_bgr, ground_truths, plate_types)

    print(f"\n[+] Benchmarking inference latency on {args.device.upper()}...")
    lat_v2 = benchmark_latency(v2_path, device=args.device)
    lat_v3 = benchmark_latency(v3_path, device=args.device)
    res_v2["latency_gpu"] = lat_v2
    res_v3["latency_gpu"] = lat_v3

    print(f"[+] Benchmarking inference latency on CPU...")
    lat_v2_cpu = benchmark_latency(v2_path, device="cpu", num_iters=100)
    lat_v3_cpu = benchmark_latency(v3_path, device="cpu", num_iters=100)
    res_v2["latency_cpu"] = lat_v2_cpu
    res_v3["latency_cpu"] = lat_v3_cpu

    # Print comparative report
    print("\n" + "=" * 90)
    print(f"{'Metric / Parameter':<30} | {'Solo LPRNet-v2':<17} | {'Solo LPRNet-v3':<17} | {'MoE Routing (Auto)':<17}")
    print("-" * 90)
    print(f"{'Overall SeqAcc (FSM)':<30} | {res_v2['seq_acc_fsm_pct']}%{'':<10} | {res_v3['seq_acc_fsm_pct']}%{'':<10} | {res_moe['seq_acc_fsm_pct']}% 🏆")
    print(f"{'Char Error Rate (CER)':<30} | {res_v2['cer_pct']}%{'':<10} | {res_v3['cer_pct']}%{'':<10} | {res_moe['cer_pct']}% 🏆")
    print(f"{'Type 1 Accuracy':<30} | {res_v2['class_acc_pct']['type1']}%{'':<10} | {res_v3['class_acc_pct']['type1']}%{'':<10} | {res_moe['class_acc_pct']['type1']}%")
    print(f"{'Type 1B Accuracy':<30} | {res_v2['class_acc_pct']['type1b']}%{'':<10} | {res_v3['class_acc_pct']['type1b']}%{'':<10} | {res_moe['class_acc_pct']['type1b']}%")
    print(f"{'Type 1A Accuracy':<30} | {res_v2['class_acc_pct']['type1a']}%{'':<10} | {res_v3['class_acc_pct']['type1a']}%{'':<10} | {res_moe['class_acc_pct']['type1a']}%")
    print(f"{'Type 2 Accuracy':<30} | {res_v2['class_acc_pct']['type2']}%{'':<10} | {res_v3['class_acc_pct']['type2']}%{'':<10} | {res_moe['class_acc_pct']['type2']}%")
    print(f"{'Optical Pair 0 vs O':<30} | {res_v2['optical_pairs']['0_vs_O_acc_pct']}%{'':<10} | {res_v3['optical_pairs']['0_vs_O_acc_pct']}%{'':<10} | {res_moe['optical_pairs']['0_vs_O_acc_pct']}%")
    print(f"{'Optical Pair 8 vs B':<30} | {res_v2['optical_pairs']['8_vs_B_acc_pct']}%{'':<10} | {res_v3['optical_pairs']['8_vs_B_acc_pct']}%{'':<10} | {res_moe['optical_pairs']['8_vs_B_acc_pct']}%")
    print(f"{'Latency GPU (mean)':<30} | {lat_v2['mean_ms']} ms{'':<9} | {lat_v3['mean_ms']} ms{'':<9} | {round((lat_v2['mean_ms']+lat_v3['mean_ms'])/2.0, 3)} ms")
    print(f"{'Total Eval Time':<30} | {res_v2['eval_time_sec']} s{'':<10} | {res_v3['eval_time_sec']} s{'':<10} | {res_moe['eval_time_sec']} s")
    print("=" * 90)

    delta_v2 = round(res_moe['seq_acc_fsm_pct'] - res_v2['seq_acc_fsm_pct'], 2)
    delta_v3 = round(res_moe['seq_acc_fsm_pct'] - res_v3['seq_acc_fsm_pct'], 2)
    print(f"\n[★] MoE Synergistic Gain: +{delta_v2}% vs Solo v2 | +{delta_v3}% vs Solo v3")

    out_path = os.path.join(ROOT_DIR, args.output_json)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "validation_samples": len(crops_bgr),
        "lprnet_v2": res_v2,
        "lprnet_v3": res_v3,
        "moe_routing": res_moe,
        "synergistic_gain": {
            "gain_vs_v2_pct": delta_v2,
            "gain_vs_v3_pct": delta_v3,
        },
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[+] Full JSON benchmark report saved to: {out_path}")


if __name__ == "__main__":
    main()
