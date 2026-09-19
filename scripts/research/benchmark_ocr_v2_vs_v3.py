#!/usr/bin/env python3
"""
scripts/research/benchmark_ocr_v2_vs_v3.py
Head-to-head comparative benchmark between LPRNet-v2 and LPRNet-v3.

Metrics evaluated:
1. Sequence Accuracy (Exact Match %) under Greedy + GOST and FSM Beam Search.
2. Character Error Rate (CER %) using Levenshtein distance.
3. Class-specific Accuracy (Type 1, Type 1A, Type 1B, Type 2).
4. Optical confusion accuracy ('0'<->'O', '8'<->'B').
5. Latency benchmark (ms/crop, p50, p95, min, max on GPU & CPU).
6. Model file size (KB).
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
    CHAR2IDX,
    IDX2CHAR,
    NUM_CLASSES,
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


def benchmark_latency(onnx_path: str, device: str = "cuda", num_iters: int = 300) -> Dict[str, float]:
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
    session = ort.InferenceSession(onnx_path, providers=providers)
    input_name = session.get_inputs()[0].name

    dummy_input = np.random.randn(1, 3, 36, 160).astype(np.float32)

    for _ in range(50):
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


def evaluate_model(
    model_name: str,
    onnx_path: str,
    crops_bgr: List[np.ndarray],
    ground_truths: List[str],
    plate_types: List[str],
    device: str = "cuda",
) -> Dict[str, any]:
    print(f"\n[+] Evaluating {model_name} on {len(crops_bgr)} validation crops ({device.upper()})...")

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
    session = ort.InferenceSession(onnx_path, providers=providers)
    input_name = session.get_inputs()[0].name

    batch_size = 64
    all_logits = []

    t0 = time.time()
    for b_start in range(0, len(crops_bgr), batch_size):
        b_crops = crops_bgr[b_start : b_start + batch_size]
        batch_tensors = []
        for c in b_crops:
            rgb = cv2.cvtColor(c, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            t = np.transpose(rgb, (2, 0, 1))
            batch_tensors.append(t)
        batch_arr = np.stack(batch_tensors, axis=0)
        out = session.run(None, {input_name: batch_arr})[0]
        all_logits.append(out)

    all_logits = np.concatenate(all_logits, axis=0)
    total_eval_time = time.time() - t0

    greedy_exact = 0
    fsm_exact = 0
    total_samples = len(ground_truths)

    total_edit_distance = 0
    total_gt_chars = 0

    class_exact = {"type1": 0, "type1a": 0, "type1b": 0, "type2": 0}
    class_total = {"type1": 0, "type1a": 0, "type1b": 0, "type2": 0}

    optical_zero_o_total = 0
    optical_zero_o_correct = 0
    optical_eight_b_total = 0
    optical_eight_b_correct = 0

    for idx in range(total_samples):
        logits_step = all_logits[idx]
        gt = ground_truths[idx]
        pt = plate_types[idx]

        greedy_indices = np.argmax(logits_step, axis=-1)
        pred_raw = CTCDecoder.decode_greedy(greedy_indices, blank_idx=BLANK_IDX)
        pred_greedy = CTCDecoder.apply_gost_heuristics(pred_raw, plate_type=pt)

        pred_fsm, det_type, conf = CTCDecoder.score_hypotheses(
            logits_step,
            plate_type_prior=pt,
            blank_idx=BLANK_IDX,
            use_beam_search=True,
            beam_width=10,
        )

        if pred_greedy == gt:
            greedy_exact += 1
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

    seq_acc_greedy = (greedy_exact / total_samples) * 100.0
    seq_acc_fsm = (fsm_exact / total_samples) * 100.0
    cer = (total_edit_distance / max(1, total_gt_chars)) * 100.0

    class_accuracies = {}
    for c_name in class_total:
        cnt = class_total[c_name]
        acc = (class_exact[c_name] / max(1, cnt)) * 100.0 if cnt > 0 else 0.0
        class_accuracies[c_name] = round(acc, 2)

    zero_o_acc = (optical_zero_o_correct / max(1, optical_zero_o_total)) * 100.0 if optical_zero_o_total > 0 else 0.0
    eight_b_acc = (optical_eight_b_correct / max(1, optical_eight_b_total)) * 100.0 if optical_eight_b_total > 0 else 0.0

    file_size_kb = round(os.path.getsize(onnx_path) / 1024.0, 1)

    return {
        "model_name": model_name,
        "onnx_path": onnx_path,
        "file_size_kb": file_size_kb,
        "total_samples": total_samples,
        "eval_time_sec": round(total_eval_time, 2),
        "seq_acc_greedy_pct": round(seq_acc_greedy, 2),
        "seq_acc_fsm_pct": round(seq_acc_fsm, 2),
        "cer_pct": round(cer, 3),
        "class_acc_pct": class_accuracies,
        "class_counts": class_total,
        "optical_pairs": {
            "0_vs_O_acc_pct": round(zero_o_acc, 2),
            "8_vs_B_acc_pct": round(eight_b_acc, 2),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Head-to-Head Benchmark: LPRNet-v2 vs LPRNet-v3")
    parser.add_argument("--v2-onnx", type=str, default="models/ocr_lprnet_best.onnx")
    parser.add_argument("--v3-onnx", type=str, default="models/ocr_lprnet_v3.onnx")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--output-json", type=str, default="test_output/benchmark_v2_vs_v3_report.json")
    args = parser.parse_args()

    v2_path = os.path.join(ROOT_DIR, args.v2_onnx)
    v3_path = os.path.join(ROOT_DIR, args.v3_onnx)

    if not os.path.exists(v2_path):
        print(f"[!] Error: v2 model not found: {v2_path}")
        sys.exit(1)
    if not os.path.exists(v3_path):
        print(f"[!] Error: v3 model not found: {v3_path}")
        sys.exit(1)

    print("=" * 80)
    print("      HEAD-TO-HEAD OCR BENCHMARK: LPRNet-v2 vs LPRNet-v3 (50 Epochs)")
    print("=" * 80)

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

    res_v2 = evaluate_model("LPRNet-v2 (Dilated d=2,3)", v2_path, crops_bgr, ground_truths, plate_types, device=args.device)
    res_v3 = evaluate_model("LPRNet-v3 (1D-ASPP + ECA)", v3_path, crops_bgr, ground_truths, plate_types, device=args.device)

    print(f"\n[+] Benchmarking inference latency on {args.device.upper()} (300 runs)...")
    lat_v2 = benchmark_latency(v2_path, device=args.device)
    lat_v3 = benchmark_latency(v3_path, device=args.device)
    res_v2["latency"] = lat_v2
    res_v3["latency"] = lat_v3

    print(f"[+] Benchmarking inference latency on CPU (100 runs)...")
    lat_v2_cpu = benchmark_latency(v2_path, device="cpu", num_iters=100)
    lat_v3_cpu = benchmark_latency(v3_path, device="cpu", num_iters=100)
    res_v2["latency_cpu"] = lat_v2_cpu
    res_v3["latency_cpu"] = lat_v3_cpu

    print("\n" + "=" * 85)
    print(f"{'Metric / Parameter':<35} | {'LPRNet-v2':<22} | {'LPRNet-v3 (50 ep)':<22}")
    print("-" * 85)
    print(f"{'ONNX Model Size':<35} | {res_v2['file_size_kb']} KB{'':<14} | {res_v3['file_size_kb']} KB")
    print(f"{'FSM Exact Match (SeqAcc)':<35} | {res_v2['seq_acc_fsm_pct']}%{'':<15} | {res_v3['seq_acc_fsm_pct']}%")
    print(f"{'Greedy Exact Match':<35} | {res_v2['seq_acc_greedy_pct']}%{'':<15} | {res_v3['seq_acc_greedy_pct']}%")
    print(f"{'Character Error Rate (CER)':<35} | {res_v2['cer_pct']}%{'':<15} | {res_v3['cer_pct']}%")
    print(f"{'Type 1 Accuracy':<35} | {res_v2['class_acc_pct']['type1']}%{'':<15} | {res_v3['class_acc_pct']['type1']}%")
    print(f"{'Type 1A Accuracy':<35} | {res_v2['class_acc_pct']['type1a']}%{'':<15} | {res_v3['class_acc_pct']['type1a']}%")
    print(f"{'Type 1B Accuracy':<35} | {res_v2['class_acc_pct']['type1b']}%{'':<15} | {res_v3['class_acc_pct']['type1b']}%")
    print(f"{'Type 2 Accuracy':<35} | {res_v2['class_acc_pct']['type2']}%{'':<15} | {res_v3['class_acc_pct']['type2']}%")
    print(f"{'Optical Pair 0 vs O':<35} | {res_v2['optical_pairs']['0_vs_O_acc_pct']}%{'':<15} | {res_v3['optical_pairs']['0_vs_O_acc_pct']}%")
    print(f"{'Optical Pair 8 vs B':<35} | {res_v2['optical_pairs']['8_vs_B_acc_pct']}%{'':<15} | {res_v3['optical_pairs']['8_vs_B_acc_pct']}%")
    print(f"{'Latency GPU (mean)':<35} | {lat_v2['mean_ms']} ms{'':<14} | {lat_v3['mean_ms']} ms")
    print(f"{'Latency GPU (p95)':<35} | {lat_v2['p95_ms']} ms{'':<14} | {lat_v3['p95_ms']} ms")
    print(f"{'Latency CPU (mean)':<35} | {lat_v2_cpu['mean_ms']} ms{'':<14} | {lat_v3_cpu['mean_ms']} ms")
    print("=" * 85)

    out_path = os.path.join(ROOT_DIR, args.output_json)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "validation_samples": len(crops_bgr),
        "lprnet_v2": res_v2,
        "lprnet_v3": res_v3,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[+] Full JSON benchmark report saved to: {out_path}")


if __name__ == "__main__":
    main()
