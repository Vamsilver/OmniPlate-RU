#!/usr/bin/env python3
"""
scripts/research/benchmark_context_expansion.py
Stage 4.5 Architectural Research & Simulation:
1. Context Margin Padding (grid search over alpha_x, alpha_y on real plates).
2. Multi-Scale Rescaling & Distant Plate Optimization (interpolation & unsharp filters).
3. LPRNet Receptive Field Modeling (layer-by-layer theoretical RF & latency analysis).
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.ocr import PlateOCR
from src.pipeline.rectifier import PlateRectifier
from src.pipeline.decoder import is_valid_gost_plate

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


def apply_unsharp_mask(image: np.ndarray, sigma: float = 1.0, strength: float = 0.6) -> np.ndarray:
    """Sharpens subtle stroke edges on low-res crops using unsharp masking."""
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def calculate_receptive_field() -> Dict[str, Any]:
    """
    Theoretically calculates the effective receptive field (RF) and feature map dimensions
    at each stage of LPRNet, comparing Standard vs Dilated Stage 3/4.
    """
    # Architecture layers: (name, kernel_w, stride_w, dilation_w)
    layers_standard = [
        ("stem_conv", 3, 1, 1),
        ("stem_pool", 3, 2, 1),
        ("block1_conv", 3, 1, 1),
        ("pool1", 3, 2, 1),
        ("block2_conv", 3, 1, 1),
        ("block3_conv", 3, 1, 1),
        ("pool2", 1, 1, 1),  # stride 1 in width!
        ("block4_conv", 3, 1, 1),
    ]

    layers_dilated = [
        ("stem_conv", 3, 1, 1),
        ("stem_pool", 3, 2, 1),
        ("block1_conv", 3, 1, 1),
        ("pool1", 3, 2, 1),
        ("block2_conv", 3, 1, 1),
        ("block3_conv", 3, 1, 2),  # dilated d=2
        ("pool2", 1, 1, 1),
        ("block4_conv", 3, 1, 3),  # dilated d=3
    ]

    def _calc_rf(layers):
        current_rf = 1
        current_jump = 1
        history = []
        for name, k, s, d in layers:
            eff_k = k + (k - 1) * (d - 1)
            current_rf += (eff_k - 1) * current_jump
            current_jump *= s
            history.append({"layer": name, "rf_w": current_rf, "jump_w": current_jump})
        return current_rf, history

    rf_std, hist_std = _calc_rf(layers_standard)
    rf_dil, hist_dil = _calc_rf(layers_dilated)

    return {
        "standard_rf_px": rf_std,
        "standard_pct_of_160": round((rf_std / 160.0) * 100.0, 1),
        "dilated_rf_px": rf_dil,
        "dilated_pct_of_160": round((rf_dil / 160.0) * 100.0, 1),
        "history_standard": hist_std,
        "history_dilated": hist_dil,
    }


def load_real_samples(meta_csv: str, root_dir: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    samples = []
    with open(meta_csv, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter=";")
        header = next(reader)
        for row in reader:
            if len(row) < 10:
                continue
            img_rel, plate_num, p_type, bbox_str, quad_str, is_veh, is_syn = row[:7]
            if is_syn == "1":
                continue
            if p_type not in ("type1", "type1a", "type1b"):
                continue

            full_path = os.path.join(root_dir, img_rel)
            if not os.path.exists(full_path):
                continue

            norm_num = normalize_plate(plate_num)
            if not norm_num:
                continue

            samples.append({
                "path": full_path,
                "plate_num": norm_num,
                "plate_type": p_type,
                "bbox": bbox_str,
                "quad": quad_str,
            })
            if max_samples and len(samples) >= max_samples:
                break
    return samples


def run_context_experiment(
    samples: List[Dict[str, Any]],
    ocr: PlateOCR,
    rectifier: PlateRectifier,
) -> Dict[str, Any]:
    print(f"\n=================================================================")
    print(f"  Stage 4.5: Context Margin Padding Experiment (N={len(samples)})")
    print(f"=================================================================")

    # Test margins: (margin_x, margin_y)
    margin_configs = [
        ((0.000, 0.000), "Zero margin [0.0%, 0.0%]"),
        ((0.010, 0.005), "Current prod [1.0%, 0.5%]"),
        ((0.020, 0.015), "Balanced [2.0%, 1.5%]"),
        ((0.030, 0.020), "Expanded [3.0%, 2.0%]"),
        ((0.040, 0.025), "Wide context [4.0%, 2.5%]"),
        ((0.050, 0.030), "Heavy context [5.0%, 3.0%]"),
    ]

    results_by_margin = {}

    for margin, label in margin_configs:
        t0 = time.perf_counter()
        seq_matches = 0
        first_char_matches = 0
        last_char_matches = 0
        total_ned = 0.0
        total_crops = 0

        for s in samples:
            img = cv2.imread(s["path"])
            if img is None:
                continue

            quad = s["quad"]
            p_type = s["plate_type"]
            gt = s["plate_num"]

            try:
                if p_type == "type1a":
                    warped = rectifier.rectify(img, quad, plate_type="type1a", margin=margin, refine_corners=False)
                    top_l, bot_l = rectifier.split_type1a(warped)
                    crop = rectifier.stitch_type1a_horizontal(top_l, bot_l, target_size=(160, 36))
                else:
                    crop = rectifier.rectify(img, quad, plate_type=p_type, margin=margin, refine_corners=False)

                pred_text, conf = ocr.predict_single(crop, plate_type=p_type)
            except Exception:
                pred_text = ""

            pred_norm = normalize_plate(pred_text)

            is_match = is_seq_match(pred_norm, gt)
            if is_match:
                seq_matches += 1

            if gt and pred_norm:
                if pred_norm[0] == gt[0]:
                    first_char_matches += 1
                if pred_norm[-1] == gt[-1]:
                    last_char_matches += 1

            dist = levenshtein_dist(pred_norm, gt)
            max_l = max(len(pred_norm), len(gt), 1)
            ned = max(0.0, 1.0 - (dist / float(max_l)))
            total_ned += ned
            total_crops += 1

        elapsed = time.perf_counter() - t0
        seq_acc = (seq_matches / total_crops * 100.0) if total_crops > 0 else 0.0
        first_acc = (first_char_matches / total_crops * 100.0) if total_crops > 0 else 0.0
        last_acc = (last_char_matches / total_crops * 100.0) if total_crops > 0 else 0.0
        mean_ned = (total_ned / total_crops * 100.0) if total_crops > 0 else 0.0
        ms_per_crop = (elapsed / total_crops * 1000.0) if total_crops > 0 else 0.0

        results_by_margin[label] = {
            "margin": list(margin),
            "seq_acc": round(seq_acc, 2),
            "first_char_acc": round(first_acc, 2),
            "last_char_acc": round(last_acc, 2),
            "mean_ned": round(mean_ned, 2),
            "ms_per_crop": round(ms_per_crop, 2),
        }
        print(f"  • {label:<28} | SeqAcc: {seq_acc:5.2f}% | FirstChar: {first_acc:5.2f}% | LastChar: {last_acc:5.2f}% | NED: {mean_ned:5.2f}% | Latency: {ms_per_crop:.2f}ms")

    return results_by_margin


def run_multiscale_experiment(
    samples: List[Dict[str, Any]],
    ocr: PlateOCR,
    rectifier: PlateRectifier,
) -> Dict[str, Any]:
    print(f"\n=================================================================")
    print(f"  Stage 4.5: Multi-Scale Rescaling & Distant Plate Experiment")
    print(f"=================================================================")

    # Filter for distant / low-res plates: bbox height <= 25 px or bbox width <= 95 px
    distant_samples = []
    for s in samples:
        try:
            bx, by, bw, bh = [float(v.strip()) for v in s["bbox"].split(",") if v.strip()]
            if bh <= 25.0 or bw <= 95.0 or (bw * bh) < 2200.0:
                distant_samples.append(s)
        except Exception:
            continue

    print(f"[*] Found {len(distant_samples)} distant/small plate scenes out of {len(samples)} total.")
    if not distant_samples:
        distant_samples = samples[:50]  # Fallback subset

    modes = [
        ("linear_base", cv2.INTER_LINEAR, False, "Bilinear (Standard Baseline)"),
        ("cubic", cv2.INTER_CUBIC, False, "Bicubic Interpolation"),
        ("lanczos", cv2.INTER_LANCZOS4, False, "Lanczos4 Interpolation"),
        ("cubic_unsharp", cv2.INTER_CUBIC, True, "Bicubic + Unsharp Mask (γ=0.5)"),
    ]

    results = {}
    for key, interp, use_unsharp, label in modes:
        t0 = time.perf_counter()
        seq_matches = 0
        total_ned = 0.0
        total_crops = 0

        rect_test = PlateRectifier(default_interpolation=interp)

        for s in distant_samples:
            img = cv2.imread(s["path"])
            if img is None:
                continue

            quad = s["quad"]
            p_type = s["plate_type"]
            gt = s["plate_num"]

            try:
                if p_type == "type1a":
                    warped = rect_test.rectify(img, quad, plate_type="type1a", margin=(0.02, 0.015), refine_corners=False)
                    top_l, bot_l = rect_test.split_type1a(warped)
                    crop = rect_test.stitch_type1a_horizontal(top_l, bot_l, target_size=(160, 36))
                else:
                    crop = rect_test.rectify(img, quad, plate_type=p_type, margin=(0.02, 0.015), refine_corners=False)

                if use_unsharp:
                    crop = apply_unsharp_mask(crop, sigma=1.0, strength=0.5)

                pred_text, conf = ocr.predict_single(crop, plate_type=p_type)
            except Exception:
                pred_text = ""

            pred_norm = normalize_plate(pred_text)
            if is_seq_match(pred_norm, gt):
                seq_matches += 1

            dist = levenshtein_dist(pred_norm, gt)
            max_l = max(len(pred_norm), len(gt), 1)
            total_ned += max(0.0, 1.0 - (dist / float(max_l)))
            total_crops += 1

        elapsed = time.perf_counter() - t0
        seq_acc = (seq_matches / total_crops * 100.0) if total_crops > 0 else 0.0
        mean_ned = (total_ned / total_crops * 100.0) if total_crops > 0 else 0.0
        ms_per_crop = (elapsed / total_crops * 1000.0) if total_crops > 0 else 0.0

        results[key] = {
            "label": label,
            "seq_acc": round(seq_acc, 2),
            "mean_ned": round(mean_ned, 2),
            "ms_per_crop": round(ms_per_crop, 2),
        }
        print(f"  • {label:<35} | SeqAcc: {seq_acc:5.2f}% | NED: {mean_ned:5.2f}% | Latency: {ms_per_crop:.2f}ms")

    return results


def main():
    parser = argparse.ArgumentParser(description="Stage 4.5 Context & Multi-Scale Expansion Benchmark")
    parser.add_argument("--max_samples", type=int, default=200, help="Max real samples for grid search")
    parser.add_argument("--device", type=str, default="cuda", help="Inference device")
    parser.add_argument("--model_path", type=str, default=None, help="Path to OCR model checkpoint or ONNX")
    args = parser.parse_args()

    meta_csv = str(PROJECT_ROOT / "dataset" / "meta.csv")
    root_dir = str(PROJECT_ROOT / "dataset")

    print("[*] Loading real road samples from dataset/meta.csv...")
    samples = load_real_samples(meta_csv, root_dir, max_samples=args.max_samples)
    print(f"[+] Loaded {len(samples)} real road plate samples.")

    ocr_model_p = args.model_path or str(PROJECT_ROOT / "models" / "ocr_lprnet_best.onnx")
    print(f"[*] Benchmarking OCR Model: {ocr_model_p}")
    ocr = PlateOCR(model_path=ocr_model_p, device=args.device, use_onnx=ocr_model_p.endswith(".onnx"))
    rectifier = PlateRectifier()

    # 1. Theoretical Receptive Field Calculation
    print("\n=================================================================")
    print("  Stage 4.5: LPRNet Receptive Field Modeling")
    print("=================================================================")
    rf_report = calculate_receptive_field()
    print(f"  • Standard LPRNet Horizontal RF: {rf_report['standard_rf_px']} px ({rf_report['standard_pct_of_160']}% of plate width)")
    print(f"  • Dilated (d=2, d=3) Horizontal RF: {rf_report['dilated_rf_px']} px ({rf_report['dilated_pct_of_160']}% of plate width)")
    print("  • Detailed Layer Trajectory:")
    for step in rf_report["history_dilated"]:
        print(f"    - Layer {step['layer']:<15}: RF_W = {step['rf_w']:2d} px | Jump = {step['jump_w']:2d} px")

    # 2. Context Margin Padding Grid Search
    margin_results = run_context_experiment(samples, ocr, rectifier)

    # 3. Multi-Scale & Distant Plate Optimization
    multiscale_results = run_multiscale_experiment(samples, ocr, rectifier)

    # Save complete report
    out_dir = PROJECT_ROOT / "docs" / "report"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "context_expansion_results.json"

    full_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "samples_evaluated": len(samples),
        "receptive_field_analysis": rf_report,
        "context_margin_padding": margin_results,
        "multiscale_distant_optimization": multiscale_results,
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)

    print(f"\n[+] Research results written to {out_json}")


if __name__ == "__main__":
    main()
