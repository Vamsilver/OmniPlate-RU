#!/usr/bin/env python3
"""
scripts/research/build_confusion_matrix.py
Extracts empirical character-level substitution confusion matrix by running LPRNet-v2
over verified crops and aligning predictions with ground-truth labels using Needleman-Wunsch.
Exports src/pipeline/confusion_matrix.json for data-driven CTC Beam Search scoring.
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT_DIR)

from src.pipeline.ocr import VOCAB, CHAR2IDX, BLANK_IDX, PlateOCR
from src.pipeline.decoder import CTCDecoder

CYR_TO_LAT: Dict[str, str] = {
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H",
    "О": "O", "Р": "P", "С": "C", "Т": "T", "У": "Y", "Х": "X",
}

def normalize_text(text: str) -> str:
    res = []
    for ch in text.upper().strip():
        ch = CYR_TO_LAT.get(ch, ch)
        if ch in CHAR2IDX:
            res.append(ch)
    return "".join(res)


def needleman_wunsch(seq1: str, seq2: str) -> List[Tuple[str, str]]:
    """
    Global sequence alignment between target (seq1) and predicted (seq2).
    Returns list of pairs (target_char, pred_char). Gap is represented as '-'.
    """
    n, m = len(seq1), len(seq2)
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    
    # Gap penalty = -1, Match = +2, Mismatch = -1
    for i in range(n + 1):
        dp[i, 0] = -i
    for j in range(m + 1):
        dp[0, j] = -j
        
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match_score = 2 if seq1[i - 1] == seq2[j - 1] else -1
            score_diag = dp[i - 1, j - 1] + match_score
            score_up = dp[i - 1, j] - 1
            score_left = dp[i, j - 1] - 1
            dp[i, j] = max(score_diag, score_up, score_left)
            
    # Traceback
    aligned_pairs = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and (dp[i, j] == dp[i - 1, j - 1] + (2 if seq1[i - 1] == seq2[j - 1] else -1)):
            aligned_pairs.append((seq1[i - 1], seq2[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and dp[i, j] == dp[i - 1, j] - 1:
            aligned_pairs.append((seq1[i - 1], "-"))
            i -= 1
        else:
            aligned_pairs.append(("-", seq2[j - 1]))
            j -= 1
            
    aligned_pairs.reverse()
    return aligned_pairs


def main():
    parser = argparse.ArgumentParser(description="Build Empirical Character Confusion Matrix")
    parser.add_argument("--manifest", type=str, default="dataset/verified_crops/manifest.csv")
    parser.add_argument("--crops-dir", type=str, default="dataset/verified_crops")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model", type=str, default=None, help="Path to OCR model checkpoint (.onnx or .pt)")
    parser.add_argument("--max-samples", type=int, default=0, help="0 for all")
    parser.add_argument("--out-json", type=str, default="src/pipeline/confusion_matrix.json")
    args = parser.parse_args()

    manifest_path = os.path.join(ROOT_DIR, args.manifest) if not os.path.isabs(args.manifest) else args.manifest
    crops_dir = os.path.join(ROOT_DIR, args.crops_dir) if not os.path.isabs(args.crops_dir) else args.crops_dir
    out_json_path = os.path.join(ROOT_DIR, args.out_json) if not os.path.isabs(args.out_json) else args.out_json

    if not os.path.exists(manifest_path):
        print(f"Error: Manifest not found at {manifest_path}")
        sys.exit(1)

    if args.model is not None and os.path.exists(args.model):
        default_model = args.model
    else:
        default_model = os.path.join(ROOT_DIR, "models", "ocr_lprnet_best.onnx")
        if not os.path.exists(default_model):
            default_model = os.path.join(ROOT_DIR, "models", "ocr_lprnet_best.pt")
    print(f"Loading OCR model from {default_model} on {args.device}...")
    ocr = PlateOCR(model_path=default_model, device=args.device)

    samples: List[Tuple[str, str, str]] = []
    with open(manifest_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter=";")
        header = next(reader, None)
        for row in reader:
            if len(row) >= 2:
                crop_file, plate_num = row[0], row[1]
                plate_type = row[2] if len(row) > 2 else "type1"
                samples.append((crop_file, normalize_text(plate_num), plate_type))

    if args.max_samples > 0:
        samples = samples[:args.max_samples]

    print(f"Total samples to evaluate: {len(samples)}")

    # Confusion counts: counts[actual_char][predicted_char]
    valid_chars = [c for c in VOCAB if c != "-"]
    confusion_counts = {c: {c2: 0 for c2 in valid_chars} for c in valid_chars}
    
    # Also track deletions and insertions
    deletions = {c: 0 for c in valid_chars}
    insertions = {c: 0 for c in valid_chars}

    total_chars = 0
    total_mismatches = 0
    exact_plate_matches = 0

    batch_size = 64
    for b_idx in range(0, len(samples), batch_size):
        batch_samples = samples[b_idx:b_idx + batch_size]
        batch_imgs = []
        valid_batch = []
        for c_file, gt_text, p_type in batch_samples:
            img_path = os.path.join(crops_dir, c_file)
            if not os.path.exists(img_path):
                continue
            img = cv2.imread(img_path)
            if img is None:
                continue
            batch_imgs.append(img)
            valid_batch.append((c_file, gt_text, p_type))

        if not batch_imgs:
            continue

        # Predict using PlateOCR batch
        p_types = [pt for _, _, pt in valid_batch]
        preds = ocr.predict_batch(batch_imgs, plate_types=p_types, return_type=False)
        for (c_file, gt_text, p_type), (pred_text, conf) in zip(valid_batch, preds):
            norm_gt = normalize_text(gt_text)
            norm_pred = normalize_text(pred_text)

            if norm_gt == norm_pred:
                exact_plate_matches += 1

            alignments = needleman_wunsch(norm_gt, norm_pred)
            for true_c, pred_c in alignments:
                if true_c == "-":
                    if pred_c in insertions:
                        insertions[pred_c] += 1
                elif pred_c == "-":
                    if true_c in deletions:
                        deletions[true_c] += 1
                else:
                    if true_c in confusion_counts and pred_c in confusion_counts[true_c]:
                        confusion_counts[true_c][pred_c] += 1
                        total_chars += 1
                        if true_c != pred_c:
                            total_mismatches += 1

        if (b_idx // batch_size) % 20 == 0 or b_idx + batch_size >= len(samples):
            print(f"Processed {min(b_idx + batch_size, len(samples))}/{len(samples)} samples... "
                  f"(Acc: {exact_plate_matches}/{max(1, min(b_idx + batch_size, len(samples)))} "
                  f"[{100.0 * exact_plate_matches / max(1, min(b_idx + batch_size, len(samples))):.2f}%])")

    print("\n--- Empirical Confusion Matrix Analysis ---")
    print(f"Total Evaluated Characters: {total_chars}")
    print(f"Total Character Mismatches: {total_mismatches} ({100.0 * total_mismatches / max(1, total_chars):.2f}% CER)")
    print(f"Exact Plate Accuracy: {exact_plate_matches}/{len(samples)} ({100.0 * exact_plate_matches / max(1, len(samples)):.2f}%)")

    # Compute conditional probabilities P(pred | true) and log probabilities
    # Laplace smoothing alpha = 0.5
    alpha = 0.5
    matrix_export = {
        "chars": valid_chars,
        "total_chars_evaluated": total_chars,
        "total_mismatches": total_mismatches,
        "exact_plate_accuracy": round(exact_plate_matches / max(1, len(samples)), 4),
        "probabilities": {},
        "log_probabilities": {},
        "top_confusions": []
    }

    all_confusions = []
    for true_c in valid_chars:
        total_true = sum(confusion_counts[true_c].values()) + alpha * len(valid_chars)
        matrix_export["probabilities"][true_c] = {}
        matrix_export["log_probabilities"][true_c] = {}

        for pred_c in valid_chars:
            p = (confusion_counts[true_c][pred_c] + alpha) / total_true
            matrix_export["probabilities"][true_c][pred_c] = round(float(p), 6)
            matrix_export["log_probabilities"][true_c][pred_c] = round(float(np.log(p)), 4)
            if true_c != pred_c and confusion_counts[true_c][pred_c] > 0:
                all_confusions.append((true_c, pred_c, confusion_counts[true_c][pred_c], p))

    # Sort top confusions
    all_confusions.sort(key=lambda x: x[2], reverse=True)
    matrix_export["top_confusions"] = [
        {"true": tc, "pred": pc, "count": cnt, "prob": round(pr, 4)}
        for tc, pc, cnt, pr in all_confusions[:25]
    ]

    print("\nTop 15 Most Frequent Optical Confusions:")
    for item in matrix_export["top_confusions"][:15]:
        print(f"  True '{item['true']}' -> Pred '{item['pred']}': {item['count']} times (P = {item['prob']:.4f})")

    os.makedirs(os.path.dirname(out_json_path), exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(matrix_export, f, indent=2, ensure_ascii=False)

    print(f"\nSaved confusion matrix to: {out_json_path}")


if __name__ == "__main__":
    main()
