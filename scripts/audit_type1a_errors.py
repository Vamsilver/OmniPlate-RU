#!/usr/bin/env python3
"""
Comprehensive Audit of Remaining Type 1A Errors.
Volga IT 2026: Automatic Vehicle License Plate Recognition.
"""

import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.pipeline import OmniPlatePipeline

CYR_TO_LAT = {
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H",
    "О": "O", "Р": "P", "С": "C", "Т": "T", "У": "Y", "Х": "X",
}


def normalize_plate(s: str) -> str:
    s = s.upper().replace(" ", "").replace("\t", "").replace("\n", "")
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


def audit_type1a():
    dataset_dir = PROJECT_ROOT / "dataset"
    meta_path = dataset_dir / "meta.csv"

    rows = []
    with open(meta_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for r in reader:
            if r.get("is_synthetic", "0").strip() == "0" and r.get("plate_type", "").strip() == "type1a":
                rows.append(r)

    print(f"[*] Loaded {len(rows)} real Type 1A images from meta.csv")

    pipeline = OmniPlatePipeline(device="cuda", conf_threshold=0.12, ocr_version="moe")
    pipeline.warmup(2)

    results = []
    exact_matches = 0
    errors = []

    for idx, r in enumerate(rows):
        img_rel = r["image"].strip()
        img_path = dataset_dir / img_rel
        gt_num = normalize_plate(r["plate_num"])
        gt_has_hash = "#" in gt_num
        hash_count = gt_num.count("#")

        if not img_path.exists():
            errors.append({
                "image": img_rel,
                "error_type": "MISSING_FILE",
                "gt_num": gt_num,
                "pred_num": "",
                "gt_has_hash": gt_has_hash,
            })
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            errors.append({
                "image": img_rel,
                "error_type": "CORRUPT_IMAGE",
                "gt_num": gt_num,
                "pred_num": "",
                "gt_has_hash": gt_has_hash,
            })
            continue

        dets = pipeline.predict(img)
        if not dets:
            errors.append({
                "image": img_rel,
                "error_type": "DETECTION_MISSED",
                "gt_num": gt_num,
                "pred_num": "",
                "gt_has_hash": gt_has_hash,
                "hash_count": hash_count,
            })
            continue

        best_det = dets[0]
        pred_num = normalize_plate(best_det.text)
        pred_type = best_det.plate_type
        det_conf = round(best_det.confidence, 4)
        ocr_conf = round(best_det.ocr_confidence, 4)

        match = is_seq_match(pred_num, gt_num)
        dist = levenshtein_dist(pred_num, gt_num)

        rec = {
            "image": img_rel,
            "gt_num": gt_num,
            "pred_num": pred_num,
            "pred_type": pred_type,
            "det_conf": det_conf,
            "ocr_conf": ocr_conf,
            "is_match": match,
            "dist": dist,
            "gt_has_hash": gt_has_hash,
            "hash_count": hash_count,
        }

        if match:
            exact_matches += 1
        else:
            # Categorize error
            if pred_type != "type1a":
                rec["category"] = f"MISCLASSIFIED_AS_{pred_type.upper()}"
            elif gt_has_hash:
                rec["category"] = "GT_CONTAINS_HASH"
            elif dist == 1:
                rec["category"] = "SINGLE_CHAR_OCR_ERROR"
            elif dist == 2:
                rec["category"] = "TWO_CHAR_OCR_ERROR"
            else:
                rec["category"] = "MULTI_CHAR_OR_SYNTAX_ERROR"

            # Analyze mismatch positions
            diffs = []
            if len(pred_num) == len(gt_num):
                for p_idx, (cp, ct) in enumerate(zip(pred_num, gt_num)):
                    if ct != "#" and cp != ct:
                        diffs.append((p_idx, ct, cp))
            rec["char_diffs"] = diffs
            errors.append(rec)

        results.append(rec)

    total = len(rows)
    acc = (exact_matches / total) * 100.0
    print("\n" + "=" * 65)
    print(f"TYPE 1A AUDIT SUMMARY (Total: {total})")
    print(f"  Exact Matches: {exact_matches}/{total} ({acc:.2f}%)")
    print(f"  Total Errors:  {len(errors)}")
    print("=" * 65)

    # Breakdown by category
    cat_counts = Counter(e.get("category", e.get("error_type", "UNKNOWN")) for e in errors)
    for cat, count in cat_counts.most_common():
        print(f"  • {cat:<32}: {count:>3} ({count / len(errors) * 100:.1f}% of errors)")

    # Breakdown of GT with #
    gt_hash_errors = [e for e in errors if e.get("gt_has_hash")]
    print(f"\n[# Analysis] Errors where GT contains '#': {len(gt_hash_errors)}")
    for e in gt_hash_errors:
        print(f"    - {e['image']}: GT='{e['gt_num']}' vs Pred='{e.get('pred_num', '')}' (type: {e.get('pred_type')}, dist: {e.get('dist')})")

    # Optical confusions in single-char errors
    single_char_diffs = []
    for e in errors:
        for diff in e.get("char_diffs", []):
            pos, expected, got = diff
            single_char_diffs.append((pos, f"{expected}->{got}"))

    print("\n[Character Confusion Patterns]")
    diff_counter = Counter(d[1] for d in single_char_diffs)
    for pair, count in diff_counter.most_common(15):
        print(f"    {pair:<10}: {count} occurrences")

    print("\n[Error Positions in Plate (0-indexed)]")
    pos_counter = Counter(d[0] for d in single_char_diffs)
    for pos, count in sorted(pos_counter.items()):
        pos_desc = (
            "Letter 1" if pos == 0 else
            f"Digit {pos}" if pos in (1, 2, 3) else
            f"Letter {pos - 2}" if pos in (4, 5) else
            f"Region digit {pos - 5}"
        )
        print(f"    Pos {pos} ({pos_desc:<14}): {count} errors")

    # Save detailed JSON report
    out_dir = PROJECT_ROOT / "test_output"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = out_dir / "audit_type1a_75_errors.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump({
            "total_type1a": total,
            "exact_matches": exact_matches,
            "sequence_accuracy": round(acc, 2),
            "total_errors": len(errors),
            "category_breakdown": dict(cat_counts),
            "errors": errors,
        }, f, indent=2, ensure_ascii=False)

    print(f"\n[+] Detailed report saved to: {report_file}")


if __name__ == "__main__":
    audit_type1a()
