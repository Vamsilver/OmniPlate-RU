#!/usr/bin/env python3
"""
scripts/extract_type1a_lines.py
Procedurally extracts 4,606 canonical (160x36) line crops for Type 1A square plates
from dataset/meta.csv:
- 2,303 Top lines (L DDD, length 4) as 'type1a_top'
- 2,303 Bottom lines (LL RR/RRR, length 4 or 5) as 'type1a_bot'
Uses PlateRectifier with subpixel corner refinement and adaptive seam search.
Saves crops into dataset/type1a_lines/crops/ and generates dataset/type1a_lines/manifest.csv.
"""

import csv
import os
import sys
import time
from typing import Dict, List

import cv2
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.pipeline.rectifier import PlateRectifier

CYR_TO_LAT: Dict[str, str] = {
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H",
    "О": "O", "Р": "P", "С": "C", "Т": "T", "У": "Y", "Х": "X",
}


def normalize_plate_str(raw: str) -> str:
    if not raw:
        return ""
    out = []
    for c in raw.strip().upper():
        out.append(CYR_TO_LAT.get(c, c))
    return "".join(out)


def main():
    dataset_dir = os.path.join(PROJECT_ROOT, "dataset")
    meta_path = os.path.join(dataset_dir, "meta.csv")
    out_dir = os.path.join(dataset_dir, "type1a_lines")
    crops_dir = os.path.join(out_dir, "crops")
    manifest_path = os.path.join(out_dir, "manifest.csv")

    os.makedirs(crops_dir, exist_ok=True)

    print("=" * 65)
    print("  OmniPlate-RU — Procedural Type 1A Line Crop Extractor")
    print(f"  Meta CSV:     {meta_path}")
    print(f"  Output Dir:   {out_dir}")
    print("=" * 65)

    rectifier = PlateRectifier()

    rows = []
    with open(meta_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter=";")
        for r in reader:
            if r.get("plate_type", "").strip() == "type1a":
                rows.append(r)

    total_plates = len(rows)
    print(f"[+] Found {total_plates} Type 1A plate annotations in meta.csv")

    manifest_entries: List[Dict[str, str]] = []
    t0 = time.time()
    extracted_top = 0
    extracted_bot = 0

    for idx, row in enumerate(rows):
        img_rel = row["image"].strip()
        img_path = os.path.join(dataset_dir, img_rel)
        plate_num_raw = row.get("plate_num", "").strip()
        plate_num = normalize_plate_str(plate_num_raw)
        quad_str = row.get("quad", "").strip()
        bbox_str = row.get("bbox", "").strip()
        is_syn = row.get("is_synthetic", "0").strip()

        img = cv2.imread(img_path)
        if img is None:
            print(f"[!] Warning: Cannot read image {img_path}")
            continue

        # Rectify plate to canonical Type 1A dimensions (96x160)
        try:
            if quad_str and len([v for v in quad_str.split(",") if v.strip()]) == 8:
                rect = rectifier.rectify(
                    img,
                    quad_str,
                    plate_type="type1a",
                    margin=(0.020, 0.015),
                    refine_corners=True,
                )
            elif bbox_str and len([v for v in bbox_str.split(",") if v.strip()]) == 4:
                rect = rectifier.rectify_bbox(img, bbox_str, plate_type="type1a")
            else:
                rect = cv2.resize(img, (160, 96), interpolation=cv2.INTER_LINEAR)
        except Exception as e:
            print(f"[!] Rectify error for {img_rel}: {e}")
            rect = cv2.resize(img, (160, 96), interpolation=cv2.INTER_LINEAR)

        # Adaptive seam split
        mid_seam = rectifier.find_adaptive_split_seam(rect)
        top_crop = rect[:mid_seam, :]
        bot_crop = rect[mid_seam:, :]

        top_160x36 = cv2.resize(top_crop, (160, 36), interpolation=cv2.INTER_LINEAR)
        bot_160x36 = cv2.resize(bot_crop, (160, 36), interpolation=cv2.INTER_LINEAR)

        top_num = plate_num[:4]
        bot_num = plate_num[4:]

        top_fname = f"top_{idx:05d}_{top_num}.png"
        bot_fname = f"bot_{idx:05d}_{bot_num}.png"

        top_save_path = os.path.join(crops_dir, top_fname)
        bot_save_path = os.path.join(crops_dir, bot_fname)

        cv2.imwrite(top_save_path, top_160x36)
        cv2.imwrite(bot_save_path, bot_160x36)

        src_label = "meta_synth" if is_syn == "1" else "meta_real"

        # Manifest entries with relative path from dataset root
        rel_top = os.path.join("type1a_lines", "crops", top_fname).replace("\\", "/")
        rel_bot = os.path.join("type1a_lines", "crops", bot_fname).replace("\\", "/")

        manifest_entries.append({
            "crop_file": rel_top,
            "plate_num": top_num,
            "plate_type": "type1a_top",
            "confidence": "1.0",
            "source": src_label,
            "is_synthetic": is_syn,
        })
        extracted_top += 1

        manifest_entries.append({
            "crop_file": rel_bot,
            "plate_num": bot_num,
            "plate_type": "type1a_bot",
            "confidence": "1.0",
            "source": src_label,
            "is_synthetic": is_syn,
        })
        extracted_bot += 1

        if (idx + 1) % 500 == 0 or (idx + 1) == total_plates:
            print(f"  [{idx + 1:04d}/{total_plates:04d}] Processed {idx + 1} plates -> {len(manifest_entries)} line crops")

    # Write manifest.csv
    with open(manifest_path, "w", encoding="utf-8", newline="") as mf:
        fieldnames = ["crop_file", "plate_num", "plate_type", "confidence", "source", "is_synthetic"]
        writer = csv.DictWriter(mf, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(manifest_entries)

    dt = time.time() - t0
    print("\n" + "=" * 65)
    print(" Extraction Complete!")
    print(f"  Total Plates Processed: {total_plates}")
    print(f"  Top Line Crops (L DDD): {extracted_top}")
    print(f"  Bot Line Crops (LL RR): {extracted_bot}")
    print(f"  Total Line Crops:       {len(manifest_entries)}")
    print(f"  Manifest written to:    {manifest_path}")
    print(f"  Time Elapsed:           {dt:.2f}s ({len(manifest_entries)/dt:.1f} crops/s)")
    print("=" * 65)


if __name__ == "__main__":
    main()
