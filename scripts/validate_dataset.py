#!/usr/bin/env python3
"""
Official Dataset Validator for Volga IT 2026 (License Plate Recognition).
Verifies directory structure, meta.csv schema, plate format regex,
geometry (bbox, quad), and dataset quotas.
"""

import argparse
import csv
import os
import re
import sys
from typing import Dict, List, Set
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Allowed characters according to competition rules
ALLOWED_LETTERS = set("ABEKMHOPCTYX#")
ALLOWED_TYPES = {"type1", "type1a", "type1b", "other"}
ALLOWED_CONDITIONS = {"day", "night", "rain", "snow", "dirt", "glare", "motion_blur", "angle"}

PLATE_REGEX = re.compile(r"^[ABEKMHOPCTYX#][\d#]{3}[ABEKMHOPCTYX#]{2}[\d#]{2,3}$")
VALID_3DIGIT_STARTS = {"1", "2", "7", "#"}


def validate_dataset(dataset_dir: str) -> bool:
    print("=" * 60)
    print(f"🔍 Starting Dataset Validation for: {dataset_dir}")
    print("=" * 60)

    errors: List[str] = []
    warnings: List[str] = []

    # 1. Directory Structure Checks
    images_real_dir = os.path.join(dataset_dir, "images", "real")
    images_synth_dir = os.path.join(dataset_dir, "images", "synthetic")
    labels_dir = os.path.join(dataset_dir, "labels")
    generator_dir = os.path.join(dataset_dir, "generator")
    meta_path = os.path.join(dataset_dir, "meta.csv")
    readme_path = os.path.join(dataset_dir, "README.md")
    license_path = os.path.join(dataset_dir, "LICENSE")

    for p, desc in [
        (images_real_dir, "Directory images/real"),
        (images_synth_dir, "Directory images/synthetic"),
        (labels_dir, "Directory labels"),
        (generator_dir, "Directory generator"),
        (meta_path, "File meta.csv"),
        (readme_path, "File README.md"),
        (license_path, "File LICENSE"),
    ]:
        if not os.path.exists(p):
            errors.append(f"Missing required path: {desc} ({p})")

    if errors:
        print("\n❌ CRITICAL STRUCTURAL ERRORS:")
        for err in errors:
            print(f"  - {err}")
        return False

    # 2. Validate meta.csv Schema and Rows
    print("\n📋 Checking meta.csv...")
    expected_header = [
        "image", "plate_num", "plate_type", "bbox", "quad",
        "is_vehicle", "is_synthetic", "source", "license", "conditions"
    ]

    stats = {
        "total_rows": 0,
        "real_count": 0,
        "synth_count": 0,
        "types": {"type1": 0, "type1a": 0, "type1b": 0, "other": 0},
        "real_types": {"type1": 0, "type1a": 0, "type1b": 0, "other": 0},
        "synth_types": {"type1": 0, "type1a": 0, "type1b": 0, "other": 0},
        "real_unique_plates": {"type1a": set(), "type1b": set(), "other": set(), "type1": set()}
    }

    # Pre-cache existing image files for fast O(1) lookup over network
    existing_images_cache = set()
    for sub in ["real", "synthetic"]:
        sdir = os.path.join(dataset_dir, "images", sub)
        if os.path.exists(sdir):
            for entry in os.scandir(sdir):
                if entry.is_file():
                    existing_images_cache.add(f"images/{sub}/{entry.name}")

    seen_images: Set[str] = set()

    with open(meta_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter=";")
        try:
            header = next(reader)
        except StopIteration:
            print("❌ meta.csv is empty!")
            return False

        if header != expected_header:
            errors.append(f"Header mismatch in meta.csv.\nExpected: {expected_header}\nGot:      {header}")

        for row_idx, row in enumerate(reader, start=2):
            if not row:
                continue

            stats["total_rows"] += 1

            if len(row) != 10:
                errors.append(f"Row {row_idx}: expected 10 columns, found {len(row)} ({row})")
                continue

            img_rel, plate_num, p_type, bbox_str, quad_str, is_veh_str, is_syn_str, source, lic, conds_str = row

            # Check image existence via pre-cached index
            norm_rel = img_rel.replace("\\", "/")
            if norm_rel not in existing_images_cache:
                errors.append(f"Row {row_idx}: Image not found: {img_rel}")
            seen_images.add(img_rel)

            # Check plate type
            if p_type not in ALLOWED_TYPES:
                errors.append(f"Row {row_idx}: Invalid plate_type '{p_type}'. Must be one of {ALLOWED_TYPES}")
            else:
                stats["types"][p_type] += 1

            # Check plate_num regex
            if p_type != "other":
                if not PLATE_REGEX.match(plate_num):
                    errors.append(f"Row {row_idx}: Plate number '{plate_num}' violates GOST mask for {p_type}")
                if len(plate_num) == 9 and plate_num[6] not in VALID_3DIGIT_STARTS:
                    warnings.append(f"Row {row_idx}: 3-digit region code in '{plate_num}' starts with unexpected digit '{plate_num[6]}'")

            # Check bbox: x,y,w,h
            try:
                b_parts = [int(x.strip()) for x in bbox_str.split(",")]
                if len(b_parts) != 4 or b_parts[2] <= 0 or b_parts[3] <= 0:
                    errors.append(f"Row {row_idx}: Invalid bbox '{bbox_str}'. Must be x,y,w,h with w>0, h>0")
            except ValueError:
                errors.append(f"Row {row_idx}: Non-integer bbox values '{bbox_str}'")

            # Check quad: 8 integers
            try:
                q_parts = [int(x.strip()) for x in quad_str.split(",")]
                if len(q_parts) != 8:
                    errors.append(f"Row {row_idx}: Quad must contain exactly 8 integers, got {len(q_parts)} in '{quad_str}'")
            except ValueError:
                errors.append(f"Row {row_idx}: Non-integer quad values '{quad_str}'")

            # Check flags
            if is_veh_str not in {"0", "1"}:
                errors.append(f"Row {row_idx}: is_vehicle must be '0' or '1', got '{is_veh_str}'")
            if is_syn_str not in {"0", "1"}:
                errors.append(f"Row {row_idx}: is_synthetic must be '0' or '1', got '{is_syn_str}'")

            is_syn = is_syn_str == "1"
            if is_syn:
                stats["synth_count"] += 1
                if p_type in stats["synth_types"]:
                    stats["synth_types"][p_type] += 1
            else:
                stats["real_count"] += 1
                if p_type in stats["real_types"]:
                    stats["real_types"][p_type] += 1
                if p_type in stats["real_unique_plates"]:
                    stats["real_unique_plates"][p_type].add(plate_num)

            # Check conditions
            cond_list = [c.strip() for c in conds_str.split(",") if c.strip()]
            for c in cond_list:
                if c not in ALLOWED_CONDITIONS:
                    warnings.append(f"Row {row_idx}: Unknown condition tag '{c}'")

    # 3. Quota Evaluation
    print("\n📊 Dataset Statistics & Quota Evaluation:")
    print(f"  • Total Plate Annotations: {stats['total_rows']}")
    print(f"  • Synthetic Images Count:  {stats['synth_count']} (Quota: >= 5000) -> {'✅ PASS' if stats['synth_count'] >= 5000 else '⏳ PENDING'}")
    print(f"  • Real Images Count:       {stats['real_count']}")
    print(f"    - Type 1A (Square):      {stats['real_types']['type1a']} real (Quota: >= 150 imgs, >= 50 unique) [Unique: {len(stats['real_unique_plates']['type1a'])}]")
    print(f"    - Type 1B (Yellow):      {stats['real_types']['type1b']} real (Quota: >= 300 imgs, >= 100 unique) [Unique: {len(stats['real_unique_plates']['type1b'])}]")
    print(f"    - Other (Negative):      {stats['real_types']['other']} real (Quota: >= 50 imgs)")

    if warnings:
        print(f"\n⚠️ WARNINGS ({len(warnings)}):")
        for w in warnings[:10]:
            print(f"  - {w}")
        if len(warnings) > 10:
            print(f"  ... and {len(warnings) - 10} more warnings.")

    if errors:
        print(f"\n❌ VALIDATION FAILED ({len(errors)} errors):")
        for err in errors[:15]:
            print(f"  - {err}")
        if len(errors) > 15:
            print(f"  ... and {len(errors) - 15} more errors.")
        return False

    print("\n✅ VALIDATION PASSED: All schema, geometry, and format checks passed successfully!")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate Volga IT License Plate Dataset")
    parser.add_argument("--dataset_dir", type=str, default="dataset", help="Path to dataset directory")
    args = parser.parse_args()

    success = validate_dataset(args.dataset_dir)
    sys.exit(0 if success else 1)
