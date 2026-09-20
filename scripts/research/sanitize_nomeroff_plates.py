#!/usr/bin/env python3
"""
OmniPlate-RU — Sanitize 238 Nomeroff Russian license plate rows in dataset/meta.csv and labels/.
Converts mislabeled 'type1b' passenger cars to their true 'type1' numbers (or 'other' if unreadable).
"""

import csv
import os
import shutil
import sys
from pathlib import Path
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.pipeline import OmniPlatePipeline
from src.pipeline.decoder import is_valid_gost_plate

META_PATH = PROJECT_ROOT / "dataset" / "meta.csv"
BACKUP_PATH = PROJECT_ROOT / "dataset" / "meta.csv.bak_nomeroff"
LABELS_DIR = PROJECT_ROOT / "dataset" / "labels"


def update_label_file(stem: str, new_class_id: int):
    lbl_path = LABELS_DIR / f"{stem}.txt"
    if not lbl_path.exists():
        print(f"[-] Warning: label file missing: {lbl_path}")
        return
    with open(lbl_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    new_lines = []
    for line in lines:
        if not line.strip():
            continue
        toks = line.strip().split()
        toks[0] = str(new_class_id)
        new_lines.append(" ".join(toks))
    with open(lbl_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(new_lines) + "\n")


def main():
    print("=" * 65)
    print("Sanitizing 238 Nomeroff Rows in meta.csv and labels/")
    print("=" * 65)

    if not BACKUP_PATH.exists():
        print(f"[*] Creating backup of meta.csv -> {BACKUP_PATH}")
        shutil.copy2(META_PATH, BACKUP_PATH)
    else:
        print(f"[*] Backup already exists: {BACKUP_PATH}")

    with open(META_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        fieldnames = reader.fieldnames
        rows = list(reader)

    nomeroff_indices = [
        i for i, r in enumerate(rows)
        if "nomeroff" in r.get("source", "")
    ]
    print(f"[*] Found {len(nomeroff_indices)} Nomeroff rows to sanitize.")

    # Initialize Pipeline
    pipeline = OmniPlatePipeline(device="cuda")
    pipeline.warmup(2)

    converted_type1 = 0
    converted_other = 0

    for idx in nomeroff_indices:
        r = rows[idx]
        img_rel = r["image"]
        img_path = PROJECT_ROOT / "dataset" / img_rel
        stem = Path(img_rel).stem

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[-] Error reading {img_path}, marking other")
            r["plate_type"] = "other"
            r["plate_num"] = "###"
            update_label_file(stem, 3)
            converted_other += 1
            continue

        dets = pipeline.predict(img)
        if not dets:
            print(f"[-] No detection on {img_rel}, marking other")
            r["plate_type"] = "other"
            r["plate_num"] = "###"
            update_label_file(stem, 3)
            converted_other += 1
            continue

        def det_sort_key(d):
            is_target = 1 if d.plate_type in ("type1", "type1a", "type1b") else 0
            has_text = 1 if d.text and "#" not in d.text else 0
            quality = (d.confidence ** 0.5) * (d.ocr_confidence ** 2)
            return (is_target, has_text, quality, d.confidence)

        dets = sorted(dets, key=det_sort_key, reverse=True)
        best = dets[0]

        if best.plate_type == "type1" and is_valid_gost_plate(best.text, "type1"):
            r["plate_type"] = "type1"
            r["plate_num"] = best.text
            update_label_file(stem, 0)
            converted_type1 += 1
        else:
            print(f"[!] Edge case {img_rel}: det_type={best.plate_type}, det_text='{best.text}', marking other")
            r["plate_type"] = "other"
            r["plate_num"] = "###"
            update_label_file(stem, 3)
            converted_other += 1

    print("\n" + "=" * 65)
    print(f"Sanitization complete:")
    print(f"  Converted to Type 1: {converted_type1}")
    print(f"  Converted to Other:  {converted_other}")
    print(f"  Total processed:     {len(nomeroff_indices)}")
    print("=" * 65)

    # Save updated meta.csv
    with open(META_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)

    print(f"[*] Successfully saved sanitized dataset/meta.csv ({len(rows)} rows).")


if __name__ == "__main__":
    main()
