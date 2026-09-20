#!/usr/bin/env python3
"""
OmniPlate-RU: Comprehensive Dataset Sanitation (Phase 1)
Applies visual audit findings for real_type1a_ and real_type1_ frames:
1. Deletes 30 confirmed garbage and duplicate files.
2. Reclassifies 3 single-line car plates from type1a to type1 (YOLO class 0).
3. Reclassifies 17 non-standard plates (motorcycles, trailers, transit, military, soviet, foreign, souvenir) to category other (YOLO class 3).
4. Synchronizes dataset/meta.csv, dataset/labels/*.txt, and dataset/images/real/.
"""

import csv
import os
import shutil
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
IMAGES_DIR = DATASET_DIR / "images" / "real"
LABELS_DIR = DATASET_DIR / "labels"
META_PATH = DATASET_DIR / "meta.csv"
BACKUP_PATH = DATASET_DIR / "meta.csv.bak_audit1a"

# 1. Garbage & duplicates to remove completely
TO_DELETE = {
    # Absolute garbage (non-plates)
    'real_type1a_0017.jpg': 'Synthetic clipart drawing',
    'real_type1a_0232.jpg': 'Infiniti fluid chart table',
    'real_type1a_0254.jpg': 'Japanese auction sheet document',
    'real_type1a_0273.jpg': 'Speedometer dial 180 km/h',
    'real_type1a_0285.jpg': 'Car parts on floor SYSTEM HID CIBIE',
    'real_type1a_0287.jpg': 'Box label 1A 80WH',
    'real_type1a_0016.jpg': 'Traffic crop artifact',
    'real_type1a_0146.jpg': 'Commercial car ad',
    'real_type1a_0176.jpg': 'Bumper frame artifact',
    # Cross-duplicates to delete
    'real_type1a_0004.jpg': 'Duplicate of real_type1_0222 (B118XE97)',
    'real_type1a_0014.jpg': 'Duplicate of real_type1_0222 (B118XE97)',
    'real_type1a_0308.jpg': 'Duplicate of trailer 9309 BB 34',
    'real_type1a_0309.jpg': 'Duplicate of trailer 9309 BB 34',
    'real_type1_0861.jpg': 'Duplicate of trailer 9309 BB 34 with fake label',
    'real_type1_0413.jpg': 'Duplicate of Volvo transit 014D020 51',
    'real_type1_0414.jpg': 'Duplicate of Volvo transit 014D020 51',
    'real_type1_0415.jpg': 'Duplicate of Volvo transit 014D020 51',
    'real_type1_0416.jpg': 'Duplicate of Volvo transit 014D020 51',
    'real_type1_0418.jpg': 'Duplicate of transit 014K011 77',
    'real_type1a_0306.jpg': 'Duplicate of Soviet 19-47 bcm',
    'real_type1_0186.jpg': 'Duplicate of Soviet 19-47 bcm',
    'real_type1_0450.jpg': 'Duplicate of Soviet 19-47 bcm',
    'real_type1_0451.jpg': 'Duplicate of Soviet 19-47 bcm',
    'real_type1_0452.jpg': 'Duplicate of Soviet 19-47 bcm',
    'real_type1a_0015.jpg': 'Duplicate of souvenir 777. RUS',
    'real_type1a_0311.jpg': 'Duplicate of souvenir 777. RUS',
    'real_type1_0491.jpg': 'Duplicate of souvenir 777. RUS',
    'real_type1_0514.jpg': 'Duplicate of souvenir 777. RUS',
    'real_type1_0515.jpg': 'Duplicate of souvenir 777. RUS',
    'real_type1a_0211.jpg': 'Duplicate of special machinery 14HP 197',
}

# 2. Single-line plates to reclassify to Type 1 (YOLO class 0)
TO_TYPE1 = {
    'real_type1a_0003.jpg': 'A389BT10',
    'real_type1a_0005.jpg': 'O341CP68',
    'real_type1a_0223.jpg': 'C800TP799',
}

# 3. Non-standard plates to reclassify to Category Other (YOLO class 3)
TO_OTHER = {
    'real_type1a_0001.jpg': '1223KO42',  # Type 4 motorcycle Agostino
    'real_type1a_0011.jpg': '0248AP50',  # Type 4 motorcycle rear
    'real_type1a_0010.jpg': '014D02051', # Type 15 transit Volvo XC90
    'real_type1a_0012.jpg': '1947BCM',   # Soviet historical
    'real_type1a_0013.jpg': '777RUS',    # Souvenir novelty
    'real_type1a_0210.jpg': '14HP197',   # Type 3 special machinery
    'real_type1a_0225.jpg': '5831428',   # Japanese yellow kei
    'real_type1a_0226.jpg': '3309626',   # Japanese kei
    'real_type1a_0235.jpg': '6790BY',    # Belarus
    'real_type1a_0237.jpg': 'OO199RUS',  # Russian special non-1a
    'real_type1a_0253.jpg': 'JPPLATE',   # Japanese convertible
    'real_type1a_0275.jpg': '777RUS',    # Souvenir Dodge
    'real_type1a_0276.jpg': '5005888',   # Japanese Yokohama
    'real_type1a_0278.jpg': '79509',     # Dubai UAE export
    'real_type1a_0307.jpg': '9309BB34',  # Type 2 trailer
    'real_type1a_0310.jpg': '014K01177', # Type 15 transit
    'real_type1_0862.jpg': '4553AM76',   # Type 5 military black
}

CLASS_MAP = {"type1": 0, "type1a": 1, "type1b": 2, "other": 3}

def apply_sanitation():
    print("=" * 70)
    print("🚀 OmniPlate-RU: Applying Dataset Sanitation (Audit Execution)")
    print("=" * 70)

    # 1. Backup meta.csv
    if not BACKUP_PATH.exists():
        shutil.copyfile(META_PATH, BACKUP_PATH)
        print(f"[*] Created backup: {BACKUP_PATH}")

    # 2. Read meta.csv
    with open(META_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        fieldnames = reader.fieldnames
        rows = list(reader)

    print(f"[*] Initial rows in meta.csv: {len(rows)}")

    # 3. Delete files from disk
    deleted_files = 0
    for fn in TO_DELETE:
        img_p = IMAGES_DIR / fn
        lbl_p = LABELS_DIR / f"{Path(fn).stem}.txt"
        if img_p.exists():
            img_p.unlink()
            deleted_files += 1
        if lbl_p.exists():
            lbl_p.unlink()

    print(f"[*] Deleted {deleted_files} physical images and their labels from disk.")

    # 4. Update labels on disk for reclassified items
    # TO_TYPE1 (class 0)
    for fn, plate_num in TO_TYPE1.items():
        lbl_p = LABELS_DIR / f"{Path(fn).stem}.txt"
        if lbl_p.exists():
            lines = lbl_p.read_text(encoding="utf-8").strip().splitlines()
            if lines:
                parts = lines[0].split()
                parts[0] = "0" # type1 class
                lbl_p.write_text(" ".join(parts) + "\n", encoding="utf-8")
                print(f"  [~] Updated label {lbl_p.name} -> class 0 (type1)")

    # TO_OTHER (class 3)
    for fn, plate_num in TO_OTHER.items():
        lbl_p = LABELS_DIR / f"{Path(fn).stem}.txt"
        if lbl_p.exists():
            lines = lbl_p.read_text(encoding="utf-8").strip().splitlines()
            if lines:
                parts = lines[0].split()
                parts[0] = "3" # other class
                lbl_p.write_text(" ".join(parts) + "\n", encoding="utf-8")
                print(f"  [~] Updated label {lbl_p.name} -> class 3 (other)")

    # 5. Process meta.csv rows
    updated_rows = []
    removed_meta = 0
    reclassified_type1 = 0
    reclassified_other = 0

    for r in rows:
        fn = Path(r["image"]).name
        if fn in TO_DELETE:
            removed_meta += 1
            continue

        if fn in TO_TYPE1:
            r["plate_type"] = "type1"
            r["plate_num"] = TO_TYPE1[fn]
            reclassified_type1 += 1

        elif fn in TO_OTHER:
            r["plate_type"] = "other"
            r["plate_num"] = TO_OTHER[fn]
            reclassified_other += 1

        updated_rows.append(r)

    print(f"[*] Meta.csv updates:")
    print(f"    - Removed rows: {removed_meta}")
    print(f"    - Reclassified to type1: {reclassified_type1}")
    print(f"    - Reclassified to other: {reclassified_other}")
    print(f"    - Surviving rows: {len(updated_rows)}")

    # 6. Save updated meta.csv
    with open(META_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(updated_rows)

    print(f"[+] meta.csv successfully written!")

if __name__ == "__main__":
    apply_sanitation()
