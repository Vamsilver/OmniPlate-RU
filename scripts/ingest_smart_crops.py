#!/usr/bin/env python3
"""
OmniPlate-RU — Smart Ingestion of Real Plate Crops for OCR & Detector Training.
Extracts:
1. 5,000 diverse, pristine real crops from Nomeroff Net (Kaggle archive.zip).
2. Verified Type 1A (square JDM/USDM) stitched crops from Roboflow.
3. Verified Type 2 (trailer) crops from HuggingFace Parquet.
Appends entries to dataset/verified_crops/manifest.csv.
"""

import csv
import json
import os
import random
import re
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Set, Tuple

import cv2
import numpy as np
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.rectifier import PlateRectifier

CYR_TO_LAT: Dict[str, str] = {
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H",
    "О": "O", "Р": "P", "С": "C", "Т": "T", "У": "Y", "Х": "X",
    "а": "A", "в": "B", "е": "E", "к": "K", "м": "M", "н": "H",
    "о": "O", "р": "P", "с": "C", "т": "T", "у": "Y", "х": "X",
}

ROBOFLOW_CLASSES = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "E", "H", "K", "M", "O", "P", "T", "X", "Y"
]

GOST_TYPE1_REGEX = re.compile(r"^[ABEKMHOPCTYX]\d{3}[ABEKMHOPCTYX]{2}\d{2,3}$")
GOST_TYPE1A_REGEX = re.compile(r"^[ABEKMHOPCTYX]\d{3}[ABEKMHOPCTYX]{2}\d{2,3}$|^[ABEKMHOPCTYX]{2}\d{4}\d{2,3}$")
GOST_TYPE2_REGEX = re.compile(r"^\d{4}[ABEKMHOPCTYX]{2}\d{2,3}$")


def clean_text(text: str) -> str:
    if not text:
        return ""
    res = []
    for ch in text.strip().upper():
        if ch in CYR_TO_LAT:
            res.append(CYR_TO_LAT[ch])
        elif ch.isalnum():
            res.append(ch)
    return "".join(res)


def ingest_nomeroff(
    zip_path: str,
    output_crops_dir: Path,
    target_count: int = 5000,
    seed: int = 42,
) -> List[Tuple[str, str, str, float, str]]:
    """Extracts target_count diverse, verified crops from Nomeroff archive.zip."""
    print(f"\n[*] Scanning Nomeroff Net ({zip_path})...")
    manifest_entries = []

    with zipfile.ZipFile(zip_path) as z:
        all_files = z.namelist()
        ann_files = [f for f in all_files if f.endswith(".json") and "/ann/" in f]
        ann_files.sort()

        rng = random.Random(seed)
        rng.shuffle(ann_files)

        # Region diversity tracking
        region_counts: Dict[str, int] = {}
        max_per_region = 120  # Prevent Moscow/Peter saturation, promote regional diversity

        saved_count = 0
        for j_path in ann_files:
            if saved_count >= target_count:
                break

            try:
                data = json.loads(z.read(j_path).decode("utf-8"))
            except Exception:
                continue

            # Must be human-moderated
            if data.get("moderation", {}).get("isModerated") != 1:
                continue

            raw_desc = data.get("description") or data.get("name") or ""
            plate_num = clean_text(raw_desc)
            if not GOST_TYPE1_REGEX.match(plate_num):
                continue

            # Check region code
            reg = plate_num[-3:] if plate_num[-3:].isdigit() else plate_num[-2:]
            if region_counts.get(reg, 0) >= max_per_region:
                continue

            # Find image
            img_path = j_path.replace("/ann/", "/img/").rsplit(".", 1)[0] + ".png"
            if img_path not in all_files:
                img_path = j_path.replace("/ann/", "/img/").rsplit(".", 1)[0] + ".jpg"
            if img_path not in all_files:
                continue

            img_bytes = z.read(img_path)
            img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            if img is None or img.size == 0:
                continue

            # Standardize crop size to canonical 160x36
            crop_std = cv2.resize(img, (160, 36), interpolation=cv2.INTER_AREA)

            fname = f"crop_nomeroff_{saved_count:05d}_{plate_num}.png"
            out_file = output_crops_dir / fname
            cv2.imwrite(str(out_file), crop_std)

            manifest_entries.append((fname, plate_num, "type1", 1.0, "nomeroff_curated_v1"))
            region_counts[reg] = region_counts.get(reg, 0) + 1
            saved_count += 1

            if saved_count % 1000 == 0:
                print(f"    Extracted {saved_count}/{target_count} Nomeroff crops...")

    print(f"[+] Successfully extracted {len(manifest_entries)} pristine Nomeroff crops across {len(region_counts)} regions.")
    return manifest_entries


def ingest_roboflow_type1a(
    zip_path: str,
    output_crops_dir: Path,
) -> List[Tuple[str, str, str, float, str]]:
    """Extracts verified 2-line Type 1A plates from Roboflow and stitches to 160x36."""
    print(f"\n[*] Scanning Roboflow for Type 1A plates ({zip_path})...")
    manifest_entries = []
    rectifier = PlateRectifier()

    with zipfile.ZipFile(zip_path) as z:
        all_files = z.namelist()
        label_files = [f for f in all_files if f.endswith(".txt") and "/labels/" in f]

        count = 0
        for lbl_path in label_files:
            content = z.read(lbl_path).decode("utf-8").strip()
            lines = [l.strip().split() for l in content.split("\n") if l.strip()]
            if len(lines) < 5:
                continue

            ys = [float(l[2]) for l in lines]
            if max(ys) - min(ys) < 0.28:
                continue

            mean_y = (max(ys) + min(ys)) / 2.0
            line1 = sorted([l for l in lines if float(l[2]) < mean_y], key=lambda l: float(l[1]))
            line2 = sorted([l for l in lines if float(l[2]) >= mean_y], key=lambda l: float(l[1]))

            t1 = "".join(ROBOFLOW_CLASSES[int(l[0])] for l in line1 if 0 <= int(l[0]) < len(ROBOFLOW_CLASSES))
            t2 = "".join(ROBOFLOW_CLASSES[int(l[0])] for l in line2 if 0 <= int(l[0]) < len(ROBOFLOW_CLASSES))
            full_num = clean_text(t1 + t2)

            if not GOST_TYPE1A_REGEX.match(full_num):
                continue

            img_path = lbl_path.replace("/labels/", "/images/").rsplit(".", 1)[0] + ".jpg"
            if img_path not in all_files:
                img_path = lbl_path.replace("/labels/", "/images/").rsplit(".", 1)[0] + ".png"
            if img_path not in all_files:
                continue

            img_bytes = z.read(img_path)
            img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue

            # Split and stitch into canonical 160x36
            try:
                # Resize to canonical 160x96 first if not square
                crop_96 = cv2.resize(img, (160, 96), interpolation=cv2.INTER_AREA)
                top_line, bottom_line = rectifier.split_type1a(crop_96)
                stitched = rectifier.stitch_type1a_horizontal(top_line, bottom_line, target_size=(160, 36))
            except Exception:
                stitched = cv2.resize(img, (160, 36), interpolation=cv2.INTER_AREA)

            fname = f"crop_roboflow_1a_{count:03d}_{full_num}.png"
            cv2.imwrite(str(output_crops_dir / fname), stitched)
            manifest_entries.append((fname, full_num, "type1a", 1.0, "roboflow_1a_verified"))
            count += 1

    print(f"[+] Successfully extracted and stitched {len(manifest_entries)} Type 1A plates from Roboflow.")
    return manifest_entries


def ingest_parquet_type2(
    parquet_paths: List[str],
    output_crops_dir: Path,
) -> List[Tuple[str, str, str, float, str]]:
    """Extracts verified Type 2 trailer crops from HuggingFace Parquet."""
    print(f"\n[*] Scanning HuggingFace Parquet for Type 2 trailer plates...")
    manifest_entries = []

    count = 0
    seen_nums: Set[str] = set()

    for p in parquet_paths:
        if not os.path.exists(p):
            continue
        table = pq.read_table(p)
        for row in table.to_pylist():
            img_dict = row.get("image", {})
            img_bytes = img_dict.get("bytes")
            img_path = img_dict.get("path", "")
            if not img_bytes:
                continue

            stem = Path(img_path).name
            prefix = stem.split("_")[0]
            plate_num = clean_text(prefix)

            if not GOST_TYPE2_REGEX.match(plate_num):
                continue
            if plate_num in seen_nums:
                continue

            objs = row.get("objects", {})
            bboxes = objs.get("bbox", []) if objs else []
            if not bboxes:
                continue

            img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue

            ih, iw = img.shape[:2]
            gx, gy, gw, gh = [int(round(v)) for v in bboxes[0]]
            # Add 5% padding around bbox
            pad_x = int(gw * 0.05)
            pad_y = int(gh * 0.05)
            x1 = max(0, gx - pad_x)
            y1 = max(0, gy - pad_y)
            x2 = min(iw, gx + gw + pad_x)
            y2 = min(ih, gy + gh + pad_y)

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop_std = cv2.resize(crop, (160, 36), interpolation=cv2.INTER_AREA)
            fname = f"crop_parquet_type2_{count:03d}_{plate_num}.png"
            cv2.imwrite(str(output_crops_dir / fname), crop_std)

            manifest_entries.append((fname, plate_num, "type2", 1.0, "parquet_type2_trailer"))
            seen_nums.add(plate_num)
            count += 1

    print(f"[+] Successfully extracted {len(manifest_entries)} unique Type 2 trailer plates from Parquet.")
    return manifest_entries


def main():
    print("=" * 70)
    print("  OmniPlate-RU — Smart Real Crops Ingestion Pipeline")
    print("=" * 70)

    downloads_dir = Path(r"C:\Users\Vamsi\Downloads")
    nomeroff_zip = downloads_dir / "archive.zip"
    roboflow_zip = downloads_dir / "russian_car_plates.v2i.yolov8.zip"
    parquet_files = [
        str(downloads_dir / "train-00000-of-00001.parquet"),
        str(downloads_dir / "validation-00000-of-00001.parquet"),
        str(downloads_dir / "test-00000-of-00001.parquet"),
    ]

    crops_dir = PROJECT_ROOT / "dataset" / "verified_crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    manifest_csv = crops_dir / "manifest.csv"

    # Read existing entries to avoid duplicates
    existing_files = set()
    if manifest_csv.exists():
        with open(manifest_csv, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=";")
            next(reader, None)
            for r in reader:
                if r:
                    existing_files.add(r[0])

    print(f"[*] Found {len(existing_files)} existing crop files in manifest.csv")

    new_entries = []

    # 1. Ingest 5000 pristine Nomeroff
    if nomeroff_zip.exists():
        nomeroff_entries = ingest_nomeroff(str(nomeroff_zip), crops_dir, target_count=5000, seed=42)
        new_entries.extend(nomeroff_entries)

    # 2. Ingest Roboflow Type 1A
    if roboflow_zip.exists():
        roboflow_entries = ingest_roboflow_type1a(str(roboflow_zip), crops_dir)
        new_entries.extend(roboflow_entries)

    # 3. Ingest Parquet Type 2
    parquet_entries = ingest_parquet_type2(parquet_files, crops_dir)
    new_entries.extend(parquet_entries)

    # Append to manifest.csv
    print(f"\n[*] Appending {len(new_entries)} verified entries to {manifest_csv}...")
    with open(manifest_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        for entry in new_entries:
            if entry[0] not in existing_files:
                writer.writerow(entry)
                existing_files.add(entry[0])

    print("=" * 70)
    print(f"[✓] INGESTION COMPLETE! Total active crops in manifest: {len(existing_files)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
