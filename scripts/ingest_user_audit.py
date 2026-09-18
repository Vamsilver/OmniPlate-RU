#!/usr/bin/env python3
"""
OmniPlate-RU — Ingest User Audit Ground Truth into verified crops dataset.
Reads test_output/omniplate_ground_truth_audit.csv, extracts high-quality crops
for user-corrected and verified plates, and writes them cleanly to
dataset/verified_crops/manifest.csv with strict semicolon delimiter support.
"""

import csv
import re
import sys
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CSV_PATH = PROJECT_ROOT / "test_output" / "omniplate_ground_truth_audit.csv"
IMAGES_DIR = PROJECT_ROOT / "dataset" / "images" / "real"
CROPS_DIR = PROJECT_ROOT / "dataset" / "verified_crops"
MANIFEST_PATH = CROPS_DIR / "manifest.csv"
META_PATH = PROJECT_ROOT / "dataset" / "meta.csv"

# Russian Cyrillic to Latin mapping for license plates
CYR_TO_LAT = {
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H",
    "О": "O", "Р": "P", "С": "C", "Т": "T", "У": "Y", "Х": "X",
}


def normalize_plate_text(raw_text: str, plate_type: str = "type1") -> str:
    """Normalizes Russian plate text to uppercase Latin + digits + wildcards."""
    text = raw_text.strip().upper()
    chars = []
    for c in text:
        if c in CYR_TO_LAT:
            chars.append(CYR_TO_LAT[c])
        elif c.isalnum() or c == "#":
            chars.append(c)
    normalized = "".join(chars)

    # If first char is '0' on a standard type1 plate (e.g. 0125TM154), fix to 'O'
    if normalized and normalized[0] == "0" and plate_type in ("type1", "type1a", "ref_real"):
        normalized = "O" + normalized[1:]

    return normalized


def load_meta_quads():
    """Loads backup quads from meta.csv in case detector misses difficult real images."""
    quads = {}
    if META_PATH.exists():
        with open(META_PATH, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f, delimiter=";")
            header = next(reader, None)
            for row in reader:
                if len(row) >= 5:
                    img_rel, _, p_type, bbox_str, quad_str = row[0], row[1], row[2], row[3], row[4]
                    fn = Path(img_rel).name
                    quads[fn] = (p_type, quad_str)
    return quads


def main():
    if not CSV_PATH.exists():
        print(f"[-] CSV file not found: {CSV_PATH}")
        return

    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    meta_quads = load_meta_quads()

    # 1. Read existing manifest, handling both ; and , delimiters
    manifest_records = {}  # crop_file -> list
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
            for line in lines[1:]:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(";") if ";" in line else line.split(",")
                if len(parts) >= 3:
                    c_file = parts[0].strip()
                    p_num = parts[1].strip()
                    p_type = parts[2].strip()
                    conf = parts[3].strip() if len(parts) > 3 else "1.0"
                    src = parts[4].strip() if len(parts) > 4 else "curated_real"
                    manifest_records[c_file] = [c_file, p_num, p_type, conf, src]

    print(f"[*] Loaded {len(manifest_records)} pre-existing crops from {MANIFEST_PATH}")

    from src.pipeline.pipeline import OmniPlatePipeline
    from src.pipeline.rectifier import PlateRectifier
    pipeline = OmniPlatePipeline(device="cuda", conf_threshold=0.05)
    rectifier = PlateRectifier()

    added_count = 0
    corrected_count = 0

    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            fn = row.get("filename", "").strip()
            status = row.get("status", "").strip().lower()
            raw_gt = row.get("ground_truth_text", "").strip()

            # Skip rejected, unreviewed, or empty
            if status in ("no", "unreviewed") or not raw_gt or raw_gt == "НОМЕР НЕ НАЙДЕН":
                continue

            p_type = row.get("type", "type1").strip()
            if p_type == "no_plate":
                p_type = "type1"

            clean_gt = normalize_plate_text(raw_gt, plate_type=p_type)
            if not clean_gt:
                continue

            # Safe filename token replacing '#' with 'X' for clean filesystem path
            safe_num = clean_gt.replace("#", "X")
            crop_filename = f"user_gt_{Path(fn).stem}_{safe_num}.png"
            crop_out_path = CROPS_DIR / crop_filename

            # Remove any outdated prior entry for this same source image
            stem_prefix = f"user_gt_{Path(fn).stem}_"
            stale_keys = [k for k in manifest_records if k.startswith(stem_prefix) and k != crop_filename]
            for k in stale_keys:
                del manifest_records[k]

            # If crop does not exist on disk, reuse prior crop or extract it
            if not crop_out_path.exists():
                prior_crops = list(CROPS_DIR.glob(f"{stem_prefix}*.png"))
                if prior_crops and prior_crops[0].exists():
                    cached_img = cv2.imread(str(prior_crops[0]))
                    if cached_img is not None and cached_img.size > 0:
                        cv2.imwrite(str(crop_out_path), cached_img)
                        if prior_crops[0] != crop_out_path:
                            try:
                                prior_crops[0].unlink()
                            except Exception:
                                pass

            if not crop_out_path.exists():
                img_path = IMAGES_DIR / fn
                if not img_path.exists():
                    print(f"[-] Image not found: {img_path}")
                    continue

                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                crop_img = None

                # Check if image itself is already a crop (e.g. ref_real 29x111)
                h, w = img.shape[:2]
                if h <= 100 and w <= 350:
                    crop_img = cv2.resize(img, (160, 36), interpolation=cv2.INTER_LINEAR)
                else:
                    # Run detection pipeline
                    dets = pipeline.predict(img)
                    if dets and dets[0].rectified_crop is not None and dets[0].rectified_crop.size > 0:
                        det = dets[0]
                        if det.plate_type == "type1a" and det.rectified_crop.shape[:2] != (36, 160):
                            top_line, bot_line = rectifier.split_type1a(det.rectified_crop)
                            crop_img = rectifier.stitch_type1a_horizontal(top_line, bot_line, target_size=(160, 36))
                        else:
                            crop_img = det.rectified_crop
                            if crop_img.shape[:2] != (36, 160):
                                crop_img = cv2.resize(crop_img, (160, 36), interpolation=cv2.INTER_LINEAR)
                        if p_type == "no_plate":
                            p_type = det.plate_type

                    # Fallback on meta.csv quad if detector missed
                    elif fn in meta_quads:
                        m_type, quad_str = meta_quads[fn]
                        rect = rectifier.rectify(img, quad_str, plate_type=m_type)
                        if m_type == "type1a":
                            top_line, bot_line = rectifier.split_type1a(rect)
                            crop_img = rectifier.stitch_type1a_horizontal(top_line, bot_line, target_size=(160, 36))
                        else:
                            crop_img = cv2.resize(rect, (160, 36), interpolation=cv2.INTER_LINEAR)
                        p_type = m_type

                if crop_img is not None and crop_img.size > 0:
                    cv2.imwrite(str(crop_out_path), crop_img)
                else:
                    print(f"[!] Warning: Could not produce crop for {fn}")
                    continue

            # Determine canonical type
            if len(clean_gt) in (8, 9) and clean_gt[:2].isalpha() and clean_gt[2:6].isdigit():
                p_type = "type2"
            elif len(clean_gt) in (7, 8) and clean_gt[:2].isalpha() and clean_gt[2:5].isdigit():
                p_type = "type1b"
            elif p_type == "type1a" and len(clean_gt) in (8, 9) and clean_gt[0].isalpha() and clean_gt[1:4].isdigit():
                p_type = "type1"
            elif p_type in ("no_plate", ""):
                p_type = "type1"

            src_tag = "user_audit_corrected" if status == "corrected" else "user_audit_verified"
            manifest_records[crop_filename] = [crop_filename, clean_gt, p_type, "1.0", src_tag]
            added_count += 1
            if status == "corrected":
                corrected_count += 1
            print(f"[+] Ingested GT crop: {crop_filename} -> {clean_gt} ({p_type}, {src_tag})")

    # 2. Write manifest strictly with semicolon delimiter
    with open(MANIFEST_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["crop_file", "plate_num", "plate_type", "confidence", "source"])
        for record in manifest_records.values():
            writer.writerow(record)

    print("\n" + "=" * 70)
    print(f"[SUCCESS] Total records in manifest.csv: {len(manifest_records)}")
    print(f"          Processed items: {added_count} (User corrections: {corrected_count})")
    print(f"          Manifest path:   {MANIFEST_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()

