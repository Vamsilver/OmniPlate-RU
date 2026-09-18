#!/usr/bin/env python3
"""
OmniPlate-RU — Sanitize 'other' class in meta.csv (Stage 3, Step 1).
Removes duplicates and reclassifies genuine Russian vehicle plates from 'other' into 'type1'/'type1a'.
"""

import csv
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.pipeline import OmniPlatePipeline
from src.pipeline.decoder import is_valid_gost_plate

META_PATH = PROJECT_ROOT / "dataset" / "meta.csv"
LABELS_DIR = PROJECT_ROOT / "dataset" / "labels"
IMAGES_DIR = PROJECT_ROOT / "dataset" / "images"

PLATE_REGEX = re.compile(r"^[ABEKMHOPCTYX][\d]{3}[ABEKMHOPCTYX]{2}[\d]{2,3}$")
SQUARE_REGEX = re.compile(r"^[ABEKMHOPCTYX]{2}[\d]{3,4}[\d]{2,3}$|^[ABEKMHOPCTYX][\d]{3}[ABEKMHOPCTYX]{2}[\d]{2,3}$")


def run_sanitization(dry_run: bool = True):
    print("=" * 65)
    print(f"🧹 Sanitizing 'other' Dataset (dry_run={dry_run})")
    print("=" * 65)

    with open(META_PATH, "r", encoding="utf-8") as f:
        reader = list(csv.reader(f, delimiter=";"))

    header = reader[0]
    rows = reader[1:]
    print(f"[*] Total rows before sanitization: {len(rows)}")

    # 1. Detect and remove the duplicate rows
    # Specifically, lines around 5135-5144 where real_other_0045..0054 are duplicated from Roboflow
    seen_images: Set[str] = set()
    deduped_rows = []
    duplicate_count = 0

    for idx, r in enumerate(rows):
        img_rel = r[0].strip()
        # If this image was already seen, drop the duplicate
        if img_rel in seen_images:
            duplicate_count += 1
            print(f"  [-] Dropping duplicate row {idx + 2}: {img_rel} (src: {r[7][:40]}...)")
            continue
        seen_images.add(img_rel)
        deduped_rows.append(r)

    print(f"[*] Removed {duplicate_count} duplicate rows. Remaining: {len(deduped_rows)}")

    # 2. Reclassify genuine car plates from 'other'
    pipeline = OmniPlatePipeline(device="cuda", conf_threshold=0.12)

    reclassified_count = 0
    final_rows = []

    for r in deduped_rows:
        img_rel = r[0].strip()
        p_type = r[2].strip()
        is_syn = r[6].strip()

        # We only check real 'other' rows
        if p_type == "other" and is_syn == "0":
            img_path = PROJECT_ROOT / "dataset" / img_rel
            im = cv2.imread(str(img_path))
            if im is None:
                final_rows.append(r)
                continue

            h, w = im.shape[:2]
            # Run detection & OCR
            dets = pipeline.predict(im)
            valid_dets = [
                d for d in dets
                if d.plate_type in ("type1", "type1a", "type1b")
                and d.text
                and is_valid_gost_plate(d.text, d.plate_type)
                and (d.ocr_confidence >= 0.70 or ("0174" in img_rel and d.ocr_confidence >= 0.60))
            ]

            # Specifically, verify if this image is one of the known real car series
            is_candidate_series = (
                "russian_car_plates" in r[7]
                or any(f"real_other_{i:04d}" in img_rel for i in range(238, 317))
                or any(f"real_other_{i:04d}" in img_rel for i in [47, 48, 49, 50, 51, 52, 53, 104, 174, 221])
            )

            # Avoid reclassifying genuine non-targets (motorcycles 0061, diplomatic 0118, 0120)
            is_protected_other = any(f"real_other_{i:04d}" in img_rel for i in [61, 118, 120, 126, 127, 145])

            if valid_dets and is_candidate_series and not is_protected_other:
                top = valid_dets[0]
                target_type = top.plate_type
                target_text = top.text

                bx, by, bw, bh = top.bbox
                quad_str = ",".join(str(int(round(x))) for x in top.quad)
                bbox_str = f"{bx},{by},{bw},{bh}"

                print(f"  [+] Reclassifying {img_rel}: other -> {target_type} '{target_text}' (conf={top.ocr_confidence:.2f})")
                new_row = list(r)
                new_row[1] = target_text
                new_row[2] = target_type
                new_row[3] = bbox_str
                new_row[4] = quad_str
                new_row[5] = "1"  # is_vehicle
                final_rows.append(new_row)
                reclassified_count += 1

                # Update label file if not dry run
                if not dry_run:
                    lbl_stem = Path(img_rel).stem
                    lbl_path = LABELS_DIR / f"{lbl_stem}.txt"
                    # YOLO pose format: class_id xc yc w h kpts...
                    class_id = 0 if target_type == "type1" else (1 if target_type == "type1a" else 2)
                    xc = (bx + bw / 2.0) / float(w)
                    yc = (by + bh / 2.0) / float(h)
                    nw = bw / float(w)
                    nh = bh / float(h)
                    q = top.quad
                    kpts = " ".join(f"{q[k]/float(w):.6f} {q[k+1]/float(h):.6f}" for k in range(0, 8, 2))
                    lbl_content = f"{class_id} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f} {kpts}\n"
                    lbl_path.write_text(lbl_content, encoding="utf-8")
                continue

        final_rows.append(r)

    print("\n" + "=" * 65)
    print(f"📊 Summary:")
    print(f"  • Duplicates removed: {duplicate_count}")
    print(f"  • Rows reclassified:  {reclassified_count}")
    print(f"  • Total rows after:   {len(final_rows)}")

    # Count real categories
    types_count = {"type1": 0, "type1a": 0, "type1b": 0, "other": 0}
    for r in final_rows:
        if r[6] == "0":
            t = r[2]
            if t in types_count:
                types_count[t] += 1

    print(f"  • Real dataset breakdown:")
    for t, cnt in types_count.items():
        print(f"    - {t}: {cnt}")

    if not dry_run:
        # Write back to meta.csv
        with open(META_PATH, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(header)
            for r in final_rows:
                writer.writerow(r)
        print(f"✅ Successfully wrote sanitized meta.csv!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="Perform actual file edits")
    args = parser.parse_args()
    run_sanitization(dry_run=not args.execute)
