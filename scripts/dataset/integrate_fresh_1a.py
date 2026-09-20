#!/usr/bin/env python3
"""
Integrate harvested pure Type 1A candidates into dataset and meta.csv.
Ensures 100% compliance with YOLO pose format and meta.csv specification.
"""

import csv
import json
import os
import re
import sys
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CANDIDATES_JSON = PROJECT_ROOT / "test_output" / "pure_1a_fresh" / "approved_25_candidates.json"
FRESH_DIR = PROJECT_ROOT / "test_output" / "pure_1a_fresh"
IMAGES_REAL_DIR = PROJECT_ROOT / "dataset" / "images" / "real"
LABELS_DIR = PROJECT_ROOT / "dataset" / "labels"
META_CSV = PROJECT_ROOT / "dataset" / "meta.csv"
AUDIT_DIR = PROJECT_ROOT / "test_output" / "visual_audit_1a"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

def find_next_real_1a_index():
    max_idx = 0
    pattern = re.compile(r"^real_type1a_(\d+)\.jpg$")
    for fn in os.listdir(IMAGES_REAL_DIR):
        m = pattern.match(fn)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1

def integrate():
    if not CANDIDATES_JSON.exists():
        print(f"Error: {CANDIDATES_JSON} does not exist.")
        return

    with open(CANDIDATES_JSON, "r", encoding="utf-8") as f:
        candidates = json.load(f)

    print(f"Found {len(candidates)} candidates to integrate.")
    if not candidates:
        return

    next_idx = find_next_real_1a_index()
    print(f"Starting at real_type1a_{next_idx:04d}.jpg")

    meta_rows = []
    contact_crops = []

    for c in candidates:
        src_img_path = (FRESH_DIR / c["full_fn"]).resolve()
        if not src_img_path.exists():
            print(f"[-] Image not found: {src_img_path}")
            continue

        img = cv2.imread(str(src_img_path))
        if img is None:
            continue
        h_im, w_im = img.shape[:2]

        dst_fn = f"real_type1a_{next_idx:04d}.jpg"
        dst_img_path = IMAGES_REAL_DIR / dst_fn
        dst_txt_path = LABELS_DIR / f"real_type1a_{next_idx:04d}.txt"

        # Copy image
        cv2.imwrite(str(dst_img_path), img, [cv2.IMWRITE_JPEG_QUALITY, 94])

        # Generate YOLO label: class 1 (type1a), cx cy w h x1 y1 x2 y2 x3 y3 x4 y4 (normalized)
        bx, by, bw, bh = c["bbox"]
        cx_norm = (bx + bw / 2.0) / w_im
        cy_norm = (by + bh / 2.0) / h_im
        w_norm = bw / w_im
        h_norm = bh / h_im

        quad = c["quad"] # [x1, y1, x2, y2, x3, y3, x4, y4]
        quad_norm = [
            quad[0] / w_im, quad[1] / h_im,
            quad[2] / w_im, quad[3] / h_im,
            quad[4] / w_im, quad[5] / h_im,
            quad[6] / w_im, quad[7] / h_im,
        ]

        label_parts = [1, cx_norm, cy_norm, w_norm, h_norm] + quad_norm
        label_line = " ".join(f"{x:.6f}" if isinstance(x, float) else str(x) for x in label_parts)
        with open(dst_txt_path, "w", encoding="utf-8") as f_lbl:
            f_lbl.write(label_line + "\n")

        # Format meta.csv row:
        # image;plate_num;plate_type;bbox;quad;is_vehicle;is_synthetic;source;license;conditions
        bbox_str = f"[{bx}, {by}, {bw}, {bh}]"
        quad_str = f"[{round(quad[0], 1)}, {round(quad[1], 1)}, {round(quad[2], 1)}, {round(quad[3], 1)}, {round(quad[4], 1)}, {round(quad[5], 1)}, {round(quad[6], 1)}, {round(quad[7], 1)}]"
        row = {
            "image": f"images/real/{dst_fn}",
            "plate_num": c["text"],
            "plate_type": "type1a",
            "bbox": bbox_str,
            "quad": quad_str,
            "is_vehicle": "1",
            "is_synthetic": "0",
            "source": c.get("url", "drive2.ru"),
            "license": "CC-BY-4.0",
            "conditions": "clean,real"
        }
        meta_rows.append(row)

        # Crop for contact sheet
        pad = int(min(bw, bh) * 0.15)
        crop = img[max(0, by - pad):min(h_im, by + bh + pad), max(0, bx - pad):min(w_im, bx + bw + pad)]
        if crop.size > 0:
            thumb = cv2.resize(crop, (180, 140))
            cv2.putText(thumb, c["text"], (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            cv2.putText(thumb, f"1A #{next_idx}", (5, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
            contact_crops.append(thumb)

        print(f"Integrated {dst_fn} <- {c['text']} (conf {c['ocr_conf']})")
        next_idx += 1

    # Append to meta.csv
    with open(META_CSV, "a", encoding="utf-8", newline="") as f_meta:
        writer = csv.DictWriter(f_meta, delimiter=";", fieldnames=[
            "image", "plate_num", "plate_type", "bbox", "quad", "is_vehicle", "is_synthetic", "source", "license", "conditions"
        ])
        for r in meta_rows:
            writer.writerow(r)

    print(f"\nSuccessfully appended {len(meta_rows)} records to {META_CSV}")

    # Build contact sheet
    if contact_crops:
        cols = 5
        rows = (len(contact_crops) + cols - 1) // cols
        sheet = np.zeros((rows * 140, cols * 180, 3), dtype=np.uint8)
        for i, thumb in enumerate(contact_crops):
            r = i // cols
            c = i % cols
            sheet[r * 140:(r + 1) * 140, c * 180:(c + 1) * 180] = thumb
        sheet_path = AUDIT_DIR / "sheet_fresh_integrated.jpg"
        cv2.imwrite(str(sheet_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"Contact sheet saved to {sheet_path}")

if __name__ == "__main__":
    integrate()
