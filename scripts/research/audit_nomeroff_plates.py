#!/usr/bin/env python3
"""
Audit and extract predicted plates for the 238 Nomeroff images.
"""

import csv
import os
import re
import sys
from pathlib import Path
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.pipeline import OmniPlatePipeline
from src.pipeline.decoder import is_valid_gost_plate

META_PATH = PROJECT_ROOT / "dataset" / "meta.csv"

def main():
    with open(META_PATH, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f, delimiter=";"))

    nomeroff_rows = [r for r in reader if "nomeroff" in r.get("source", "")]
    print(f"Total Nomeroff rows: {len(nomeroff_rows)}")

    pipeline = OmniPlatePipeline(device="cuda")
    pipeline.warmup(2)

    valid_t1_count = 0
    non_t1 = []

    for i, r in enumerate(nomeroff_rows):
        img_path = PROJECT_ROOT / "dataset" / r["image"]
        img = cv2.imread(str(img_path))
        if img is None:
            non_t1.append((r["image"], "IMG_MISSING", "", 0.0, 0.0))
            continue

        dets = pipeline.predict(img)
        if not dets:
            non_t1.append((r["image"], "NO_DET", "", 0.0, 0.0))
            continue

        def det_sort_key(d):
            is_target = 1 if d.plate_type in ("type1", "type1a", "type1b") else 0
            has_text = 1 if d.text and "#" not in d.text else 0
            quality = (d.confidence ** 0.5) * (d.ocr_confidence ** 2)
            return (is_target, has_text, quality, d.confidence)

        dets = sorted(dets, key=det_sort_key, reverse=True)
        best = dets[0]

        is_gost = is_valid_gost_plate(best.text, "type1")
        if is_gost and best.plate_type == "type1":
            valid_t1_count += 1
        else:
            non_t1.append((r["image"], best.plate_type, best.text, best.confidence, best.ocr_confidence))

    print(f"\nAudit Finished:")
    print(f"Valid GOST Type 1 detected: {valid_t1_count} / {len(nomeroff_rows)}")
    print(f"Non-Type1 or invalid count: {len(non_t1)}")
    for item in non_t1:
        print(f"  {item[0]}: type={item[1]}, text='{item[2]}', conf={item[3]:.3f}, ocr_conf={item[4]:.3f}")

if __name__ == "__main__":
    main()
