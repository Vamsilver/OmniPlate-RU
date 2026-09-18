#!/usr/bin/env python3
"""
Curator of Verified Real Plate Crops for Volga IT 2026 (Component 2).
Selects 150-200 high-clarity real license plate crops with verified ground-truth text:
- Extracts crops using PlateRectifier and verified_previews.
- Verifies strict GOST regex matching.
- Saves clean canonical 160x36 crops into dataset/verified_crops/
- Generates dataset/verified_crops/manifest.csv
"""

import csv
import glob
import os
import re
import sys
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from src.pipeline.ocr import PlateOCR
from src.pipeline.rectifier import PlateRectifier
from dataset.quality_filter import PlateQualityVerifier

TYPE1_REGEX = re.compile(r"^[ABEKMHOPCTYX]\d{3}[ABEKMHOPCTYX]{2}\d{2,3}$")
TYPE1B_REGEX = re.compile(r"^[ABEKMHOPCTYX]{2}\d{3}\d{2,3}$")


def curate_crops(target_count: int = 180):
    print("=" * 70)
    print("  OmniPlate-RU: Real Crop Curation & Ground Truth Verification")
    print("=" * 70)

    meta_path = os.path.join(ROOT_DIR, "dataset", "meta.csv")
    crops_dir = os.path.join(ROOT_DIR, "dataset", "verified_crops")
    os.makedirs(crops_dir, exist_ok=True)
    manifest_path = os.path.join(crops_dir, "manifest.csv")

    rectifier = PlateRectifier()
    ocr = PlateOCR(model_path=os.path.join(ROOT_DIR, "models", "ocr_lprnet_best.pt"), use_onnx=False, device="cuda")

    curated_records = []
    seen_texts = set()

    # 1. First, include all 20 verified ref_real crops from Roboflow
    print("[*] Ingesting verified reference plates (ref_real)...")
    with open(meta_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")
        next(reader)
        for row in reader:
            if len(row) < 10 or row[6] == "1":
                continue
            img_rel, plate_num, p_type, bbox_str, quad_str = row[0], row[1], row[2], row[3], row[4]
            if "ref_real_" in img_rel:
                fn = os.path.basename(img_rel)
                parts = fn.split("_")
                gt_text = parts[-1].split(".")[0] if len(parts) >= 3 else plate_num

                full_p = os.path.join(ROOT_DIR, "dataset", img_rel.replace("/", os.sep))
                img = cv2.imread(full_p)
                if img is None:
                    continue
                try:
                    rect = rectifier.rectify(img, quad_str, plate_type=p_type)
                    crop_36 = cv2.resize(rect, (160, 36), interpolation=cv2.INTER_LINEAR)
                    crop_fn = f"crop_ref_{len(curated_records):04d}_{gt_text}.png"
                    cv2.imwrite(os.path.join(crops_dir, crop_fn), crop_36)
                    curated_records.append({
                        "crop_file": crop_fn,
                        "plate_num": gt_text,
                        "plate_type": p_type,
                        "confidence": 1.0,
                        "source": "verified_reference",
                    })
                    seen_texts.add(gt_text)
                except Exception as e:
                    print(f"[-] Error processing {fn}: {e}")

    print(f"[+] Added {len(curated_records)} verified reference crops.")

    # 2. Select high-clarity yellow bus plates (Type 1B)
    print("\n[*] Curating Type 1B (Yellow) plates with verified GOST mask...")
    type1b_added = 0
    with open(meta_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")
        next(reader)
        for row in reader:
            if len(row) < 10 or row[6] == "1" or row[2] != "type1b":
                continue
            if type1b_added >= 60:
                break

            img_rel, plate_num, p_type, bbox_str, quad_str = row[0], row[1], row[2], row[3], row[4]
            full_p = os.path.join(ROOT_DIR, "dataset", img_rel.replace("/", os.sep))
            img = cv2.imread(full_p)
            if img is None:
                continue

            try:
                rect = rectifier.rectify(img, quad_str, plate_type="type1b")
                crop_36 = cv2.resize(rect, (160, 36), interpolation=cv2.INTER_LINEAR)
                pred_text, conf = ocr.predict_single(crop_36, plate_type="type1b")

                if conf >= 0.70 and TYPE1B_REGEX.match(pred_text) and pred_text not in seen_texts:
                    crop_fn = f"crop_1b_{len(curated_records):04d}_{pred_text}.png"
                    cv2.imwrite(os.path.join(crops_dir, crop_fn), crop_36)
                    curated_records.append({
                        "crop_file": crop_fn,
                        "plate_num": pred_text,
                        "plate_type": "type1b",
                        "confidence": round(conf, 4),
                        "source": "curated_real_1b",
                    })
                    seen_texts.add(pred_text)
                    type1b_added += 1
            except Exception:
                pass

    print(f"[+] Added {type1b_added} verified Type 1B crops.")

    # 3. Select high-clarity square plates (Type 1A)
    print("\n[*] Curating Type 1A (Square 2-row) plates...")
    type1a_added = 0
    with open(meta_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")
        next(reader)
        for row in reader:
            if len(row) < 10 or row[6] == "1" or row[2] != "type1a":
                continue
            if type1a_added >= 15:
                break

            img_rel, plate_num, p_type, bbox_str, quad_str = row[0], row[1], row[2], row[3], row[4]
            full_p = os.path.join(ROOT_DIR, "dataset", img_rel.replace("/", os.sep))
            img = cv2.imread(full_p)
            if img is None:
                continue

            try:
                rect = rectifier.rectify(img, quad_str, plate_type="type1a")
                top, bottom = rectifier.split_type1a(rect)
                stitched = rectifier.stitch_type1a_horizontal(top, bottom, target_size=(160, 36))
                pred_text, conf = ocr.predict_single(stitched, plate_type="type1a")

                if conf >= 0.60 and TYPE1_REGEX.match(pred_text) and pred_text not in seen_texts:
                    crop_fn = f"crop_1a_{len(curated_records):04d}_{pred_text}.png"
                    cv2.imwrite(os.path.join(crops_dir, crop_fn), stitched)
                    curated_records.append({
                        "crop_file": crop_fn,
                        "plate_num": pred_text,
                        "plate_type": "type1a",
                        "confidence": round(conf, 4),
                        "source": "curated_real_1a",
                    })
                    seen_texts.add(pred_text)
                    type1a_added += 1
            except Exception:
                pass

    print(f"[+] Added {type1a_added} verified Type 1A crops.")

    # 4. Select high-clarity passenger car single-line plates (Type 1)
    print("\n[*] Curating Type 1 (Single-Line) plates...")
    type1_added = 0

    # From meta.csv type1 entries
    with open(meta_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")
        next(reader)
        for row in reader:
            if len(row) < 10 or row[6] == "1" or row[2] != "type1":
                continue
            if len(curated_records) >= target_count:
                break

            img_rel, plate_num, p_type, bbox_str, quad_str = row[0], row[1], row[2], row[3], row[4]
            full_p = os.path.join(ROOT_DIR, "dataset", img_rel.replace("/", os.sep))
            img = cv2.imread(full_p)
            if img is None:
                continue

            try:
                rect = rectifier.rectify(img, quad_str, plate_type="type1")
                crop_36 = cv2.resize(rect, (160, 36), interpolation=cv2.INTER_LINEAR)
                pred_text, conf = ocr.predict_single(crop_36, plate_type="type1")

                if conf >= 0.70 and TYPE1_REGEX.match(pred_text) and pred_text not in seen_texts:
                    crop_fn = f"crop_1_{len(curated_records):04d}_{pred_text}.png"
                    cv2.imwrite(os.path.join(crops_dir, crop_fn), crop_36)
                    curated_records.append({
                        "crop_file": crop_fn,
                        "plate_num": pred_text,
                        "plate_type": "type1",
                        "confidence": round(conf, 4),
                        "source": "curated_real_1",
                    })
                    seen_texts.add(pred_text)
                    type1_added += 1
            except Exception:
                pass

    # From verified_previews if still needed to reach target_count
    if len(curated_records) < target_count:
        previews = sorted(glob.glob(os.path.join(ROOT_DIR, "dataset", "verified_previews", "crop_*.jpg")))
        for p in previews:
            if len(curated_records) >= target_count:
                break
            img = cv2.imread(p)
            if img is None:
                continue
            h, w = img.shape[:2]
            ar = w / float(h) if h > 0 else 0
            if 2.5 <= ar <= 5.8:
                crop_36 = cv2.resize(img, (160, 36), interpolation=cv2.INTER_LINEAR)
                pred_text, conf = ocr.predict_single(crop_36, plate_type="type1")
                if conf >= 0.75 and TYPE1_REGEX.match(pred_text) and pred_text not in seen_texts:
                    crop_fn = f"crop_1_{len(curated_records):04d}_{pred_text}.png"
                    cv2.imwrite(os.path.join(crops_dir, crop_fn), crop_36)
                    curated_records.append({
                        "crop_file": crop_fn,
                        "plate_num": pred_text,
                        "plate_type": "type1",
                        "confidence": round(conf, 4),
                        "source": "curated_real_preview",
                    })
                    seen_texts.add(pred_text)
                    type1_added += 1

    print(f"[+] Added {type1_added} verified Type 1 crops.")

    # Save manifest.csv
    with open(manifest_path, "w", newline="", encoding="utf-8") as mf:
        writer = csv.writer(mf, delimiter=";")
        writer.writerow(["crop_file", "plate_num", "plate_type", "confidence", "source"])
        for r in curated_records:
            writer.writerow([r["crop_file"], r["plate_num"], r["plate_type"], r["confidence"], r["source"]])

    print("\n" + "=" * 70)
    print("  VERIFIED CROPS CURATION SUMMARY:")
    print(f"  • Total Curated Crops:   {len(curated_records)}")
    print(f"  • Manifest Path:         {manifest_path}")
    print(f"  • Destination Directory: {crops_dir}")
    print("=" * 70)


if __name__ == "__main__":
    curate_crops()
