#!/usr/bin/env python3
"""
OmniPlate-RU — Sanitization and Reclassification of Misclassified 'Other' Frames.

Tasks:
1. Audit all 289 'real_other_*.jpg' images.
2. Reclassify genuine GOST Type 1 and Type 1A plates into 'real_type1_*.jpg' and 'real_type1a_*.jpg'.
3. Update labels in dataset/labels/*.txt with proper class IDs (0 for type1, 1 for type1a, 3 for other).
4. For remaining 'other' frames, set plate_num="" (empty string) to prevent false validator warnings.
5. Ensure 1:1 consistency between meta.csv and labels/*.txt.
"""

import csv
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.pipeline import OmniPlatePipeline
from src.pipeline.decoder import is_valid_gost_plate

DATASET_DIR = PROJECT_ROOT / "dataset"
IMAGES_REAL_DIR = DATASET_DIR / "images" / "real"
LABELS_DIR = DATASET_DIR / "labels"
META_PATH = DATASET_DIR / "meta.csv"
BACKUP_PATH = DATASET_DIR / "meta.csv.bak_other_sanitized"


def canonicalize_quad(bx: int, by: int, bw: int, bh: int, quad: Optional[List[float]] = None) -> List[int]:
    """Ensures quad is clockwise, starts at top-left, and matches bbox."""
    if quad is None or len(quad) != 8:
        pts = [(bx, by), (bx + bw, by), (bx + bw, by + bh), (bx, by + bh)]
    else:
        pts = [(quad[i], quad[i + 1]) for i in range(0, 8, 2)]

    # 1. Ensure clockwise orientation (positive shoelace area)
    s = 0.0
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        s += x1 * y2 - x2 * y1
    if s / 2.0 < 0:
        pts = [pts[0], pts[3], pts[2], pts[1]]

    # 2. First point: minimum sum of (x + y)
    first_idx = min(range(4), key=lambda k: pts[k][0] + pts[k][1])
    pts = pts[first_idx:] + pts[:first_idx]

    flat = []
    for x, y in pts:
        flat.extend([int(round(x)), int(round(y))])
    return flat


def write_yolo_label(
    label_path: Path,
    class_id: int,
    bbox: Tuple[int, int, int, int],
    quad: List[int],
    img_w: int,
    img_h: int
):
    """Writes a standardized YOLO pose annotation line."""
    bx, by, bw, bh = bbox
    xc = (bx + bw / 2.0) / float(img_w)
    yc = (by + bh / 2.0) / float(img_h)
    nw = bw / float(img_w)
    nh = bh / float(img_h)

    # Keypoints normalized
    kpts_str = " ".join(
        f"{quad[i] / float(img_w):.6f} {quad[i+1] / float(img_h):.6f}"
        for i in range(0, 8, 2)
    )

    content = f"{class_id} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f} {kpts_str}\n"
    label_path.write_text(content, encoding="utf-8", newline="\n")


def main():
    print("=" * 70)
    print("OmniPlate-RU — Other Category Sanitization & Reclassification")
    print("=" * 70)

    # 1. Backup meta.csv
    if not BACKUP_PATH.exists():
        print(f"[*] Creating backup: {BACKUP_PATH.name}")
        shutil.copy2(META_PATH, BACKUP_PATH)
    else:
        print(f"[*] Backup already exists: {BACKUP_PATH.name}")

    with open(META_PATH, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f, delimiter=";"))
        fieldnames = list(reader[0].keys())

    # 2. Find next indices for type1 and type1a
    existing_t1 = list(IMAGES_REAL_DIR.glob("real_type1_*.jpg"))
    existing_t1a = list(IMAGES_REAL_DIR.glob("real_type1a_*.jpg"))
    max_t1_idx = max([int(p.stem.split("_")[-1]) for p in existing_t1]) if existing_t1 else 0
    max_t1a_idx = max([int(p.stem.split("_")[-1]) for p in existing_t1a]) if existing_t1a else 0

    print(f"[*] Current max real_type1 index:  {max_t1_idx}")
    print(f"[*] Current max real_type1a index: {max_t1a_idx}")

    next_t1_idx = max_t1_idx + 1
    next_t1a_idx = max_t1a_idx + 1

    # 3. Initialize Pipeline
    pipeline = OmniPlatePipeline(device="cuda")
    pipeline.warmup(2)

    reclassified_t1 = 0
    reclassified_t1a = 0
    sanitized_other = 0

    reclassified_map = {}

    for row_idx, r in enumerate(reader):
        img_rel = r["image"]
        if "real_other_" not in img_rel:
            continue

        img_path = DATASET_DIR / img_rel
        if not img_path.exists():
            print(f"[-] Image not found: {img_path}")
            continue

        im = cv2.imread(str(img_path))
        if im is None:
            continue
        h, w = im.shape[:2]

        # Pipeline predictions
        dets = pipeline.predict(im)
        best_det = None
        if dets:
            def sort_key(d):
                is_target = 1 if d.plate_type in ("type1", "type1a", "type1b") else 0
                has_text = 1 if d.text and "#" not in d.text else 0
                quality = (d.confidence ** 0.5) * (d.ocr_confidence ** 2)
                return (is_target, has_text, quality, d.confidence)
            dets = sorted(dets, key=sort_key, reverse=True)
            best_det = dets[0]

        t1_txt, t1_c = pipeline.ocr.predict_single(im, "type1")

        is_protected_other = (
            "diplomatic" in r["source"].lower()
            or "kazakhstan" in r["source"].lower()
            or "trailer" in r["source"].lower()
            or "motorcycle" in r["source"].lower()
            or "ural" in r["source"].lower()
            or "kvas" in r["source"].lower()
        )

        is_reclass = False
        target_type = None
        target_text = None
        target_bbox = None
        target_quad = None

        if not is_protected_other:
            # Check 1: Valid pipeline detection
            if best_det and best_det.plate_type in ("type1", "type1a", "type1b") and is_valid_gost_plate(best_det.text, best_det.plate_type):
                is_reclass = True
                target_type = best_det.plate_type
                target_text = best_det.text
                target_bbox = best_det.bbox
                target_quad = canonicalize_quad(best_det.bbox[0], best_det.bbox[1], best_det.bbox[2], best_det.bbox[3], best_det.quad)
            # Check 2: Known Roboflow car plates
            elif r["plate_num"] in ("X606AB71", "K069XK70", "Y079XA57", "O032TE68"):
                is_reclass = True
                target_type = "type1"
                target_text = r["plate_num"]
                bx, by, bw, bh = [int(v) for v in r["bbox"].split(",")]
                raw_q = [float(v) for v in r["quad"].split(",")]
                target_bbox = (bx, by, bw, bh)
                target_quad = canonicalize_quad(bx, by, bw, bh, raw_q)
            # Check 3: Tight crop of Type 1 plate
            elif (w / float(h) > 2.0 and h < 250) and t1_c >= 0.85 and is_valid_gost_plate(t1_txt, "type1"):
                is_reclass = True
                target_type = "type1"
                target_text = t1_txt
                target_bbox = (0, 0, w, h)
                target_quad = canonicalize_quad(0, 0, w, h)

        old_stem = Path(img_rel).stem
        old_label_path = LABELS_DIR / f"{old_stem}.txt"

        if is_reclass:
            if target_type == "type1":
                new_stem = f"real_type1_{next_t1_idx:04d}"
                next_t1_idx += 1
                class_id = 0
                reclassified_t1 += 1
            else:
                new_stem = f"real_type1a_{next_t1a_idx:04d}"
                next_t1a_idx += 1
                class_id = 1
                reclassified_t1a += 1

            new_img_rel = f"images/real/{new_stem}.jpg"
            new_img_path = IMAGES_REAL_DIR / f"{new_stem}.jpg"
            new_label_path = LABELS_DIR / f"{new_stem}.txt"

            # 1. Move image file
            shutil.move(str(img_path), str(new_img_path))

            # 2. Delete old label file
            if old_label_path.exists():
                old_label_path.unlink()

            # 3. Create new label file
            write_yolo_label(new_label_path, class_id, target_bbox, target_quad, w, h)

            # 4. Update row in meta.csv
            r["image"] = new_img_rel
            r["plate_num"] = target_text
            r["plate_type"] = target_type
            r["bbox"] = f"{target_bbox[0]},{target_bbox[1]},{target_bbox[2]},{target_bbox[3]}"
            r["quad"] = ",".join(str(v) for v in target_quad)
            r["is_vehicle"] = "1"

            reclassified_map[img_rel] = (new_img_rel, target_type, target_text)
        else:
            # Stays in other: clear plate_num to empty string
            r["plate_num"] = ""
            r["plate_type"] = "other"
            sanitized_other += 1

            # Ensure label file matches meta.csv
            try:
                bx, by, bw, bh = [int(v) for v in r["bbox"].split(",")]
                raw_q = [float(v) for v in r["quad"].split(",")]
                quad = canonicalize_quad(bx, by, bw, bh, raw_q)
                # If bbox is valid (w > 0, h > 0), write class 3 annotation
                if bw > 0 and bh > 0:
                    write_yolo_label(old_label_path, 3, (bx, by, bw, bh), quad, w, h)
                else:
                    old_label_path.write_text("", encoding="utf-8")
            except Exception:
                old_label_path.write_text("", encoding="utf-8")

    # 4. Write updated meta.csv
    with open(META_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(reader)

    print("\n" + "=" * 70)
    print("Sanitization Completed Successfully!")
    print(f"  • Reclassified to Type 1:  {reclassified_t1} frames (new: real_type1_{max_t1_idx+1:04d}..{next_t1_idx-1:04d})")
    print(f"  • Reclassified to Type 1A: {reclassified_t1a} frames (new: real_type1a_{max_t1a_idx+1:04d}..{next_t1a_idx-1:04d})")
    print(f"  • Retained in Other:       {sanitized_other} frames (plate_num cleared to empty string)")
    print(f"  • Total meta rows written: {len(reader)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
