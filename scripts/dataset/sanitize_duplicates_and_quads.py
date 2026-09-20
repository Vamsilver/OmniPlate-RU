#!/usr/bin/env python3
"""
OmniPlate-RU — Dataset Sanitation Script:
1. Remove 146 byte-by-byte duplicate images in real_type1a_ (leaving 165 unique images).
   - Deletes images from dataset/images/real/
   - Deletes labels from dataset/labels/
   - Removes rows from dataset/meta.csv
2. Synchronize bbox = qbox for all rows where bbox != qbox (60 meta-quad warnings total,
   12 deleted with duplicate type1a, 48 updated in surviving rows).
   - Updates bbox in dataset/meta.csv
   - Updates normalized YOLO cx, cy, w, h in dataset/labels/<stem>.txt
"""

import csv
import glob
import hashlib
import os
import sys
from collections import defaultdict
from pathlib import Path
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
IMAGES_DIR = DATASET_DIR / "images" / "real"
LABELS_DIR = DATASET_DIR / "labels"
META_PATH = DATASET_DIR / "meta.csv"


def bbox_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    inter_w = max(0.0, xi2 - xi1)
    inter_h = max(0.0, yi2 - yi1)
    inter = inter_w * inter_h
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0


def sanitize_dataset():
    print("=" * 70)
    print("🚀 OmniPlate-RU: Type 1A De-duplication & Meta-Quad BBox Alignment")
    print("=" * 70)

    # 1. Identify 146 Type 1A byte duplicates
    type1a_files = sorted(glob.glob(str(IMAGES_DIR / "real_type1a_*.jpg")))
    print(f"Found {len(type1a_files)} total real_type1a_ images.")

    by_hash = defaultdict(list)
    for p in type1a_files:
        with open(p, "rb") as f:
            h = hashlib.md5(f.read()).hexdigest()
        rel = os.path.relpath(p, DATASET_DIR).replace("\\", "/")
        by_hash[h].append((rel, p))

    kept_images = {}
    removed_images = {}

    for h, entries in by_hash.items():
        # Canonical is the first file (lowest index)
        entries.sort(key=lambda x: x[0])
        kept_images[entries[0][0]] = entries[0][1]
        for rel, abs_p in entries[1:]:
            removed_images[rel] = abs_p

    print(f"Unique Type 1A images kept: {len(kept_images)}")
    print(f"Duplicate Type 1A images to remove: {len(removed_images)}")
    assert len(kept_images) == 165, f"Expected 165 kept images, got {len(kept_images)}"
    assert len(removed_images) == 146, f"Expected 146 removed images, got {len(removed_images)}"

    # 2. Read meta.csv
    with open(META_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        fieldnames = reader.fieldnames
        rows = list(reader)

    print(f"Total rows in meta.csv before processing: {len(rows)}")

    # 3. Process rows: remove duplicates, align bbox to qbox
    updated_rows = []
    deleted_meta_count = 0
    fixed_quad_count = 0
    labels_updated_count = 0

    for row in rows:
        img_rel = row.get("image", "").strip()

        # Check if row belongs to removed duplicate
        if img_rel in removed_images:
            deleted_meta_count += 1
            continue

        # Check meta-quad alignment
        bbox_str = row.get("bbox", "").strip()
        quad_str = row.get("quad", "").strip()

        if bbox_str and quad_str:
            try:
                bbox_vals = [int(v.strip()) for v in bbox_str.split(",")]
                quad_vals = [int(v.strip()) for v in quad_str.split(",")]
            except ValueError:
                bbox_vals = []
                quad_vals = []

            if len(bbox_vals) == 4 and len(quad_vals) == 8:
                pts = [(float(quad_vals[i]), float(quad_vals[i + 1])) for i in range(0, 8, 2)]
                qx = [p[0] for p in pts]
                qy = [p[1] for p in pts]
                xmin, xmax = int(round(min(qx))), int(round(max(qx)))
                ymin, ymax = int(round(min(qy))), int(round(max(qy)))
                bw = xmax - xmin
                bh = ymax - ymin
                qbox = (float(xmin), float(ymin), float(bw), float(bh))

                # If IoU < 0.8, synchronize bbox = qbox
                if bbox_iou(tuple(map(float, bbox_vals)), qbox) < 0.8:
                    row["bbox"] = f"{xmin},{ymin},{bw},{bh}"
                    fixed_quad_count += 1

                    # Update corresponding label file
                    stem = Path(img_rel).stem
                    lbl_path = LABELS_DIR / f"{stem}.txt"
                    if lbl_path.exists():
                        img_path = DATASET_DIR / img_rel
                        with Image.open(img_path) as im:
                            img_w, img_h = im.size

                        # Read label line
                        with open(lbl_path, "r", encoding="utf-8") as lf:
                            lbl_lines = [ln.strip() for ln in lf.read().splitlines() if ln.strip()]

                        if len(lbl_lines) == 1:
                            toks = lbl_lines[0].split()
                            cls_id = toks[0]

                            cx = (xmin + bw / 2.0) / float(img_w)
                            cy = (ymin + bh / 2.0) / float(img_h)
                            nw = bw / float(img_w)
                            nh = bh / float(img_h)

                            # If label has 13 elements, format new line with updated cx, cy, w, h
                            if len(toks) == 13:
                                quad_norm_strs = toks[5:]
                                new_line = f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f} " + " ".join(quad_norm_strs)
                            else:
                                new_line = f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"

                            with open(lbl_path, "w", encoding="utf-8") as lf:
                                lf.write(new_line + "\n")
                            labels_updated_count += 1

        updated_rows.append(row)

    print(f"Rows removed from meta.csv: {deleted_meta_count}")
    print(f"Meta-quad rows fixed to tight qbox: {fixed_quad_count}")
    print(f"Label files updated with new bbox: {labels_updated_count}")
    print(f"Total rows in new meta.csv: {len(updated_rows)}")
    assert deleted_meta_count == 146, f"Expected 146 rows deleted, got {deleted_meta_count}"
    assert len(updated_rows) == len(rows) - 146

    # 4. Physically delete duplicate image and label files
    deleted_images_count = 0
    deleted_labels_count = 0

    for rel, abs_p in removed_images.items():
        p_img = Path(abs_p)
        if p_img.exists():
            p_img.unlink()
            deleted_images_count += 1

        stem = p_img.stem
        p_lbl = LABELS_DIR / f"{stem}.txt"
        if p_lbl.exists():
            p_lbl.unlink()
            deleted_labels_count += 1

    print(f"Physical image files deleted: {deleted_images_count}")
    print(f"Physical label files deleted: {deleted_labels_count}")
    assert deleted_images_count == 146, f"Expected 146 images deleted, got {deleted_images_count}"
    assert deleted_labels_count == 146, f"Expected 146 labels deleted, got {deleted_labels_count}"

    # 5. Save updated meta.csv
    with open(META_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(updated_rows)
    print("Saved updated meta.csv successfully.")

    # 6. Verify quotas
    type1a_surviving = [r for r in updated_rows if "real_type1a_" in r.get("image", "")]
    unique_plates = set(r["plate_num"] for r in type1a_surviving if "#" not in r.get("plate_num", ""))
    print(f"Surviving real_type1a_ rows: {len(type1a_surviving)} (TZ quota >= 150)")
    print(f"Surviving real_type1a_ unique plates without #: {len(unique_plates)} (TZ quota >= 50)")
    assert len(type1a_surviving) >= 150
    assert len(unique_plates) >= 50

    print("=" * 70)
    print("✅ All de-duplication and bbox alignment operations completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    sanitize_dataset()
