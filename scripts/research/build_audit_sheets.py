#!/usr/bin/env python3
"""
build_audit_sheets.py
Generates high-resolution visual audit contact sheets for any plate category.
Shows:
1. Banner with index, filename, ground truth plate_num, conditions.
2. Scene image with annotated BBox (cyan) and Quad (green polygon + colored corners).
3. Zoomed high-resolution crop of the plate area with sharp contrast.
"""

import argparse
import csv
import cv2
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
IMAGES_DIR = DATASET_DIR / "images" / "real"
LABELS_DIR = DATASET_DIR / "labels"
META_CSV = DATASET_DIR / "meta.csv"


def generate_sheets(category="type1a", cols=4, rows=4, out_dir_name="audit_1a"):
    out_dir = PROJECT_ROOT / "test_output" / out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(META_CSV, "r", encoding="utf-8") as f:
        meta_rows = {
            r["image"].replace("\\", "/"): r
            for r in csv.DictReader(f, delimiter=";")
            if r.get("is_synthetic") == "0" and r.get("plate_type") == category
        }

    pattern = f"real_{category}_*.jpg"
    img_files = sorted(list(IMAGES_DIR.glob(pattern)), key=lambda p: p.name)
    print(f"[*] Found {len(img_files)} images for category '{category}'")

    cell_w, cell_h = 560, 480
    banner_h = 44
    per_sheet = cols * rows
    num_sheets = (len(img_files) + per_sheet - 1) // per_sheet

    for sheet_idx in range(num_sheets):
        start_idx = sheet_idx * per_sheet
        end_idx = min(start_idx + per_sheet, len(img_files))
        batch = img_files[start_idx:end_idx]

        sheet_img = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)

        for i, img_path in enumerate(batch):
            r_idx = i // cols
            c_idx = i % cols
            x0 = c_idx * cell_w
            y0 = r_idx * cell_h

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            orig_h, orig_w = img.shape[:2]
            rel_name = f"images/real/{img_path.name}"
            m_row = meta_rows.get(rel_name, {})
            gt_text = m_row.get("plate_num", "???")
            cond = m_row.get("conditions", "")

            # Read label for bbox and quad
            lbl_path = LABELS_DIR / f"{img_path.stem}.txt"
            label_data = None
            if lbl_path.exists():
                txt = lbl_path.read_text().strip()
                if txt:
                    lines = txt.splitlines()
                    parts = [float(x) for x in lines[0].split()]
                    cls_id = int(parts[0])
                    cx, cy, bw, bh = parts[1:5]
                    quad = parts[5:13] if len(parts) >= 13 else None
                    label_data = (cls_id, cx, cy, bw, bh, quad)

            annotated = img.copy()
            crop = None

            if label_data:
                cls_id, cx, cy, bw, bh, quad = label_data
                bx1 = max(0, int((cx - bw / 2) * orig_w))
                by1 = max(0, int((cy - bh / 2) * orig_h))
                bx2 = min(orig_w, int((cx + bw / 2) * orig_w))
                by2 = min(orig_h, int((cy + bh / 2) * orig_h))

                # Crop plate with 15% margin
                mw = int((bx2 - bx1) * 0.15)
                mh = int((by2 - by1) * 0.15)
                cbx1 = max(0, bx1 - mw)
                cby1 = max(0, by1 - mh)
                cbx2 = min(orig_w, bx2 + mw)
                cby2 = min(orig_h, by2 + mh)
                if cbx2 > cbx1 and cby2 > cby1:
                    crop = img[cby1:cby2, cbx1:cbx2].copy()

                # Draw BBox
                th = max(2, int(min(orig_w, orig_h) * 0.006))
                cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (255, 255, 0), th)

                # Draw Quad
                if quad and len(quad) == 8:
                    pts = np.array([
                        [int(quad[0] * orig_w), int(quad[1] * orig_h)],
                        [int(quad[2] * orig_w), int(quad[3] * orig_h)],
                        [int(quad[4] * orig_w), int(quad[5] * orig_h)],
                        [int(quad[6] * orig_w), int(quad[7] * orig_h)]
                    ], dtype=np.int32)
                    cv2.polylines(annotated, [pts], isClosed=True, color=(0, 255, 0), thickness=th)
                    colors = [(0, 0, 255), (0, 255, 255), (255, 0, 0), (255, 0, 255)]
                    rad = max(4, int(min(orig_w, orig_h) * 0.012))
                    for pt, col in zip(pts, colors):
                        cv2.circle(annotated, tuple(pt), rad, col, -1)

            # Cell Layout:
            # Banner: y0 .. y0 + banner_h
            # Left side of body: Scene with annotations (width ~ 320)
            # Right side of body: High-res zoom crop (width ~ 220)
            avail_h = cell_h - banner_h - 10
            avail_w = 310

            scale = min(avail_w / orig_w, avail_h / orig_h)
            new_w = max(1, int(orig_w * scale))
            new_h = max(1, int(orig_h * scale))
            scene_resized = cv2.resize(annotated, (new_w, new_h), interpolation=cv2.INTER_AREA)

            off_x = 10 + (avail_w - new_w) // 2
            off_y = banner_h + 5 + (avail_h - new_h) // 2
            sheet_img[y0 + off_y : y0 + off_y + new_h, x0 + off_x : x0 + off_x + new_w] = scene_resized

            # Crop placement on right side
            if crop is not None and crop.size > 0:
                crop_box_x = x0 + 330
                crop_box_y = y0 + banner_h + 10
                crop_max_w = 215
                crop_max_h = cell_h - banner_h - 30

                ch, cw = crop.shape[:2]
                c_scale = min(crop_max_w / cw, crop_max_h / ch)
                c_w = max(1, int(cw * c_scale))
                c_h = max(1, int(ch * c_scale))
                c_resized = cv2.resize(crop, (c_w, c_h), interpolation=cv2.INTER_LINEAR)

                c_off_x = crop_box_x + (crop_max_w - c_w) // 2
                c_off_y = crop_box_y + (crop_max_h - c_h) // 2

                # draw subtle frame for crop
                cv2.rectangle(sheet_img, (c_off_x - 2, c_off_y - 2), (c_off_x + c_w + 2, c_off_y + c_h + 2), (0, 200, 255), 1)
                sheet_img[c_off_y : c_off_y + c_h, c_off_x : c_off_x + c_w] = c_resized

            # Cell borders
            cv2.rectangle(sheet_img, (x0, y0), (x0 + cell_w, y0 + cell_h), (60, 60, 60), 1)
            # Banner background
            cv2.rectangle(sheet_img, (x0, y0), (x0 + cell_w, y0 + banner_h), (20, 20, 20), -1)
            cv2.line(sheet_img, (x0, y0 + banner_h), (x0 + cell_w, y0 + banner_h), (80, 80, 80), 1)

            # Banner Text:
            # Left: filename
            fname_str = f"#{start_idx + i + 1} {img_path.name}"
            cv2.putText(sheet_img, fname_str, (x0 + 8, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
            if cond:
                cv2.putText(sheet_img, f"[{cond}]", (x0 + 8, y0 + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 140), 1, cv2.LINE_AA)

            # Right: Ground Truth Plate Number in bright green
            cv2.putText(sheet_img, gt_text, (x0 + cell_w - 200, y0 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        out_path = out_dir / f"audit_{category}_sheet_{sheet_idx + 1:02d}.jpg"
        cv2.imwrite(str(out_path), sheet_img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"    [+] Saved {out_path.name} ({batch[0].name} .. {batch[-1].name})")

    print(f"\n[PASS] Generated {num_sheets} audit sheets in {out_dir}")
    return num_sheets, out_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="type1a", choices=["type1", "type1a", "type1b", "other"])
    args = parser.parse_args()
    generate_sheets(args.category)
