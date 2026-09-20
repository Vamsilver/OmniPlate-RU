"""
Generate visual audit contact sheets for real_type1a_*.jpg and suspicious real_type1_*.jpg.
Grid: 5x5 per sheet (25 images per sheet).
Each cell includes:
- Top banner: Filename and cell index
- Bottom banner: Plate number, Type, Source snippet
- Scaled image with BBox (cyan) and Quad (green polygon + colored vertices)
- Zoom-in crop of the annotated plate area in bottom-right corner
"""

import os
import cv2
import csv
import numpy as np
from pathlib import Path

def create_contact_sheets():
    root = Path(__file__).resolve().parent.parent.parent
    img_dir = root / "dataset" / "images" / "real"
    label_dir = root / "dataset" / "labels"
    meta_path = root / "dataset" / "meta.csv"
    out_dir = root / "test_output" / "visual_audit_1a"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(meta_path, "r", encoding="utf-8") as f:
        meta_rows = {Path(r["image"]).name: r for r in csv.DictReader(f, delimiter=";")}

    # 1. All real_type1a_ images
    t1a_imgs = sorted([img_dir / name for name in meta_rows.keys() if name.startswith("real_type1a_")])
    print(f"Found {len(t1a_imgs)} real_type1a images.")

    # 2. Suspicious real_type1 images
    suspicious_t1_names = ["real_type1_0861.jpg", "real_type1_0862.jpg"]
    for name, r in meta_rows.items():
        if name.startswith("real_type1_"):
            s = r["source"]
            if any(k in s for k in ["9309bb34", "1947bcm", "014k", "014d", "A777RUS"]):
                if name not in suspicious_t1_names:
                    suspicious_t1_names.append(name)
    
    suspicious_t1_imgs = [img_dir / name for name in suspicious_t1_names if (img_dir / name).exists()]
    print(f"Found {len(suspicious_t1_imgs)} suspicious real_type1 images.")

    all_audit_imgs = t1a_imgs + suspicious_t1_imgs
    print(f"Total images for audit: {len(all_audit_imgs)}")

    cell_w, cell_h = 440, 440
    top_banner_h = 32
    bot_banner_h = 40
    cols, rows = 5, 5
    per_sheet = cols * rows

    num_sheets = (len(all_audit_imgs) + per_sheet - 1) // per_sheet

    for sheet_idx in range(num_sheets):
        start_idx = sheet_idx * per_sheet
        end_idx = min(start_idx + per_sheet, len(all_audit_imgs))
        batch = all_audit_imgs[start_idx:end_idx]

        sheet_img = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)

        for i, img_path in enumerate(batch):
            r = i // cols
            c = i % cols
            x0 = c * cell_w
            y0 = r * cell_h

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            orig_h, orig_w = img.shape[:2]

            # Read meta info
            meta_info = meta_rows.get(img_path.name, {})
            plate_num = meta_info.get("plate_num", "UNKNOWN")
            plate_type = meta_info.get("plate_type", "?")
            src_str = meta_info.get("source", "")
            if len(src_str) > 35:
                src_str = "..." + src_str[-32:]

            # Read label
            lbl_path = label_dir / f"{img_path.stem}.txt"
            label_data = None
            if lbl_path.exists():
                txt = lbl_path.read_text(encoding="utf-8").strip()
                if txt:
                    lines = txt.splitlines()
                    parts = [float(x) for x in lines[0].split()]
                    cls_id = int(parts[0])
                    cx, cy, bw, bh = parts[1:5]
                    quad = None
                    if len(parts) >= 13:
                        quad = parts[5:13]
                    label_data = (cls_id, cx, cy, bw, bh, quad)

            annotated = img.copy()
            crop = None

            if label_data:
                cls_id, cx, cy, bw, bh, quad = label_data
                bx1 = int((cx - bw / 2) * orig_w)
                by1 = int((cy - bh / 2) * orig_h)
                bx2 = int((cx + bw / 2) * orig_w)
                by2 = int((cy + bh / 2) * orig_h)

                bx1, by1 = max(0, bx1), max(0, by1)
                bx2, by2 = min(orig_w, bx2), min(orig_h, by2)

                # Crop for inset
                if bx2 > bx1 and by2 > by1:
                    mw = int((bx2 - bx1) * 0.15)
                    mh = int((by2 - by1) * 0.15)
                    cbx1, cby1 = max(0, bx1 - mw), max(0, by1 - mh)
                    cbx2, cby2 = min(orig_w, bx2 + mw), min(orig_h, by2 + mh)
                    crop = img[cby1:cby2, cbx1:cbx2].copy()

                # Draw bbox (cyan)
                cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (255, 255, 0), max(2, int(min(orig_w, orig_h) * 0.005)))

                # Draw quad if available
                if quad and len(quad) == 8:
                    pts = np.array([
                        [int(quad[0] * orig_w), int(quad[1] * orig_h)],
                        [int(quad[2] * orig_w), int(quad[3] * orig_h)],
                        [int(quad[4] * orig_w), int(quad[5] * orig_h)],
                        [int(quad[6] * orig_w), int(quad[7] * orig_h)]
                    ], dtype=np.int32)
                    cv2.polylines(annotated, [pts], isClosed=True, color=(0, 255, 0), thickness=max(2, int(min(orig_w, orig_h) * 0.005)))
                    colors = [(0, 0, 255), (0, 255, 255), (255, 0, 0), (255, 0, 255)]
                    radius = max(3, int(min(orig_w, orig_h) * 0.01))
                    for pt, col in zip(pts, colors):
                        cv2.circle(annotated, tuple(pt), radius, col, -1)

            avail_h = cell_h - top_banner_h - bot_banner_h
            avail_w = cell_w

            scale = min(avail_w / orig_w, avail_h / orig_h)
            new_w = max(1, int(orig_w * scale))
            new_h = max(1, int(orig_h * scale))
            resized = cv2.resize(annotated, (new_w, new_h), interpolation=cv2.INTER_AREA)

            offset_x = (avail_w - new_w) // 2
            offset_y = top_banner_h + (avail_h - new_h) // 2
            sheet_img[y0 + offset_y : y0 + offset_y + new_h, x0 + offset_x : x0 + offset_x + new_w] = resized

            # Crop inset in bottom right above bottom banner
            if crop is not None and crop.size > 0:
                ch, cw = crop.shape[:2]
                inset_max_w, inset_max_h = 160, 110
                c_scale = min(inset_max_w / cw, inset_max_h / ch)
                icw, ich = max(1, int(cw * c_scale)), max(1, int(ch * c_scale))
                crop_resized = cv2.resize(crop, (icw, ich), interpolation=cv2.INTER_AREA)

                iy = y0 + cell_h - bot_banner_h - ich - 4
                ix = x0 + cell_w - icw - 4
                cv2.rectangle(sheet_img, (ix - 2, iy - 2), (ix + icw + 2, iy + ich + 2), (0, 255, 255), 2)
                sheet_img[iy : iy + ich, ix : ix + icw] = crop_resized

            # Cell borders
            cv2.rectangle(sheet_img, (x0, y0), (x0 + cell_w, y0 + cell_h), (60, 60, 60), 1)

            # Top banner
            cv2.rectangle(sheet_img, (x0, y0), (x0 + cell_w, y0 + top_banner_h), (25, 25, 25), -1)
            cv2.line(sheet_img, (x0, y0 + top_banner_h), (x0 + cell_w, y0 + top_banner_h), (90, 90, 90), 1)
            idx_str = f"#{start_idx + i + 1} {img_path.name}"
            cv2.putText(sheet_img, idx_str, (x0 + 8, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

            # Bottom banner
            by_start = y0 + cell_h - bot_banner_h
            cv2.rectangle(sheet_img, (x0, by_start), (x0 + cell_w, y0 + cell_h), (20, 20, 20), -1)
            cv2.line(sheet_img, (x0, by_start), (x0 + cell_w, by_start), (90, 90, 90), 1)
            
            line1 = f"Plate: {plate_num} ({plate_type})"
            line2 = f"Src: {src_str}"
            cv2.putText(sheet_img, line1, (x0 + 6, by_start + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(sheet_img, line2, (x0 + 6, by_start + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)

        out_path = out_dir / f"sheet_1a_{sheet_idx + 1:02d}.jpg"
        cv2.imwrite(str(out_path), sheet_img, [cv2.IMWRITE_JPEG_QUALITY, 93])
        print(f"Saved {out_path.name} ({len(batch)} images)")

if __name__ == "__main__":
    create_contact_sheets()
