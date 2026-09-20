"""
Generate visual audit contact sheets for real_other_*.jpg images.
Grid: 5x5 per sheet (25 images per sheet).
Each cell includes:
- Filename banner
- Image with BBox (cyan) and Quad (green polygon + colored vertices)
- Zoom-in crop of the annotated area in corner
"""

import os
import cv2
import numpy as np
from pathlib import Path

def create_contact_sheets():
    root = Path(__file__).resolve().parent.parent.parent
    img_dir = root / "dataset" / "images" / "real"
    label_dir = root / "dataset" / "labels"
    out_dir = root / "test_output" / "visual_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    other_imgs = sorted(list(img_dir.glob("real_other_*.jpg")))
    print(f"Found {len(other_imgs)} real_other images to audit.")

    cell_w, cell_h = 420, 420
    banner_h = 36
    cols, rows = 5, 5
    per_sheet = cols * rows

    num_sheets = (len(other_imgs) + per_sheet - 1) // per_sheet

    for sheet_idx in range(num_sheets):
        start_idx = sheet_idx * per_sheet
        end_idx = min(start_idx + per_sheet, len(other_imgs))
        batch = other_imgs[start_idx:end_idx]

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

            # Read label
            lbl_path = label_dir / f"{img_path.stem}.txt"
            label_data = None
            if lbl_path.exists():
                txt = lbl_path.read_text().strip()
                if txt:
                    lines = txt.splitlines()
                    # take first line
                    parts = [float(x) for x in lines[0].split()]
                    cls_id = int(parts[0])
                    cx, cy, bw, bh = parts[1:5]
                    # check if quad keypoints present
                    quad = None
                    if len(parts) >= 13:
                        quad = parts[5:13]
                    label_data = (cls_id, cx, cy, bw, bh, quad)

            # Draw bbox and quad on copy of image
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
                    # add some margin
                    mw = int((bx2 - bx1) * 0.2)
                    mh = int((by2 - by1) * 0.2)
                    cbx1, cby1 = max(0, bx1 - mw), max(0, by1 - mh)
                    cbx2, cby2 = min(orig_w, bx2 + mw), min(orig_h, by2 + mh)
                    crop = img[cby1:cby2, cbx1:cbx2].copy()

                # Draw bbox
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
                    # Draw corner markers
                    colors = [(0, 0, 255), (0, 255, 255), (255, 0, 0), (255, 0, 255)] # TL=red, TR=yellow, BR=blue, BL=magenta
                    radius = max(3, int(min(orig_w, orig_h) * 0.01))
                    for pt, col in zip(pts, colors):
                        cv2.circle(annotated, tuple(pt), radius, col, -1)

            # Render into cell
            avail_h = cell_h - banner_h
            avail_w = cell_w

            # Resize maintaining aspect ratio
            scale = min(avail_w / orig_w, avail_h / orig_h)
            new_w = max(1, int(orig_w * scale))
            new_h = max(1, int(orig_h * scale))
            resized = cv2.resize(annotated, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # Center in cell (below banner)
            offset_x = (avail_w - new_w) // 2
            offset_y = banner_h + (avail_h - new_h) // 2
            sheet_img[y0 + offset_y : y0 + offset_y + new_h, x0 + offset_x : x0 + offset_x + new_w] = resized

            # Overlay crop inset in top-right or bottom-right corner if crop exists
            if crop is not None and crop.size > 0:
                ch, cw = crop.shape[:2]
                inset_max_w, inset_max_h = 130, 70
                c_scale = min(inset_max_w / cw, inset_max_h / ch)
                icw, ich = max(1, int(cw * c_scale)), max(1, int(ch * c_scale))
                crop_resized = cv2.resize(crop, (icw, ich), interpolation=cv2.INTER_AREA)
                
                # Bottom right inset
                iy = y0 + cell_h - ich - 4
                ix = x0 + cell_w - icw - 4
                # background border
                cv2.rectangle(sheet_img, (ix - 2, iy - 2), (ix + icw + 2, iy + ich + 2), (0, 255, 255), 1)
                sheet_img[iy : iy + ich, ix : ix + icw] = crop_resized

            # Cell borders
            cv2.rectangle(sheet_img, (x0, y0), (x0 + cell_w, y0 + cell_h), (50, 50, 50), 1)

            # Banner background
            cv2.rectangle(sheet_img, (x0, y0), (x0 + cell_w, y0 + banner_h), (25, 25, 25), -1)
            cv2.line(sheet_img, (x0, y0 + banner_h), (x0 + cell_w, y0 + banner_h), (80, 80, 80), 1)

            # Text
            idx_str = f"#{start_idx + i + 1} {img_path.name}"
            cv2.putText(sheet_img, idx_str, (x0 + 8, y0 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        out_path = out_dir / f"sheet_{sheet_idx + 1:02d}.jpg"
        cv2.imwrite(str(out_path), sheet_img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"Saved {out_path.name} ({len(batch)} images)")

if __name__ == "__main__":
    create_contact_sheets()
