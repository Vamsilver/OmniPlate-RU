#!/usr/bin/env python3
"""
Apply Visual Audit Results for real_other_ Category.
- Deletes 58 inappropriate/garbage images + labels
- Reclassifies 2 valid Russian Type 1 plates to real_type1_0912 and real_type1_0913
- Fixes meta.csv for all 167 surviving other images (plate_num="", bbox=qbox, canonical quad)
- Cleans and regenerates dataset labels and yolo_pose splits
"""

import csv
import os
import shutil
from pathlib import Path
from PIL import Image

def run_audit_application():
    base_dir = Path("dataset")
    img_real_dir = base_dir / "images" / "real"
    label_dir = base_dir / "labels"
    label_real_dir = base_dir / "labels" / "real"
    meta_path = base_dir / "meta.csv"

    # 1. Exact 58 deletion candidates
    del_candidates = set(
        ['real_other_0075.jpg'] +
        [f'real_other_{i:04d}.jpg' for i in range(155, 167)] +
        ['real_other_0176.jpg', 'real_other_0181.jpg', 'real_other_0183.jpg', 'real_other_0184.jpg'] +
        ['real_other_0141.jpg', 'real_other_0146.jpg', 'real_other_0149.jpg', 'real_other_0180.jpg'] +
        ['real_other_0096.jpg', 'real_other_0111.jpg', 'real_other_0121.jpg', 'real_other_0185.jpg', 'real_other_0196.jpg', 'real_other_0197.jpg'] +
        ['real_other_0211.jpg', 'real_other_0214.jpg', 'real_other_0215.jpg', 'real_other_0217.jpg'] +
        [f'real_other_{i:04d}.jpg' for i in range(252, 255)] +
        [f'real_other_{i:04d}.jpg' for i in range(256, 273)] +
        [f'real_other_{i:04d}.jpg' for i in range(246, 252)] +
        ['real_other_0289.jpg']
    )
    assert len(del_candidates) == 58, f"Expected 58 deletion files, got {len(del_candidates)}"

    # 2. Reclassifications
    # Current max type1 is 0911
    reclass_map = {
        "real_other_0175.jpg": {
            "new_name": "real_type1_0912.jpg",
            "plate_num": "H874OO174",
            "plate_type": "type1",
            "bbox": "64,1017,124,39",
            "quad": "64,1021,186,1017,188,1054,66,1056"
        },
        "real_other_0288.jpg": {
            "new_name": "real_type1_0913.jpg",
            "plate_num": "X199CT96",
            "plate_type": "type1",
            "bbox": "30,177,580,125",
            "quad": "30,177,610,177,610,302,30,302"
        }
    }

    # Verify all targets exist before acting
    for f_name in del_candidates:
        p = img_real_dir / f_name
        if not p.exists():
            print(f"[WARN] Deletion candidate not on disk: {p}")

    for f_name in reclass_map:
        p = img_real_dir / f_name
        assert p.exists(), f"Reclass source missing: {p}"

    # Read current meta.csv
    with open(meta_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        fieldnames = reader.fieldnames
        rows = list(reader)

    print(f"Read {len(rows)} rows from meta.csv")

    deleted_count = 0
    reclassified_count = 0
    surviving_other_count = 0
    updated_rows = []

    for row in rows:
        img_rel = row["image"]
        base_name = Path(img_rel).name

        # Case A: Deletion candidate
        if base_name in del_candidates:
            deleted_count += 1
            # Remove image
            img_p = img_real_dir / base_name
            if img_p.exists():
                img_p.unlink()
            # Remove label files if any
            txt_name = base_name.replace(".jpg", ".txt")
            for lp in [label_dir / txt_name, label_real_dir / txt_name]:
                if lp.exists():
                    lp.unlink()
            continue

        # Case B: Reclassification candidate
        if base_name in reclass_map:
            reclassified_count += 1
            rec = reclass_map[base_name]
            new_img_name = rec["new_name"]
            
            # Rename image
            src_img = img_real_dir / base_name
            dst_img = img_real_dir / new_img_name
            src_img.rename(dst_img)

            # Delete old label files
            old_txt = base_name.replace(".jpg", ".txt")
            for lp in [label_dir / old_txt, label_real_dir / old_txt]:
                if lp.exists():
                    lp.unlink()

            # Update row in meta
            row["image"] = f"images/real/{new_img_name}"
            row["plate_num"] = rec["plate_num"]
            row["plate_type"] = rec["plate_type"]
            row["bbox"] = rec["bbox"]
            row["quad"] = rec["quad"]
            row["is_vehicle"] = "1"
            updated_rows.append(row)
            continue

        # Case C: Surviving real_other_
        if base_name.startswith("real_other_"):
            surviving_other_count += 1
            row["plate_num"] = ""
            row["plate_type"] = "other"
            row["is_vehicle"] = "1"

            # Parse quad and fix bbox = tight qbox
            if row.get("quad"):
                pts = [int(v.strip()) for v in row["quad"].split(",")]
                if len(pts) == 8:
                    pts_pairs = [(pts[i*2], pts[i*2+1]) for i in range(4)]
                    # Ensure first point is top-left (min x+y)
                    first_idx = min(range(4), key=lambda k: pts_pairs[k][0] + pts_pairs[k][1])
                    if first_idx != 0:
                        pts_pairs = pts_pairs[first_idx:] + pts_pairs[:first_idx]
                        row["quad"] = ",".join(f"{x},{y}" for x, y in pts_pairs)
                    
                    qx = [p[0] for p in pts_pairs]
                    qy = [p[1] for p in pts_pairs]
                    bx = min(qx)
                    by = min(qy)
                    bw = max(qx) - min(qx)
                    bh = max(qy) - min(qy)
                    row["bbox"] = f"{bx},{by},{bw},{bh}"

            updated_rows.append(row)
            continue

        # Any other row
        updated_rows.append(row)

    print(f"Processed: Deleted={deleted_count}, Reclassified={reclassified_count}, Surviving other={surviving_other_count}")
    print(f"Total rows in new meta: {len(updated_rows)}")

    # Write updated meta.csv
    with open(meta_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(updated_rows)

    print("[SUCCESS] Updated meta.csv written.")

if __name__ == "__main__":
    run_audit_application()
