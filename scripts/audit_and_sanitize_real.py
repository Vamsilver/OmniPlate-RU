import csv
import os
import shutil
import sys
from typing import Dict, List, Set, Tuple
import cv2
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from dataset.quality_filter import PlateQualityVerifier
from src.pipeline.rectifier import PlateRectifier

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def sanitize_and_audit():
    print("=" * 70)
    print("  OmniPlate-RU: Real Dataset Deep Audit & Sanitation (v2.0)")
    print("=" * 70)

    meta_path = os.path.join(ROOT_DIR, "dataset", "meta.csv")
    bak_path = os.path.join(ROOT_DIR, "dataset", "meta.csv.bak2")
    real_dir = os.path.join(ROOT_DIR, "dataset", "images", "real")
    preview_dir = os.path.join(ROOT_DIR, "dataset", "verified_previews")
    os.makedirs(preview_dir, exist_ok=True)

    if not os.path.exists(meta_path):
        print(f"[-] meta.csv not found at {meta_path}")
        return

    # Backup meta.csv
    shutil.copyfile(meta_path, bak_path)
    print(f"[+] Backup created at {bak_path}")

    rectifier = PlateRectifier()
    verifier = PlateQualityVerifier()

    clean_rows: List[List[str]] = []
    referenced_images: Set[str] = set()

    stats = {
        "synth_kept": 0,
        "type1_reclassified": 0,
        "type1a_kept": 0,
        "type1b_kept": 0,
        "type1_kept": 0,
        "other_kept": 0,
        "purged_junk": 0,
        "purged_missing": 0,
    }

    with open(bak_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter=";")
        header = next(reader)
        clean_rows.append(header)

        for row_idx, row in enumerate(reader, start=2):
            if len(row) < 10:
                continue

            img_rel, plate_num, p_type, bbox_str, quad_str, is_veh, is_syn, src, lic, cond = row[:10]

            # 1. Synthetic data: keep all 5000 rows untouched
            if is_syn == "1":
                clean_rows.append(row)
                referenced_images.add(os.path.basename(img_rel))
                stats["synth_kept"] += 1
                continue

            # 2. Real data: verify file existence
            full_img_p = os.path.join(ROOT_DIR, "dataset", img_rel.replace("/", os.sep))
            if not os.path.exists(full_img_p):
                stats["purged_missing"] += 1
                continue

            img = cv2.imread(full_img_p)
            if img is None:
                stats["purged_missing"] += 1
                continue

            ih, iw = img.shape[:2]

            # Parse BBox
            try:
                b_parts = [int(v.strip()) for v in bbox_str.split(",")]
                if len(b_parts) != 4:
                    stats["purged_junk"] += 1
                    continue
                bx, by, bw, bh = b_parts
            except Exception:
                stats["purged_junk"] += 1
                continue

            # Clamp bbox to image
            bx = max(0, min(iw - 1, bx))
            by = max(0, min(ih - 1, by))
            bw = max(1, min(iw - bx, bw))
            bh = max(1, min(ih - by, bh))
            bbox_str = f"{bx},{by},{bw},{bh}"

            # Check aspect ratio
            ar = float(bw) / float(bh)

            # Reclassification logic:
            # If labeled 'type1a' but AR > 2.20, it is physically a single-line Type 1 plate!
            if p_type == "type1a" and ar > 2.20:
                p_type = "type1"
                stats["type1_reclassified"] += 1

            # Quality check for crop
            crop = img[by:by+bh, bx:bx+bw]
            if crop.size == 0:
                stats["purged_junk"] += 1
                continue

            # Check quad coordinates
            try:
                q_parts = [float(v.strip()) for v in quad_str.split(",")]
                if len(q_parts) == 8:
                    pts = np.array(q_parts, dtype=np.float32).reshape(4, 2)
                    pts[:, 0] = np.clip(pts[:, 0], 0, iw)
                    pts[:, 1] = np.clip(pts[:, 1], 0, ih)
                    pts_int = pts.astype(np.int32)
                    # Check convexity and reasonable fill
                    if cv2.isContourConvex(pts_int) and (cv2.contourArea(pts) / float(bw * bh)) >= 0.45:
                        quad_str = ",".join(f"{int(v)}" for v in pts.flatten())
                    else:
                        # Fallback to tight bbox corners
                        quad_str = f"{bx},{by},{bx+bw},{by},{bx+bw},{by+bh},{bx},{by+bh}"
                else:
                    quad_str = f"{bx},{by},{bx+bw},{by},{bx+bw},{by+bh},{bx},{by+bh}"
            except Exception:
                quad_str = f"{bx},{by},{bx+bw},{by},{bx+bw},{by+bh},{bx},{by+bh}"

            # Verifier check for authenticity
            if p_type != "other":
                is_valid, reason, _ = verifier.verify_crop(crop, plate_type=p_type)
                # If non-yellow bus or degenerate, reject
                if not is_valid and ("NOT_GENUINE_YELLOW" in reason or "NO_CHARACTERS_FOUND" in reason or "TOO_SMALL" in reason):
                    stats["purged_junk"] += 1
                    continue

            # Update stats
            if p_type == "type1a":
                stats["type1a_kept"] += 1
            elif p_type == "type1b":
                stats["type1b_kept"] += 1
            elif p_type == "type1":
                stats["type1_kept"] += 1
            elif p_type == "other":
                stats["other_kept"] += 1

            fname = os.path.basename(img_rel)
            referenced_images.add(fname)

            # Update preview crop
            preview_p = os.path.join(preview_dir, f"crop_{fname}")
            cv2.imwrite(preview_p, crop)

            clean_rows.append([
                img_rel,
                plate_num,
                p_type,
                bbox_str,
                quad_str,
                is_veh,
                is_syn,
                src,
                lic,
                cond
            ])

    # Write cleaned meta.csv
    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerows(clean_rows)

    print(f"[+] Cleaned meta.csv written with {len(clean_rows)-1} entries.")

    # Clean orphaned files in real_dir
    print("[*] Scanning for orphaned files in dataset/images/real/...")
    orphaned = 0
    for f in os.listdir(real_dir):
        if f.endswith((".jpg", ".png", ".jpeg")) and f not in referenced_images:
            fp = os.path.join(real_dir, f)
            try:
                os.remove(fp)
                orphaned += 1
                # Also delete preview
                prev = os.path.join(preview_dir, f"crop_{f}")
                if os.path.exists(prev):
                    os.remove(prev)
            except Exception as e:
                print(f"    [-] Failed to remove {f}: {e}")

    print(f"[+] Removed {orphaned} orphaned files from disk.")
    print("\n" + "=" * 70)
    print("  AUDIT & SANITATION SUMMARY:")
    print(f"  • Synthetic Kept:        {stats['synth_kept']}")
    print(f"  • Type 1B (Yellow):      {stats['type1b_kept']}")
    print(f"  • Type 1A (True Square): {stats['type1a_kept']}")
    print(f"  • Type 1 (Single-Line):  {stats['type1_kept']} (including {stats['type1_reclassified']} reclassified from 1a)")
    print(f"  • Other (Negative):      {stats['other_kept']}")
    print(f"  • Purged Junk / Missing: {stats['purged_junk'] + stats['purged_missing']}")
    total_real = stats['type1b_kept'] + stats['type1a_kept'] + stats['type1_kept'] + stats['other_kept']
    print(f"  • Total Valid Real:      {total_real}")
    print("=" * 70)


if __name__ == "__main__":
    sanitize_and_audit()
