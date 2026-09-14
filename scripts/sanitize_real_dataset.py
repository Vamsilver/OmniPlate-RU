#!/usr/bin/env python3
"""
Sanitizes the Volga IT real dataset by strictly purging all non-plate junk:
- Removes false positives (scaffolding, windows, LED route displays, etc.)
- Retains only 100% authentic, verified license plates
- Cleans dataset/meta.csv and deletes rejected image files from dataset/images/real/
- Creates meta.csv.bak before any modification
"""

import csv
import os
import shutil
import sys
from typing import Dict, List, Set
import cv2

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from dataset.quality_filter import PlateQualityVerifier
from src.pipeline.rectifier import PlateRectifier

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def sanitize_dataset():
    print("=" * 68)
    print("  OmniPlate-RU: Real Dataset Deep Sanitize & Junk Purge")
    print("  Zero Tolerance for Scaffolding, Windows, and LED Signs")
    print("=" * 68)

    meta_path = os.path.join(ROOT_DIR, "dataset", "meta.csv")
    bak_path = os.path.join(ROOT_DIR, "dataset", "meta.csv.bak")

    if not os.path.exists(meta_path):
        print(f"[-] meta.csv not found at {meta_path}")
        return

    # Create backup
    shutil.copyfile(meta_path, bak_path)
    print(f"[+] Created backup: {bak_path}")

    rectifier = PlateRectifier()
    verifier = PlateQualityVerifier()

    clean_rows: List[List[str]] = []
    purged_files: List[str] = []
    stats = {
        "synth_kept": 0,
        "type1b_kept": 0,
        "type1b_purged": 0,
        "type1a_kept": 0,
        "type1a_purged": 0,
        "other_kept": 0,
        "other_purged": 0,
    }

    with open(bak_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter=";")
        header = next(reader)
        clean_rows.append(header)

        for row_idx, row in enumerate(reader, start=2):
            if len(row) < 10:
                continue

            img_rel, plate_num, p_type, bbox_str, quad_str, is_veh, is_syn, src, lic, cond = row

            # Keep all synthetic data intact
            if is_syn == "1":
                clean_rows.append(row)
                stats["synth_kept"] += 1
                continue

            full_img_p = os.path.join(ROOT_DIR, "dataset", img_rel.replace("/", os.sep))
            if not os.path.exists(full_img_p):
                stats[f"{p_type}_purged"] += 1
                purged_files.append(full_img_p)
                continue

            img = cv2.imread(full_img_p)
            if img is None:
                stats[f"{p_type}_purged"] += 1
                purged_files.append(full_img_p)
                continue

            # Rectify crop
            try:
                crop = rectifier.rectify(img, quad_str, plate_type=p_type)
            except Exception:
                try:
                    b = [int(v.strip()) for v in bbox_str.split(",")]
                    crop = img[b[1]:b[1]+b[3], b[0]:b[0]+b[2]]
                except Exception:
                    crop = None

            is_valid, reason, dbg = verifier.verify_crop(crop, plate_type=p_type)

            if is_valid:
                clean_rows.append(row)
                stats[f"{p_type}_kept"] += 1
            else:
                stats[f"{p_type}_purged"] += 1
                purged_files.append(full_img_p)

    # Overwrite meta.csv with clean data
    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerows(clean_rows)

    # Delete junk image files from disk
    print(f"\n[*] Deleting {len(purged_files)} junk files from disk...")
    deleted_count = 0
    for fp in purged_files:
        if os.path.exists(fp):
            try:
                os.remove(fp)
                deleted_count += 1
                # Also delete preview crop if exists
                fname = os.path.basename(fp)
                preview_p = os.path.join(ROOT_DIR, "dataset", "verified_previews", f"crop_{fname}")
                if os.path.exists(preview_p):
                    os.remove(preview_p)
            except Exception as e:
                print(f"    [-] Failed to remove {fp}: {e}")

    print(f"[+] Successfully deleted {deleted_count} junk images.")
    print("\n" + "=" * 68)
    print("  PURGE SUMMARY:")
    print(f"  • Synthetic Kept:     {stats['synth_kept']}")
    print(f"  • Type 1B (Yellow):   {stats['type1b_kept']} KEPT, {stats['type1b_purged']} PURGED")
    print(f"  • Type 1A (Square):   {stats['type1a_kept']} KEPT, {stats['type1a_purged']} PURGED")
    print(f"  • Other (Negatives):  {stats['other_kept']} KEPT, {stats['other_purged']} PURGED")
    total_real_clean = stats['type1b_kept'] + stats['type1a_kept'] + stats['other_kept']
    print(f"  • Total Clean Real:   {total_real_clean}")
    print("=" * 68)


if __name__ == "__main__":
    sanitize_dataset()
