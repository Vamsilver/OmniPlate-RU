#!/usr/bin/env python3
"""
OmniPlate-RU - Colorimetric Sanitation of Disputed Type 1B Records.

Analyzes ground truth audit registry (test_output/omniplate_ground_truth_audit.csv),
verifies plate background color in HSV color space (Pantone 116C yellow vs white/gray),
and automatically reclassifies false Type 1B plates (with Type 1 formula and S < 60)
to Type 1.
"""

import argparse
import csv
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.pipeline import OmniPlatePipeline

CYR_TO_LAT: Dict[str, str] = {
    "\u0410": "A", "\u0412": "B", "\u0415": "E", "\u041a": "K", "\u041c": "M", "\u041d": "H",
    "\u041e": "O", "\u0420": "P", "\u0421": "C", "\u0422": "T", "\u0423": "Y", "\u0425": "X",
}

TYPE1_REGEX = re.compile(r"^[ABEKMHOPCTYX]\d{3}[ABEKMHOPCTYX]{2}\d{2,3}$")


def normalize_plate_text(text: Optional[str]) -> str:
    if not text or not isinstance(text, str):
        return ""
    cleaned = text.strip().upper()
    return "".join(CYR_TO_LAT.get(ch, ch) for ch in cleaned)


def evaluate_crop_saturation(crop: np.ndarray) -> Tuple[float, float, float]:
    if crop is None or crop.size == 0:
        return 0.0, 0.0, 0.0

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    bg_mask = hsv[:, :, 2] > 80
    if np.sum(bg_mask) > 0:
        bg_s = hsv[:, :, 1][bg_mask]
        bg_mean_s = float(np.mean(bg_s))
    else:
        bg_mean_s = float(np.mean(hsv[:, :, 1]))

    yellow_mask = (hsv[:, :, 0] >= 12) & (hsv[:, :, 0] <= 38) & (hsv[:, :, 1] >= 60) & (hsv[:, :, 2] >= 60)
    yellow_ratio = float(np.mean(yellow_mask))

    h_yellow = (hsv[:, :, 0] >= 12) & (hsv[:, :, 0] <= 38)
    if np.sum(h_yellow) > 10:
        yellow_mean_s = float(np.mean(hsv[:, :, 1][h_yellow]))
    else:
        yellow_mean_s = 0.0

    return bg_mean_s, yellow_ratio, yellow_mean_s


def sync_registry_after_sanitation(csv_path: Path, registry_path: Path, img_dir: Path) -> dict:
    all_imgs = [f.name.lower() for f in img_dir.glob("*") if f.suffix.lower() in (".jpg", ".png")] if img_dir.exists() else []
    total_real = len(all_imgs) if all_imgs else 531

    audited_map = {}
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f, delimiter=";")
            header = next(reader, None)
            for row in reader:
                if len(row) >= 6 and row[1]:
                    fn = Path(row[1].strip()).name.lower()
                    status = row[4].strip()
                    if status != "unreviewed":
                        audited_map[fn] = status

    audited_count = len(audited_map)
    unseen_count = max(0, total_real - audited_count)
    progress_pct = round((audited_count / max(1, total_real)) * 100.0, 1)

    reg_data = {
        "total_real_images": total_real,
        "audited_count": audited_count,
        "unseen_count": unseen_count,
        "progress_percent": progress_pct,
        "audited_filenames": sorted(list(audited_map.keys())),
    }
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(reg_data, f, indent=2, ensure_ascii=False)
    return reg_data


def main():
    parser = argparse.ArgumentParser(description="Colorimetric Sanitation of False Type 1B Plates in Audit Registry")
    parser.add_argument("--csv", type=str, default="test_output/omniplate_ground_truth_audit.csv", help="Path to audit CSV")
    parser.add_argument("--images_dir", type=str, default="dataset/images/real", help="Directory with real images")
    parser.add_argument("--s-threshold", type=float, default=60.0, help="Max saturation threshold for white/gray plates (default: 60.0)")
    parser.add_argument("--dry-run", action="store_true", help="Simulate sanitation without writing changes to disk")
    parser.add_argument("--no-backup", action="store_true", help="Do not create backup of CSV before writing")
    args = parser.parse_args()

    csv_path = PROJECT_ROOT / args.csv
    images_dir = PROJECT_ROOT / args.images_dir
    reg_path = PROJECT_ROOT / "test_output" / "audited_registry.json"

    if not csv_path.exists():
        print(f"[!] Error: Audit CSV not found at {csv_path}")
        sys.exit(1)

    print("=" * 75)
    print("OmniPlate-RU - Colorimetric Sanitation of False Type 1B Plates")
    print("=" * 75)
    print(f"[*] CSV file:        {csv_path}")
    print(f"[*] Images dir:      {images_dir}")
    print(f"[*] S threshold:     S < {args.s_threshold:.1f}")
    print(f"[*] Mode:            {'DRY-RUN' if args.dry_run else 'LIVE (disk write)'}")
    print("=" * 75)

    rows = []
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter=";")
        header = next(reader)
        for row in reader:
            if row:
                rows.append(row)

    print(f"[*] Total rows in CSV: {len(rows)}")

    pipeline = OmniPlatePipeline(device="cuda")
    pipeline.warmup(iterations=1)

    type1b_candidates = []
    for idx, row in enumerate(rows):
        if len(row) >= 3 and row[2].strip().lower() == "type1b":
            type1b_candidates.append((idx, row))

    print(f"[*] Total type1b candidates: {len(type1b_candidates)}")

    reclassified = []
    retained_type1b = []
    missing_images = []

    for row_idx, row in type1b_candidates:
        row_id = row[0]
        fname = row[1].strip()
        pred_text = normalize_plate_text(row[3])
        gt_text = normalize_plate_text(row[5]) if len(row) > 5 else ""

        effective_text = gt_text if gt_text and gt_text != "###" else pred_text
        is_t1_formula = bool(TYPE1_REGEX.match(effective_text))

        img_path = images_dir / fname
        if not img_path.exists():
            missing_images.append(fname)
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            missing_images.append(fname)
            continue

        dets = pipeline.predict(img)
        if not dets or dets[0].rectified_crop is None:
            retained_type1b.append((fname, effective_text, is_t1_formula, 999.0, "no_crop"))
            continue

        crop = dets[0].rectified_crop
        bg_mean_s, yellow_ratio, yellow_mean_s = evaluate_crop_saturation(crop)

        if is_t1_formula and bg_mean_s < args.s_threshold:
            reclassified.append({
                "row_idx": row_idx,
                "filename": fname,
                "text": effective_text,
                "old_type": "type1b",
                "new_type": "type1",
                "bg_mean_s": bg_mean_s,
                "yellow_ratio": yellow_ratio,
            })
            if not args.dry_run:
                rows[row_idx][2] = "type1"
        else:
            retained_type1b.append((fname, effective_text, is_t1_formula, bg_mean_s, "kept"))

    print("\n" + "=" * 75)
    print("SANITATION RESULTS")
    print("=" * 75)
    print(f"[*] Checked Type 1B records:            {len(type1b_candidates)}")
    print(f"[*] Reclassified to Type 1 (S < {args.s_threshold:.1f}): {len(reclassified)}")
    print(f"[*] Retained in Type 1B:                 {len(retained_type1b)}")
    if missing_images:
        print(f"[!] Images missing on disk:              {len(missing_images)}")

    if reclassified:
        print(f"\nSample reclassified records (first 15 of {len(reclassified)}):")
        print(f"{'Filename':<24} | {'Text':<12} | {'Type':<18} | {'Bg Sat (S)':<10} | {'Yellow Pct':<10}")
        print("-" * 75)
        for item in reclassified[:15]:
            print(f"{item['filename']:<24} | {item['text']:<12} | {item['old_type']} -> {item['new_type']:<7} | {item['bg_mean_s']:<10.1f} | {item['yellow_ratio']*100:<9.1f}%")

    if not args.dry_run:
        if not args.no_backup:
            bak_path = csv_path.with_suffix(".csv.bak")
            shutil.copy2(csv_path, bak_path)
            print(f"\n[+] Backup saved to: {bak_path.name}")

        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(header)
            writer.writerows(rows)
        print(f"[+] Updated CSV written to: {csv_path.name}")

        reg_info = sync_registry_after_sanitation(csv_path, reg_path, images_dir)
        print(f"[+] Registry {reg_path.name} synced (Audited: {reg_info['audited_count']}/{reg_info['total_real_images']} [{reg_info['progress_percent']}%])")
    else:
        print("\n[i] DRY-RUN Mode: No files were modified.")

    print("=" * 75)


if __name__ == "__main__":
    main()
