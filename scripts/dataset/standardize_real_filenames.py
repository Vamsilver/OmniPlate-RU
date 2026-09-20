#!/usr/bin/env python3
"""
standardize_real_filenames.py
Standardizes real image and label filenames in OmniPlate-RU dataset.

Renames all real dataset files so their prefix strictly reflects their true GOST plate_type:
  - type1  -> images/real/real_type1_0001.jpg .. real_type1_0856.jpg
  - type1a -> images/real/real_type1a_0001.jpg .. real_type1a_0305.jpg
  - type1b -> images/real/real_type1b_0001.jpg .. real_type1b_0310.jpg
  - other  -> images/real/real_other_0001.jpg .. real_other_0289.jpg

Updates:
  1. dataset/images/real/*.jpg
  2. dataset/labels/*.txt
  3. dataset/meta.csv (image column)
"""

import csv
import os
import shutil
import sys
import uuid
from collections import defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
IMAGES_DIR = DATASET_DIR / "images" / "real"
LABELS_DIR = DATASET_DIR / "labels"
META_CSV = DATASET_DIR / "meta.csv"


def main():
    print("=" * 70)
    print("🚀 Standardizing Real Dataset Filenames by True Plate Type")
    print("=" * 70)

    # 1. Read meta.csv
    with open(META_CSV, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f, delimiter=";"))

    fieldnames = list(reader[0].keys())
    total_rows = len(reader)
    print(f"[*] Read {total_rows} rows from meta.csv")

    real_rows = [r for r in reader if r.get("is_synthetic") == "0"]
    synth_rows = [r for r in reader if r.get("is_synthetic") == "1"]
    print(f"    - Synthetic rows: {len(synth_rows)}")
    print(f"    - Real rows:      {len(real_rows)}")

    # 2. Group by plate_type
    by_type = defaultdict(list)
    for r in real_rows:
        ptype = r.get("plate_type", "other")
        by_type[ptype].append(r)

    # 3. Create deterministic mapping
    # Sort by existing image filename to preserve established ordering
    file_mapping = {}  # old_rel -> new_rel
    stem_mapping = {}  # old_stem -> new_stem

    for ptype in ["type1", "type1a", "type1b", "other"]:
        items = sorted(by_type[ptype], key=lambda x: x["image"])
        for idx, r in enumerate(items, start=1):
            old_rel = r["image"].replace("\\", "/")
            new_fname = f"real_{ptype}_{idx:04d}.jpg"
            new_rel = f"images/real/{new_fname}"
            file_mapping[old_rel] = new_rel

            old_stem = Path(old_rel).stem
            new_stem = Path(new_fname).stem
            stem_mapping[old_stem] = new_stem

    print(f"\n[*] Mapping generated for {len(file_mapping)} real images:")
    for ptype in ["type1", "type1a", "type1b", "other"]:
        sub = [v for v in file_mapping.values() if f"real_{ptype}_" in v]
        print(f"    - {ptype:<8}: {len(sub):>4} files ({min(sub)} .. {max(sub)})")

    assert len(file_mapping) == len(set(file_mapping.values())) == len(real_rows)

    # 4. Verify all source files exist
    missing_images = []
    missing_labels = []
    for old_rel in file_mapping:
        img_p = DATASET_DIR / old_rel
        if not img_p.exists():
            missing_images.append(str(img_p))
        old_stem = Path(old_rel).stem
        lbl_p = LABELS_DIR / f"{old_stem}.txt"
        if not lbl_p.exists():
            missing_labels.append(str(lbl_p))

    if missing_images:
        print(f"[ERROR] Missing source images ({len(missing_images)}): {missing_images[:5]}")
        sys.exit(1)
    if missing_labels:
        print(f"[ERROR] Missing source labels ({len(missing_labels)}): {missing_labels[:5]}")
        sys.exit(1)

    print("\n[+] All source images and labels verified!")

    # 5. Two-step atomic renaming to prevent collisions
    session_id = uuid.uuid4().hex[:8]
    print(f"\n[*] Step 1: Renaming {len(file_mapping)} files to temporary unique staging names...")

    tmp_image_renames = []  # (tmp_path, final_path)
    tmp_label_renames = []  # (tmp_path, final_path)

    for idx, (old_rel, new_rel) in enumerate(file_mapping.items(), start=1):
        old_img = DATASET_DIR / old_rel
        new_img = DATASET_DIR / new_rel
        tmp_img = IMAGES_DIR / f"__tmp_{session_id}_{idx:05d}.jpg"
        old_img.rename(tmp_img)
        tmp_image_renames.append((tmp_img, new_img))

        old_stem = Path(old_rel).stem
        new_stem = stem_mapping[old_stem]
        old_lbl = LABELS_DIR / f"{old_stem}.txt"
        new_lbl = LABELS_DIR / f"{new_stem}.txt"
        tmp_lbl = LABELS_DIR / f"__tmp_{session_id}_{idx:05d}.txt"
        old_lbl.rename(tmp_lbl)
        tmp_label_renames.append((tmp_lbl, new_lbl))

    print(f"[*] Step 2: Moving staging files to final standardized names...")
    for tmp_img, final_img in tmp_image_renames:
        tmp_img.rename(final_img)

    for tmp_lbl, final_lbl in tmp_label_renames:
        tmp_lbl.rename(final_lbl)

    print("[+] All image and label files successfully renamed!")

    # 6. Update meta.csv
    print(f"\n[*] Step 3: Updating {META_CSV}...")
    updated_rows = []
    for r in reader:
        old_img = r["image"].replace("\\", "/")
        if old_img in file_mapping:
            r["image"] = file_mapping[old_img]
        updated_rows.append(r)

    with open(META_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(updated_rows)

    print(f"[+] Successfully wrote {len(updated_rows)} rows to meta.csv!")

    # 7. Post-rename validation checks
    print("\n[*] Step 4: Post-rename verification checks...")
    for new_rel in file_mapping.values():
        img_p = DATASET_DIR / new_rel
        if not img_p.exists():
            raise RuntimeError(f"Verification failed: {img_p} does not exist!")
        lbl_p = LABELS_DIR / f"{img_p.stem}.txt"
        if not lbl_p.exists():
            raise RuntimeError(f"Verification failed: {lbl_p} does not exist!")

    print(f"[+] Verified 100% of {len(file_mapping)} real image & label pairs!")
    print("\n=======================================================")
    print("🎉 File Standardization Completed Successfully!")
    print("=======================================================")


if __name__ == "__main__":
    main()
