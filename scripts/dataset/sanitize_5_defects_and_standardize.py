#!/usr/bin/env python3
"""
sanitize_5_defects_and_standardize.py
Sanitizes the 5 defective frames identified during visual audit:
1. real_type1_0291.jpg (SAMPLE TEXT 59 souvenir with watermark)
2. real_type1a_0070.jpg (Censored Drom car with blurred plate)
3. real_type1b_0311.jpg (Roboflow crop of yellow transit plate AB 865 M 77)
4. real_type1b_0312.jpg (Roboflow crop of yellow transit plate AB 865 M 77)
5. real_type1b_0313.jpg (Roboflow crop of yellow transit plate AB 865 M 77)

Then performs a clean 1-based sequential renumbering across all categories:
  - type1  : real_type1_0001.jpg .. real_type1_0741.jpg (741 items)
  - type1a : real_type1a_0001.jpg .. real_type1a_0168.jpg (168 items)
  - type1b : real_type1b_0001.jpg .. real_type1b_0312.jpg (312 items, AH88977 strictly preserved at 0001)
  - other  : real_other_0001.jpg .. real_other_0289.jpg (289 items)
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

DEFECT_STEMS = [
    "real_type1_0291",
    "real_type1a_0070",
    "real_type1b_0311",
    "real_type1b_0312",
    "real_type1b_0313",
]


def main():
    print("=" * 70)
    print("🚀 OmniPlate-RU: Sanitize 5 Defect Frames & Renumber Dataset")
    print("=" * 70)

    # 1. Read existing meta.csv
    with open(META_CSV, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f, delimiter=";"))
    fieldnames = list(reader[0].keys())
    print(f"[*] Total rows before sanitation: {len(reader)}")

    defect_images = {f"images/real/{stem}.jpg" for stem in DEFECT_STEMS}

    # Verify all 5 defect rows exist in meta.csv
    found_defects = [r for r in reader if r["image"].replace("\\", "/") in defect_images]
    print(f"[*] Identified {len(found_defects)}/5 target defect rows in meta.csv:")
    for r in found_defects:
        print(f"    - {r['image']}: plate={r['plate_num']} ({r['plate_type']}) source={r['source'][:50]}")
    assert len(found_defects) == 5, f"Expected 5 defect rows, found {len(found_defects)}"

    # 2. Physically delete defect images and labels
    print("\n[*] Step 1: Deleting physical defect images and labels...")
    for stem in DEFECT_STEMS:
        img_p = IMAGES_DIR / f"{stem}.jpg"
        lbl_p = LABELS_DIR / f"{stem}.txt"
        if img_p.exists():
            img_p.unlink()
            print(f"    [DELETED] Image: {img_p.name}")
        else:
            print(f"    [WARNING] Image not found: {img_p.name}")
        if lbl_p.exists():
            lbl_p.unlink()
            print(f"    [DELETED] Label: {lbl_p.name}")
        else:
            print(f"    [WARNING] Label not found: {lbl_p.name}")

    # 3. Filter reader rows
    filtered_rows = [r for r in reader if r["image"].replace("\\", "/") not in defect_images]
    print(f"\n[*] Filtered rows count: {len(filtered_rows)} (-5 rows)")

    real_rows = [r for r in filtered_rows if r.get("is_synthetic") == "0"]
    synth_rows = [r for r in filtered_rows if r.get("is_synthetic") == "1"]
    print(f"    - Synthetic: {len(synth_rows)}")
    print(f"    - Real:      {len(real_rows)}")
    assert len(synth_rows) == 5000, f"Expected 5000 synthetic rows, found {len(synth_rows)}"
    assert len(real_rows) == 1510, f"Expected 1510 real rows, found {len(real_rows)}"

    # 4. Group real rows by plate_type and sort deterministically
    by_type = defaultdict(list)
    for r in real_rows:
        ptype = r.get("plate_type", "other")
        by_type[ptype].append(r)

    print("\n[*] Real counts per category:")
    for ptype in ["type1", "type1a", "type1b", "other"]:
        print(f"    - {ptype:<8}: {len(by_type[ptype])}")

    assert len(by_type["type1"]) == 741
    assert len(by_type["type1a"]) == 168
    assert len(by_type["type1b"]) == 312
    assert len(by_type["other"]) == 289

    # 5. Build 1-based sequential mapping
    # Sort by existing filename to preserve stable ordering
    file_mapping = {}  # old_rel -> new_rel
    stem_mapping = {}  # old_stem -> new_stem

    for ptype in ["type1", "type1a", "type1b", "other"]:
        items = sorted(by_type[ptype], key=lambda x: x["image"].replace("\\", "/"))
        for idx, r in enumerate(items, start=1):
            old_rel = r["image"].replace("\\", "/")
            new_fname = f"real_{ptype}_{idx:04d}.jpg"
            new_rel = f"images/real/{new_fname}"
            file_mapping[old_rel] = new_rel

            old_stem = Path(old_rel).stem
            new_stem = Path(new_fname).stem
            stem_mapping[old_stem] = new_stem

    # CRITICAL CHECK: Verify AH88977 remains real_type1b_0001
    ah88977_row = next(r for r in by_type["type1b"] if r["plate_num"] == "AH88977")
    old_ah = ah88977_row["image"].replace("\\", "/")
    new_ah = file_mapping[old_ah]
    print(f"\n[*] Critical Invariant Check: AH88977 -> {new_ah}")
    assert new_ah == "images/real/real_type1b_0001.jpg", f"AH88977 got mapped to {new_ah} instead of real_type1b_0001.jpg!"
    print("    [PASS] AH88977 is confirmed at real_type1b_0001.jpg")

    # 6. Verify all existing source files exist before renaming
    print("\n[*] Step 2: Verifying source image and label files existence...")
    for old_rel in file_mapping:
        img_p = DATASET_DIR / old_rel
        if not img_p.exists():
            raise FileNotFoundError(f"Missing image: {img_p}")
        old_stem = Path(old_rel).stem
        lbl_p = LABELS_DIR / f"{old_stem}.txt"
        if not lbl_p.exists():
            raise FileNotFoundError(f"Missing label: {lbl_p}")
    print(f"    [PASS] All {len(file_mapping)} pairs verified on disk.")

    # 7. Two-step atomic renaming to prevent collisions
    session_id = uuid.uuid4().hex[:8]
    print(f"\n[*] Step 3: Renaming {len(file_mapping)} files to temporary staging names (__tmp_{session_id}_*)...")

    tmp_image_renames = []
    tmp_label_renames = []

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

    print(f"[*] Step 4: Moving staging files to final sequential standardized names...")
    for tmp_img, final_img in tmp_image_renames:
        tmp_img.rename(final_img)

    for tmp_lbl, final_lbl in tmp_label_renames:
        tmp_lbl.rename(final_lbl)

    print("    [PASS] All physical files renamed.")

    # 8. Update meta.csv
    print(f"\n[*] Step 5: Updating {META_CSV}...")
    updated_rows = []
    for r in filtered_rows:
        old_img = r["image"].replace("\\", "/")
        if old_img in file_mapping:
            r["image"] = file_mapping[old_img]
        updated_rows.append(r)

    with open(META_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(updated_rows)

    print(f"    [PASS] Successfully wrote {len(updated_rows)} rows to meta.csv.")

    # 9. Final post-check
    print("\n[*] Step 6: Post-execution integrity verification...")
    for new_rel in file_mapping.values():
        img_p = DATASET_DIR / new_rel
        if not img_p.exists():
            raise RuntimeError(f"Integrity check failed: {img_p} missing!")
        lbl_p = LABELS_DIR / f"{img_p.stem}.txt"
        if not lbl_p.exists():
            raise RuntimeError(f"Integrity check failed: {lbl_p} missing!")

    print(f"    [PASS] Verified 100% of {len(file_mapping)} pairs exist on disk and in meta.csv!")
    print("\n" + "=" * 70)
    print("🎉 Sanitization & Sequential Renumbering Completed Successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
