#!/usr/bin/env python3
"""
Reclassifies 10 confirmed Type 1A square plates from other to type1a,
removes 2 tampered/masked Dodge frames, and keeps real dataset files strictly standardized.
Ensures real_type1b_0001.jpg remains AH88977 for the unit test.
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

# 1. 10 square plates to move to Type 1A
RECLASS_1A = {
    "real_other_0185.jpg": "A999CB21",
    "real_other_0186.jpg": "C056YB54",
    "real_other_0187.jpg": "E198MM196",
    "real_other_0188.jpg": "K616OB154",
    "real_other_0189.jpg": "K153HC154",
    "real_other_0190.jpg": "P852AM154",
    "real_other_0191.jpg": "M480TA154",
    "real_other_0192.jpg": "B747TX154",
    "real_other_0193.jpg": "C888BK54",
    "real_other_0194.jpg": "O657XB154",
}

# 2. 2 tampered Dodge frames to delete
DELETE_TAMPERED = {
    "real_other_0174.jpg",
    "real_other_0176.jpg",
}


def update_label_class(lbl_path: Path, new_cls: int):
    if not lbl_path.exists():
        return
    with open(lbl_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    new_lines = []
    for ln in lines:
        parts = ln.split()
        if parts:
            parts[0] = str(new_cls)
            new_lines.append(" ".join(parts))
    with open(lbl_path, "w", encoding="utf-8") as f:
        f.write("\n".join(new_lines) + "\n")


def main():
    print("=" * 70)
    print("🚀 OmniPlate-RU: Reclassifying 10 Type 1A Squares & Deleting Tampered Frames")
    print("=" * 70)

    # 1. Read meta.csv
    with open(META_CSV, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f, delimiter=";"))
    fieldnames = list(reader[0].keys())

    # Delete tampered files from disk
    for fn in DELETE_TAMPERED:
        img_p = IMAGES_DIR / fn
        lbl_p = LABELS_DIR / f"{Path(fn).stem}.txt"
        if img_p.exists():
            img_p.unlink()
        if lbl_p.exists():
            lbl_p.unlink()
        print(f"[+] Deleted tampered file: {fn}")

    # Process meta.csv rows
    updated_rows = []
    for r in reader:
        base_fn = Path(r["image"]).name
        stem = Path(r["image"]).stem

        if base_fn in DELETE_TAMPERED:
            continue

        if base_fn in RECLASS_1A:
            r["plate_type"] = "type1a"
            r["plate_num"] = RECLASS_1A[base_fn]
            r["is_vehicle"] = "1"
            update_label_class(LABELS_DIR / f"{stem}.txt", 1)
            print(f"[+] Reclassified {base_fn} -> Type 1A ({RECLASS_1A[base_fn]})")

        updated_rows.append(r)

    with open(META_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(updated_rows)

    print(f"[+] Updated meta.csv written: {len(updated_rows)} rows")

    # 2. Standardize real filenames cleanly
    # Split real vs synth
    real_rows = [r for r in updated_rows if r.get("is_synthetic") == "0"]
    synth_rows = [r for r in updated_rows if r.get("is_synthetic") == "1"]

    by_type = defaultdict(list)
    for r in real_rows:
        by_type[r["plate_type"]].append(r)

    # Special handling for type1b: keep AH88977 at index 1!
    type1b_rows = by_type["type1b"]
    ah_rows = [r for r in type1b_rows if "AH88977" in r["plate_num"]]
    other_1b = [r for r in type1b_rows if "AH88977" not in r["plate_num"]]
    other_1b = sorted(other_1b, key=lambda x: x["image"])
    sorted_1b = ah_rows + other_1b

    file_mapping = {}  # old_rel -> new_rel
    stem_mapping = {}  # old_stem -> new_stem

    for ptype in ["type1", "type1a", "type1b", "other"]:
        if ptype == "type1b":
            items = sorted_1b
        else:
            items = sorted(by_type[ptype], key=lambda x: x["image"])

        for idx, r in enumerate(items, start=1):
            old_rel = r["image"].replace("\\", "/")
            new_fname = f"real_{ptype}_{idx:04d}.jpg"
            new_rel = f"images/real/{new_fname}"
            file_mapping[old_rel] = new_rel
            old_stem = Path(old_rel).stem
            new_stem = Path(new_fname).stem
            stem_mapping[old_stem] = new_stem

    print(f"\n[*] Standardizing {len(file_mapping)} real images:")
    for ptype in ["type1", "type1a", "type1b", "other"]:
        sub = [v for v in file_mapping.values() if f"real_{ptype}_" in v]
        print(f"    - {ptype:<8}: {len(sub):>4} files ({min(sub)} .. {max(sub)})")

    # Verify all source files exist
    for old_rel in file_mapping:
        img_p = DATASET_DIR / old_rel
        assert img_p.exists(), f"Missing image: {img_p}"
        old_stem = Path(old_rel).stem
        lbl_p = LABELS_DIR / f"{old_stem}.txt"
        assert lbl_p.exists(), f"Missing label: {lbl_p}"

    # 2-step atomic rename
    session_id = uuid.uuid4().hex[:8]
    tmp_renames = []

    for old_rel, new_rel in file_mapping.items():
        if old_rel == new_rel:
            continue
        old_img = DATASET_DIR / old_rel
        new_img = DATASET_DIR / new_rel
        tmp_img = IMAGES_DIR / f"__tmp_{session_id}_{Path(old_rel).stem}.jpg"
        old_img.rename(tmp_img)

        old_stem = Path(old_rel).stem
        new_stem = stem_mapping[old_stem]
        old_lbl = LABELS_DIR / f"{old_stem}.txt"
        new_lbl = LABELS_DIR / f"{new_stem}.txt"
        tmp_lbl = LABELS_DIR / f"__tmp_{session_id}_{old_stem}.txt"
        old_lbl.rename(tmp_lbl)
        tmp_renames.append((tmp_img, new_img, tmp_lbl, new_lbl))

    for tmp_img, new_img, tmp_lbl, new_lbl in tmp_renames:
        tmp_img.rename(new_img)
        tmp_lbl.rename(new_lbl)

    print(f"[+] All files successfully renamed on disk!")

    # Update meta.csv with new standardized names
    final_rows = []
    for r in updated_rows:
        old_rel = r["image"].replace("\\", "/")
        if old_rel in file_mapping:
            r["image"] = file_mapping[old_rel]
        final_rows.append(r)

    with open(META_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(final_rows)

    print(f"[+] Final meta.csv saved with {len(final_rows)} rows")
    print("=" * 70)
    print("🎉 Reclassification and Standardization Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
