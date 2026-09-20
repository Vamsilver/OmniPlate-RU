#!/usr/bin/env python3
"""
OmniPlate-RU: Comprehensive Sanitation of Type 1 Frames and Renaming of Type 1A.
Applies visual audit findings from test_output/visual_audit_type1/ (sheets 01-07 and type1a_01):
1. Deletes 24 inappropriate/garbage frames + 1 ghost entry.
2. Reclassifies 118 non-standard signs (transits, motorcycles, trailers, Soviet, red, blue, military) to 'other' (YOLO class 3).
3. Reclassifies confirmed yellow taxi plates to 'type1b' (YOLO class 2).
4. Reclassifies square passenger car plates to 'type1a' (YOLO class 1).
5. Physically renames the 16 audited Type 1A candidates according to their true GOST class
   (real_type1_... / real_other_...) with full synchronization of meta.csv and labels/.
"""

import csv
import os
import shutil
import sys
from pathlib import Path

# Force UTF-8 on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
DATASET_DIR = PROJECT_ROOT / "dataset"
IMAGES_DIR = DATASET_DIR / "images" / "real"
LABELS_DIR = DATASET_DIR / "labels"
META_CSV = DATASET_DIR / "meta.csv"
BACKUP_CSV = DATASET_DIR / "meta.csv.bak_sanitation_type1"

# 1. 24 Deletion candidates from visual audit + ghost real_type1_0580
DELETIONS = {
    # Human hands holding fake plate
    "real_type1_0419.jpg",
    "real_type1_0420.jpg",
    # Augmented crops
    "real_type1_0226.jpg",
    "real_type1_0227.jpg",
    "real_type1_0228.jpg",
    "real_type1_0402.jpg",
    # Souvenir SAMPLE TEXT
    "real_type1_0404.jpg",
    "real_type1_0541.jpg",
    "real_type1_0542.jpg",
    "real_type1_0543.jpg",
    # Flag crops & cut-offs
    "real_type1_0467.jpg",
    "real_type1_0468.jpg",
    "real_type1_0469.jpg",
    "real_type1_0530.jpg",
    "real_type1_0531.jpg",
    "real_type1_0539.jpg",
    "real_type1_0540.jpg",
    "real_type1_0483.jpg",
    "real_type1_0484.jpg",
    "real_type1_0485.jpg",
    "real_type1_0534.jpg",
    "real_type1_0538.jpg",
    "real_type1_0573.jpg",
    # Chinese plate
    "real_type1_0864.jpg",
    # Ghost row / orphaned label with missing image
    "real_type1_0580.jpg",
}

# 2. Confirmed Yellow Taxi plates -> Type 1B (YOLO class 2)
# With calibrated plate numbers matching Russian Type 1B GOST format LL DDD RR
RECLASS_TYPE1B = {
    "real_type1_0516.jpg": "AB86577",
    "real_type1_0517.jpg": "AB86577",
    "real_type1_0518.jpg": "AB8657#",
    "real_type1_0556.jpg": "AP65478",
    "real_type1_0557.jpg": "AX57555",
}

# 3. Confirmed Square Passenger plates -> Type 1A (YOLO class 1)
RECLASS_TYPE1A = {
    "real_type1_0328.jpg": "P084BE50",
    "real_type1_0369.jpg": "H473XC777",
    "real_type1_0370.jpg": "C161YP190",
    "real_type1_0371.jpg": "B411XX152",
    "real_type1_0379.jpg": "O968TC102",
}

# 4. 16 Audited Type 1A files to physically rename
RENAME_TYPE1A_TO_TYPE1 = [
    "real_type1a_0003.jpg",
    "real_type1a_0005.jpg",
    "real_type1a_0223.jpg",
]

RENAME_TYPE1A_TO_OTHER = [
    "real_type1a_0010.jpg",
    "real_type1a_0012.jpg",
    "real_type1a_0013.jpg",
    "real_type1a_0225.jpg",
    "real_type1a_0226.jpg",
    "real_type1a_0235.jpg",
    "real_type1a_0237.jpg",
    "real_type1a_0253.jpg",
    "real_type1a_0275.jpg",
    "real_type1a_0276.jpg",
    "real_type1a_0278.jpg",
    "real_type1a_0307.jpg",
    "real_type1a_0310.jpg",
]


def update_label_class(label_path: Path, new_class: int):
    """Updates the YOLO class ID (first integer) in the label file."""
    if not label_path.exists():
        return
    with open(label_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    new_lines = []
    for ln in lines:
        parts = ln.split()
        if parts:
            parts[0] = str(new_class)
            new_lines.append(" ".join(parts))

    with open(label_path, "w", encoding="utf-8") as f:
        f.write("\n".join(new_lines) + "\n")


def make_tight_qbox(quad_str: str) -> tuple:
    """Computes tight bounding box and canonical quad (clockwise starting at top-left)."""
    pts = [int(v.strip()) for v in quad_str.split(",")]
    if len(pts) != 8:
        return None, quad_str
    pts_pairs = [(pts[i * 2], pts[i * 2 + 1]) for i in range(4)]
    first_idx = min(range(4), key=lambda k: pts_pairs[k][0] + pts_pairs[k][1])
    if first_idx != 0:
        pts_pairs = pts_pairs[first_idx:] + pts_pairs[:first_idx]
    canonical_quad = ",".join(f"{x},{y}" for x, y in pts_pairs)

    qx = [p[0] for p in pts_pairs]
    qy = [p[1] for p in pts_pairs]
    bx = min(qx)
    by = min(qy)
    bw = max(qx) - min(qx)
    bh = max(qy) - min(qy)
    return f"{bx},{by},{bw},{bh}", canonical_quad


def main():
    print("=" * 70)
    print("🚀 OmniPlate-RU: Executing Comprehensive Type 1 Sanitation & 1A Renaming")
    print("=" * 70)

    # 1. Backup meta.csv
    if not BACKUP_CSV.exists():
        shutil.copyfile(META_CSV, BACKUP_CSV)
        print(f"[*] Backup created: {BACKUP_CSV.name}")

    # 2. Read meta.csv
    with open(META_CSV, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f, delimiter=";"))
    fieldnames = list(reader[0].keys())
    print(f"[*] Read {len(reader)} rows from meta.csv")

    # Determine 118 reclass to 'other' candidates from the 152 suspects
    from scripts.research.generate_suspects_sheets import suspects_type1
    existing_152 = [s for s in suspects_type1 if (IMAGES_DIR / f"{s}.jpg").exists()]

    del_stems = {Path(f).stem for f in DELETIONS}
    t1b_stems = {Path(f).stem for f in RECLASS_TYPE1B}
    t1a_stems = {Path(f).stem for f in RECLASS_TYPE1A}

    reclass_other_stems = set()
    for s in existing_152:
        if s not in del_stems and s not in t1b_stems and s not in t1a_stems:
            reclass_other_stems.add(s)

    print(f"[*] Distribution of audited 152 suspects:")
    print(f"    - Deletions:          {len(DELETIONS)} (24 real + 1 ghost)")
    print(f"    - Reclass to type1b:  {len(RECLASS_TYPE1B)}")
    print(f"    - Reclass to type1a:  {len(RECLASS_TYPE1A)}")
    print(f"    - Reclass to other:   {len(reclass_other_stems)}")

    assert len(reclass_other_stems) == 118, f"Expected 118 other candidates, got {len(reclass_other_stems)}"

    # 3. Determine new filenames for physical renaming
    # For type1: current max is 913 -> next starts at 914
    max_t1_idx = 0
    for f in os.listdir(IMAGES_DIR):
        if f.startswith("real_type1_") and f.endswith(".jpg"):
            num = int(f.replace("real_type1_", "").replace(".jpg", ""))
            if num > max_t1_idx:
                max_t1_idx = num

    # For other: current max is 287 -> next starts at 288
    max_other_idx = 0
    for f in os.listdir(IMAGES_DIR):
        if f.startswith("real_other_") and f.endswith(".jpg"):
            num = int(f.replace("real_other_", "").replace(".jpg", ""))
            if num > max_other_idx:
                max_other_idx = num

    print(f"[*] Current max index: real_type1_{max_t1_idx:04d}, real_other_{max_other_idx:04d}")

    rename_map = {}  # old_fn -> new_fn
    cur_t1 = max_t1_idx
    for old_fn in RENAME_TYPE1A_TO_TYPE1:
        cur_t1 += 1
        rename_map[old_fn] = f"real_type1_{cur_t1:04d}.jpg"

    cur_other = max_other_idx
    for old_fn in RENAME_TYPE1A_TO_OTHER:
        cur_other += 1
        rename_map[old_fn] = f"real_other_{cur_other:04d}.jpg"

    print(f"[*] Physical renaming plan ({len(rename_map)} files):")
    for old_fn, new_fn in rename_map.items():
        print(f"    - {old_fn} -> {new_fn}")

    # 4. Perform physical deletions on disk
    deleted_images_count = 0
    deleted_labels_count = 0
    for fn in DELETIONS:
        img_p = IMAGES_DIR / fn
        if img_p.exists():
            img_p.unlink()
            deleted_images_count += 1
        lbl_p = LABELS_DIR / f"{Path(fn).stem}.txt"
        if lbl_p.exists():
            lbl_p.unlink()
            deleted_labels_count += 1

    print(f"\n[+] Deleted from disk: {deleted_images_count} images, {deleted_labels_count} labels")

    # 5. Perform physical renaming on disk
    for old_fn, new_fn in rename_map.items():
        src_img = IMAGES_DIR / old_fn
        dst_img = IMAGES_DIR / new_fn
        if src_img.exists():
            src_img.rename(dst_img)

        old_stem = Path(old_fn).stem
        new_stem = Path(new_fn).stem
        src_lbl = LABELS_DIR / f"{old_stem}.txt"
        dst_lbl = LABELS_DIR / f"{new_stem}.txt"
        if src_lbl.exists():
            src_lbl.rename(dst_lbl)

    print(f"[+] Physically renamed {len(rename_map)} images and labels on disk")

    # 6. Process meta.csv rows
    updated_rows = []
    reclassed_t1b_count = 0
    reclassed_t1a_count = 0
    reclassed_other_count = 0
    renamed_meta_count = 0

    for r in reader:
        img_rel = r["image"].replace("\\", "/")
        base_fn = Path(img_rel).name
        stem = Path(img_rel).stem

        # Case A: Deleted file -> omit from meta.csv
        if base_fn in DELETIONS or stem in del_stems:
            continue

        # Case B: Renamed Type 1A file
        if base_fn in rename_map:
            renamed_meta_count += 1
            new_fn = rename_map[base_fn]
            new_stem = Path(new_fn).stem
            r["image"] = f"images/real/{new_fn}"
            if new_fn.startswith("real_type1_"):
                r["plate_type"] = "type1"
                update_label_class(LABELS_DIR / f"{new_stem}.txt", 0)
            else:
                r["plate_type"] = "other"
                r["plate_num"] = ""
                update_label_class(LABELS_DIR / f"{new_stem}.txt", 3)
            updated_rows.append(r)
            continue

        # Case C: Reclassify to Type 1B (taxi)
        if base_fn in RECLASS_TYPE1B:
            reclassed_t1b_count += 1
            r["plate_type"] = "type1b"
            r["plate_num"] = RECLASS_TYPE1B[base_fn]
            update_label_class(LABELS_DIR / f"{stem}.txt", 2)
            updated_rows.append(r)
            continue

        # Case D: Reclassify to Type 1A (square passenger)
        if base_fn in RECLASS_TYPE1A:
            reclassed_t1a_count += 1
            r["plate_type"] = "type1a"
            r["plate_num"] = RECLASS_TYPE1A[base_fn]
            update_label_class(LABELS_DIR / f"{stem}.txt", 1)
            updated_rows.append(r)
            continue

        # Case E: Reclassify to other
        if stem in reclass_other_stems:
            reclassed_other_count += 1
            r["plate_type"] = "other"
            r["plate_num"] = ""
            r["is_vehicle"] = "1"
            if r.get("quad"):
                tight_bbox, canonical_quad = make_tight_qbox(r["quad"])
                if tight_bbox:
                    r["bbox"] = tight_bbox
                    r["quad"] = canonical_quad
            update_label_class(LABELS_DIR / f"{stem}.txt", 3)
            updated_rows.append(r)
            continue

        # Default: keep unmodified
        updated_rows.append(r)

    print(f"\n[*] Processing Summary:")
    print(f"    - Renamed in meta:      {renamed_meta_count}")
    print(f"    - Reclassified to 1B:   {reclassed_t1b_count}")
    print(f"    - Reclassified to 1A:   {reclassed_t1a_count}")
    print(f"    - Reclassified to other:{reclassed_other_count}")
    print(f"    - Final rows in meta:   {len(updated_rows)} (was {len(reader)})")

    # 7. Write updated meta.csv
    with open(META_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(updated_rows)

    print(f"[+] Successfully wrote {len(updated_rows)} rows to {META_CSV.name}")

    # 8. Post-sanitation verification
    print("\n🔍 Verifying dataset file integrity...")
    missing_images = []
    missing_labels = []
    for r in updated_rows:
        img_p = DATASET_DIR / r["image"]
        if not img_p.exists():
            missing_images.append(str(img_p))
        lbl_p = LABELS_DIR / f"{img_p.stem}.txt"
        if not lbl_p.exists():
            missing_labels.append(str(lbl_p))

    assert len(missing_images) == 0, f"Missing images: {missing_images[:5]}"
    assert len(missing_labels) == 0, f"Missing labels: {missing_labels[:5]}"
    print(f"✅ Verified 100% of {len(updated_rows)} image and label pairs on disk!")
    print("=" * 70)
    print("🎉 Sanitation Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
