#!/usr/bin/env python3
"""
apply_1a_visual_corrections.py
Applies the visual audit corrections discovered during eye inspection of Type 1A:
1. Deletes real_type1a_0120.jpg (souvenir ad plate insert "BK 96 RUS ДЛЯ ИНОМАРОК")
2. Corrects 27 human transcription typos in meta.csv
3. Renumbers real_type1a_0001..0167 continuously
"""

import csv
import sys
import uuid
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

# Exact character corrections verified by multimodal vision
CORRECTIONS_1A = {
    "images/real/real_type1a_0025.jpg": "Y183MP142",
    "images/real/real_type1a_0029.jpg": "X444EE19",
    "images/real/real_type1a_0031.jpg": "C911CT22",
    "images/real/real_type1a_0033.jpg": "P054BX154",
    "images/real/real_type1a_0034.jpg": "E313YO27",
    "images/real/real_type1a_0041.jpg": "K331TB199",
    "images/real/real_type1a_0042.jpg": "K331TB199",
    "images/real/real_type1a_0048.jpg": "B455KC790",
    "images/real/real_type1a_0054.jpg": "O340OO186",
    "images/real/real_type1a_0057.jpg": "E530PO28",
    "images/real/real_type1a_0060.jpg": "Y423TK750",
    "images/real/real_type1a_0062.jpg": "C437HB761",
    "images/real/real_type1a_0063.jpg": "M991PX163",
    "images/real/real_type1a_0064.jpg": "M141CO134",
    "images/real/real_type1a_0065.jpg": "A209PP27",
    "images/real/real_type1a_0071.jpg": "T378YB154",
    "images/real/real_type1a_0079.jpg": "P402HE790",
    "images/real/real_type1a_0098.jpg": "T856HP138",
    "images/real/real_type1a_0102.jpg": "A947TK82",
    "images/real/real_type1a_0108.jpg": "P223BE125",
    "images/real/real_type1a_0112.jpg": "P920YO125",
    "images/real/real_type1a_0115.jpg": "T919XP125",
    "images/real/real_type1a_0116.jpg": "T328BE154",
    "images/real/real_type1a_0123.jpg": "K113HC34",
    "images/real/real_type1a_0124.jpg": "A155CA154",
    "images/real/real_type1a_0125.jpg": "M497PY193",
    "images/real/real_type1a_0127.jpg": "X337YO124",
}

DELETE_IMAGE = "images/real/real_type1a_0120.jpg"


def main():
    print("=" * 70)
    print("🚀 Applying Visual Corrections for Type 1A")
    print("=" * 70)

    # 1. Read meta.csv
    with open(META_CSV, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f, delimiter=";"))
    fieldnames = list(reader[0].keys())

    # 2. Delete the souvenir ad insert file
    del_img = DATASET_DIR / DELETE_IMAGE
    del_lbl = LABELS_DIR / f"{del_img.stem}.txt"
    if del_img.exists():
        del_img.unlink()
        print(f"[+] Deleted souvenir image: {del_img.name}")
    if del_lbl.exists():
        del_lbl.unlink()
        print(f"[+] Deleted souvenir label: {del_lbl.name}")

    # 3. Filter and correct rows in meta.csv
    updated_rows = []
    corrected_count = 0
    for r in reader:
        img_rel = r["image"].replace("\\", "/")
        if img_rel == DELETE_IMAGE:
            continue
        if img_rel in CORRECTIONS_1A:
            old_val = r["plate_num"]
            new_val = CORRECTIONS_1A[img_rel]
            r["plate_num"] = new_val
            corrected_count += 1
            print(f"    [CORRECTED] {img_rel}: {old_val} -> {new_val}")
        updated_rows.append(r)

    print(f"\n[*] Corrected {corrected_count}/27 entries in meta.csv")

    # 4. Renumber Type 1A files sequentially (1..167)
    type1a_rows = [r for r in updated_rows if r.get("is_synthetic") == "0" and r.get("plate_type") == "type1a"]
    type1a_rows.sort(key=lambda x: x["image"].replace("\\", "/"))
    print(f"[*] Type 1A rows after deletion: {len(type1a_rows)} (Quota >= 150)")

    file_mapping = {}
    stem_mapping = {}
    for idx, r in enumerate(type1a_rows, start=1):
        old_rel = r["image"].replace("\\", "/")
        new_rel = f"images/real/real_type1a_{idx:04d}.jpg"
        file_mapping[old_rel] = new_rel
        stem_mapping[Path(old_rel).stem] = Path(new_rel).stem

    # Two-step atomic rename
    session_id = uuid.uuid4().hex[:8]
    tmp_imgs = []
    tmp_lbls = []

    for idx, (old_rel, new_rel) in enumerate(file_mapping.items(), start=1):
        if old_rel == new_rel:
            continue
        old_img = DATASET_DIR / old_rel
        new_img = DATASET_DIR / new_rel
        tmp_img = IMAGES_DIR / f"__tmp1a_{session_id}_{idx:04d}.jpg"
        old_img.rename(tmp_img)
        tmp_imgs.append((tmp_img, new_img))

        old_stem = Path(old_rel).stem
        new_stem = stem_mapping[old_stem]
        old_lbl = LABELS_DIR / f"{old_stem}.txt"
        new_lbl = LABELS_DIR / f"{new_stem}.txt"
        tmp_lbl = LABELS_DIR / f"__tmp1a_{session_id}_{idx:04d}.txt"
        old_lbl.rename(tmp_lbl)
        tmp_lbls.append((tmp_lbl, new_lbl))

    for tmp_img, final_img in tmp_imgs:
        tmp_img.rename(final_img)
    for tmp_lbl, final_lbl in tmp_lbls:
        tmp_lbl.rename(final_lbl)

    print(f"[*] Renamed {len(tmp_imgs)} files to ensure contiguous 0001..{len(type1a_rows):04d} indexing.")

    # 5. Update meta.csv image paths
    final_rows = []
    for r in updated_rows:
        img_rel = r["image"].replace("\\", "/")
        if img_rel in file_mapping:
            r["image"] = file_mapping[img_rel]
        final_rows.append(r)

    with open(META_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(final_rows)

    print(f"[+] Successfully wrote {len(final_rows)} rows to meta.csv!")
    print("=" * 70)


if __name__ == "__main__":
    main()
