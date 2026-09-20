import csv
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_DIR = ROOT / "dataset"
IMAGES_DIR = DATASET_DIR / "images" / "real"
LABELS_DIR = DATASET_DIR / "labels"
META_CSV = DATASET_DIR / "meta.csv"
BACKUP_CSV = DATASET_DIR / "meta.csv.bak_audit1b"

# 1. Text corrections
CORRECTIONS = {
    'real_type1b_0067.jpg': 'EP22277',
    'real_type1b_0068.jpg': 'EP22277',
    'real_type1b_0073.jpg': 'EO95277',
    'real_type1b_0081.jpg': 'EP81477',
    'real_type1b_0084.jpg': 'BC43777',
    'real_type1b_0091.jpg': 'EO59577',
    'real_type1b_0096.jpg': 'OK28377',
    'real_type1b_0099.jpg': 'BC70377',
    'real_type1b_0119.jpg': 'BA56277',
    'real_type1b_0123.jpg': 'BA56277',
    'real_type1b_0126.jpg': 'BK21677',
    'real_type1b_0139.jpg': 'BC67077',
    'real_type1b_0141.jpg': 'EO82777',
    'real_type1b_0167.jpg': 'EP11077',
    'real_type1b_0181.jpg': 'AO94699',
    'real_type1b_0183.jpg': 'XM26377',
    'real_type1b_0191.jpg': 'EP24377',
    'real_type1b_0198.jpg': 'AB96677',
    'real_type1b_0208.jpg': 'CE98277',
    'real_type1b_0213.jpg': 'YK84877',
    'real_type1b_0214.jpg': 'YH81577',
    'real_type1b_0232.jpg': 'BY89577',
    'real_type1b_0235.jpg': 'EE81277',
    'real_type1b_0236.jpg': 'EA81677',
    'real_type1b_0237.jpg': 'EE81077',
    'real_type1b_0251.jpg': 'EO50577',
    'real_type1b_0252.jpg': 'EO50277',
    'real_type1b_0266.jpg': 'EO66277',
    'real_type1b_0267.jpg': 'EO66277',
    'real_type1b_0280.jpg': 'EB68977',
    'real_type1b_0284.jpg': 'EB74977',
    'real_type1b_0286.jpg': 'EK18277',
    'real_type1b_0300.jpg': 'EP71877',
    'real_type1b_0306.jpg': 'EO67477',
    'real_type1b_0307.jpg': 'MT06677',
    'real_type1b_0308.jpg': 'XX96777',
    'real_type1b_0309.jpg': 'EM95177',
    'real_type1b_0310.jpg': 'EP03277',
}

def main():
    print("[*] Backing up meta.csv -> meta.csv.bak_audit1b")
    shutil.copyfile(META_CSV, BACKUP_CSV)

    with open(META_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        fieldnames = reader.fieldnames
        rows = list(reader)

    print(f"[*] Total rows before: {len(rows)}")

    # 1. Apply corrections to rows
    corrected_count = 0
    for r in rows:
        img_name = Path(r["image"]).name
        if img_name in CORRECTIONS:
            old_p = r["plate_num"]
            new_p = CORRECTIONS[img_name]
            r["plate_num"] = new_p
            corrected_count += 1
            print(f"  [FIX] {img_name}: {old_p} -> {new_p}")

    print(f"[*] Corrected {corrected_count} labels in meta.csv")

    # 2. Delete real_type1b_0302.jpg
    del_img = IMAGES_DIR / "real_type1b_0302.jpg"
    del_lbl = LABELS_DIR / "real_type1b_0302.txt"
    if del_img.exists():
        del_img.unlink()
        print(f"[*] Deleted image {del_img.name}")
    if del_lbl.exists():
        del_lbl.unlink()
        print(f"[*] Deleted label {del_lbl.name}")

    # Remove from rows
    rows = [r for r in rows if Path(r["image"]).name != "real_type1b_0302.jpg"]
    print(f"[*] Total rows after deleting 0302: {len(rows)}")

    # 3. Renumber 0303..0312 -> 0302..0311
    # Do disk rename in ascending order 303..312
    for old_idx in range(303, 313):
        new_idx = old_idx - 1
        old_stem = f"real_type1b_{old_idx:04d}"
        new_stem = f"real_type1b_{new_idx:04d}"

        old_img = IMAGES_DIR / f"{old_stem}.jpg"
        new_img = IMAGES_DIR / f"{new_stem}.jpg"
        old_lbl = LABELS_DIR / f"{old_stem}.txt"
        new_lbl = LABELS_DIR / f"{new_stem}.txt"

        if old_img.exists():
            old_img.rename(new_img)
        if old_lbl.exists():
            old_lbl.rename(new_lbl)

        # Update row image path
        old_img_rel = f"images/real/{old_stem}.jpg"
        new_img_rel = f"images/real/{new_stem}.jpg"
        for r in rows:
            if r["image"].replace("\\", "/") == old_img_rel:
                r["image"] = f"images\\real\\{new_stem}.jpg"
                print(f"  [RENAME] {old_stem} -> {new_stem} (plate_num: {r['plate_num']})")

    # Write updated meta.csv
    with open(META_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)

    print(f"[*] Successfully wrote updated meta.csv ({len(rows)} rows)")

if __name__ == "__main__":
    main()
