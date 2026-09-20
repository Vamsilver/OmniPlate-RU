import csv
import shutil
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parent.parent.parent
DATASET = ROOT / "dataset"
IMAGES = DATASET / "images" / "real"
LABELS = DATASET / "labels"
META = DATASET / "meta.csv"

# 11 files to migrate
items = [
    ("real_other_0069.jpg", "O965HC154", (17, 18, 290, 130), [20, 20, 305, 35, 295, 145, 18, 130]),
    ("real_other_0079.jpg", "X014PM78", (323, 638, 388, 77), [323, 638, 711, 638, 711, 715, 323, 715]),
    ("real_other_0190.jpg", "B073CE750", (322, 666, 56, 17), [322, 666, 378, 666, 378, 683, 322, 683]),
    ("real_other_0192.jpg", "M153BK977", (566, 431, 56, 18), [566, 431, 622, 431, 622, 449, 566, 449]),
    ("real_other_0194.jpg", "M476OX777", (416, 686, 38, 12), [416, 686, 454, 686, 454, 698, 416, 698]),
    ("real_other_0203.jpg", "C###AA###", (10, 2, 342, 76), [10, 2, 352, 2, 352, 78, 10, 78]),
    ("real_other_0205.jpg", "A###AA###", (0, 7, 188, 46), [0, 7, 188, 7, 188, 53, 0, 53]),
    ("real_other_0206.jpg", "X###XX###", (8, 5, 125, 37), [8, 5, 133, 5, 133, 42, 8, 42]),
    ("real_other_0213.jpg", "K666EX177", (11, 80, 322, 78), [11, 80, 333, 80, 333, 158, 11, 158]),
    ("real_other_0255.jpg", "A046CX77", (394, 222, 98, 98), [394, 222, 492, 240, 475, 320, 394, 290]),
    ("real_other_0284.jpg", "A001AA777", (109, 712, 115, 70), [109, 742, 220, 712, 224, 755, 115, 782]),
]

with open(META, "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f, delimiter=";"))
    fieldnames = list(rows[0].keys())

existing_t1 = list(IMAGES.glob("real_type1_*.jpg"))
max_idx = max([int(p.stem.split("_")[-1]) for p in existing_t1]) if existing_t1 else 900
next_idx = max_idx + 1

for old_name, text, bbox, quad in items:
    new_stem = f"real_type1_{next_idx:04d}"
    next_idx += 1
    new_name = f"{new_stem}.jpg"
    
    old_img = IMAGES / old_name
    new_img = IMAGES / new_name
    old_lbl = LABELS / f"{Path(old_name).stem}.txt"
    new_lbl = LABELS / f"{new_stem}.txt"
    
    if not old_img.exists():
        print(f"Skipping {old_name}, not on disk")
        continue
        
    im = cv2.imread(str(old_img))
    h, w = im.shape[:2]
    
    bx, by, bw, bh = bbox
    xc = (bx + bw / 2.0) / float(w)
    yc = (by + bh / 2.0) / float(h)
    nw = bw / float(w)
    nh = bh / float(h)
    kpts = " ".join(f"{quad[i]/float(w):.6f} {quad[i+1]/float(h):.6f}" for i in range(0, 8, 2))
    new_lbl.write_text(f"0 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f} {kpts}\n", encoding="utf-8")
    
    shutil.move(str(old_img), str(new_img))
    if old_lbl.exists():
        old_lbl.unlink()
        
    for r in rows:
        if r["image"] == f"images/real/{old_name}":
            r["image"] = f"images/real/{new_name}"
            r["plate_type"] = "type1"
            r["plate_num"] = text
            r["bbox"] = f"{bx},{by},{bw},{bh}"
            r["quad"] = ",".join(str(v) for v in quad)
            r["is_vehicle"] = "1"
            
    print(f"Moved {old_name} -> {new_name} ({text})")

with open(META, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
    writer.writeheader()
    writer.writerows(rows)

print("Updated meta.csv!")
