import csv
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DATASET = ROOT / "dataset"
IMAGES = DATASET / "images" / "real"
LABELS = DATASET / "labels"
META = DATASET / "meta.csv"

with open(META, "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f, delimiter=";"))
    fieldnames = list(rows[0].keys())

moves = [
    {
        "old": "images/real/real_other_0027.jpg",
        "new": "images/real/real_type1_0899.jpg",
        "type": "type1",
        "text": "K050OE63",
        "cls": 0,
        "bbox": (0, 0, 467, 105),
        "quad": (0, 0, 467, 0, 467, 105, 0, 105),
        "w": 467, "h": 105
    },
    {
        "old": "images/real/real_other_0028.jpg",
        "new": "images/real/real_type1_0900.jpg",
        "type": "type1",
        "text": "B152MM142",
        "cls": 0,
        "bbox": (0, 186, 480, 108),
        "quad": (0, 186, 480, 186, 480, 294, 0, 294),
        "w": 480, "h": 480
    },
    {
        "old": "images/real/real_other_0029.jpg",
        "new": "images/real/real_type1b_0311.jpg",
        "type": "type1b",
        "text": "XX616457",
        "cls": 2,
        "bbox": (0, 143, 480, 194),
        "quad": (0, 143, 480, 143, 480, 337, 0, 337),
        "w": 480, "h": 480
    }
]

for m in moves:
    old_img = DATASET / m["old"]
    new_img = DATASET / m["new"]
    old_lbl = LABELS / f"{Path(m['old']).stem}.txt"
    new_lbl = LABELS / f"{Path(m['new']).stem}.txt"

    if old_img.exists():
        shutil.move(str(old_img), str(new_img))
        print(f"Moved {old_img.name} -> {new_img.name}")
    if old_lbl.exists():
        old_lbl.unlink()
        print(f"Removed old label {old_lbl.name}")

    bx, by, bw, bh = m["bbox"]
    w, h = m["w"], m["h"]
    xc = (bx + bw / 2.0) / float(w)
    yc = (by + bh / 2.0) / float(h)
    nw = bw / float(w)
    nh = bh / float(h)
    q = m["quad"]
    kpts = " ".join(f"{q[k]/float(w):.6f} {q[k+1]/float(h):.6f}" for k in range(0, 8, 2))
    new_lbl.write_text(f"{m['cls']} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f} {kpts}\n", encoding="utf-8")
    print(f"Created new label {new_lbl.name}")

    for r in rows:
        if r["image"] == m["old"]:
            r["image"] = m["new"]
            r["plate_type"] = m["type"]
            r["plate_num"] = m["text"]
            r["bbox"] = f"{bx},{by},{bw},{bh}"
            r["quad"] = ",".join(str(v) for v in q)
            r["is_vehicle"] = "1"

with open(META, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
    writer.writeheader()
    writer.writerows(rows)

print("Updated meta.csv successfully!")
