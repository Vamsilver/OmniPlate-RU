import csv
from pathlib import Path
import cv2

root = Path("D:/AIProjects/VolgaIT")
img_dir = root / "dataset/images/real"
label_dir = root / "dataset/labels"
meta_path = root / "dataset/meta.csv"

with open(meta_path, "r", encoding="utf-8") as f:
    meta_rows = {Path(r["image"]).name: r for r in csv.DictReader(f, delimiter=";")}

other_imgs = sorted(list(img_dir.glob("real_other_*.jpg")))

print(f"Total other images: {len(other_imgs)}")

# Let's inspect each sheet
for s in range(10):
    start = s * 25
    end = min(start + 25, len(other_imgs))
    chunk = other_imgs[start:end]
    print(f"\n=================== SHEET {s+1:02d} (#{start+1:03d} - #{end:03d}) ===================")
    for idx, p in enumerate(chunk, start=start+1):
        r = meta_rows.get(p.name, {})
        lbl_p = label_dir / f"{p.stem}.txt"
        lbl = lbl_p.read_text().strip() if lbl_p.exists() else ""
        src = r.get("source", "")
        # truncate source
        short_src = src.split("/")[-1] if "://" in src else src
        if len(short_src) > 40:
            short_src = short_src[:37] + "..."
        print(f"#{idx:03d} {p.name} | bbox={r.get('bbox','')} | src={short_src}")
