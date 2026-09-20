import csv
from pathlib import Path
import cv2

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.pipeline.pipeline import OmniPlatePipeline

with open(PROJECT_ROOT / "dataset" / "meta.csv", "r", encoding="utf-8") as f:
    reader = list(csv.DictReader(f, delimiter=";"))

real_other_rows = {r["image"]: r for r in reader if "real_other_" in r["image"]}

warned_imgs = [
    "images/real/real_other_0043.jpg",
    "images/real/real_other_0047.jpg",
    "images/real/real_other_0048.jpg",
    "images/real/real_other_0055.jpg",
    "images/real/real_other_0070.jpg",
    "images/real/real_other_0089.jpg",
    "images/real/real_other_0094.jpg",
    "images/real/real_other_0120.jpg"
]

pipeline = OmniPlatePipeline(device="cuda")

print("=== WARNED IMAGES BBOX CROP OCR ===")
for img_rel in warned_imgs:
    r = real_other_rows[img_rel]
    img_path = PROJECT_ROOT / "dataset" / img_rel
    img = cv2.imread(str(img_path))
    bx, by, bw, bh = [int(v) for v in r["bbox"].split(",")]
    crop = img[by:by+bh, bx:bx+bw]
    if crop.size == 0:
        print(f"{img_rel} | empty crop!")
        continue
    t1_txt, t1_conf = pipeline.ocr.predict_single(crop, "type1")
    t1b_txt, t1b_conf = pipeline.ocr.predict_single(crop, "type1b")
    meta_num = r["plate_num"]
    print(f"{img_rel} | shape={img.shape} | meta_bbox=({bx},{by},{bw},{bh}) | meta_num={meta_num} | OCR_t1={t1_txt}({t1_conf:.2f}) | OCR_t1b={t1b_txt}({t1b_conf:.2f})")
