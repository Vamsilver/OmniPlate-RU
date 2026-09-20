import json
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(r"D:\AIProjects\VolgaIT")
FRESH_DIR = PROJECT_ROOT / "test_output" / "pure_1a_fresh"
CROPS_DIR = FRESH_DIR / "crops"
METADATA_JSON = FRESH_DIR / "candidates_metadata.json"
OUT_SHEET = PROJECT_ROOT / "test_output" / "visual_audit_1a" / "sheet_fresh_21.jpg"

with open(METADATA_JSON, "r", encoding="utf-8") as f:
    candidates = json.load(f)

thumbs = []
for c in candidates:
    crop_path = CROPS_DIR / c["crop_fn"]
    if not crop_path.exists():
        continue
    img = cv2.imread(str(crop_path))
    if img is None:
        continue
    thumb = cv2.resize(img, (200, 150))
    cv2.putText(thumb, f"#{c['id']} {c['text']}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    cv2.putText(thumb, f"OCR:{c['ocr_conf']:.2f} AR:{c['ar']:.2f}", (5, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
    thumbs.append(thumb)

cols = 5
rows = (len(thumbs) + cols - 1) // cols
sheet = np.zeros((rows * 150, cols * 200, 3), dtype=np.uint8)

for i, t in enumerate(thumbs):
    r = i // cols
    c = i % cols
    sheet[r * 150:(r + 1) * 150, c * 200:(c + 1) * 200] = t

cv2.imwrite(str(OUT_SHEET), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
print(f"Saved sheet with {len(thumbs)} candidates to {OUT_SHEET}")
