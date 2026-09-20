import csv
import os
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(r"D:\AIProjects\VolgaIT")
META_PATH = PROJECT_ROOT / "dataset" / "meta.csv"
OUT_DIR = PROJECT_ROOT / "test_output" / "visual_audit_type1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# List of all suspects from user prompt
suspects_type1 = [
    "real_type1_0184", "real_type1_0190", "real_type1_0191", "real_type1_0192",
    "real_type1_0226", "real_type1_0227", "real_type1_0228", "real_type1_0310",
    "real_type1_0328", "real_type1_0329", "real_type1_0330", "real_type1_0332",
    "real_type1_0351", "real_type1_0354", "real_type1_0355", "real_type1_0360",
    "real_type1_0369", "real_type1_0370", "real_type1_0371", "real_type1_0379",
    "real_type1_0381", "real_type1_0388", "real_type1_0395", "real_type1_0396",
    "real_type1_0397", "real_type1_0398", "real_type1_0399", "real_type1_0400",
    "real_type1_0401", "real_type1_0402", "real_type1_0403", "real_type1_0404",
    "real_type1_0406", "real_type1_0407", "real_type1_0408", "real_type1_0409",
    "real_type1_0412", "real_type1_0419", "real_type1_0420", "real_type1_0421",
    "real_type1_0422", "real_type1_0423", "real_type1_0424", "real_type1_0425",
    "real_type1_0426", "real_type1_0427", "real_type1_0428", "real_type1_0429",
    "real_type1_0430", "real_type1_0432", "real_type1_0433", "real_type1_0434",
    "real_type1_0435", "real_type1_0436", "real_type1_0437", "real_type1_0438",
    "real_type1_0439", "real_type1_0441", "real_type1_0442", "real_type1_0443",
    "real_type1_0444", "real_type1_0445", "real_type1_0446", "real_type1_0447"
]

def add_r(start, end):
    for i in range(start, end + 1):
        suspects_type1.append(f"real_type1_{i:04d}")

add_r(448, 506)
add_r(509, 519)
suspects_type1.extend(["real_type1_0529", "real_type1_0530", "real_type1_0531", "real_type1_0534"])
add_r(538, 543)
suspects_type1.extend(["real_type1_0553", "real_type1_0556", "real_type1_0557", "real_type1_0573", "real_type1_0575", "real_type1_0576", "real_type1_0577"])
add_r(588, 593)
suspects_type1.extend(["real_type1_0862", "real_type1_0864"])

suspects_type1a = [
    "real_type1a_0003", "real_type1a_0005", "real_type1a_0010", "real_type1a_0012",
    "real_type1a_0013", "real_type1a_0223", "real_type1a_0225", "real_type1a_0226",
    "real_type1a_0235", "real_type1a_0237", "real_type1a_0253", "real_type1a_0275",
    "real_type1a_0276", "real_type1a_0278", "real_type1a_0307", "real_type1a_0310"
]

meta_dict = {}
with open(META_PATH, "r", encoding="utf-8") as f:
    for r in csv.DictReader(f, delimiter=";"):
        stem = Path(r["image"]).stem
        meta_dict[stem] = r

def make_sheet(items, prefix, sheet_idx):
    thumbs = []
    for stem in items:
        fn = f"{stem}.jpg"
        img_path = PROJECT_ROOT / "dataset" / "images" / "real" / fn
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        r = meta_dict.get(stem, {})
        plate = r.get("plate_num", "UNKNOWN")
        ptype = r.get("plate_type", "?")

        # Try to crop plate if bbox exists
        crop = None
        if "bbox" in r and r["bbox"]:
            try:
                bx, by, bw, bh = [int(x) for x in r["bbox"].split(",")]
                pad_x = int(bw * 0.2)
                pad_y = int(bh * 0.2)
                crop = img[max(0, by - pad_y):min(h, by + bh + pad_y), max(0, bx - pad_x):min(w, bx + bw + pad_x)]
            except Exception:
                crop = None

        thumb = np.zeros((180, 260, 3), dtype=np.uint8)

        # Draw full image mini preview on the left
        mini_full = cv2.resize(img, (110, 110))
        thumb[10:120, 10:120] = mini_full

        # Draw plate crop on the right
        if crop is not None and crop.size > 0:
            mini_crop = cv2.resize(crop, (120, 70))
            thumb[10:80, 130:250] = mini_crop

        # Text labels
        cv2.putText(thumb, stem, (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
        cv2.putText(thumb, f"{plate} [{ptype}]", (10, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 1)
        thumbs.append(thumb)

    if not thumbs:
        return None

    cols = 5
    rows = (len(thumbs) + cols - 1) // cols
    sheet = np.zeros((rows * 180, cols * 260, 3), dtype=np.uint8)

    for i, t in enumerate(thumbs):
        r_idx = i // cols
        c_idx = i % cols
        sheet[r_idx * 180:(r_idx + 1) * 180, c_idx * 260:(c_idx + 1) * 260] = t

    out_file = OUT_DIR / f"sheet_{prefix}_{sheet_idx:02d}.jpg"
    cv2.imwrite(str(out_file), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"Saved: {out_file} ({len(thumbs)} items)")
    return out_file

# Generate sheets in chunks of 25
chunk_size = 25
for i in range(0, len(suspects_type1), chunk_size):
    chunk = suspects_type1[i:i + chunk_size]
    make_sheet(chunk, "type1", (i // chunk_size) + 1)

make_sheet(suspects_type1a, "type1a", 1)
