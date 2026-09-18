import csv
import json
import os
import re
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(r"D:\AIProjects\VolgaIT")
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.pipeline import OmniPlatePipeline

META_PATH = PROJECT_ROOT / "dataset" / "meta.csv"
BACKUP_PATH = PROJECT_ROOT / "dataset" / "meta.csv.bak_pure1a"
REAL_DIR = PROJECT_ROOT / "dataset" / "images" / "real"
CROPS_DIR = PROJECT_ROOT / "test_output" / "new_1a_verified" / "crops"
NEW_FULL_DIR = PROJECT_ROOT / "test_output" / "new_1a_verified"

PLATE_REGEX = re.compile(r"^[ABEKMHOPCTYX#][\d#]{3}[ABEKMHOPCTYX#]{2}[\d#]{2,3}$")

def evaluate_seam_ratio(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sob_y = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    norm_sob = cv2.resize(sob_y, (100, 100))
    p100 = np.mean(norm_sob, axis=1)
    mid_e = np.mean(p100[42:54])
    top_e = np.mean(p100[18:36])
    bot_e = np.mean(p100[60:78])
    return float((top_e + bot_e) / (2.0 * max(1e-3, mid_e)))

def main():
    print("=" * 70)
    print("OmniPlate-RU: Финальная интеграция 100% чистых Type 1A")
    print("=" * 70)

    # 1. Backup meta.csv
    if not BACKUP_PATH.exists():
        shutil.copyfile(META_PATH, BACKUP_PATH)
        print(f"[*] Создан бэкап: {BACKUP_PATH}")

    # 2. Read meta.csv
    with open(META_PATH, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")
        header = next(reader)
        rows = list(reader)

    print(f"[*] Всего строк в meta.csv до обновления: {len(rows)}")

    pipeline = OmniPlatePipeline(device="cuda")

    # 3. Identify and reclassify old 1-line traps among real_type1a
    print("[*] Аудит старых строк real_type1a...")
    old_pure_1a = 0
    old_reclassified = 0
    max_idx = 0

    for r in rows:
        norm_img = r[0].replace("\\", "/")
        m = re.search(r"real_type1a_(\d+)", norm_img)
        if m:
            idx = int(m.group(1))
            max_idx = max(max_idx, idx)

        if len(r) > 6 and r[2] == "type1a" and r[6] == "0":
            img_p = PROJECT_ROOT / "dataset" / r[0]
            im = cv2.imread(str(img_p))
            if im is None:
                continue

            bx, by, bw, bh = [int(x) for x in r[3].split(",")]
            quad = [float(x) for x in r[4].split(",")]
            ar = bw / max(1.0, bh)

            crop = im[max(0, by):min(im.shape[0], by+bh), max(0, bx):min(im.shape[1], bx+bw)]
            seam = evaluate_seam_ratio(crop) if crop.size > 0 else 0.0

            rect_1a = pipeline.rectifier.rectify(im, quad, plate_type="type1a")
            top_l, bot_l = pipeline.rectifier.split_type1a(rect_1a)
            stitched = pipeline.rectifier.stitch_type1a_horizontal(top_l, bot_l)
            txt_1a, conf_1a = pipeline.ocr.predict_single(stitched, plate_type="type1a")

            rect_1 = pipeline.rectifier.rectify(im, quad, plate_type="type1")
            txt_1, conf_1 = pipeline.ocr.predict_single(rect_1, plate_type="type1")[:2]

            # Condition for pure 2-row GOST 1A:
            # Physical AR [1.05..1.85], Seam >= 1.05, OCR 1A conf >= 0.55 and 1A > 1
            if 1.05 <= ar <= 1.85 and seam >= 1.05 and conf_1a >= 0.55 and conf_1a > conf_1:
                old_pure_1a += 1
            else:
                # Reclassify trap to type1
                r[2] = "type1"
                if PLATE_REGEX.match(txt_1) and conf_1 >= 0.60:
                    r[1] = txt_1
                old_reclassified += 1

    print(f"[*] Старых сохраненных чистых Type 1A: {old_pure_1a}")
    print(f"[*] Старых наклонных однострочников реклассифицировано в 'type1': {old_reclassified}")
    print(f"[*] Текущий максимальный индекс real_type1a: {max_idx}")

    # 4. Integrate 128 new verified pure Type 1A frames
    crops = sorted(list(CROPS_DIR.glob("*.jpg")))
    print(f"[*] Интеграция {len(crops)} новых проверенных кадров Type 1A...")

    new_rows = []
    current_idx = max_idx + 1

    for c_path in crops:
        stem = c_path.stem.replace("_crop", "")
        # Extract plate text from filename
        parts = stem.split("_")
        txt_plate = parts[3] if len(parts) >= 4 else parts[2]

        full_fn = f"{stem}.jpg"
        full_src = NEW_FULL_DIR / full_fn
        if not full_src.exists():
            continue

        im = cv2.imread(str(full_src))
        if im is None:
            continue
        h_im, w_im = im.shape[:2]

        # Detect precise quad and bbox on full image
        res = pipeline.detector.predict(source=im, imgsz=640, conf=0.10, device="cuda", verbose=False)
        parsed = pipeline._parse_results(res[0], w_im, h_im)

        best_det = None
        best_diff = 999.0
        for det in parsed:
            bx, by, bw, bh = det.bbox
            ar = bw / max(1.0, bh)
            if 1.05 <= ar <= 1.90:
                diff = abs(ar - 1.45)
                if diff < best_diff:
                    best_diff = diff
                    best_det = det

        if best_det is None and parsed:
            best_det = parsed[0]

        if best_det is not None:
            bx, by, bw, bh = best_det.bbox
            quad = [int(round(x)) for x in best_det.quad]
        else:
            bx, by, bw, bh = int(w_im * 0.35), int(h_im * 0.4), int(w_im * 0.3), int(h_im * 0.2)
            quad = [bx, by, bx+bw, by, bx+bw, by+bh, bx, by+bh]

        # Target filename in dataset
        target_fn = f"real_type1a_{current_idx:04d}.jpg"
        target_path = REAL_DIR / target_fn
        shutil.copyfile(full_src, target_path)

        bbox_str = f"{bx},{by},{bw},{bh}"
        quad_str = ",".join(str(x) for x in quad)
        rel_img = f"images/real/{target_fn}"
        source_tag = "https://www.drive2.ru/r/jdm"
        license_tag = "Drive2"
        conds = "day"

        row = [
            rel_img,
            txt_plate,
            "type1a",
            bbox_str,
            quad_str,
            "1",
            "0",
            source_tag,
            license_tag,
            conds
        ]
        new_rows.append(row)
        current_idx += 1

    all_rows = rows + new_rows

    with open(META_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(header)
        writer.writerows(all_rows)

    print(f"\n[+] Успешно интегрировано {len(new_rows)} новых чистых Type 1A!")
    print(f"[+] Всего строк в обновленном meta.csv: {len(all_rows)}")

if __name__ == "__main__":
    main()
