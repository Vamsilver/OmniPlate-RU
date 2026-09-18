#!/usr/bin/env python3
"""
OmniPlate-RU — Integration of Verified True 2-Row Type 1A Plates.

1. Audits and reclassifies 95 false 1-row frames in dataset/meta.csv to 'type1'.
2. Extracts and integrates 91 clean 2-row candidates from russian_car_plates.v2i.yolov8.zip.
3. Extracts and integrates 20 pristine 2-row candidates from train-00000-of-00001.parquet.
4. Applies FaceBlurrer (YuNet ONNX) to all new images for 100% privacy compliance.
5. Emits strict BBox and 8-point Quad coordinates with continuous real_type1a_XXXX indexing.
"""

import csv
import json
import os
import re
import shutil
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

PROJECT_ROOT = Path(r"D:\AIProjects\VolgaIT")
sys.path.insert(0, str(PROJECT_ROOT))

from dataset.privacy.face_blur import FaceBlurrer
from src.pipeline.ocr import PlateOCR
from src.pipeline.rectifier import PlateRectifier

TOP_REGEX = re.compile(r"^[ABEKMHOPCTYX][0-9]{3}$")
BOT_REGEX = re.compile(r"^[ABEKMHOPCTYX]{2}[0-9]{2,3}$")
NAMES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'E', 'H', 'K', 'M', 'O', 'P', 'T', 'X', 'Y']

REAL_DIR = PROJECT_ROOT / "dataset" / "images" / "real"
META_PATH = PROJECT_ROOT / "dataset" / "meta.csv"
BACKUP_PATH = PROJECT_ROOT / "dataset" / "meta.csv.bak"

def integrate():
    print("=" * 75)
    print("OmniPlate-RU: Финальная интеграция истинных двухстрочных Type 1A")
    print("=" * 75)

    # 1. Backup meta.csv
    if META_PATH.exists() and not BACKUP_PATH.exists():
        shutil.copyfile(META_PATH, BACKUP_PATH)
        print(f"[*] Сделан бэкап: {BACKUP_PATH}")

    # 2. Load audit decisions for existing type1a
    audit_json = PROJECT_ROOT / "test_output" / "type1a_audit_decisions.json"
    with open(audit_json, "r", encoding="utf-8") as f:
        records = json.load(f)
    false_rel_paths = set(r["rel_path"].replace("\\", "/") for r in records if r["is_false_1a"])
    print(f"[*] Найдено {len(false_rel_paths)} ложных кадров Type 1A для реклассификации в 'type1'.")

    # 3. Read meta.csv and apply reclassification
    with open(META_PATH, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")
        header = next(reader)
        all_rows = list(reader)

    reclassified_count = 0
    max_1a_idx = 0

    for r in all_rows:
        img_norm = r[0].replace("\\", "/")
        if "real_type1a_" in img_norm:
            try:
                idx = int(img_norm.split("real_type1a_")[-1].split(".")[0])
                if idx > max_1a_idx:
                    max_1a_idx = idx
            except Exception:
                pass

        if img_norm in false_rel_paths and r[2] == "type1a":
            r[2] = "type1"
            reclassified_count += 1

    print(f"[*] Успешно реклассифицировано строк в 'type1': {reclassified_count}")
    print(f"[*] Текущий максимальный индекс real_type1a: {max_1a_idx}")

    blurrer = FaceBlurrer()
    rectifier = PlateRectifier()
    ocr = PlateOCR(model_path="models/ocr_lprnet_best.pt", device="cuda")

    new_rows = []
    current_idx = max_1a_idx + 1

    # 4. Integrate from russian_car_plates.v2i.yolov8.zip
    z_path = r"C:\Users\Vamsi\Downloads\russian_car_plates.v2i.yolov8.zip"
    z = zipfile.ZipFile(z_path)
    lbl_files = [f for f in z.namelist() if f.endswith('.txt') and not f.startswith('README')]

    print("\n[*] [1/2] Интеграция кандидатов из russian_car_plates.v2i.yolov8.zip...")
    zip_added = 0
    for lf in lbl_files:
        if "bmp" in lf:
            continue
        content = z.read(lf).decode('utf-8').strip()
        if not content:
            continue
        chars = []
        for l in content.splitlines():
            p = l.split()
            if len(p) == 5:
                chars.append((int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])))
        if len(chars) not in (8, 9):
            continue
        chars_by_y = sorted(chars, key=lambda c: c[2])
        top = sorted(chars_by_y[:4], key=lambda c: c[1])
        bot = sorted(chars_by_y[4:], key=lambda c: c[1])
        t_str = ''.join(NAMES[c[0]] for c in top)
        b_str = ''.join(NAMES[c[0]] for c in bot)
        if TOP_REGEX.match(t_str) and BOT_REGEX.match(b_str):
            plate_num = t_str + b_str
            if len(plate_num) == 9 and plate_num[6] not in {'1', '2', '7'}:
                continue
            im_p = lf.replace('/labels/', '/images/').replace('.txt', '.jpg')
            if im_p not in z.namelist():
                continue

            im_bytes = z.read(im_p)
            im = cv2.imdecode(np.asarray(bytearray(im_bytes), dtype=np.uint8), cv2.IMREAD_COLOR)
            if im is None:
                continue
            h_img, w_img = im.shape[:2]

            # Compute bounding box
            x1_all = min(c[1] - c[3]/2.0 for c in chars) * w_img
            y1_all = min(c[2] - c[4]/2.0 for c in chars) * h_img
            x2_all = max(c[1] + c[3]/2.0 for c in chars) * w_img
            y2_all = max(c[2] + c[4]/2.0 for c in chars) * h_img

            pw = x2_all - x1_all
            ph = y2_all - y1_all
            pad_x = pw * 0.05
            pad_y = ph * 0.07

            bx1 = max(0, int(round(x1_all - pad_x)))
            by1 = max(0, int(round(y1_all - pad_y)))
            bx2 = min(w_img, int(round(x2_all + pad_x)))
            by2 = min(h_img, int(round(y2_all + pad_y)))
            bw = max(1, bx2 - bx1)
            bh = max(1, by2 - by1)

            bbox_str = f"{bx1},{by1},{bw},{bh}"
            quad_str = f"{bx1},{by1},{bx2},{by1},{bx2},{by2},{bx1},{by2}"

            # De-identify faces
            blurred, _ = blurrer.process_image(im)

            out_fn = f"real_type1a_{current_idx:04d}.jpg"
            out_path = REAL_DIR / out_fn
            cv2.imwrite(str(out_path), blurred, [cv2.IMWRITE_JPEG_QUALITY, 94])

            # Determine conditions
            conds = "day"
            if bw / float(bh) < 1.4 or bw / float(bh) > 1.9:
                conds += ",angle"

            new_rows.append([
                f"images/real/{out_fn}",
                plate_num,
                "type1a",
                bbox_str,
                quad_str,
                "1",
                "0",
                f"russian_car_plates.v2i.yolov8.zip:{im_p}",
                "CC BY 4.0",
                conds
            ])
            current_idx += 1
            zip_added += 1

    print(f"  [+] Добавлено из ZIP: {zip_added} кадров.")

    # 5. Integrate pristine 2-row candidates from Parquet
    print("\n[*] [2/2] Интеграция эталонных JDM-кадров из Parquet...")
    pq_path = r"C:\Users\Vamsi\Downloads\train-00000-of-00001.parquet"
    table = pq.read_table(pq_path)
    seen_plates = set(r[1] for r in all_rows if r[2] == "type1a") | set(r[1] for r in new_rows)
    pq_added = 0

    for r in table.to_pylist():
        objs = r.get("objects", {})
        bboxes = objs.get("bbox", [])
        if not bboxes:
            continue
        bx, by, bw, bh = bboxes[0]
        ar = bw / max(1.0, bh)
        if 0.8 <= ar <= 1.85:
            im_bytes = r["image"]["bytes"]
            im = cv2.imdecode(np.asarray(bytearray(im_bytes), dtype=np.uint8), cv2.IMREAD_COLOR)
            if im is None:
                continue
            h_img, w_img = im.shape[:2]
            bx = max(0, min(w_img - 1, int(bx)))
            by = max(0, min(h_img - 1, int(by)))
            bw = max(1, min(w_img - bx, int(bw)))
            bh = max(1, min(h_img - by, int(bh)))

            pts = np.array([[bx, by], [bx+bw, by], [bx+bw, by+bh], [bx, by+bh]], dtype=np.float32)
            rect_1a = rectifier.rectify(im, pts, plate_type="type1a")
            top_l, bot_l = rectifier.split_type1a(rect_1a)
            stitched = rectifier.stitch_type1a_horizontal(top_l, bot_l)
            txt, conf = ocr.predict_single(stitched, plate_type="type1a")

            if conf >= 0.70 and txt.count("#") == 0 and len(txt) in (8, 9) and txt not in seen_plates:
                seen_plates.add(txt)
                blurred, _ = blurrer.process_image(im)

                out_fn = f"real_type1a_{current_idx:04d}.jpg"
                out_path = REAL_DIR / out_fn
                cv2.imwrite(str(out_path), blurred, [cv2.IMWRITE_JPEG_QUALITY, 94])

                bbox_str = f"{bx},{by},{bw},{bh}"
                quad_str = f"{bx},{by},{bx+bw},{by},{bx+bw},{by+bh},{bx},{by+bh}"

                conds = "day"
                if ar < 1.3 or ar > 1.9:
                    conds += ",angle"

                new_rows.append([
                    f"images/real/{out_fn}",
                    txt,
                    "type1a",
                    bbox_str,
                    quad_str,
                    "1",
                    "0",
                    f"train-00000-of-00001.parquet:{r['image']['path']}",
                    "CC BY 4.0",
                    conds
                ])
                current_idx += 1
                pq_added += 1
                if pq_added >= 20:
                    break

    print(f"  [+] Добавлено из Parquet: {pq_added} кадров.")

    # 6. Write final updated meta.csv
    final_rows = all_rows + new_rows
    with open(META_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(header)
        writer.writerows(final_rows)

    print(f"\n[+] Обновлен meta.csv:")
    print(f"  • Всего строк: {len(final_rows)}")
    type1a_final = [r for r in final_rows if r[2] == "type1a" and r[6] == "0"]
    print(f"  • Реальных Type 1A: {len(type1a_final)}")
    print(f"  • Уникальных номеров Type 1A: {len(set(r[1] for r in type1a_final))}")

if __name__ == "__main__":
    integrate()
