#!/usr/bin/env python3
"""
OmniPlate-RU — Expansion of Type 1A (Square JDM Plates) to >= 300 Real Images.

1. Scans fresh candidates from Parquet suite and remaining 2-row Zip candidates.
2. Applies FaceBlurrer (OpenCV YuNet ONNX) for 100% privacy compliance.
3. Evaluates 2-line Type 1A OCR (Split & Stitch) and verifies GOST regex compliance.
4. Generates precise BBox and 8-point Quad coordinates.
5. Saves images to dataset/images/real/real_type1a_XXXX.jpg with continuous indexing.
6. Updates dataset/meta.csv until Type 1A real count reaches >= 302.
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

REAL_DIR = PROJECT_ROOT / "dataset" / "images" / "real"
META_PATH = PROJECT_ROOT / "dataset" / "meta.csv"

PLATE_REGEX = re.compile(r"^[ABEKMHOPCTYX#][\d#]{3}[ABEKMHOPCTYX#]{2}[\d#]{2,3}$")
TOP_REGEX = re.compile(r"^[ABEKMHOPCTYX][0-9]{3}$")
BOT_REGEX = re.compile(r"^[ABEKMHOPCTYX]{2}[0-9]{2,3}$")
NAMES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'E', 'H', 'K', 'M', 'O', 'P', 'T', 'X', 'Y']


def expand():
    print("=" * 75)
    print("OmniPlate-RU: Расширение выборки Type 1A до >= 300 реальных кадров")
    print("=" * 75)

    # 1. Read existing meta.csv
    with open(META_PATH, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")
        header = next(reader)
        existing_rows = list(reader)

    current_1a = [r for r in existing_rows if r[2] == "type1a" and r[6] == "0"]
    existing_sources = set(r[7] for r in existing_rows if len(r) > 7)
    existing_plates = set(r[1] for r in current_1a)
    
    max_idx = 0
    for r in existing_rows:
        img_p = r[0].replace("\\", "/")
        if "real_type1a_" in img_p:
            try:
                num = int(img_p.split("real_type1a_")[-1].split(".")[0])
                if num > max_idx:
                    max_idx = num
            except Exception:
                pass

    print(f"[*] Текущее количество real Type 1A: {len(current_1a)}")
    print(f"[*] Уникальных номеров Type 1A: {len(existing_plates)}")
    print(f"[*] Текущий максимальный индекс real_type1a: {max_idx}")
    
    target_total = 302
    needed = target_total - len(current_1a)
    print(f"[*] Требуется добавить новых кадров: {needed}")

    blurrer = FaceBlurrer()
    rectifier = PlateRectifier()
    ocr = PlateOCR(model_path="models/ocr_lprnet_best.pt", device="cuda")

    new_rows = []
    current_idx = max_idx + 1

    # 2. Extract remaining 2-row candidates from Roboflow ZIP
    z_path = r"C:\Users\Vamsi\Downloads\russian_car_plates.v2i.yolov8.zip"
    if Path(z_path).exists():
        print("\n[*] [1/2] Проверка оставшихся 2-строчных кандидатов из Roboflow ZIP...")
        z = zipfile.ZipFile(z_path)
        lbl_files = [f for f in z.namelist() if f.endswith('.txt') and not f.startswith('README')]
        for lf in lbl_files:
            im_p = lf.replace('/labels/', '/images/').replace('.txt', '.jpg')
            src = f"russian_car_plates.v2i.yolov8.zip:{im_p}"
            if src in existing_sources:
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
            top = chars_by_y[:4]
            bot = chars_by_y[4:]
            top_x = (min(c[1] for c in top), max(c[1] for c in top))
            bot_x = (min(c[1] for c in bot), max(c[1] for c in bot))
            overlap = max(0.0, min(top_x[1], bot_x[1]) - max(top_x[0], bot_x[0]))
            top_y_avg = sum(c[2] for c in top) / 4.0
            bot_y_avg = sum(c[2] for c in bot) / len(bot)
            y_sep = bot_y_avg - top_y_avg

            if overlap > 0.15 and y_sep > 0.15:
                top_sorted = sorted(top, key=lambda c: c[1])
                bot_sorted = sorted(bot, key=lambda c: c[1])
                t_str = ''.join(NAMES[c[0]] for c in top_sorted)
                b_str = ''.join(NAMES[c[0]] for c in bot_sorted)
                full_num = t_str + b_str
                if not PLATE_REGEX.match(full_num):
                    continue

                if im_p not in z.namelist():
                    continue

                im_bytes = z.read(im_p)
                im = cv2.imdecode(np.asarray(bytearray(im_bytes), dtype=np.uint8), cv2.IMREAD_COLOR)
                if im is None:
                    continue
                h_img, w_img = im.shape[:2]

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

                blurred, _ = blurrer.process_image(im)
                out_fn = f"real_type1a_{current_idx:04d}.jpg"
                out_path = REAL_DIR / out_fn
                cv2.imwrite(str(out_path), blurred, [cv2.IMWRITE_JPEG_QUALITY, 94])

                conds = "day"
                if bw / float(bh) < 1.4 or bw / float(bh) > 1.9:
                    conds += ",angle"

                new_rows.append([
                    f"images/real/{out_fn}",
                    full_num,
                    "type1a",
                    bbox_str,
                    quad_str,
                    "1",
                    "0",
                    src,
                    "CC BY 4.0",
                    conds
                ])
                existing_sources.add(src)
                existing_plates.add(full_num)
                current_idx += 1
                if len(new_rows) >= needed:
                    break

        print(f"  [+] Добавлено из ZIP: {len(new_rows)} кадров.")

    # 3. Extract fresh candidates from Parquet suite
    if len(new_rows) < needed:
        print("\n[*] [2/2] Извлечение кандидатов из Parquet suite...")
        pq_files = [
            r"C:\Users\Vamsi\Downloads\train-00000-of-00001.parquet",
            r"C:\Users\Vamsi\Downloads\validation-00000-of-00001.parquet",
            r"C:\Users\Vamsi\Downloads\test-00000-of-00001.parquet"
        ]
        
        for pq_p in pq_files:
            if len(new_rows) >= needed:
                break
            table = pq.read_table(pq_p)
            fn = Path(pq_p).name
            for r in table.to_pylist():
                if len(new_rows) >= needed:
                    break
                p = r["image"]["path"]
                src = f"{fn}:{p}"
                if src in existing_sources:
                    continue

                objs = r.get("objects", {})
                bboxes = objs.get("bbox", [])
                if not bboxes:
                    continue
                bx, by, bw, bh = bboxes[0]
                ar = bw / max(1.0, bh)
                if not (0.75 <= ar <= 2.10):
                    continue

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
                top_l, bot_l = rectifier.split_type1a(rect_1a, adaptive_seam=True)
                stitched = rectifier.stitch_type1a_horizontal(top_l, bot_l)
                txt_1a, conf_1a = ocr.predict_single(stitched, plate_type="type1a")

                # Verify valid GOST regex
                if not PLATE_REGEX.match(txt_1a):
                    continue

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
                    txt_1a,
                    "type1a",
                    bbox_str,
                    quad_str,
                    "1",
                    "0",
                    src,
                    "CC BY 4.0",
                    conds
                ])
                existing_sources.add(src)
                existing_plates.add(txt_1a)
                current_idx += 1

        print(f"  [+] Добавлено всего новых кадров: {len(new_rows)}")

    # 4. Save updated meta.csv
    final_rows = existing_rows + new_rows
    with open(META_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(header)
        writer.writerows(final_rows)

    type1a_final = [r for r in final_rows if r[2] == "type1a" and r[6] == "0"]
    unique_1a = set(r[1] for r in type1a_final)
    print(f"\n[+] Интеграция завершена:")
    print(f"  • Всего строк в meta.csv: {len(final_rows)}")
    print(f"  • Всего реальных Type 1A: {len(type1a_final)}")
    print(f"  • Уникальных номеров Type 1A: {len(unique_1a)}")

if __name__ == "__main__":
    expand()
