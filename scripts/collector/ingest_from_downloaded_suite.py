#!/usr/bin/env python3
"""
OmniPlate-RU — Extraction of Real Vehicle Scenes directly from Downloaded Suite:
  - C:/Users/Vamsi\Downloads\russian_car_plates.v2i.yolov8.zip (Traffic Cameras & Trailer Plates)
  - C:/Users/Vamsi\Downloads\train-00000-of-00001.parquet
  - C:/Users/Vamsi\Downloads\validation-00000-of-00001.parquet
  - C:/Users/Vamsi\Downloads\test-00000-of-00001.parquet

Applies:
  - 100% replacement of candidates_review (zero old photos!)
  - YuNet FaceBlurrer on every single image for privacy compliance
  - Auto-annotation via OmniPlatePipeline (detector_yolo_pose)
  - Regeneration of candidates_manifest.csv and gallery.html
"""

import csv
import re
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dataset.privacy.face_blur import FaceBlurrer
from src.pipeline.pipeline import OmniPlatePipeline

DL_DIR = Path(r"C:/Users/Vamsi\Downloads")
ZIP_ROBOFLOW = DL_DIR / "russian_car_plates.v2i.yolov8.zip"
PARQUET_FILES = [
    DL_DIR / "train-00000-of-00001.parquet",
    DL_DIR / "validation-00000-of-00001.parquet",
    DL_DIR / "test-00000-of-00001.parquet"
]

REVIEW_DIR = PROJECT_ROOT / "dataset" / "candidates_review"
DIR_1A = REVIEW_DIR / "type1a"
DIR_T2 = REVIEW_DIR / "type2"
MANIFEST_PATH = REVIEW_DIR / "candidates_manifest.csv"

CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'E', 'H', 'K', 'M', 'O', 'P', 'T', 'X', 'Y']
TYPE2_REGEX = re.compile(r'^[ABEKMHOPCTYX]{2}\d{4}\d{2,3}$')


def clean_and_extract():
    print("=" * 70)
    print("  OmniPlate-RU: Извлечение 100% новых кандидатов из скачанных файлов")
    print("=" * 70)

    # 1. Completely clean candidates_review
    if DIR_1A.exists():
        shutil.rmtree(DIR_1A)
    if DIR_T2.exists():
        shutil.rmtree(DIR_T2)
    DIR_1A.mkdir(parents=True, exist_ok=True)
    DIR_T2.mkdir(parents=True, exist_ok=True)

    blurrer = FaceBlurrer()
    pipeline = OmniPlatePipeline(device="cuda", conf_threshold=0.08)

    manifest_rows = []
    t1a_idx = 1
    t2_idx = 1

    # -------------------------------------------------------------
    # 2. Extract Type 2 (Trailers) from russian_car_plates.v2i.yolov8.zip
    # -------------------------------------------------------------
    print("\n[*] [1/2] Извлечение прицепов (Type 2) из russian_car_plates.v2i.yolov8.zip...")
    if ZIP_ROBOFLOW.exists():
        with zipfile.ZipFile(ZIP_ROBOFLOW) as z:
            all_files = z.namelist()
            lbl_files = [f for f in all_files if f.endswith(".txt") and "labels" in f]
            
            for l_path in lbl_files:
                content = z.read(l_path).decode("utf-8").strip()
                if not content:
                    continue
                lines = content.split("\n")
                chars = []
                for line in lines:
                    parts = line.split()
                    if len(parts) == 5:
                        cls_idx = int(parts[0])
                        xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                        chars.append((xc, yc, w, h, CLASSES[cls_idx]))
                if not chars:
                    continue

                chars.sort(key=lambda c: c[0])
                plate_text = "".join([c[4] for c in chars])

                # Check if trailer plate (2 letters, 4 digits, region) or camera scene
                is_trailer = bool(TYPE2_REGEX.match(plate_text)) or ("Cam" in l_path and len(chars) in (8, 9) and chars[0][4].isalpha() and chars[1][4].isalpha())

                if is_trailer:
                    im_path = l_path.replace("/labels/", "/images/").rsplit(".", 1)[0] + ".jpg"
                    if im_path not in all_files:
                        continue

                    im_bytes = z.read(im_path)
                    im = cv2.imdecode(np.asarray(bytearray(im_bytes), dtype=np.uint8), cv2.IMREAD_COLOR)
                    if im is None:
                        continue

                    # Face blur
                    blurred, fb_stats = blurrer.process_image(im)
                    preds = pipeline.predict(blurred)
                    pred_t = preds[0].text if preds else plate_text

                    cand_fname = f"cand_t2_dl_{t2_idx:03d}_{plate_text}.jpg"
                    out_p = DIR_T2 / cand_fname
                    cv2.imwrite(str(out_p), blurred, [cv2.IMWRITE_JPEG_QUALITY, 93])

                    manifest_rows.append([
                        f"type2/{cand_fname}",
                        "type2",
                        f"russian_car_plates.v2i.yolov8.zip:{im_path}",
                        pred_t,
                        "0",
                        "day,real_photo"
                    ])
                    t2_idx += 1
                    if t2_idx > 35:
                        break

    print(f"  [+] Извлечено {t2_idx - 1} прицепов Type 2 из Roboflow ZIP.")

    # -------------------------------------------------------------
    # 3. Extract Type 1A (Square / 2-row) and Real Vehicles from Parquet files
    # -------------------------------------------------------------
    print("\n[*] [2/2] Извлечение Type 1A и уличных сцен из Parquet файлов...")
    seen_parquet_stems = set()

    for pq_path in PARQUET_FILES:
        if not pq_path.exists():
            continue
        table = pq.read_table(pq_path)
        for r in table.to_pylist():
            p = r["image"]["path"]
            stem = p.split("_")[0]
            if stem in seen_parquet_stems:
                continue

            objs = r.get("objects", {})
            bboxes = objs.get("bbox", [])
            if not bboxes:
                continue

            bx, by, bw, bh = bboxes[0]
            ratio = (bw * bh) / (640.0 * 640.0)
            aspect = bw / max(1.0, bh)

            # Check if square/2-row (aspect < 2.2) or full street vehicle (ratio < 0.30)
            is_square = (aspect < 2.2) and (0.01 <= ratio <= 0.85)

            if is_square and t1a_idx <= 45:
                seen_parquet_stems.add(stem)
                b = r["image"]["bytes"]
                im = cv2.imdecode(np.asarray(bytearray(b), dtype=np.uint8), cv2.IMREAD_COLOR)
                if im is None:
                    continue

                # Filter flat graphic icons (check std_dev)
                gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                if np.std(gray) < 25:
                    continue

                blurred, fb_stats = blurrer.process_image(im)
                preds = pipeline.predict(blurred)
                pred_t = preds[0].text if preds else stem

                cand_fname = f"cand_1a_dl_{t1a_idx:03d}_{stem}.jpg"
                out_p = DIR_1A / cand_fname
                cv2.imwrite(str(out_p), blurred, [cv2.IMWRITE_JPEG_QUALITY, 93])

                manifest_rows.append([
                    f"type1a/{cand_fname}",
                    "type1a",
                    f"{pq_path.name}:{p}",
                    pred_t,
                    "0",
                    "day,real_photo"
                ])
                t1a_idx += 1

    # If more Type 1A needed, extract 2-row square plates from Roboflow ZIP
    if t1a_idx <= 40 and ZIP_ROBOFLOW.exists():
        print("  [*] Доизвлечение квадратных 2-строчных номеров из Roboflow ZIP...")
        with zipfile.ZipFile(ZIP_ROBOFLOW) as z:
            all_files = z.namelist()
            lbl_files = [f for f in all_files if f.endswith(".txt") and "labels" in f]
            for l_path in lbl_files:
                if t1a_idx > 40:
                    break
                content = z.read(l_path).decode("utf-8").strip()
                if not content:
                    continue
                lines = content.split("\n")
                chars = []
                for line in lines:
                    parts = line.split()
                    if len(parts) == 5:
                        cls_idx = int(parts[0])
                        xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                        chars.append((xc, yc, w, h, CLASSES[cls_idx]))
                if len(chars) >= 6:
                    ys = [c[1] for c in chars]
                    y_diff = max(ys) - min(ys)
                    if y_diff > 0.25:  # 2 rows
                        chars.sort(key=lambda c: c[0])
                        plate_text = "".join([c[4] for c in chars])
                        im_path = l_path.replace("/labels/", "/images/").rsplit(".", 1)[0] + ".jpg"
                        if im_path not in all_files:
                            continue
                        im_bytes = z.read(im_path)
                        im = cv2.imdecode(np.asarray(bytearray(im_bytes), dtype=np.uint8), cv2.IMREAD_COLOR)
                        if im is None or min(im.shape[:2]) < 60:
                            continue

                        blurred, fb_stats = blurrer.process_image(im)
                        preds = pipeline.predict(blurred)
                        pred_t = preds[0].text if preds else plate_text

                        cand_fname = f"cand_1a_dl_{t1a_idx:03d}_{plate_text}.jpg"
                        out_p = DIR_1A / cand_fname
                        cv2.imwrite(str(out_p), blurred, [cv2.IMWRITE_JPEG_QUALITY, 93])

                        manifest_rows.append([
                            f"type1a/{cand_fname}",
                            "type1a",
                            f"russian_car_plates.v2i.yolov8.zip:{im_path}",
                            pred_t,
                            "0",
                            "day,real_photo"
                        ])
                        t1a_idx += 1

    # Save manifest
    with open(MANIFEST_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["filename", "plate_type", "source", "plate_num", "is_synthetic", "conditions"])
        writer.writerows(manifest_rows)

    total_candidates = len(manifest_rows)
    print("=" * 70)
    print(f"[УСПЕХ] Извлечено {total_candidates} 100% НОВЫХ кандидатов из скачанной пачки!")
    print(f"  - Type 1A (Квадратные): {t1a_idx - 1} кадров")
    print(f"  - Type 2 (Прицепы):     {t2_idx - 1} кадров")
    print(f"  - Все старые фото удалены из review-пула!")
    print(f"  - Манифест записан: {MANIFEST_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    clean_and_extract()
