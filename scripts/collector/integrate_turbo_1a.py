#!/usr/bin/env python3
"""
Integrate verified Turbo JDM Type 1A candidates into dataset/meta.csv and dataset/images/real/.
Indexing starts at real_type1a_0846.jpg.
"""

import csv
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import List, Set

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.pipeline import OmniPlatePipeline

META_PATH = PROJECT_ROOT / "dataset" / "meta.csv"
BACKUP_PATH = PROJECT_ROOT / "dataset" / "meta.csv.bak_turbo1a"
REAL_DIR = PROJECT_ROOT / "dataset" / "images" / "real"
TURBO_DIR = PROJECT_ROOT / "test_output" / "turbo_1a_verified"
CROPS_DIR = TURBO_DIR / "crops"
METADATA_JSON = TURBO_DIR / "candidates_metadata.json"

PLATE_REGEX = re.compile(r"^[ABEKMHOPCTYX][\d]{3}[ABEKMHOPCTYX]{2}(?:\d{2}|[127]\d{2})$")


def integrate_turbo_candidates(min_target_total_1a: int = 302, rejected_ids: Set[str] = None):
    if rejected_ids is None:
        rejected_ids = set()

    print("=" * 70)
    print("OmniPlate-RU: Интеграция кандидатов Turbo JDM Type 1A")
    print("=" * 70)

    # 1. Backup meta.csv
    if not BACKUP_PATH.exists():
        shutil.copyfile(META_PATH, BACKUP_PATH)
        print(f"[*] Создан бэкап: {BACKUP_PATH}")

    # 2. Read current meta.csv
    with open(META_PATH, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")
        header = next(reader)
        rows = list(reader)

    current_real_1a = sum(1 for r in rows if len(r) > 6 and r[2] == "type1a" and r[6] == "0")
    print(f"[*] Текущее количество чистых реальных Type 1A в meta.csv: {current_real_1a}")

    # Find highest real_type1a index
    max_idx = 845
    for r in rows:
        m = re.search(r"real_type1a_(\d+)", r[0])
        if m:
            max_idx = max(max_idx, int(m.group(1)))

    print(f"[*] Максимальный текущий индекс real_type1a: {max_idx}")
    start_idx = max(846, max_idx + 1)
    print(f"[*] Начальный индекс для новых кадров: real_type1a_{start_idx:04d}.jpg")

    # 3. Load candidates
    crop_files = sorted(list(CROPS_DIR.glob("*.jpg")))
    print(f"[*] Всего доступных кандидатов в turbo_1a_verified: {len(crop_files)}")

    pipeline = OmniPlatePipeline(device="cuda")

    new_rows = []
    current_idx = start_idx
    integrated_plates = set()

    for c_path in crop_files:
        stem = c_path.stem.replace("_crop", "")
        # e.g. cand_1a_0001_A123BC77
        parts = stem.split("_")
        txt_plate = parts[3] if len(parts) >= 4 else parts[2]
        cand_id = parts[2] if len(parts) >= 4 else parts[1]

        if cand_id in rejected_ids or stem in rejected_ids:
            print(f"  [-] Пропуск отклоненного кандидата: {stem}")
            continue

        if not PLATE_REGEX.match(txt_plate):
            continue

        full_fn = f"{stem}.jpg"
        full_src = TURBO_DIR / full_fn
        if not full_src.exists():
            continue

        im = cv2.imread(str(full_src))
        if im is None:
            continue
        h_im, w_im = im.shape[:2]

        # Detect plate on full image
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

        # Target file
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
        integrated_plates.add(txt_plate)
        current_idx += 1

        if (current_real_1a + len(new_rows)) >= min_target_total_1a and len(new_rows) >= 145:
            break

    # Save to meta.csv
    all_rows = rows + new_rows
    with open(META_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(header)
        writer.writerows(all_rows)

    final_real_1a = sum(1 for r in all_rows if len(r) > 6 and r[2] == "type1a" and r[6] == "0")
    print(f"\n[+] Успешно интегрировано {len(new_rows)} новых кадров Type 1A!")
    print(f"[+] Новые файлы: real_type1a_{start_idx:04d}.jpg .. real_type1a_{current_idx-1:04d}.jpg")
    print(f"[+] Итоговое количество чистых реальных Type 1A: {final_real_1a} (Цель: 300+)")
    print(f"[+] Уникальных номеров среди новых: {len(integrated_plates)}")
    print(f"[+] Всего строк в обновленном meta.csv: {len(all_rows)}")


if __name__ == "__main__":
    integrate_turbo_candidates()
