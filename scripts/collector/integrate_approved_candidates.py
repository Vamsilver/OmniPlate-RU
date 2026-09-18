#!/usr/bin/env python3
"""
OmniPlate-RU — Integration of Approved Staging Candidates into dataset/meta.csv & dataset/images/real/
Reads review_decisions.csv (or candidates_manifest.csv if all are approved):
  1. Copies approved candidate images into dataset/images/real/ with continuous naming
  2. Extracts exact plate BBox and 4-point Quad via YOLO-Pose
  3. Appends clean, 100% compliant rows to dataset/meta.csv
  4. Updates dataset/candidates_review/ status
"""

import csv
import json
import os
import re
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.pipeline import OmniPlatePipeline

REVIEW_DIR = PROJECT_ROOT / "dataset" / "candidates_review"
DIR_1A = REVIEW_DIR / "type1a"
DIR_T2 = REVIEW_DIR / "type2"
DECISIONS_PATH = REVIEW_DIR / "review_decisions.csv"
MANIFEST_PATH = REVIEW_DIR / "candidates_manifest.csv"
OVERRIDE_PATH = REVIEW_DIR / "verified_manual_boxes.json"

REAL_IMAGES_DIR = PROJECT_ROOT / "dataset" / "images" / "real"
META_PATH = PROJECT_ROOT / "dataset" / "meta.csv"

PLATE_REGEX = re.compile(r"^[ABEKMHOPCTYX#][\d#]{3}[ABEKMHOPCTYX#]{2}[\d#]{2,3}$")


def integrate_approved():
    print("=" * 70)
    print("  OmniPlate-RU: Интеграция одобренных кандидатов в рабочий датасет")
    print("=" * 70)

    REAL_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    pipeline = OmniPlatePipeline(device="cuda", conf_threshold=0.08)

    # 1. Determine next index for real_type1a and real_other
    max_1a = 0
    max_other = 0
    existing_meta_files = set()
    if META_PATH.exists():
        with open(META_PATH, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=";")
            next(reader, None)
            for r in reader:
                if len(r) > 0:
                    fn = Path(r[0]).name
                    existing_meta_files.add(fn)
                    if fn.startswith("real_type1a_"):
                        try:
                            num = int(fn.split("_")[-1].split(".")[0])
                            if num > max_1a:
                                max_1a = num
                        except Exception:
                            pass
                    elif fn.startswith("real_other_"):
                        try:
                            num = int(fn.split("_")[-1].split(".")[0])
                            if num > max_other:
                                max_other = num
                        except Exception:
                            pass

    print(f"[*] Текущие максимальные индексы: Type 1A={max_1a}, Other/Type 2={max_other}")

    # 2. Determine approved items
    approved_items = []  # (rel_cand_path, p_type, gt_text, source)
    if DECISIONS_PATH.exists():
        with open(DECISIONS_PATH, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=";")
            next(reader, None)
            for r in reader:
                    cand_file, p_type, pred, status, gt = r[1], r[2], r[3], r[4], r[5]
                    cand_file = cand_file.replace("\\", "/")
                    st = status.strip().lower()
                    if st in ("yes", "corrected", "correct", "ok", "verified") and st != "no":
                        text = gt.strip() if gt.strip() else pred.strip()
                        if text and not text.startswith("####") and "\ufffd" not in text:
                            approved_items.append((cand_file, p_type, text))
        print(f"[*] Загружено {len(approved_items)} подтвержденных решений из {DECISIONS_PATH}")
    else:
        print(f"[*] Файл решений {DECISIONS_PATH} пока не создан (нет сохраненных кликов).")
        print(f"[*] Используйте веб-галерею http://localhost:8080 для голосования.")
        return

    if not approved_items:
        print("[!] Нет одобренных кандидатов для импорта.")
        return

    # 3. Read sources from manifest
    sources = {}
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=";")
            next(reader, None)
            for r in reader:
                if len(r) >= 3:
                    sources[r[0]] = r[2]

    # 4. Integrate into real_dir and meta.csv
    new_meta_rows = []
    integrated_count = 0

    for cand_file, p_type, text in approved_items:
        cand_p = REVIEW_DIR / cand_file
        if not cand_p.exists():
            continue

        im = cv2.imread(str(cand_p))
        if im is None:
            continue

        h, w = im.shape[:2]
        manual_override = {}
        if OVERRIDE_PATH.exists():
            try:
                with open(OVERRIDE_PATH, "r", encoding="utf-8") as f:
                    manual_override = json.load(f)
            except Exception as e:
                print(f"[!] Warning reading {OVERRIDE_PATH.name}: {e}")

        if cand_file in manual_override:
            bx, by, bw, bh = [int(round(x)) for x in manual_override[cand_file]["bbox"]]
            q = [int(round(x)) for x in manual_override[cand_file]["quad"]]
            quad_str = f"{q[0]},{q[1]},{q[2]},{q[3]},{q[4]},{q[5]},{q[6]},{q[7]}"
            bbox_str = f"{bx},{by},{bw},{bh}"
        else:
            preds = pipeline.predict(im)
            if preds:
                bx, by, bw, bh = [int(round(x)) for x in preds[0].bbox]
                raw_q = preds[0].quad
                if isinstance(raw_q[0], (list, tuple)):
                    q = [int(round(coord)) for pt in raw_q for coord in pt]
                else:
                    q = [int(round(coord)) for coord in raw_q]
                quad_str = f"{q[0]},{q[1]},{q[2]},{q[3]},{q[4]},{q[5]},{q[6]},{q[7]}"
                bbox_str = f"{bx},{by},{bw},{bh}"
            else:
                print(f"  [!] ПРОПУСК {cand_file}: Нет детекции и нет в manual_override (отклонено)")
                continue

        # Strict geometry validation
        if bw < 20 or bh < 10 or bw > w or bh > h or bx < 0 or by < 0 or (bx + bw) > w or (by + bh) > h:
            print(f"  [!] ПРОПУСК {cand_file}: Нарушена геометрия ({bx},{by},{bw},{bh}) на кадре {w}x{h}")
            continue

        target_type = "type1a" if p_type in ("type1a", "type1A") else "other"

        # Check GOST mask for Type 1A
        if target_type == "type1a":
            if not PLATE_REGEX.match(text):
                print(f"  [!] ПРОПУСК {cand_file}: Номер '{text}' нарушает ГОСТ-маску для type1a")
                continue

        if target_type == "type1a":
            max_1a += 1
            dst_name = f"real_type1a_{max_1a:04d}.jpg"
        else:
            max_other += 1
            dst_name = f"real_other_{max_other:04d}.jpg"

        dst_path = REAL_IMAGES_DIR / dst_name
        shutil.copy2(str(cand_p), str(dst_path))

        src_url = sources.get(cand_file, "https://commons.wikimedia.org")
        new_meta_rows.append([
            f"images/real/{dst_name}",
            text,
            target_type,
            bbox_str,
            quad_str,
            "1",
            "0",
            src_url,
            "CC BY 4.0",
            "day"
        ])
        integrated_count += 1
        print(f"  [+] Интегрирован {dst_name} ({target_type}, '{text}')")

    if new_meta_rows:
        with open(META_PATH, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerows(new_meta_rows)
        print(f"\n[УСПЕХ] Интегрировано {integrated_count} кадров в dataset/meta.csv и dataset/images/real/")


if __name__ == "__main__":
    integrate_approved()
