#!/usr/bin/env python3
"""
OmniPlate-RU — Harvester of Real Street Candidates (Type 1A and Type 2)
Extracts authentic full-size street photos from open photographic archives (Wikimedia Commons):
  - Type 1A (JDM / Square): Vladivostok, Nakhodka, Khabarovsk, Primorsky Krai, Irkutsk
  - Type 2 (Trailers): Trailers in Russia, Trailers in Moscow, Trailer plates of Russia
Applies:
  - FaceBlurrer (YuNet DNN) for 100% VolgaIT privacy compliance
  - Auto-annotation / plate detection with detector_yolo_pose
  - Staging into dataset/candidates_review/ with manifest update
"""

import csv
import json
import os
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dataset.privacy.face_blur import FaceBlurrer
from src.pipeline.pipeline import OmniPlatePipeline

REVIEW_DIR = PROJECT_ROOT / "dataset" / "candidates_review"
DIR_1A = REVIEW_DIR / "type1a"
DIR_T2 = REVIEW_DIR / "type2"
MANIFEST_PATH = REVIEW_DIR / "candidates_manifest.csv"

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) VolgaITRealHarvester/4.0 (contact: vamsi@local)"

T1A_CATEGORIES = [
    "Category:Automobiles in Vladivostok",
    "Category:Automobiles in Nakhodka",
    "Category:Automobiles with license plates of Primorsky Krai",
    "Category:Automobiles with license plates of Khabarovsk Krai",
    "Category:Automobiles in Khabarovsk",
    "Category:Automobiles in Irkutsk",
]

T2_CATEGORIES = [
    "Category:Trailers in Russia",
    "Category:Trailers in Moscow",
    "Category:Trailer license plates of Russia",
]


def fetch_commons_file_list(category: str, limit: int = 40) -> List[str]:
    encoded = urllib.parse.quote(category)
    url = f"{COMMONS_API}?action=query&list=categorymembers&cmtitle={encoded}&cmtype=file&cmlimit={limit}&format=json"
    try:
        out = subprocess.check_output(["curl.exe", "-s", "-m", "10", "-A", USER_AGENT, url], timeout=12)
        data = json.loads(out.decode("utf-8", errors="replace"))
        members = data.get("query", {}).get("categorymembers", [])
        return [m["title"] for m in members if m["title"].lower().endswith((".jpg", ".jpeg", ".png"))]
    except Exception as e:
        print(f"  [!] Category query error ({category}): {e}")
        return []


def fetch_commons_image(file_title: str) -> Optional[Tuple[np.ndarray, str]]:
    encoded = urllib.parse.quote(file_title)
    url = f"{COMMONS_API}?action=query&titles={encoded}&prop=imageinfo&iiprop=url&iiurlwidth=1280&format=json"
    try:
        out = subprocess.check_output(["curl.exe", "-s", "-m", "10", "-A", USER_AGENT, url], timeout=12)
        data = json.loads(out.decode("utf-8", errors="replace"))
        pages = data.get("query", {}).get("pages", {})
        for _, p in pages.items():
            info = p.get("imageinfo", [])
            if info:
                thumb_url = info[0].get("thumburl") or info[0].get("url")
                if thumb_url:
                    img_bytes = subprocess.check_output(["curl.exe", "-s", "-m", "15", "-A", USER_AGENT, thumb_url], timeout=18)
                    arr = np.asarray(bytearray(img_bytes), dtype=np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if img is not None and min(img.shape[:2]) >= 300:
                        return img, thumb_url
    except Exception:
        pass
    return None


def harvest_candidates(target_per_category: int = 25):
    print("=" * 70)
    print("  OmniPlate-RU: Сбор настоящих уличных фото (Type 1A и Type 2)")
    print("=" * 70)

    DIR_1A.mkdir(parents=True, exist_ok=True)
    DIR_T2.mkdir(parents=True, exist_ok=True)

    blurrer = FaceBlurrer()
    pipeline = OmniPlatePipeline()

    manifest_rows = []
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=";")
            header = next(reader)
            for r in reader:
                if len(r) > 1:
                    manifest_rows.append(r)
    else:
        header = ["filename", "plate_type", "source", "plate_num", "is_synthetic", "conditions"]

    print("\n[*] Сбор реальных кадров Type 1A (JDM / Дальний Восток)...")
    t1a_added = 0
    t1a_index = len(list(DIR_1A.glob("cand_1a_real_*.jpg"))) + 1

    for cat in T1A_CATEGORIES:
        if t1a_added >= target_per_category:
            break
        print(f"  -> Сканирование {cat}...")
        files = fetch_commons_file_list(cat, limit=35)
        for f_title in files:
            if t1a_added >= target_per_category:
                break
            time.sleep(1.2)
            res = fetch_commons_image(f_title)
            if res is None:
                continue
            img, source_url = res

            blurred, fb_stats = blurrer.process_image(img)
            if fb_stats.get("is_human_dominant", False):
                continue

            preds = pipeline.predict(blurred)
            pred_text = preds[0].text if preds else ""

            cand_fname = f"cand_1a_real_{t1a_index:03d}.jpg"
            out_path = DIR_1A / cand_fname

            cv2.imwrite(str(out_path), blurred, [cv2.IMWRITE_JPEG_QUALITY, 93])
            manifest_rows.append([
                f"type1a/{cand_fname}",
                "type1a",
                source_url,
                pred_text,
                "0",
                "day,real_photo"
            ])
            t1a_added += 1
            t1a_index += 1
            faces_n = fb_stats.get("faces_detected", 0)
            print(f"     [+] Сохранен {cand_fname} ({img.shape[1]}x{img.shape[0]}, Лиц: {faces_n}, Детекция: '{pred_text}')")

    print("\n[*] Сбор реальных кадров Type 2 (Прицепы / Полуприцепы)...")
    t2_added = 0
    t2_index = len(list(DIR_T2.glob("cand_t2_real_*.jpg"))) + 1

    for cat in T2_CATEGORIES:
        if t2_added >= target_per_category:
            break
        print(f"  -> Сканирование {cat}...")
        files = fetch_commons_file_list(cat, limit=40)
        for f_title in files:
            if t2_added >= target_per_category:
                break
            time.sleep(1.2)
            res = fetch_commons_image(f_title)
            if res is None:
                continue
            img, source_url = res

            blurred, fb_stats = blurrer.process_image(img)
            if fb_stats.get("is_human_dominant", False):
                continue

            preds = pipeline.predict(blurred)
            pred_text = preds[0].text if preds else ""

            cand_fname = f"cand_t2_real_{t2_index:03d}.jpg"
            out_path = DIR_T2 / cand_fname

            cv2.imwrite(str(out_path), blurred, [cv2.IMWRITE_JPEG_QUALITY, 93])
            manifest_rows.append([
                f"type2/{cand_fname}",
                "type2",
                source_url,
                pred_text,
                "0",
                "day,real_photo"
            ])
            t2_added += 1
            t2_index += 1
            faces_n = fb_stats.get("faces_detected", 0)
            print(f"     [+] Сохранен {cand_fname} ({img.shape[1]}x{img.shape[0]}, Лиц: {faces_n}, Детекция: '{pred_text}')")

    with open(MANIFEST_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(header)
        writer.writerows(manifest_rows)

    total_real = len(manifest_rows)
    print("=" * 70)
    print(f"[УСПЕХ] Сбор завершен!")
    print(f"  - Добавлено Type 1A: {t1a_added}")
    print(f"  - Добавлено Type 2:  {t2_added}")
    print(f"  - Всего реальных кандидатов в staging: {total_real}")
    print(f"  - Манифест обновлен: {MANIFEST_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    harvest_candidates(target_per_category=25)
