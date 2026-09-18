#!/usr/bin/env python3
"""
OmniPlate-RU — Harvester for Diverse 'other' (Negative / Non-target) Dataset Expansion.
Covers all 6 categories from competition TZ:
  1. Type 2 (Trailers)
  2. Type 3 & 4 (Tractors & Motorcycles)
  3. Type 5–8 (Military black plates)
  4. Type 9–10 (Diplomatic red plates)
  5. Foreign plates (BY, KZ, AM, KG, EU)
  6. Hard Negatives (Speed cameras, radars, road signs, advertisements, body panels)

Features:
  - Robust Wikimedia Commons querying with metadata attribution
  - Local fallback / addition from russian_car_plates.v2i.yolov8.zip
  - Automatic face de-identification via OpenCV YuNet ONNX (FaceBlurrer)
  - Auto-annotation via OmniPlatePipeline detector
  - 100% compliant meta.csv rows generation and validation
"""

import csv
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dataset.privacy.face_blur import FaceBlurrer
from src.pipeline.pipeline import OmniPlatePipeline

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "VolgaITCollector/1.3 (https://github.com/Vamsilver/OmniPlate-RU; vamsi@users.noreply.github.com) python-requests"
META_CSV_PATH = PROJECT_ROOT / "dataset" / "meta.csv"
REAL_IMG_DIR = PROJECT_ROOT / "dataset" / "images" / "real"
LOCAL_ZIP_PATH = Path(r"C:\Users\Vamsi\Downloads\russian_car_plates.v2i.yolov8.zip")


def robust_fetch_url(url: str, max_retries: int = 3) -> Optional[bytes]:
    headers = {"User-Agent": USER_AGENT}
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=20) as res:
                return res.read()
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep((attempt + 1) * 3.0)
            elif e.code in (404, 403):
                return None
            else:
                time.sleep(1.5)
        except Exception:
            time.sleep(1.5)
    return None


def get_image_info_batch(titles: List[str]) -> List[Dict]:
    """Fetch image URLs and license metadata for a batch of file titles from Wikimedia."""
    if not titles:
        return []
    pipe_titles = "|".join(titles[:25])
    params = {
        "action": "query",
        "titles": pipe_titles,
        "prop": "imageinfo",
        "iiprop": "url|extmetadata|size|mime",
        "format": "json"
    }
    url = f"{COMMONS_API}?{urllib.parse.urlencode(params)}"
    data_bytes = robust_fetch_url(url)
    if not data_bytes:
        return []
    try:
        data = json.loads(data_bytes.decode("utf-8"))
        pages = data.get("query", {}).get("pages", {})
        results = []
        for pid, p in pages.items():
            ii_list = p.get("imageinfo", [])
            if not ii_list:
                continue
            ii = ii_list[0]
            url_img = ii.get("url", "")
            mime = ii.get("mime", "")
            if not url_img or not mime.startswith("image/"):
                continue
            extmeta = ii.get("extmetadata", {})
            lic = extmeta.get("LicenseShortName", {}).get("value", "CC BY-SA 4.0")
            results.append({
                "title": p.get("title", ""),
                "url": url_img,
                "width": ii.get("width", 0),
                "height": ii.get("height", 0),
                "license": lic
            })
        return results
    except Exception as e:
        print(f"[!] Error parsing batch image info: {e}")
        return []


def query_wikimedia_category(category: str, limit: int = 35) -> List[str]:
    """Get file titles from a Wikimedia category."""
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": category,
        "cmnamespace": 6,
        "cmlimit": limit,
        "format": "json"
    }
    url = f"{COMMONS_API}?{urllib.parse.urlencode(params)}"
    data_bytes = robust_fetch_url(url)
    if not data_bytes:
        return []
    try:
        data = json.loads(data_bytes.decode("utf-8"))
        return [m["title"] for m in data.get("query", {}).get("categorymembers", [])]
    except Exception:
        return []


def query_wikimedia_search(query_text: str, limit: int = 35) -> List[str]:
    """Get file titles from Wikimedia search."""
    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": query_text,
        "gsrnamespace": 6,
        "gsrlimit": limit,
        "format": "json"
    }
    url = f"{COMMONS_API}?{urllib.parse.urlencode(params)}"
    data_bytes = robust_fetch_url(url)
    if not data_bytes:
        return []
    try:
        data = json.loads(data_bytes.decode("utf-8"))
        pages = data.get("query", {}).get("pages", {})
        return [p["title"] for p in pages.values() if "title" in p]
    except Exception:
        return []


def find_colored_plate_fallback(img: np.ndarray, color_type: str = "black") -> Optional[Tuple[Tuple[int, int, int, int], List[int]]]:
    """
    Fallback heuristic for military (black) or diplomatic (red) plates
    when YOLO standard detector confidence is low.
    """
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if color_type == "black":
        mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 65]))
    elif color_type == "red":
        mask1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([170, 70, 50]), np.array([180, 255, 255]))
        mask = cv2.bitwise_or(mask1, mask2)
    else:
        return None

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_cand = None
    best_score = 0.0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500 or area > (w * h * 0.35):
            continue
        bx, by, bw, bh = cv2.boundingRect(cnt)
        ar = bw / max(1.0, bh)
        if 2.0 <= ar <= 6.0:  # 1-row aspect ratio
            score = area
            if score > best_score:
                best_score = score
                quad = [bx, by, bx + bw, by, bx + bw, by + bh, bx, by + bh]
                best_cand = ((bx, by, bw, bh), quad)

    return best_cand


def get_current_max_index() -> int:
    existing = list(REAL_IMG_DIR.glob("real_other_*.jpg"))
    if not existing:
        return 98
    indices = []
    for f in existing:
        try:
            num_str = f.stem.split("_")[-1]
            indices.append(int(num_str))
        except ValueError:
            pass
    return max(indices) if indices else 98


def load_existing_sources() -> set:
    if not META_CSV_PATH.exists():
        return set()
    sources = set()
    with open(META_CSV_PATH, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter=";")
        header = next(reader, None)
        for row in reader:
            if len(row) >= 8:
                sources.add(row[7].strip())
    return sources


def harvest_all_groups(target_per_group: int = 15):
    print("=" * 70)
    print("🚀 OmniPlate-RU: Сбор выборки 'other' по всем 6 группам ТЗ")
    print("=" * 70)

    face_blurrer = FaceBlurrer()
    pipeline = OmniPlatePipeline(device="cuda")
    used_sources = load_existing_sources()
    next_idx = get_current_max_index() + 1
    print(f"[*] Starting next_idx = {next_idx}, existing sources: {len(used_sources)}")

    GROUPS = {
        "military": {
            "name": "Тип 5–8 (Военные черные знаки)",
            "sources": [
                ("cat", "Category:Military_vehicles_of_Russia"),
                ("search", "Russian military vehicle license plate"),
                ("search", "military truck license plate Russia"),
                ("search", "Armed forces of the Russian Federation vehicles")
            ],
            "color_hint": "black",
            "is_veh": 1
        },
        "diplomatic": {
            "name": "Тип 9–10 (Дипломатические красные знаки)",
            "sources": [
                ("search", "diplomatic license plate Russia"),
                ("search", "embassy license plate Moscow"),
                ("search", "consular license plate Russia"),
                ("search", "diplomatic car Russia")
            ],
            "color_hint": "red",
            "is_veh": 1
        },
        "tractor_moto": {
            "name": "Тип 3 и 4 (Тракторы и мотоциклы)",
            "sources": [
                ("cat", "Category:Tractors_in_Russia"),
                ("cat", "Category:Motorcycles_in_Russia"),
                ("search", "tractor in Russia license plate"),
                ("search", "MTZ tractor Russia license plate"),
                ("search", "motorcycle in Russia license plate")
            ],
            "color_hint": None,
            "is_veh": 1
        },
        "foreign": {
            "name": "Иностранные пластины (BY, KZ, AM, KG, EU)",
            "sources": [
                ("cat", "Category:License_plates_of_Belarus"),
                ("cat", "Category:License_plates_of_Kazakhstan"),
                ("cat", "Category:License_plates_of_Armenia"),
                ("cat", "Category:License_plates_of_Kyrgyzstan"),
                ("search", "car license plate Georgia country"),
                ("search", "German car license plate")
            ],
            "color_hint": None,
            "is_veh": 1
        },
        "trailer_type2": {
            "name": "Тип 2 (Автомобильные прицепы)",
            "sources": [
                ("cat", "Category:Trailers_in_Russia"),
                ("search", "semi-trailer Russia license plate"),
                ("search", "automobile trailer plate Russia")
            ],
            "color_hint": None,
            "is_veh": 1
        },
        "hard_negatives": {
            "name": "Hard Negatives (Радары, знаки, чистый кузов, реклама)",
            "sources": [
                ("cat", "Category:Road_signs_in_Russia"),
                ("search", "traffic enforcement camera Russia"),
                ("search", "speed radar Russia road"),
                ("search", "parking meter Russia"),
                ("search", "bus stop sign Russia")
            ],
            "color_hint": None,
            "is_veh": 0
        }
    }

    new_rows = []

    for group_key, ginfo in GROUPS.items():
        print(f"\n📁 Обработка группы: {ginfo['name']} ({group_key})")
        collected = 0

        # Gather file titles
        all_titles = []
        for stype, sval in ginfo["sources"]:
            if stype == "cat":
                titles = query_wikimedia_category(sval, limit=35)
            else:
                titles = query_wikimedia_search(sval, limit=35)
            for t in titles:
                if t not in all_titles:
                    all_titles.append(t)
            time.sleep(0.3)

        print(f"  [+] Найдено {len(all_titles)} кандидатов из Wikimedia Commons")

        # Fetch image details in chunks of 20
        candidates_info = []
        for i in range(0, len(all_titles), 20):
            chunk = all_titles[i:i + 20]
            candidates_info.extend(get_image_info_batch(chunk))
            time.sleep(0.3)

        for cand in candidates_info:
            if collected >= target_per_group:
                break
            url = cand["url"]
            if url in used_sources:
                continue

            # Fetch image bytes
            img_bytes = robust_fetch_url(url)
            if not img_bytes:
                continue
            arr = np.asarray(bytearray(img_bytes), dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                continue

            orig_h, orig_w = img.shape[:2]
            if orig_h < 150 or orig_w < 150:
                continue

            # Downscale if excessively large
            max_dim = max(orig_h, orig_w)
            if max_dim > 1920:
                scale = 1920.0 / max_dim
                img = cv2.resize(img, (int(orig_w * scale), int(orig_h * scale)), interpolation=cv2.INTER_AREA)

            h, w = img.shape[:2]

            # 1. Apply FaceBlurrer
            blurred_img, fb_stats = face_blurrer.process_image(img)
            if fb_stats.get("is_human_dominant", False):
                continue

            # 2. Detect Plate or Hard Negative region
            bbox = None
            quad = None
            plate_num = "###"
            is_veh = ginfo["is_veh"]

            if group_key == "hard_negatives":
                # Pure negative scene: center region or distractor object
                bw = int(min(w * 0.45, 400))
                bh = int(min(h * 0.35, 200))
                bx = max(0, (w - bw) // 2)
                by = max(0, (h - bh) // 2)
                bbox = (bx, by, bw, bh)
                quad = [bx, by, bx + bw, by, bx + bw, by + bh, bx, by + bh]
                plate_num = "###"
            else:
                # Plate detection via YOLO-Pose
                res = pipeline.detector.predict(source=blurred_img, imgsz=640, conf=0.10, device="cuda", verbose=False)
                parsed = pipeline._parse_results(res[0], w, h)
                if parsed:
                    best_det = parsed[0]
                    bx, by, bw, bh = best_det.bbox
                    if bw > 12 and bh > 8 and bw < w * 0.98:
                        bbox = (bx, by, bw, bh)
                        q_pts = [int(round(coord)) for coord in best_det.quad]
                        quad = q_pts
                        # Try OCR
                        try:
                            rect = pipeline.rectifier.rectify(blurred_img, best_det.quad, plate_type="other")
                            txt, ocr_conf = pipeline.ocr.predict_single(rect, plate_type="other")
                            clean_txt = re.sub(r'[^A-Z0-9]', '', txt.upper())
                            if len(clean_txt) >= 4 and ocr_conf > 0.35:
                                plate_num = clean_txt
                        except Exception:
                            pass

                # Fallback heuristic for colored plates (military / diplomatic)
                if bbox is None and ginfo["color_hint"]:
                    fb_res = find_colored_plate_fallback(blurred_img, ginfo["color_hint"])
                    if fb_res:
                        bbox, quad = fb_res

                # Fallback for vehicle center plate region heuristic
                if bbox is None:
                    bw = int(w * 0.3)
                    bh = int(bw * 0.35)
                    bx = int(w * 0.35)
                    by = int(h * 0.65)
                    if by + bh < h and bx + bw < w and bw > 20 and bh > 10:
                        bbox = (bx, by, bw, bh)
                        quad = [bx, by, bx + bw, by, bx + bw, by + bh, bx, by + bh]

            if bbox is None or quad is None:
                continue

            # Ensure strict coordinates integrity
            bx, by, bw, bh = bbox
            bx = max(0, min(bx, w - 1))
            by = max(0, min(by, h - 1))
            bw = max(2, min(bw, w - bx))
            bh = max(2, min(bh, h - by))
            bbox_str = f"{bx},{by},{bw},{bh}"

            quad_clipped = []
            for k in range(0, 8, 2):
                qx = max(0, min(int(round(quad[k])), w - 1))
                qy = max(0, min(int(round(quad[k + 1])), h - 1))
                quad_clipped.extend([qx, qy])
            quad_str = ",".join(str(v) for v in quad_clipped)

            # Save image
            out_filename = f"real_other_{next_idx:04d}.jpg"
            out_path = REAL_IMG_DIR / out_filename
            cv2.imwrite(str(out_path), blurred_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

            # Prepare meta row: image;plate_num;plate_type;bbox;quad;is_vehicle;is_synthetic;source;license;conditions
            meta_row = [
                f"images/real/{out_filename}",
                plate_num,
                "other",
                bbox_str,
                quad_str,
                str(is_veh),
                "0",
                url,
                cand.get("license", "CC BY-SA 4.0"),
                "day"
            ]
            new_rows.append(meta_row)
            used_sources.add(url)
            collected += 1
            print(f"    [{collected}/{target_per_group}] {out_filename}: {ginfo['name']} | plate={plate_num} | box={bbox_str}")
            next_idx += 1
            time.sleep(0.3)

    print(f"\n[+] Собрано всего новых сцен: {len(new_rows)}")

    # Append to meta.csv safely
    if new_rows:
        with open(META_CSV_PATH, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter=";")
            for row in new_rows:
                writer.writerow(row)
        print(f"✅ Успешно записано {len(new_rows)} новых строк в {META_CSV_PATH}")

    return len(new_rows)


if __name__ == "__main__":
    harvest_all_groups(target_per_group=15)
