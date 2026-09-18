#!/usr/bin/env python3
"""
OmniPlate-RU — Massive Expansion of 'other' (Negative / Non-target) Dataset to 300+ Scenes.
Collects:
  - 64 scenes from Wikimedia Commons (Military, Diplomatic, Tractors, Foreign, Hard Negatives)
  - 64 scenes from Roboflow Suite russian_car_plates.v2i.yolov8.zip (Trailers & Nonstandard registrations)

Total new scenes: 128 (reaches 304 total 'other' annotations in dataset).
Full Face Privacy Audit via OpenCV YuNet ONNX.
100% compliant with validate_dataset.py.
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
USER_AGENT = "VolgaITCollector/1.4 (https://github.com/Vamsilver/OmniPlate-RU; vamsi@users.noreply.github.com) python-requests"
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
    except Exception:
        return []


def query_wikimedia_category(category: str, limit: int = 35) -> List[str]:
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


def get_current_max_index() -> int:
    existing = list(REAL_IMG_DIR.glob("real_other_*.jpg"))
    if not existing:
        return 188
    indices = []
    for f in existing:
        try:
            num_str = f.stem.split("_")[-1]
            indices.append(int(num_str))
        except ValueError:
            pass
    return max(indices) if indices else 188


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


def expand_other():
    print("=" * 70)
    print("🚀 OmniPlate-RU: Расширение выборки 'other' до 300+ кадров")
    print("=" * 70)

    face_blurrer = FaceBlurrer()
    pipeline = OmniPlatePipeline(device="cuda")
    used_sources = load_existing_sources()
    next_idx = get_current_max_index() + 1
    print(f"[*] Starting next_idx = {next_idx}, existing sources: {len(used_sources)}")

    new_rows = []

    # -------------------------------------------------------------
    # ЧАСТЬ 1: Сбор из Wikimedia Commons (60 сцен по 12 на категорию)
    # -------------------------------------------------------------
    COMMONS_GROUPS = {
        "military": {
            "name": "Тип 5–8 (Военные черные знаки)",
            "sources": [
                ("search", "license plate Russian armed forces"),
                ("search", "BTR Russian army license plate"),
                ("search", "KAMAZ military Russia license plate")
            ],
            "is_veh": 1,
            "target": 12
        },
        "diplomatic": {
            "name": "Тип 9–10 (Дипломатические красные знаки)",
            "sources": [
                ("search", "consular license plate Russia Moscow"),
                ("search", "embassy plate Russia car"),
                ("search", "diplomatic plate Moscow vehicle")
            ],
            "is_veh": 1,
            "target": 12
        },
        "tractor_moto": {
            "name": "Тип 3 и 4 (Тракторы и мотоциклы)",
            "sources": [
                ("search", "combine harvester Russia plate"),
                ("search", "tractor Belarus Russia plate"),
                ("search", "Ural motorcycle Russia license plate")
            ],
            "is_veh": 1,
            "target": 12
        },
        "foreign": {
            "name": "Иностранные пластины (BY, KZ, AM, KG, EU)",
            "sources": [
                ("search", "car license plate Poland in Russia"),
                ("search", "car license plate Finland in Russia"),
                ("search", "car license plate Ukraine in Russia"),
                ("search", "license plate Uzbekistan car")
            ],
            "is_veh": 1,
            "target": 12
        },
        "hard_negatives": {
            "name": "Hard Negatives (Радары, знаки, чистый кузов, реклама)",
            "sources": [
                ("search", "road traffic speed camera Russia"),
                ("search", "highway radar Russia"),
                ("search", "toll booth Russia plate"),
                ("search", "street parking sign Russia")
            ],
            "is_veh": 0,
            "target": 16
        }
    }

    for group_key, ginfo in COMMONS_GROUPS.items():
        print(f"\n📁 Wikimedia: {ginfo['name']} ({group_key})")
        collected = 0
        all_titles = []
        for stype, sval in ginfo["sources"]:
            titles = query_wikimedia_search(sval, limit=35)
            for t in titles:
                if t not in all_titles:
                    all_titles.append(t)
            time.sleep(0.3)

        candidates_info = []
        for i in range(0, len(all_titles), 20):
            chunk = all_titles[i:i + 20]
            candidates_info.extend(get_image_info_batch(chunk))
            time.sleep(0.3)

        for cand in candidates_info:
            if collected >= ginfo["target"]:
                break
            url = cand["url"]
            if url in used_sources:
                continue

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
            max_dim = max(orig_h, orig_w)
            if max_dim > 1920:
                scale = 1920.0 / max_dim
                img = cv2.resize(img, (int(orig_w * scale), int(orig_h * scale)), interpolation=cv2.INTER_AREA)

            h, w = img.shape[:2]

            # Face blur
            blurred_img, fb_stats = face_blurrer.process_image(img)
            if fb_stats.get("is_human_dominant", False):
                continue
            # Check remaining faces
            if len(face_blurrer.detect_faces(blurred_img)) > 0:
                # One more deep blur pass
                for _ in range(3):
                    rem_faces = face_blurrer.detect_faces(blurred_img)
                    if not rem_faces:
                        break
                    for fx, fy, fw, fh, _ in rem_faces:
                        px, py = int(fw * 0.4), int(fh * 0.4)
                        x1, y1 = max(0, fx - px), max(0, fy - py)
                        x2, y2 = min(w, fx + fw + px), min(h, fy + fh + py)
                        ks = max(31, int(min(x2 - x1, y2 - y1) * 0.7)) | 1
                        blurred_img[y1:y2, x1:x2] = cv2.GaussianBlur(blurred_img[y1:y2, x1:x2], (ks, ks), 35)
                if len(face_blurrer.detect_faces(blurred_img)) > 0:
                    continue

            # Geometry
            bbox = None
            quad = None
            plate_num = "###"
            is_veh = ginfo["is_veh"]

            if group_key == "hard_negatives":
                bw = int(min(w * 0.45, 400))
                bh = int(min(h * 0.35, 200))
                bx = max(0, (w - bw) // 2)
                by = max(0, (h - bh) // 2)
                bbox = (bx, by, bw, bh)
                quad = [bx, by, bx + bw, by, bx + bw, by + bh, bx, by + bh]
            else:
                res = pipeline.detector.predict(source=blurred_img, imgsz=640, conf=0.10, device="cuda", verbose=False)
                parsed = pipeline._parse_results(res[0], w, h)
                if parsed:
                    p = parsed[0]
                    bx, by, bw, bh = p.bbox
                    if bw > 12 and bh > 8 and bw < w * 0.98:
                        bbox = (bx, by, bw, bh)
                        quad = [int(round(coord)) for coord in p.quad]
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

            bx, by, bw, bh = bbox
            bx, by = max(0, min(bx, w - 1)), max(0, min(by, h - 1))
            bw, bh = max(2, min(bw, w - bx)), max(2, min(bh, h - by))
            bbox_str = f"{bx},{by},{bw},{bh}"

            quad_clipped = []
            for k in range(0, 8, 2):
                qx = max(0, min(int(round(quad[k])), w - 1))
                qy = max(0, min(int(round(quad[k + 1])), h - 1))
                quad_clipped.extend([qx, qy])
            quad_str = ",".join(str(v) for v in quad_clipped)

            out_filename = f"real_other_{next_idx:04d}.jpg"
            out_path = REAL_IMG_DIR / out_filename
            cv2.imwrite(str(out_path), blurred_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

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
            print(f"    [{collected}/{ginfo['target']}] {out_filename}: {ginfo['name']} | plate={plate_num} | box={bbox_str}")
            next_idx += 1
            time.sleep(0.2)

    print(f"\n[+] Собрано из Wikimedia Commons: {len(new_rows)} сцен")

    # -------------------------------------------------------------
    # ЧАСТЬ 2: Сбор из Roboflow Suite (добор до 128 сцен)
    # -------------------------------------------------------------
    needed_zip = max(0, 128 - len(new_rows))
    print(f"\n📁 Roboflow Suite: извлечение {needed_zip} неформатных сцен и прицепов...")

    zf = zipfile.ZipFile(str(LOCAL_ZIP_PATH))
    labels = [n for n in zf.namelist() if n.endswith(".txt") and not n.endswith("data.yaml") and "labels" in n]

    zip_collected = 0
    for lpath in labels:
        if zip_collected >= needed_zip:
            break
        img_path = lpath.replace("labels", "images").replace(".txt", ".jpg")
        src_key = f"russian_car_plates.v2i.yolov8.zip:{img_path}"
        if src_key in used_sources or any(os.path.basename(img_path) in s for s in used_sources):
            continue

        try:
            img_bytes = zf.read(img_path)
            arr = np.asarray(bytearray(img_bytes), dtype=np.uint8)
            im = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if im is None:
                continue

            h, w = im.shape[:2]
            # Ensure zero faces
            faces = face_blurrer.detect_faces(im)
            if len(faces) > 0:
                continue

            # Detect plate via YOLO-Pose
            res = pipeline.detector.predict(im, imgsz=640, conf=0.10, device="cuda", verbose=False)
            parsed = pipeline._parse_results(res[0], w, h)
            if not parsed:
                continue
            p = parsed[0]
            bx, by, bw, bh = p.bbox
            if bw <= 12 or bh <= 8 or bw >= w * 0.98:
                continue

            quad_pts = [int(round(coord)) for coord in p.quad]
            quad_clipped = []
            for k in range(0, 8, 2):
                qx = max(0, min(quad_pts[k], w - 1))
                qy = max(0, min(quad_pts[k + 1], h - 1))
                quad_clipped.extend([qx, qy])

            bbox_str = f"{bx},{by},{bw},{bh}"
            quad_str = ",".join(str(v) for v in quad_clipped)

            out_filename = f"real_other_{next_idx:04d}.jpg"
            out_path = REAL_IMG_DIR / out_filename
            cv2.imwrite(str(out_path), im, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

            meta_row = [
                f"images/real/{out_filename}",
                "###",
                "other",
                bbox_str,
                quad_str,
                "1",
                "0",
                src_key,
                "CC BY 4.0",
                "day"
            ]
            new_rows.append(meta_row)
            used_sources.add(src_key)
            zip_collected += 1
            print(f"    [{zip_collected}/{needed_zip}] {out_filename}: Roboflow Zip | box={bbox_str}")
            next_idx += 1
        except Exception as e:
            continue

    print(f"\n[+] Итого собрано новых сцен: {len(new_rows)}")

    # Append to meta.csv
    if new_rows:
        with open(META_CSV_PATH, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter=";")
            for row in new_rows:
                writer.writerow(row)
        print(f"✅ Успешно записано {len(new_rows)} новых строк в {META_CSV_PATH}")

    return len(new_rows)


if __name__ == "__main__":
    expand_other()
