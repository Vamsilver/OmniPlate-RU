#!/usr/bin/env python3
"""
Targeted pure Type 1A harvester for dataset quota completion.
Collects 35 high-confidence, verified 2-row square car plates from Drive2.
"""

import csv
import json
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dataset.privacy.face_blur import FaceBlurrer
from src.pipeline.pipeline import OmniPlatePipeline

PLATE_REGEX = re.compile(r"^[ABEKMHOPCTYX][\d]{3}[ABEKMHOPCTYX]{2}(?:\d{2}|[127]\d{2})$")

OUT_DIR = PROJECT_ROOT / "test_output" / "pure_1a_quota"
CROPS_DIR = OUT_DIR / "crops"
METADATA_JSON = OUT_DIR / "candidates_metadata.json"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CROPS_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
}

MODELS = [
    'toyota/probox', 'toyota/succeed', 'toyota/vitz', 'toyota/wish',
    'toyota/caldina', 'toyota/chaser', 'toyota/mark_ii', 'toyota/cresta',
    'toyota/crown', 'toyota/bb', 'toyota/harrier', 'toyota/corolla_fielder',
    'subaru/forester', 'subaru/legacy', 'honda/fit', 'honda/stepwgn',
    'nissan/cube', 'nissan/stagea', 'nissan/wingroad', 'daihatsu/terios'
]

def download_image(url: str):
    try:
        r = requests.get(url, headers=HEADERS, timeout=6)
        if r.status_code == 200 and len(r.content) > 10000:
            arr = np.asarray(bytearray(r.content), dtype=np.uint8)
            im = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return url, im
    except Exception:
        pass
    return url, None

def evaluate_seam_ratio(crop: np.ndarray) -> float:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sob_y = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    norm_sob = cv2.resize(sob_y, (100, 100))
    p100 = np.mean(norm_sob, axis=1)
    mid_e = np.mean(p100[42:54])
    top_e = np.mean(p100[18:36])
    bot_e = np.mean(p100[60:78])
    return float((top_e + bot_e) / (2.0 * max(1e-3, mid_e)))

def harvest_quota(target_fresh: int = 35):
    print("=" * 70)
    print(f"[*] Starting targeted pure Type 1A harvest (target: {target_fresh})")
    print("=" * 70)

    # Read existing plates from meta.csv
    meta_path = PROJECT_ROOT / "dataset" / "meta.csv"
    existing_plates = set()
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            existing_plates = {r["plate_num"] for r in csv.DictReader(f, delimiter=";")}

    pipeline = OmniPlatePipeline(device="cuda")
    blurrer = FaceBlurrer()

    seen_urls = set()
    collected = []
    plate_counts = defaultdict(int)

    for page in range(1, 10):
        if len(collected) >= target_fresh:
            break
        print(f"\n--- Scanning Page {page} ---", flush=True)

        for model in MODELS:
            if len(collected) >= target_fresh:
                break

            cat_url = f"https://www.drive2.ru/cars/{model}/?page={page}"
            try:
                r_cat = requests.get(cat_url, headers=HEADERS, timeout=5)
                if r_cat.status_code != 200:
                    continue
                car_links = re.findall(r'href="(/r/[a-zA-Z0-9_/-]+)"', r_cat.text)
            except Exception:
                continue
            time.sleep(0.15)

            for car_href in car_links:
                if len(collected) >= target_fresh:
                    break

                car_url = f"https://www.drive2.ru{car_href}"
                img_urls = []
                try:
                    r_car = requests.get(car_url, headers=HEADERS, timeout=5)
                    if r_car.status_code == 200:
                        imgs = re.findall(r'https://[a-zA-Z0-9.-]+\.drive-data\.ru/[a-zA-Z0-9_-]+-(?:480|960|1920)\.jpg', r_car.text)
                        for u in imgs:
                            hi = re.sub(r'-(?:480|960)\.jpg', '-1920.jpg', u)
                            if hi not in seen_urls:
                                img_urls.append(hi)
                except Exception:
                    pass
                time.sleep(0.15)

                if not img_urls:
                    continue

                with ThreadPoolExecutor(max_workers=min(8, len(img_urls))) as pool:
                    dl_results = list(pool.map(download_image, img_urls[:6]))

                for u, img in dl_results:
                    seen_urls.add(u)
                    if img is None:
                        continue
                    h_im, w_im = img.shape[:2]
                    if h_im < 300 or w_im < 300:
                        continue

                    try:
                        res = pipeline.detector.predict(source=img, imgsz=640, conf=0.15, device="cuda", verbose=False)
                        parsed = pipeline._parse_results(res[0], w_im, h_im)

                        for det in parsed:
                            bx, by, bw, bh = det.bbox
                            ar = bw / max(1.0, bh)
                            # Strict square aspect ratio
                            if not (1.20 <= ar <= 1.82):
                                continue

                            crop = img[max(0, by):min(h_im, by+bh), max(0, bx):min(w_im, bx+bw)]
                            if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 25:
                                continue

                            seam = evaluate_seam_ratio(crop)
                            if seam < 1.20:
                                continue

                            rect_1a = pipeline.rectifier.rectify(img, det.quad, plate_type="type1a")
                            top_l, bot_l = pipeline.rectifier.split_type1a(rect_1a)
                            stitched_1a = pipeline.rectifier.stitch_type1a_horizontal(top_l, bot_l)
                            txt_1a, conf_1a = pipeline.ocr.predict_single(stitched_1a, plate_type="type1a")
                            txt_1, conf_1 = pipeline.ocr.predict_single(rect_1a, plate_type="type1")

                            txt_clean = txt_1a.replace("#", "")
                            if not PLATE_REGEX.match(txt_clean):
                                continue
                            if conf_1a < 0.82 or conf_1a <= conf_1 + 0.15:
                                continue
                            if plate_counts[txt_clean] >= 2:
                                continue
                            if txt_clean in existing_plates:
                                continue

                            # Face blur
                            blurred_full, num_faces = blurrer.blur_faces(img)
                            if num_faces.get("is_human_dominant", False):
                                continue

                            idx = len(collected) + 1
                            cid = f"fresh_1a_{idx:03d}_{txt_clean}"
                            full_fn = f"{cid}.jpg"
                            crop_fn = f"{cid}_crop.jpg"

                            cv2.imwrite(str(OUT_DIR / full_fn), blurred_full, [cv2.IMWRITE_JPEG_QUALITY, 93])
                            cv2.imwrite(str(CROPS_DIR / crop_fn), crop, [cv2.IMWRITE_JPEG_QUALITY, 94])

                            rec = {
                                "id": idx,
                                "full_fn": full_fn,
                                "crop_fn": crop_fn,
                                "url": u,
                                "text": txt_clean,
                                "conf_1a": float(conf_1a),
                                "conf_1": float(conf_1),
                                "bbox": [int(bx), int(by), int(bw), int(bh)],
                                "quad": [round(float(x), 2) for x in det.quad],
                                "ar": round(float(ar), 3),
                                "seam_ratio": round(float(seam), 3),
                                "num_faces_blurred": num_faces,
                            }
                            collected.append(rec)
                            plate_counts[txt_clean] += 1

                            print(f"  [+] #{idx:02d}/{target_fresh} | {txt_clean} | 1A:{conf_1a:.2f} > 1:{conf_1:.2f} | AR:{ar:.2f} | Seam:{seam:.2f}", flush=True)

                            with open(METADATA_JSON, "w", encoding="utf-8") as f:
                                json.dump(collected, f, indent=2, ensure_ascii=False)

                            if len(collected) >= target_fresh:
                                break
                    except Exception as e:
                        pass

    print(f"\n[+] Harvest complete! Collected {len(collected)} pure Type 1A candidates.")

if __name__ == "__main__":
    harvest_quota(target_fresh=35)
