#!/usr/bin/env python3
"""
Pure Type 1A Harvester from high-yield square plate models (Toyota bB, Probox, Cube, etc.)
Uses pipeline.predict() directly. Stops at 25 pristine candidates.
"""

import csv
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dataset.privacy.face_blur import FaceBlurrer
from src.pipeline.pipeline import OmniPlatePipeline

PLATE_REGEX = re.compile(r"^[ABEKMHOPCTYX][\d]{3}[ABEKMHOPCTYX]{2}(?:\d{2}|[127]\d{2})$")

OUT_DIR = PROJECT_ROOT / "test_output" / "pure_1a_fresh"
CROPS_DIR = OUT_DIR / "crops"
METADATA_JSON = OUT_DIR / "candidates_metadata.json"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CROPS_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

YIELD_MODELS = [
    'toyota/probox', 'toyota/succeed', 'nissan/cube', 'toyota/bb',
    'toyota/funcargo', 'honda/stepwgn', 'daihatsu/tanto', 'suzuki/jimny',
    'toyota/vitz', 'toyota/wish', 'toyota/caldina', 'toyota/chaser', 'toyota/mark_ii',
    'subaru/forester', 'mitsubishi/delica', 'nissan/serena', 'toyota/alphard', 'toyota/vellfire',
    'toyota/raum', 'toyota/isis', 'mazda/demio', 'honda/fit'
]

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:129.0) Gecko/20100101 Firefox/129.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36 Edg/127.0.0.0',
]

def harvest(target: int = 25):
    print("=" * 70)
    print(f"[*] Starting high-yield Type 1A harvest (target: {target})")
    print("=" * 70)

    meta_path = PROJECT_ROOT / "dataset" / "meta.csv"
    existing_real_plates = set()
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f, delimiter=";"):
                if r["is_synthetic"] == "0" and r["plate_type"] == "type1a":
                    existing_real_plates.add(r["plate_num"])

    pipeline = OmniPlatePipeline(device="cuda")
    blurrer = FaceBlurrer()

    s = requests.Session()

    collected = []
    seen_urls = set()
    plate_counts = defaultdict(int)

    if METADATA_JSON.exists():
        try:
            with open(METADATA_JSON, "r", encoding="utf-8") as f:
                collected = json.load(f)
            for c in collected:
                plate_counts[c["text"]] += 1
                seen_urls.add(c.get("url", ""))
            print(f"[*] Resuming from {METADATA_JSON.name} with {len(collected)} items.")
        except Exception as e:
            print(f"[-] Error resuming: {e}")

    ua_idx = 0

    for page in range(1, 8):
        if len(collected) >= target:
            break

        for model in YIELD_MODELS:
            if len(collected) >= target:
                break

            s.headers.update({
                "User-Agent": USER_AGENTS[ua_idx % len(USER_AGENTS)],
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
            })
            ua_idx += 1

            cat_url = f"https://www.drive2.ru/cars/{model}/?page={page}"
            try:
                r_cat = s.get(cat_url, timeout=7)
                if r_cat.status_code == 429:
                    print(f"[{model} | p.{page}] 429 rate limit hit. Waiting 6s...", flush=True)
                    time.sleep(6)
                    continue
                if r_cat.status_code != 200:
                    continue

                car_links = list(set(re.findall(rf'href="(/r/{model}/[0-9]+/)"', r_cat.text)))
                if not car_links:
                    car_links = list(set(re.findall(r'href="(/r/[a-zA-Z0-9_/-]+/[0-9]+/)"', r_cat.text)))
            except Exception as e:
                print(f"[-] Error fetching cat {cat_url}: {e}")
                continue

            time.sleep(1.2)
            print(f"[{model} | p.{page}] Found {len(car_links)} cars. (Current total: {len(collected)}/{target})", flush=True)

            for car_href in car_links:
                if len(collected) >= target:
                    break

                car_url = f"https://www.drive2.ru{car_href}"
                try:
                    s.headers.update({"User-Agent": USER_AGENTS[ua_idx % len(USER_AGENTS)]})
                    ua_idx += 1
                    r_car = s.get(car_url, timeout=7)
                    if r_car.status_code == 429:
                        time.sleep(5)
                        continue
                    if r_car.status_code != 200:
                        continue
                    img_urls = list(set(re.findall(r'https://[a-zA-Z0-9.-]+\.drive-data\.ru/[a-zA-Z0-9_-]+-1920\.jpg', r_car.text)))
                except Exception as e:
                    continue

                time.sleep(0.6)

                for u in img_urls[:6]:
                    if u in seen_urls or len(collected) >= target:
                        continue
                    seen_urls.add(u)

                    try:
                        r_img = s.get(u, timeout=7)
                        if r_img.status_code != 200 or len(r_img.content) < 15000:
                            continue
                        arr = np.asarray(bytearray(r_img.content), dtype=np.uint8)
                        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        if img is None:
                            continue
                        h_im, w_im = img.shape[:2]
                        if h_im < 300 or w_im < 300:
                            continue

                        dets = pipeline.predict(img)
                        for d in dets:
                            if d.plate_type != "type1a":
                                continue
                            txt_clean = d.text.replace("#", "")
                            if not PLATE_REGEX.match(txt_clean):
                                continue
                            if d.ocr_confidence < 0.85:
                                continue

                            bx, by, bw, bh = [int(v) for v in d.bbox]
                            ar = bw / max(1.0, bh)
                            if not (1.15 <= ar <= 1.85):
                                continue

                            if plate_counts[txt_clean] >= 2 or txt_clean in existing_real_plates:
                                continue

                            crop = img[max(0, by):min(h_im, by+bh), max(0, bx):min(w_im, bx+bw)]
                            if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 25:
                                continue

                            blurred_full, stats = blurrer.process_image(img)
                            if stats.get("is_human_dominant", False):
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
                                "conf": float(d.confidence),
                                "ocr_conf": float(d.ocr_confidence),
                                "bbox": [bx, by, bw, bh],
                                "quad": [round(float(x), 2) for x in d.quad],
                                "ar": round(float(ar), 3),
                                "num_faces_blurred": stats["faces_detected"],
                            }
                            collected.append(rec)
                            plate_counts[txt_clean] += 1

                            print(f"  [+] #{idx:02d}/{target} | {txt_clean} | OCR:{d.ocr_confidence:.2f} | Det:{d.confidence:.2f} | AR:{ar:.2f} ({model})", flush=True)

                            with open(METADATA_JSON, "w", encoding="utf-8") as f:
                                json.dump(collected, f, indent=2, ensure_ascii=False)

                            if len(collected) >= target:
                                break
                    except Exception as e:
                        print(f"[-] Error processing image {u}: {e}")
                time.sleep(0.1)

    print(f"\n[+] Harvest finished! Collected {len(collected)} verified Type 1A plates.")

if __name__ == "__main__":
    harvest(target=25)
