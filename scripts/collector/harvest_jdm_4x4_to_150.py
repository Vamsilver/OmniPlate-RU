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

PROJECT_ROOT = Path(r"D:\AIProjects\VolgaIT")
sys.path.insert(0, str(PROJECT_ROOT))

from dataset.privacy.face_blur import FaceBlurrer
from src.pipeline.pipeline import OmniPlatePipeline

PLATE_REGEX = re.compile(r"^[ABEKMHOPCTYX][\d]{3}[ABEKMHOPCTYX]{2}(?:\d{2}|[127]\d{2})$")

OUT_DIR = PROJECT_ROOT / "test_output" / "new_1a_verified"
CROPS_DIR = OUT_DIR / "crops"
METADATA_JSON = OUT_DIR / "candidates_metadata.json"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
}

MODELS_4X4 = [
    'toyota/hilux_surf', 'toyota/land_cruiser_prado', 'toyota/rav4',
    'honda/hr_v', 'nissan/terrano', 'mitsubishi/pajero_mini',
    'mitsubishi/pajero_io', 'mitsubishi/chariot_grandis', 'suzuki/escudo',
    'toyota/celica', 'toyota/corolla_levin', 'toyota/sprinter_trueno',
    'toyota/nadia', 'toyota/gaia', 'toyota/opa', 'honda/integra', 'honda/prelude'
]

def download_image(url: str):
    try:
        r = requests.get(url, headers=HEADERS, timeout=6)
        if r.status_code == 200 and len(r.content) > 8000:
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
    seam_ratio = (top_e + bot_e) / (2.0 * max(1e-3, mid_e))
    return float(seam_ratio)

def run():
    target_count = 126  # 126 in new_1a_verified + 29 old = 155 total pure!
    
    seen_urls = set()
    plate_counts = defaultdict(int)
    approved_records = []
    
    existing_crops = list(CROPS_DIR.glob("*.jpg"))
    for c in existing_crops:
        stem = c.stem.replace("_crop", "")
        parts = stem.split("_")
        if len(parts) >= 4:
            plate_counts[parts[3]] += 1
        elif len(parts) == 3:
            plate_counts[parts[2]] += 1
            
    counter = len(existing_crops)
    print(f"[*] Starting verified crops on disk: {counter} | Target: {target_count}", flush=True)

    pipeline = OmniPlatePipeline(device="cuda")
    blurrer = FaceBlurrer()
    
    for page in range(1, 4):
        if counter >= target_count:
            break
            
        print(f"\n==================== 4X4 JDM PAGE {page} ====================", flush=True)
        for model in MODELS_4X4:
            if counter >= target_count:
                break
                
            cat_url = f"https://www.drive2.ru/cars/{model}/?page={page}"
            try:
                r_cat = requests.get(cat_url, headers=HEADERS, timeout=5)
                if r_cat.status_code != 200:
                    continue
                car_links = re.findall(r'href="(/r/[a-zA-Z0-9_/-]+)"', r_cat.text)
            except Exception:
                continue
            time.sleep(0.12)
            
            print(f"[{model} | p.{page}] Found {len(car_links)} cars (Current total: {counter}, Unique: {len(plate_counts)})", flush=True)
            
            for car_href in car_links:
                if counter >= target_count:
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
                time.sleep(0.12)
                
                if not img_urls:
                    continue
                    
                with ThreadPoolExecutor(max_workers=min(10, len(img_urls))) as pool:
                    dl_results = list(pool.map(download_image, img_urls[:8]))
                    
                for u, img in dl_results:
                    seen_urls.add(u)
                    if img is None:
                        continue
                    h_im, w_im = img.shape[:2]
                    if h_im < 250 or w_im < 250:
                        continue
                        
                    try:
                        res = pipeline.detector.predict(source=img, imgsz=640, conf=0.10, device='cuda', verbose=False)
                        parsed = pipeline._parse_results(res[0], w_im, h_im)
                        for det in parsed:
                            bx, by, bw, bh = det.bbox
                            ar = bw / max(1.0, bh)
                            if not (1.08 <= ar <= 1.85):
                                continue
                                
                            crop = img[max(0, by):min(h_im, by+bh), max(0, bx):min(w_im, bx+bw)]
                            if crop.size == 0 or crop.shape[0] < 22 or crop.shape[1] < 28:
                                continue
                                
                            seam_ratio = evaluate_seam_ratio(crop)
                            if seam_ratio < 1.06:
                                continue
                                
                            rect_1a = pipeline.rectifier.rectify(img, det.quad, plate_type="type1a")
                            top_l, bot_l = pipeline.rectifier.split_type1a(rect_1a)
                            stitched_1a = pipeline.rectifier.stitch_type1a_horizontal(top_l, bot_l)
                            txt_1a, conf_1a = pipeline.ocr.predict_single(stitched_1a, plate_type="type1a")
                            
                            rect_1 = pipeline.rectifier.rectify(img, det.quad, plate_type="type1")
                            txt_1, conf_1 = pipeline.ocr.predict_single(rect_1, plate_type="type1")[:2]
                            
                            txt_clean = txt_1a.strip().upper()
                            if not PLATE_REGEX.match(txt_clean):
                                continue
                            if conf_1a < 0.65 or conf_1a <= conf_1:
                                continue
                                
                            if plate_counts[txt_clean] >= 3:
                                continue
                                
                            blurred_full, num_faces = blurrer.process_image(img)
                            
                            cid = f"cand_1a_{counter:04d}_{txt_clean}"
                            full_fn = f"{cid}.jpg"
                            crop_fn = f"{cid}_crop.jpg"
                            
                            cv2.imwrite(str(OUT_DIR / full_fn), blurred_full, [cv2.IMWRITE_JPEG_QUALITY, 94])
                            cv2.imwrite(str(CROPS_DIR / crop_fn), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                            
                            rec = {
                                "id": counter,
                                "full_fn": full_fn,
                                "crop_fn": crop_fn,
                                "url": u,
                                "text": txt_clean,
                                "conf_1a": float(conf_1a),
                                "conf_1": float(conf_1),
                                "txt_1": txt_1,
                                "bbox": [int(bx), int(by), int(bw), int(bh)],
                                "quad": [round(float(x), 2) for x in det.quad],
                                "ar": round(float(ar), 3),
                                "seam_ratio": round(float(seam_ratio), 3),
                                "num_faces_blurred": num_faces
                            }
                            approved_records.append(rec)
                            plate_counts[txt_clean] += 1
                            counter += 1
                            
                            print(f"  [+] #{counter:03d}/{target_count} PURE 1A (Unique: {len(plate_counts)}): {txt_clean} | 1A:{conf_1a:.2f} > 1:{conf_1:.2f} | AR:{ar:.2f} -> {crop_fn}", flush=True)
                            
                            if counter >= target_count:
                                break
                    except Exception:
                        pass
                        
    if approved_records:
        with open(METADATA_JSON, "w", encoding="utf-8") as f:
            json.dump(approved_records, f, indent=2, ensure_ascii=False)
            
    print(f"\n==========================================", flush=True)
    print(f"[+] TARGET QUOTA ACHIEVED! Total Verified Crops: {counter}, Unique Plates: {len(plate_counts)}", flush=True)
    print(f"==========================================", flush=True)

if __name__ == "__main__":
    run()
