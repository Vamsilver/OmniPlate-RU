import csv
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import requests

PROJECT_ROOT = Path(r"D:\AIProjects\VolgaIT")
sys.path.insert(0, str(PROJECT_ROOT))

from dataset.privacy.face_blur import FaceBlurrer
from src.pipeline.pipeline import OmniPlatePipeline

PLATE_REGEX = re.compile(r"^[ABEKMHOPCTYX#][\d#]{3}[ABEKMHOPCTYX#]{2}[\d#]{2,3}$")

OUT_DIR = PROJECT_ROOT / "test_output" / "new_1a_verified"
CROPS_DIR = OUT_DIR / "crops"
METADATA_JSON = OUT_DIR / "candidates_metadata.json"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CROPS_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
}

JDM_MODELS = [
    'toyota/chaser', 'toyota/mark_ii', 'toyota/cresta', 'toyota/crown',
    'toyota/caldina', 'toyota/bb', 'toyota/probox', 'toyota/succeed',
    'toyota/wish', 'toyota/vitz', 'toyota/allion', 'toyota/premio',
    'toyota/harrier', 'toyota/corolla_fielder', 'toyota/altezza',
    'toyota/soarer', 'toyota/aristo', 'toyota/alphard', 'toyota/ipsum',
    'toyota/carina', 'toyota/corona',
    'nissan/cube', 'nissan/stagea', 'nissan/skyline', 'nissan/laurel',
    'nissan/cefiro', 'nissan/wingroad', 'nissan/elgrand', 'nissan/serena',
    'nissan/silvia', 'subaru/forester', 'subaru/legacy', 'subaru/impreza_wrx_sti',
    'honda/stepwgn', 'honda/fit', 'honda/odyssey', 'honda/stream',
    'honda/airwave', 'mitsubishi/delica', 'mitsubishi/pajero',
    'mitsubishi/lancer_evolution', 'mitsubishi/legnum',
    'mazda/demio', 'mazda/rx_7', 'mazda/rx_8'
]

def fetch_model_photos(model_name: str, max_pages: int = 6) -> list:
    urls = []
    for page in range(1, max_pages + 1):
        catalog_url = f"https://www.drive2.ru/cars/{model_name}/?page={page}"
        try:
            r = requests.get(catalog_url, headers=HEADERS, timeout=7)
            if r.status_code != 200:
                continue
            car_links = re.findall(r'href="(/r/[a-zA-Z0-9_/-]+)"', r.text)
            for cl in car_links[:6]:
                car_url = f"https://www.drive2.ru{cl}"
                try:
                    r_car = requests.get(car_url, headers=HEADERS, timeout=7)
                    if r_car.status_code != 200:
                        continue
                    imgs = re.findall(r'https://[a-zA-Z0-9.-]+\.drive-data\.ru/[a-zA-Z0-9_-]+-(?:480|960|1920)\.jpg', r_car.text)
                    for u in imgs:
                        hi_res = re.sub(r'-(?:480|960)\.jpg', '-1920.jpg', u)
                        urls.append(hi_res)
                    
                    log_links = re.findall(r'href="(/l/[0-9]+/?)"', r_car.text)
                    for ll in log_links[:2]:
                        try:
                            r_log = requests.get(f"https://www.drive2.ru{ll}", headers=HEADERS, timeout=6)
                            if r_log.status_code == 200:
                                log_imgs = re.findall(r'https://[a-zA-Z0-9.-]+\.drive-data\.ru/[a-zA-Z0-9_-]+-(?:480|960|1920)\.jpg', r_log.text)
                                for lu in log_imgs:
                                    urls.append(re.sub(r'-(?:480|960)\.jpg', '-1920.jpg', lu))
                        except Exception:
                            pass
                except Exception:
                    pass
        except Exception:
            pass
    return urls

def download_image(url: str):
    try:
        r = requests.get(url, headers=HEADERS, timeout=9)
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
    seam_ratio = (top_e + bot_e) / (2.0 * max(1e-3, mid_e))
    return float(seam_ratio)

def main():
    target_count = 160
    
    seen_urls = set()
    existing_meta = []
    if METADATA_JSON.exists():
        try:
            with open(METADATA_JSON, "r", encoding="utf-8") as f:
                existing_meta = json.load(f)
                for item in existing_meta:
                    seen_urls.add(item.get("url"))
        except Exception:
            pass

    existing_crops = list(CROPS_DIR.glob("*.jpg"))
    counter = len(existing_crops)
    print(f"[*] Starting crop count: {counter}, Target: {target_count}")

    print(f"[*] Gathering photo links across {len(JDM_MODELS)} models...")
    all_urls = set()
    with ThreadPoolExecutor(max_workers=10) as pool:
        future_to_model = {pool.submit(fetch_model_photos, m, 6): m for m in JDM_MODELS}
        for fut in as_completed(future_to_model):
            m = future_to_model[fut]
            try:
                res = fut.result()
                all_urls.update(res)
                print(f"  + {m}: found {len(res)} URLs (cumulative unique: {len(all_urls)})")
            except Exception as e:
                print(f"  - {m} error: {e}")
                
    new_urls = [u for u in all_urls if u not in seen_urls]
    print(f"\n[+] Total unique new URLs to process: {len(new_urls)}")
    
    print("[*] Initializing AI Pipeline (RTX 5080) and FaceBlurrer...")
    pipeline = OmniPlatePipeline(device="cuda")
    blurrer = FaceBlurrer()
    
    batch_dl_size = 20
    approved_records = list(existing_meta)
    start_time = time.time()
    
    for i in range(0, len(new_urls), batch_dl_size):
        if counter >= target_count:
            print(f"[!] Target quota of {target_count} reached!")
            break
            
        chunk = new_urls[i : i + batch_dl_size]
        with ThreadPoolExecutor(max_workers=batch_dl_size) as dl_pool:
            results = list(dl_pool.map(download_image, chunk))
            
        for u, img in results:
            if img is None:
                continue
            h_im, w_im = img.shape[:2]
            if h_im < 250 or w_im < 250:
                continue
                
            try:
                # Detector pass
                res = pipeline.detector.predict(source=img, imgsz=640, conf=0.12, device='cuda', verbose=False)
                parsed = pipeline._parse_results(res[0], w_im, h_im)
                for det in parsed:
                    bx, by, bw, bh = det.bbox
                    ar = bw / max(1.0, bh)
                    if not (1.08 <= ar <= 1.85):
                        continue
                        
                    crop = img[max(0, by):min(h_im, by+bh), max(0, bx):min(w_im, bx+bw)]
                    if crop.size == 0 or crop.shape[0] < 18 or crop.shape[1] < 22:
                        continue
                        
                    # Seam ratio filter
                    seam_ratio = evaluate_seam_ratio(crop)
                    if seam_ratio < 1.05:
                        continue
                        
                    # 1A hypothesis
                    rect_1a = pipeline.rectifier.rectify(img, det.quad, plate_type="type1a")
                    top_l, bot_l = pipeline.rectifier.split_type1a(rect_1a)
                    stitched_1a = pipeline.rectifier.stitch_type1a_horizontal(top_l, bot_l)
                    txt_1a, conf_1a = pipeline.ocr.predict_single(stitched_1a, plate_type="type1a")
                    
                    # 1 direct hypothesis
                    rect_1 = pipeline.rectifier.rectify(img, det.quad, plate_type="type1")
                    txt_1, conf_1 = pipeline.ocr.predict_single(rect_1, plate_type="type1")[:2]
                    
                    txt_clean = txt_1a.strip().upper()
                    if not PLATE_REGEX.match(txt_clean) or txt_clean.count('#') > 1:
                        continue
                    if conf_1a < 0.65 or conf_1a <= conf_1:
                        continue
                        
                    # FaceBlurrer
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
                    seen_urls.add(u)
                    counter += 1
                    
                    print(f"[{counter:03d}/{target_count}] PURE 1A: {txt_clean} | 1A:{conf_1a:.2f} > 1:{conf_1:.2f} | AR:{ar:.2f} | Seam:{seam_ratio:.2f} | Faces:{num_faces} -> {crop_fn}")
                    
                    if counter >= target_count:
                        break
            except Exception as ex:
                pass
                
        if len(approved_records) % 5 == 0:
            with open(METADATA_JSON, "w", encoding="utf-8") as f:
                json.dump(approved_records, f, indent=2, ensure_ascii=False)
                
    with open(METADATA_JSON, "w", encoding="utf-8") as f:
        json.dump(approved_records, f, indent=2, ensure_ascii=False)
        
    elapsed = time.time() - start_time
    print(f"\n[+] Harvesting complete in {elapsed:.1f}s. Total candidates: {counter}")

if __name__ == "__main__":
    main()
