#!/usr/bin/env python3
"""
OmniPlate-RU Turbo JDM Harvester (Full Streaming Producer-Consumer Pipeline).

All stages run concurrently in real time:
Stage 1 (Catalog Streamer): 16 paced workers sweep 200+ models, immediately pushing cars to cars_queue.
Stage 2 (Photo Streamer): 16 workers pull cars, fetch HTML, immediately pushing 1920px CDN URLs to photo_queue.
Stage 3 (CDN Streamer): 32 parallel threads download images from CDN directly into RAM.
Stage 4 (RTX 5080 Consumer):
  - YOLO-Pose detector (CUDA ONNX)
  - Aspect Ratio filter: AR in [1.08, 1.85]
  - Sobel_Y SeamRatio: >= 1.05 (detecting two-row horizontal division)
  - Dual-branch OCR: LPRNet 1A vs Type 1 arbitration (conf_1a >= 0.55 and conf_1a > conf_1)
  - Strict GOST 1A regex validation
  - FaceBlurrer (YuNet ONNX) privacy protection
  - Diversity cap: max 3 images per unique plate
"""

import csv
import json
import os
import queue
import random
import re
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.privacy.face_blur import FaceBlurrer
from src.pipeline.pipeline import OmniPlatePipeline

PLATE_REGEX = re.compile(r"^[ABEKMHOPCTYX][\d]{3}[ABEKMHOPCTYX]{2}(?:\d{2}|[127]\d{2})$")

OUT_DIR = PROJECT_ROOT / "test_output" / "turbo_1a_verified"
CROPS_DIR = OUT_DIR / "crops"
METADATA_JSON = OUT_DIR / "candidates_metadata.json"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CROPS_DIR.mkdir(parents=True, exist_ok=True)

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:129.0) Gecko/20100101 Firefox/129.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36 Edg/127.0.0.0',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:128.0) Gecko/20100101 Firefox/128.0'
]

def make_headers(idx: int = 0):
    return {
        'User-Agent': USER_AGENTS[idx % len(USER_AGENTS)],
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
        'Sec-Ch-Ua': '"Chromium";v="128", "Not;A=Brand";v="24"',
        'Sec-Ch-Ua-Mobile': '?0',
        'Sec-Ch-Ua-Platform': '"Windows"',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1',
    }

# 250+ Models with high likelihood of 2-row square plates
EXPANDED_MODELS = [
    # Suzuki Kei & 4x4
    'suzuki/jimny', 'suzuki/jimny_sierra', 'suzuki/wagon_r', 'suzuki/hustler',
    'suzuki/alto', 'suzuki/kei', 'suzuki/every', 'suzuki/solio', 'suzuki/spacia', 'suzuki/carry',
    'suzuki/escudo', 'suzuki/grand_vitara', 'suzuki/sx4', 'suzuki/swift', 'suzuki/ignis',
    'suzuki/cappuccino', 'suzuki/twin', 'suzuki/palette', 'suzuki/cervo',
    # Daihatsu Kei & Compact
    'daihatsu/terios', 'daihatsu/terios_kid', 'daihatsu/mira', 'daihatsu/move',
    'daihatsu/tanto', 'daihatsu/copen', 'daihatsu/hijet', 'daihatsu/boon',
    'daihatsu/rocky', 'daihatsu/yrv', 'daihatsu/atrai', 'daihatsu/cast', 'daihatsu/taft',
    'daihatsu/naked', 'daihatsu/mira_cocoa', 'daihatsu/mira_e_s', 'daihatsu/sonica', 'daihatsu/wake', 'daihatsu/thor',
    # Mitsubishi 4x4 & Kei
    'mitsubishi/pajero_mini', 'mitsubishi/pajero_io', 'mitsubishi/pajero_junior',
    'mitsubishi/delica', 'mitsubishi/delica_d_5', 'mitsubishi/delica_d_2', 'mitsubishi/ek_wagon',
    'mitsubishi/ek_custom', 'mitsubishi/ek_space', 'mitsubishi/rvr',
    'mitsubishi/chariot_grandis', 'mitsubishi/colt', 'mitsubishi/pajero',
    'mitsubishi/lancer_evolution', 'mitsubishi/legnum', 'mitsubishi/galant',
    'mitsubishi/airtrek', 'mitsubishi/outlander', 'mitsubishi/eclipse_cross',
    'mitsubishi/minica', 'mitsubishi/toppo', 'mitsubishi/town_box', 'mitsubishi/bravo',
    'mitsubishi/gto', 'mitsubishi/fto', 'mitsubishi/mirage', 'mitsubishi/diamante',
    # Subaru Kei & JDM
    'subaru/sambar', 'subaru/pleo', 'subaru/r2', 'subaru/r1', 'subaru/stella',
    'subaru/vivio', 'subaru/domingo', 'subaru/lucra', 'subaru/dex', 'subaru/trezia',
    'subaru/levorg', 'subaru/exiga', 'subaru/forester',
    'subaru/legacy', 'subaru/legacy_b4', 'subaru/impreza_wrx_sti', 'subaru/impreza',
    'subaru/outback', 'subaru/crosstrek', 'subaru/brz', 'subaru/alcyone_svx',
    # Honda Micro, Kei & MPV
    'honda/n_box', 'honda/n_wgn', 'honda/n_one', 'honda/n_van', 'honda/stepwgn',
    'honda/freed', 'honda/freed_spike', 'honda/mobilio', 'honda/mobilio_spike',
    'honda/edix', 'honda/airwave', 'honda/elysion', 'honda/crossroad', 'honda/element',
    'honda/fit', 'honda/fit_shuttle', 'honda/odyssey', 'honda/stream',
    'honda/shuttle', 'honda/cr_v', 'honda/vezel', 'honda/insight', 'honda/cr_z',
    'honda/accord', 'honda/civic', 'honda/integra', 'honda/prelude', 'honda/s2000',
    'honda/legend', 'honda/inspire', 'honda/saber', 'honda/torneo',
    'honda/sm_x', 'honda/life', 'honda/thats', 'honda/zest', 'honda/s660', 'honda/beat',
    'honda/hr_v', 'honda/avancier', 'honda/orthia', 'honda/capa', 'honda/logo',
    # Nissan Kei, Cubes & MPVs
    'nissan/cube', 'nissan/moco', 'nissan/dayz', 'nissan/dayz_roox', 'nissan/roox', 'nissan/clipper',
    'nissan/stagea', 'nissan/wingroad', 'nissan/elgrand', 'nissan/serena',
    'nissan/r_nessa', 'nissan/bassara', 'nissan/avenir', 'nissan/prairie', 'nissan/nv200',
    'nissan/skyline', 'nissan/laurel', 'nissan/cefiro', 'nissan/cedric',
    'nissan/gloria', 'nissan/silvia', 'nissan/180sx', 'nissan/fairlady_z',
    'nissan/presage', 'nissan/march', 'nissan/tiida', 'nissan/note',
    'nissan/expert', 'nissan/ad', 'nissan/teana', 'nissan/fuga',
    'nissan/dualis', 'nissan/juke', 'nissan/x_trail', 'nissan/safari',
    'nissan/cima', 'nissan/president', 'nissan/caravan', 'nissan/rasheen',
    'nissan/pao', 'nissan/figaro', 'nissan/be_1', 'nissan/nv100', 'nissan/nv350',
    'nissan/bluebird_sylphy', 'nissan/bluebird', 'nissan/pulsar', 'nissan/terrano',
    # Toyota Vans, Boxes, Kei & JDM
    'toyota/bb', 'toyota/funcargo', 'toyota/sienta', 'toyota/roomy',
    'toyota/tank', 'toyota/porte', 'toyota/spade', 'toyota/ractis',
    'toyota/passo', 'toyota/rush', 'toyota/noah', 'toyota/voxy',
    'toyota/vellfire', 'toyota/alphard', 'toyota/hiace', 'toyota/estima',
    'toyota/ipsum', 'toyota/nadia', 'toyota/gaia', 'toyota/opa',
    'toyota/corolla_rumion', 'toyota/will_cypha', 'toyota/will_vi', 'toyota/will_vs',
    'toyota/probox', 'toyota/succeed', 'toyota/lite_ace', 'toyota/town_ace',
    'toyota/mark_ii', 'toyota/chaser', 'toyota/cresta', 'toyota/verossa',
    'toyota/mark_x', 'toyota/crown', 'toyota/crown_majesta', 'toyota/aristo',
    'toyota/celsior', 'toyota/century', 'toyota/altezza', 'toyota/soarer',
    'toyota/caldina', 'toyota/allion', 'toyota/premio', 'toyota/harrier',
    'toyota/corolla_fielder', 'toyota/vitz', 'toyota/rav4', 'toyota/land_cruiser_prado',
    'toyota/land_cruiser', 'toyota/hilux_surf', 'toyota/carina', 'toyota/corona',
    'toyota/auris', 'toyota/prius', 'toyota/c_hr', 'toyota/mark_x_zio',
    'toyota/vanguard', 'toyota/isis', 'toyota/wish', 'toyota/esquire',
    'toyota/mr2', 'toyota/mr_s', 'toyota/curren', 'toyota/sprinter_trueno',
    'toyota/corolla_levin', 'toyota/belta', 'toyota/blade', 'toyota/sai',
    'toyota/pixis_epoch', 'toyota/pixis_space', 'toyota/pixis_mega',
    # Mazda JDM
    'mazda/rx_7', 'mazda/rx_8', 'mazda/bongo_friendee', 'mazda/bongo', 'mazda/biante',
    'mazda/mpv', 'mazda/roadster', 'mazda/demio', 'mazda/verisa',
    'mazda/axela', 'mazda/atenza', 'mazda/cx_5', 'mazda/cx_3', 'mazda/premacy',
    'mazda/scrum', 'mazda/carol', 'mazda/az_wagon', 'mazda/az_1', 'mazda/flare',
    'mazda/familia', 'mazda/capella', 'mazda/luce', 'mazda/sentia', 'mazda/millenia',
    # Isuzu JDM
    'isuzu/bighorn', 'isuzu/wizard', 'isuzu/vehicross', 'isuzu/mu', 'isuzu/gemini',
    # USDM square plate models (Trucks, SUVs & Muscle)
    'ford/mustang', 'ford/f150', 'ford/f250', 'ford/bronco', 'ford/explorer',
    'ford/expedition', 'ford/crown_victoria',
    'dodge/challenger', 'dodge/charger', 'dodge/durango', 'dodge/nitro',
    'dodge/magnum', 'dodge/ram', 'dodge/dakota', 'dodge/viper', 'dodge/caravan',
    'chevrolet/camaro', 'chevrolet/corvette', 'chevrolet/tahoe', 'chevrolet/suburban',
    'chevrolet/silverado', 'chevrolet/avalanche', 'chevrolet/blazer', 'chevrolet/trailblazer',
    'chevrolet/caprice', 'chevrolet/impala', 'chevrolet/express',
    'gmc/yukon', 'gmc/sierra', 'gmc/savana',
    'cadillac/escalade', 'cadillac/cts', 'cadillac/srx', 'cadillac/sts', 'cadillac/de_ville',
    'hummer/h2', 'hummer/h3', 'hummer/h1',
    'lincoln/town_car', 'lincoln/navigator', 'lincoln/aviator',
    'jeep/wrangler', 'jeep/grand_cherokee', 'jeep/cherokee', 'jeep/commander',
    'toyota/tundra', 'toyota/tacoma', 'ram/1500', 'chrysler/300c', 'chrysler/pt_cruiser'
]


def evaluate_seam_ratio(crop: np.ndarray) -> float:
    """Calculates horizontal energy ratio between text rows and center divider."""
    if crop is None or crop.size == 0 or crop.shape[0] < 16 or crop.shape[1] < 16:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sob_y = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    norm_sob = cv2.resize(sob_y, (100, 100))
    p100 = np.mean(norm_sob, axis=1)
    mid_e = np.mean(p100[42:54])
    top_e = np.mean(p100[18:36])
    bot_e = np.mean(p100[60:78])
    seam_ratio = (top_e + bot_e) / (2.0 * max(1e-3, mid_e))
    return float(seam_ratio)


class TurboJDMHarvester:
    def __init__(self, target_candidates: int = 170, download_workers: int = 32):
        self.target_candidates = target_candidates
        self.download_workers = download_workers
        
        self.cars_queue: queue.Queue = queue.Queue(maxsize=40000)
        self.photo_urls_queue: queue.Queue = queue.Queue(maxsize=50000)
        self.downloaded_queue: queue.Queue = queue.Queue(maxsize=500)
        
        self.seen_photos: Set[str] = set()
        self.seen_cars: Set[str] = set()
        self.seen_lock = threading.Lock()
        
        self.stop_event = threading.Event()
        self.plate_counts: Dict[str, int] = defaultdict(int)
        self.approved_records: List[Dict] = []
        
        if METADATA_JSON.exists():
            try:
                with open(METADATA_JSON, "r", encoding="utf-8") as f:
                    self.approved_records = json.load(f)
                for rec in self.approved_records:
                    self.plate_counts[rec["text"]] += 1
            except Exception:
                pass

        existing_crops = list(CROPS_DIR.glob("*.jpg"))
        for c in existing_crops:
            stem = c.stem.replace("_crop", "")
            parts = stem.split("_")
            if len(parts) >= 4:
                self.plate_counts[parts[3]] += 1
            elif len(parts) == 3:
                self.plate_counts[parts[2]] += 1
                
        self.counter = max(len(existing_crops), len(self.approved_records))
        print(f"[*] Initialized TurboHarvester:")
        print(f"    - Existing candidates on disk: {self.counter}")
        print(f"    - Target total candidates in turbo: {self.target_candidates}")
        print(f"    - Unique plates so far: {len(self.plate_counts)}")
        print(f"    - Download workers: {self.download_workers}", flush=True)

    def producer_catalog(self, models_slice: List[str], worker_id: int):
        """Scrapes Drive2 catalog pages and immediately streams cars into queue."""
        s = requests.Session()
        s.headers.update(make_headers(worker_id))
        
        for model in models_slice:
            if self.stop_event.is_set() or self.counter >= self.target_candidates:
                break
                
            clean_m = model.strip()
            if not clean_m.startswith("/"):
                clean_m = "/" + clean_m
            if not clean_m.endswith("/"):
                clean_m = clean_m + "/"
            if not clean_m.startswith("/cars/"):
                clean_m = "/cars" + clean_m

            for page in range(1, 10):
                if self.stop_event.is_set() or self.counter >= self.target_candidates:
                    break
                    
                url = f"https://www.drive2.ru{clean_m}?page={page}"
                found_cars = 0
                for attempt in range(2):
                    try:
                        r = s.get(url, timeout=5)
                        if r.status_code == 200:
                            cars = re.findall(r'href="(/r/[a-zA-Z0-9_/-]+)"', r.text)
                            found_cars = len(cars)
                            for c in cars:
                                if c.count('/') >= 4:
                                    with self.seen_lock:
                                        if c not in self.seen_cars:
                                            self.seen_cars.add(c)
                                            self.cars_queue.put(c)
                            break
                        elif r.status_code == 429:
                            time.sleep(4.0)
                    except Exception:
                        time.sleep(1.0)
                if found_cars == 0:
                    break
                time.sleep(0.30)

    def producer_car_photos(self, worker_id: int):
        """Worker thread to fetch photos for individual cars with pacing and retry."""
        s = requests.Session()
        s.headers.update(make_headers(worker_id))
        
        while not self.stop_event.is_set() and self.counter < self.target_candidates:
            try:
                car_path = self.cars_queue.get(timeout=2.0)
            except queue.Empty:
                continue
                
            urls_to_check = [f"https://www.drive2.ru{car_path}"]
            if random.random() < 0.5:
                urls_to_check.append(f"https://www.drive2.ru{car_path}/logbook/")

            for target_url in urls_to_check:
                if self.stop_event.is_set() or self.counter >= self.target_candidates:
                    break
                for attempt in range(2):
                    try:
                        r = s.get(target_url, timeout=6)
                        if r.status_code == 200:
                            imgs = re.findall(r'https://[a-zA-Z0-9.-]+\.drive-data\.ru/[a-zA-Z0-9_-]+-(?:480|960|1920)\.jpg', r.text)
                            for u in imgs:
                                hi_res = re.sub(r'-(?:480|960)\.jpg', '-1920.jpg', u)
                                with self.seen_lock:
                                    if hi_res not in self.seen_photos:
                                        self.seen_photos.add(hi_res)
                                        try:
                                            self.photo_urls_queue.put(hi_res, timeout=0.5)
                                        except queue.Full:
                                            pass
                            break
                        elif r.status_code == 429:
                            time.sleep(4.0)
                    except Exception:
                        time.sleep(1.0)
                time.sleep(0.18)

            self.cars_queue.task_done()
            time.sleep(0.10)

    def consumer_download(self):
        """Worker thread to download images from CDN into RAM."""
        s = requests.Session()
        s.headers.update(make_headers(0))
        
        while not self.stop_event.is_set() and self.counter < self.target_candidates:
            try:
                url = self.photo_urls_queue.get(timeout=2.0)
            except queue.Empty:
                continue
                
            try:
                r = s.get(url, timeout=6)
                if r.status_code == 200 and len(r.content) > 15000:
                    arr = np.asarray(bytearray(r.content), dtype=np.uint8)
                    im = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if im is not None and im.shape[0] >= 280 and im.shape[1] >= 280:
                        while not self.stop_event.is_set():
                            try:
                                self.downloaded_queue.put((url, im), timeout=1.0)
                                break
                            except queue.Full:
                                continue
            except Exception:
                pass
            finally:
                self.photo_urls_queue.task_done()

    def run(self):
        # Load discovered models if available, fallback to EXPANDED_MODELS
        disc_file = PROJECT_ROOT / "test_output" / "discovered_models.txt"
        if disc_file.exists():
            with open(disc_file, "r", encoding="utf-8") as f:
                models_list = [line.strip() for line in f if line.strip()]
            print(f"[*] Loaded {len(models_list)} exact discovered models from {disc_file.name}", flush=True)
        else:
            models_list = list(EXPANDED_MODELS)

        models_shuffled = list(models_list)
        random.seed(int(time.time()))
        random.shuffle(models_shuffled)

        # 1. Start Catalog Streamers (4 workers)
        num_cat_workers = 4
        chunk_size = max(1, len(models_shuffled) // num_cat_workers + 1)
        cat_threads = []
        for i in range(num_cat_workers):
            m_slice = models_shuffled[i * chunk_size : (i + 1) * chunk_size]
            if not m_slice:
                continue
            t = threading.Thread(target=self.producer_catalog, args=(m_slice, i), daemon=True)
            t.start()
            cat_threads.append(t)

        # 2. Start Car Photo Producers (8 workers)
        photo_producers = []
        for i in range(8):
            t = threading.Thread(target=self.producer_car_photos, args=(i+10,), daemon=True)
            t.start()
            photo_producers.append(t)

        # 3. Start Download Workers (32 parallel CDN threads)
        download_workers = []
        for _ in range(self.download_workers):
            t = threading.Thread(target=self.consumer_download, daemon=True)
            t.start()
            download_workers.append(t)

        # 4. Initialize AI Pipeline on RTX 5080
        print("[*] Initializing OmniPlatePipeline (RTX 5080 CUDA) & FaceBlurrer...", flush=True)
        pipeline = OmniPlatePipeline(device="cuda")
        blurrer = FaceBlurrer()
        print("[*] Pipeline ready! GPU processing started...", flush=True)

        t_start = time.time()
        processed_images = 0

        try:
            while self.counter < self.target_candidates and not self.stop_event.is_set():
                try:
                    item = self.downloaded_queue.get(timeout=5.0)
                except queue.Empty:
                    cats_alive = any(t.is_alive() for t in cat_threads)
                    if not cats_alive and self.cars_queue.empty() and self.photo_urls_queue.empty() and self.downloaded_queue.empty():
                        print("[*] All queues empty and processing complete.", flush=True)
                        break
                    continue

                url, img = item
                processed_images += 1
                h_im, w_im = img.shape[:2]

                try:
                    res = pipeline.detector.predict(source=img, imgsz=640, conf=0.10, device='cuda', verbose=False)
                    parsed = pipeline._parse_results(res[0], w_im, h_im)

                    for det in parsed:
                        bx, by, bw, bh = det.bbox
                        ar = bw / max(1.0, bh)

                        if not (1.08 <= ar <= 1.85):
                            continue

                        crop = img[max(0, by):min(h_im, by+bh), max(0, bx):min(w_im, bx+bw)]
                        if crop.size == 0 or crop.shape[0] < 18 or crop.shape[1] < 22:
                            continue

                        seam_ratio = evaluate_seam_ratio(crop)
                        if seam_ratio < 1.05:
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

                        if conf_1a < 0.55 or conf_1a <= conf_1:
                            continue

                        if self.plate_counts[txt_clean] >= 3:
                            continue

                        blurred_full, num_faces = blurrer.process_image(img)

                        cid = f"cand_1a_{self.counter:04d}_{txt_clean}"
                        full_fn = f"{cid}.jpg"
                        crop_fn = f"{cid}_crop.jpg"

                        cv2.imwrite(str(OUT_DIR / full_fn), blurred_full, [cv2.IMWRITE_JPEG_QUALITY, 94])
                        cv2.imwrite(str(CROPS_DIR / crop_fn), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

                        rec = {
                            "id": self.counter,
                            "full_fn": full_fn,
                            "crop_fn": crop_fn,
                            "url": url,
                            "text": txt_clean,
                            "conf_1a": float(conf_1a),
                            "conf_1": float(conf_1),
                            "txt_1": txt_1,
                            "bbox": [int(bx), int(by), int(bw), int(bh)],
                            "quad": [round(float(x), 2) for x in det.quad],
                            "ar": round(float(ar), 3),
                            "seam_ratio": round(float(seam_ratio), 3),
                            "num_faces_blurred": num_faces,
                        }
                        self.approved_records.append(rec)
                        self.plate_counts[txt_clean] += 1
                        self.counter += 1

                        rate = processed_images / max(1.0, time.time() - t_start)
                        print(f"  [+] #{self.counter:03d}/{self.target_candidates} (Unique: {len(self.plate_counts)}) | "
                              f"{txt_clean} | 1A:{conf_1a:.2f} > 1:{conf_1:.2f} | AR:{ar:.2f} | Seam:{seam_ratio:.2f} | "
                              f"{rate:.1f} img/s | Cars:{self.cars_queue.qsize()} | P:{self.photo_urls_queue.qsize()}", flush=True)

                        if self.counter >= self.target_candidates:
                            self.stop_event.set()
                            break

                except Exception:
                    pass

                if processed_images % 50 == 0:
                    dt = time.time() - t_start
                    rate = processed_images / max(1.0, dt)
                    print(f"  --> [Prog] Processed {processed_images} imgs | Found: {self.counter} 1A | Cars left: {self.cars_queue.qsize()} | P:{self.photo_urls_queue.qsize()} | Rate: {rate:.1f} img/s", flush=True)

                if self.counter % 5 == 0 and self.approved_records:
                    with open(METADATA_JSON, "w", encoding="utf-8") as f:
                        json.dump(self.approved_records, f, indent=2, ensure_ascii=False)

        finally:
            self.stop_event.set()
            if self.approved_records:
                with open(METADATA_JSON, "w", encoding="utf-8") as f:
                    json.dump(self.approved_records, f, indent=2, ensure_ascii=False)

        total_time = time.time() - t_start
        print(f"\n=======================================================", flush=True)
        print(f"[+] Turbo Harvest Done in {total_time:.1f}s!", flush=True)
        print(f"    - Processed images: {processed_images}")
        print(f"    - Verified Type 1A Candidates: {self.counter}")
        print(f"    - Unique License Plates: {len(self.plate_counts)}")
        print(f"=======================================================", flush=True)


if __name__ == "__main__":
    harvester = TurboJDMHarvester(target_candidates=170, download_workers=32)
    harvester.run()
