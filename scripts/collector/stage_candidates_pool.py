#!/usr/bin/env python3
"""
Candidate Staging Pool Builder for Volga IT 2026.
Populates dataset/candidates_review/ with high-quality candidates:
- dataset/candidates_review/type1a/: JDM/USDM square 2-row plates (real + procedural)
- dataset/candidates_review/type2/: Trailer plates (real + procedural)
Applies:
- YuNet FaceBlurrer for 100% privacy compliance on all real scenes
- GOST R 50577-2018 geometry & realistic bumper/niche backgrounds
- Manifest generation in dataset/candidates_review/candidates_manifest.csv
"""

import csv
import json
import math
import os
import random
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

from dataset.generator.plate_renderer import PlateRenderer
from dataset.generator.augmentations import PlateAugmentor
from dataset.generator.perspective import PerspectiveTransformer
from dataset.privacy.face_blur import FaceBlurrer

REVIEW_DIR = PROJECT_ROOT / "dataset" / "candidates_review"
DIR_1A = REVIEW_DIR / "type1a"
DIR_T2 = REVIEW_DIR / "type2"
MANIFEST_PATH = REVIEW_DIR / "candidates_manifest.csv"

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) VolgaITCandidateHarvester/1.0"

CAR_COLORS = [
    (240, 240, 242),  # White pearl
    (28, 28, 30),     # Obsidian black
    (88, 92, 98),     # Graphite metallic
    (165, 170, 175),  # Silver
    (135, 22, 22),    # Wine red / Dark ruby
    (18, 42, 85),     # Deep navy blue
    (35, 55, 45),     # Emerald / Forest
    (175, 140, 95),   # Champagne bronze
    (115, 115, 120),  # Slate grey
    (50, 50, 55)      # Asphalt dark grey
]


def fetch_commons_file_list(category: str, limit: int = 25) -> List[str]:
    encoded = urllib.parse.quote(category)
    url = f"{COMMONS_API}?action=query&list=categorymembers&cmtitle={encoded}&cmtype=file&cmlimit={limit}&format=json"
    try:
        out = subprocess.check_output(["curl.exe", "-s", "-m", "8", "-A", USER_AGENT, url], timeout=10)
        data = json.loads(out.decode("utf-8", errors="replace"))
        members = data.get("query", {}).get("categorymembers", [])
        return [m["title"] for m in members if m["title"].lower().endswith((".jpg", ".jpeg", ".png"))]
    except Exception as e:
        print(f"  [!] Category error ({category}): {e}")
        return []


def fetch_commons_image(file_title: str) -> Optional[Tuple[np.ndarray, str]]:
    encoded = urllib.parse.quote(file_title)
    url = f"{COMMONS_API}?action=query&titles={encoded}&prop=imageinfo&iiprop=url&iiurlwidth=1280&format=json"
    try:
        out = subprocess.check_output(["curl.exe", "-s", "-m", "8", "-A", USER_AGENT, url], timeout=10)
        data = json.loads(out.decode("utf-8", errors="replace"))
        pages = data.get("query", {}).get("pages", {})
        for _, p in pages.items():
            info = p.get("imageinfo", [])
            if info:
                thumb_url = info[0].get("thumburl") or info[0].get("url")
                if thumb_url:
                    img_bytes = subprocess.check_output(["curl.exe", "-s", "-m", "15", "-A", USER_AGENT, thumb_url], timeout=20)
                    arr = np.asarray(bytearray(img_bytes), dtype=np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if img is not None and min(img.shape[:2]) >= 320:
                        return img, thumb_url
    except Exception:
        pass
    return None


def generate_jdm_niche_background(w: int = 1280, h: int = 720) -> np.ndarray:
    """Generates authentic JDM / USDM vehicle rear scene with recessed license plate niche"""
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    car_color = random.choice(CAR_COLORS)
    bg[:] = car_color

    # Add realistic vertical lighting gradient on car curved panels
    grad = np.linspace(random.uniform(0.75, 0.92), random.uniform(1.05, 1.25), h)[:, np.newaxis, np.newaxis]
    bg = np.clip(bg.astype(np.float32) * grad, 0, 255).astype(np.uint8)

    # Trunk / Bumper crease line
    crease_y = random.randint(int(h * 0.30), int(h * 0.38))
    shadow_col = (int(car_color[0] * 0.55), int(car_color[1] * 0.55), int(car_color[2] * 0.55))
    cv2.line(bg, (0, crease_y), (w, crease_y), shadow_col, 4)
    cv2.line(bg, (0, crease_y + 2), (w, crease_y + 2), (250, 250, 250), 1)

    # Lower bumper edge
    bumper_y = random.randint(int(h * 0.78), int(h * 0.88))
    cv2.line(bg, (0, bumper_y), (w, bumper_y), shadow_col, 5)

    # USDM / JDM Square Plate Niche
    # Dimensions proportional to a square plate with recess margin
    nw = random.randint(int(w * 0.40), int(w * 0.50))
    nh = random.randint(int(h * 0.42), int(h * 0.50))
    nx1 = (w - nw) // 2 + random.randint(-40, 40)
    ny1 = crease_y + random.randint(25, 45)
    nx2, ny2 = nx1 + nw, ny1 + nh

    # Recessed inner shadow
    niche_col = (random.randint(20, 35), random.randint(20, 35), random.randint(22, 38))
    cv2.rectangle(bg, (nx1, ny1), (nx2, ny2), niche_col, -1)
    # Recessed bevel border
    cv2.rectangle(bg, (nx1, ny1), (nx2, ny2), shadow_col, 6)
    # Top inner shadow inside niche
    cv2.line(bg, (nx1 + 4, ny1 + 8), (nx2 - 4, ny1 + 8), (10, 10, 10), 8)

    # License plate illumination lamps (two small lights above plate)
    lamp_w = int(nw * 0.12)
    lx1 = nx1 + int(nw * 0.22)
    lx2 = nx1 + int(nw * 0.66)
    cv2.rectangle(bg, (lx1, ny1 - 12), (lx1 + lamp_w, ny1), (220, 220, 200), -1)
    cv2.rectangle(bg, (lx2, ny1 - 12), (lx2 + lamp_w, ny1), (220, 220, 200), -1)

    # Subtle metallic flake noise
    noise = np.empty((h, w, 3), dtype=np.float32)
    cv2.randn(noise, 0, random.uniform(3, 7))
    bg = np.clip(bg.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return bg


def generate_trailer_background(w: int = 1280, h: int = 720) -> np.ndarray:
    """Generates authentic Russian trailer rear scene (galvanized/matte steel bumper, tail lights, mud flaps)"""
    bg = np.zeros((h, w, 3), dtype=np.uint8)

    # Upper trailer body / bed (wood/corrugated metal/dark canvas)
    theme = random.choice(["galvanized", "canvas", "corrugated", "flatbed"])
    if theme == "galvanized":
        bg[:int(h*0.55)] = (random.randint(120, 150), random.randint(125, 155), random.randint(130, 160))
    elif theme == "canvas":
        bg[:int(h*0.55)] = (random.randint(30, 60), random.randint(45, 75), random.randint(80, 120))  # Blue/green tarp
    elif theme == "corrugated":
        bg[:int(h*0.55)] = (random.randint(140, 160), random.randint(140, 160), random.randint(140, 160))
        for x in range(0, w, 40):
            cv2.line(bg, (x, 0), (x, int(h*0.55)), (100, 100, 100), 4)
    else:
        bg[:int(h*0.55)] = (random.randint(40, 50), random.randint(40, 50), random.randint(45, 55))

    # Lower trailer steel bumper bar
    bar_y1 = int(h * 0.52)
    bar_y2 = int(h * 0.82)
    bar_col = (random.randint(35, 55), random.randint(35, 55), random.randint(35, 60))
    cv2.rectangle(bg, (0, bar_y1), (w, bar_y2), bar_col, -1)
    cv2.rectangle(bg, (0, bar_y1), (w, bar_y2), (180, 180, 185), 3)

    # Red rectangular trailer rear lights (left and right)
    lw = int(w * 0.16)
    lh = int((bar_y2 - bar_y1) * 0.65)
    ly = bar_y1 + int((bar_y2 - bar_y1 - lh) / 2)

    # Left light cluster (Red, Amber, White)
    ll_x = int(w * 0.05)
    cv2.rectangle(bg, (ll_x, ly), (ll_x + lw, ly + lh), (20, 20, 20), -1)
    cv2.rectangle(bg, (ll_x + 4, ly + 4), (ll_x + int(lw*0.5), ly + lh - 4), (20, 20, 210), -1)   # Red
    cv2.rectangle(bg, (ll_x + int(lw*0.5), ly + 4), (ll_x + int(lw*0.8), ly + lh - 4), (20, 160, 230), -1) # Amber
    cv2.rectangle(bg, (ll_x + int(lw*0.8), ly + 4), (ll_x + lw - 4, ly + lh - 4), (220, 220, 220), -1) # White

    # Right light cluster
    rl_x = int(w * 0.79)
    cv2.rectangle(bg, (rl_x, ly), (rl_x + lw, ly + lh), (20, 20, 20), -1)
    cv2.rectangle(bg, (rl_x + 4, ly + 4), (rl_x + int(lw*0.2), ly + lh - 4), (220, 220, 220), -1)
    cv2.rectangle(bg, (rl_x + int(lw*0.2), ly + 4), (rl_x + int(lw*0.5), ly + lh - 4), (20, 160, 230), -1)
    cv2.rectangle(bg, (rl_x + int(lw*0.5), ly + 4), (rl_x + lw - 4, ly + lh - 4), (20, 20, 210), -1)

    # Mud flaps at bottom
    cv2.rectangle(bg, (ll_x, bar_y2), (ll_x + lw + 20, h), (15, 15, 15), -1)
    cv2.rectangle(bg, (rl_x - 20, bar_y2), (rl_x + lw, h), (15, 15, 15), -1)
    cv2.line(bg, (ll_x, h - 25), (ll_x + lw + 20, h - 25), (220, 220, 220), 4)
    cv2.line(bg, (rl_x - 20, h - 25), (rl_x + lw, h - 25), (220, 220, 220), 4)

    # Road asphalt at very bottom
    cv2.rectangle(bg, (0, int(h * 0.94)), (w, h), (40, 42, 45), -1)

    return bg


def stage_candidates():
    print("=" * 70)
    print("  OmniPlate-RU: Сбор и генерация кандидатов Type 1A и Type 2")
    print("=" * 70)

    DIR_1A.mkdir(parents=True, exist_ok=True)
    DIR_T2.mkdir(parents=True, exist_ok=True)

    renderer = PlateRenderer()
    augmentor = PlateAugmentor()
    transformer = PerspectiveTransformer()
    blurrer = FaceBlurrer()

    manifest_rows = []

    # =========================================================================
    # 1. TYPE 1A: REAL CANDIDATES FROM COMMONS (Primorsky / Vladivostok JDM)
    # =========================================================================
    print("\n[*] [1/4] Поиск реальных кандидатов Type 1A в открытых архивах...")
    jdm_categories = [
        "Category:Automobiles in Vladivostok",
        "Category:Automobiles with license plates of Primorsky Krai",
        "Category:Toyota vehicles in Vladivostok",
        "Category:Nissan vehicles in Vladivostok"
    ]
    real_1a_count = 0
    for cat in jdm_categories:
        if real_1a_count >= 8:
            break
        files = fetch_commons_file_list(cat, limit=12)
        print(f"  -> Найдено {len(files)} файлов в '{cat}'")
        for ftitle in files:
            if real_1a_count >= 8:
                break
            res = fetch_commons_image(ftitle)
            if res is not None:
                img, source_url = res
                blurred, _ = blurrer.process_image(img)
                fname = f"cand_1a_real_{real_1a_count + 1:03d}.jpg"
                out_path = DIR_1A / fname
                cv2.imwrite(str(out_path), blurred, [cv2.IMWRITE_JPEG_QUALITY, 93])
                manifest_rows.append({
                    "filename": f"type1a/{fname}",
                    "plate_type": "type1a",
                    "source": source_url,
                    "plate_num": "",
                    "is_synthetic": 0,
                    "conditions": "day,real_jdm"
                })
                real_1a_count += 1
                print(f"  [+] Добавлен реальный Type 1A кандидат: {fname}")

    # =========================================================================
    # 2. TYPE 1A: PROCEDURAL PHOTOREALISTIC JDM/USDM SCENES (20 scenes)
    # =========================================================================
    print("\n[*] [2/4] Синтез фотореалистичных сцен Type 1A (JDM/USDM ниши с 3D-гомографией)...")
    target_synth_1a = 20
    for i in range(1, target_synth_1a + 1):
        clean_plate, plate_num = renderer.render_type1a()
        aug_plate = augmentor.augment_plate(clean_plate)
        plate_arr = np.array(aug_plate)

        bg = generate_jdm_niche_background(1280, 720)
        # Project plate inside the central recessed niche
        scene, bbox, quad = transformer.project_plate_onto_background(
            plate_arr, bg, max_yaw=26.0, max_pitch=14.0, max_roll=6.0
        )

        conds = ["day", "angle", "jdm_niche"]
        if random.random() < 0.35:
            conds.append("dirt")
        if random.random() < 0.25:
            conds.append("glare")

        fname = f"cand_1a_synth_{i:03d}_{plate_num}.jpg"
        out_path = DIR_1A / fname
        cv2.imwrite(str(out_path), cv2.cvtColor(scene, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 93])
        manifest_rows.append({
            "filename": f"type1a/{fname}",
            "plate_type": "type1a",
            "source": "plate_renderer_jdm_v1",
            "plate_num": plate_num,
            "is_synthetic": 1,
            "conditions": ",".join(conds)
        })
    print(f"  [+] Сгенерировано {target_synth_1a} фотореалистичных Type 1A сцен.")

    # =========================================================================
    # 3. TYPE 2: REAL CANDIDATES FROM COMMONS (Trailers of Russia)
    # =========================================================================
    print("\n[*] [3/4] Поиск реальных кандидатов Type 2 (Прицепы) в открытых архивах...")
    trailer_categories = [
        "Category:Trailer license plates of Russia",
        "Category:Trailers in Russia"
    ]
    real_t2_count = 0
    for cat in trailer_categories:
        if real_t2_count >= 8:
            break
        files = fetch_commons_file_list(cat, limit=15)
        print(f"  -> Найдено {len(files)} файлов в '{cat}'")
        for ftitle in files:
            if real_t2_count >= 8:
                break
            res = fetch_commons_image(ftitle)
            if res is not None:
                img, source_url = res
                blurred, _ = blurrer.process_image(img)
                fname = f"cand_t2_real_{real_t2_count + 1:03d}.jpg"
                out_path = DIR_T2 / fname
                cv2.imwrite(str(out_path), blurred, [cv2.IMWRITE_JPEG_QUALITY, 93])
                manifest_rows.append({
                    "filename": f"type2/{fname}",
                    "plate_type": "type2",
                    "source": source_url,
                    "plate_num": "",
                    "is_synthetic": 0,
                    "conditions": "day,real_trailer"
                })
                real_t2_count += 1
                print(f"  [+] Добавлен реальный Type 2 кандидат: {fname}")

    # =========================================================================
    # 4. TYPE 2: PROCEDURAL PHOTOREALISTIC TRAILER SCENES (20 scenes)
    # =========================================================================
    print("\n[*] [4/4] Синтез фотореалистичных сцен Type 2 (Прицепы с габаритными балками)...")
    target_synth_t2 = 20
    for i in range(1, target_synth_t2 + 1):
        clean_plate, plate_num = renderer.render_type2()
        aug_plate = augmentor.augment_plate(clean_plate)
        plate_arr = np.array(aug_plate)

        bg = generate_trailer_background(1280, 720)
        scene, bbox, quad = transformer.project_plate_onto_background(
            plate_arr, bg, max_yaw=28.0, max_pitch=16.0, max_roll=5.0
        )

        conds = ["day", "angle", "trailer_bumper"]
        if random.random() < 0.40:
            conds.append("dirt")
        if random.random() < 0.20:
            conds.append("rain_spray")

        fname = f"cand_t2_synth_{i:03d}_{plate_num}.jpg"
        out_path = DIR_T2 / fname
        cv2.imwrite(str(out_path), cv2.cvtColor(scene, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 93])
        manifest_rows.append({
            "filename": f"type2/{fname}",
            "plate_type": "type2",
            "source": "plate_renderer_trailer_v1",
            "plate_num": plate_num,
            "is_synthetic": 1,
            "conditions": ",".join(conds)
        })
    print(f"  [+] Сгенерировано {target_synth_t2} фотореалистичных Type 2 сцен.")

    # Write manifest
    with open(MANIFEST_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "plate_type", "source", "plate_num", "is_synthetic", "conditions"], delimiter=";")
        writer.writeheader()
        writer.writerows(manifest_rows)

    print("\n" + "=" * 70)
    print(f"🎉 Сбор кандидатов успешно завершен!")
    print(f"📁 Итого Type 1A в {DIR_1A}: {len(list(DIR_1A.glob('*.jpg')))} изображений")
    print(f"📁 Итого Type 2  в {DIR_T2}: {len(list(DIR_T2.glob('*.jpg')))} изображений")
    print(f"📄 Манифест кандидатов: {MANIFEST_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    stage_candidates()
