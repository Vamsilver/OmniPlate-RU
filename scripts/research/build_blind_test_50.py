#!/usr/bin/env python3
"""
scripts/research/build_blind_test_50.py
Harvests and prepares 50 diverse street images across all Volga IT classes:
- 15 Type 1: Real traffic camera photos from russian_car_plates.v2i.yolov8.zip
- 15 Type 1A: Real JDM vehicle photos from test_output/turbo_1a_verified/
- 10 Type 1B: Real public transport (LiAZ buses / taxis) from Wikimedia Commons
- 10 Other: Real trailer plates, tractors, motorcycles, and negative street scenes

Performs Face De-identification via OpenCV YuNet DNN FaceBlurrer.
Runs OmniPlatePipeline with MoE OCR on CUDA.
Extracts crops and generates 5 high-resolution contact sheets for AI Vision inspection.
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

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from dataset.privacy.face_blur import FaceBlurrer
from src.pipeline.pipeline import OmniPlatePipeline

ROBOFLOW_ZIP = Path(r"C:\Users\Vamsi\Downloads\russian_car_plates.v2i.yolov8.zip")
TURBO_DIR = ROOT_DIR / "test_output" / "turbo_1a_verified"
CAND_T2_DIR = ROOT_DIR / "dataset" / "candidates_review" / "type2"
OUT_DIR = ROOT_DIR / "test_output" / "blind_test_50"

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT_API = "VolgaITBlindTest/1.0 (https://github.com/Vamsilver/OmniPlate-RU; test@example.com)"
HEADERS_DOWNLOAD = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Referer": "https://commons.wikimedia.org/",
}

CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'E', 'H', 'K', 'M', 'O', 'P', 'T', 'X', 'Y']
T1_REGEX = re.compile(r'^[ABEKMHOPCTYX]\d{3}[ABEKMHOPCTYX]{2}\d{2,3}$')


def get_meta_sources() -> set:
    meta_path = ROOT_DIR / "dataset" / "meta.csv"
    if not meta_path.exists():
        return set()
    with open(meta_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        return {row["source"] for row in reader}


def harvest_type1(quota: int = 15, existing_sources: set = None) -> List[Dict]:
    """Extracts 15 diverse Type 1 real road scenes from Roboflow zip."""
    if existing_sources is None:
        existing_sources = set()
    results = []
    seen_plates = set()

    with zipfile.ZipFile(ROBOFLOW_ZIP) as z:
        labels = [n for n in z.namelist() if n.startswith("train/labels/") and n.endswith(".txt")]
        for lbl in labels:
            if len(results) >= quota:
                break
            img_name = lbl.replace("train/labels/", "train/images/").replace(".txt", ".jpg")
            src_tag = f"russian_car_plates.v2i.yolov8.zip:{img_name}"
            if src_tag in existing_sources:
                continue

            try:
                lines = [l.strip().split() for l in z.read(lbl).decode("utf-8").strip().split("\n") if l.strip()]
                if not (8 <= len(lines) <= 9):
                    continue
                boxes = sorted(lines, key=lambda x: float(x[1]))
                plate_chars = "".join([CLASSES[int(b[0])] for b in boxes])
                if not T1_REGEX.match(plate_chars):
                    continue
                if plate_chars in seen_plates:
                    continue

                img_bytes = z.read(img_name)
                img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                if img is None or img.shape[0] < 200 or img.shape[1] < 200:
                    continue

                seen_plates.add(plate_chars)
                results.append({
                    "target_type": "type1",
                    "source": src_tag,
                    "title": f"Roboflow Traffic Cam: {plate_chars}",
                    "gt_hint": plate_chars,
                    "image": img,
                })
            except Exception:
                continue

    print(f"[+] Harvested {len(results)} Type 1 images from traffic camera pool.")
    return results


def harvest_type1a(quota: int = 15, existing_sources: set = None) -> List[Dict]:
    """Extracts 15 diverse Type 1A square plate scenes from turbo_1a_verified."""
    if existing_sources is None:
        existing_sources = set()

    meta_json = TURBO_DIR / "candidates_metadata.json"
    results = []
    seen_plates = set()

    if meta_json.exists():
        with open(meta_json, "r", encoding="utf-8") as f:
            items = json.load(f)
        for it in items:
            if len(results) >= quota:
                break
            full_fn = it.get("full_fn", "")
            img_path = TURBO_DIR / full_fn
            if not img_path.exists():
                continue
            text = it.get("text", "")
            url = it.get("url", "")
            if url in existing_sources or text in seen_plates:
                continue
            img = cv2.imread(str(img_path))
            if img is None or img.shape[0] < 200:
                continue

            seen_plates.add(text)
            results.append({
                "target_type": "type1a",
                "source": url or f"turbo_1a:{full_fn}",
                "title": f"JDM Vehicle: {text} ({full_fn})",
                "gt_hint": text,
                "image": img,
            })

    print(f"[+] Harvested {len(results)} Type 1A images from JDM pool.")
    return results


def query_commons(cmtitle: str, limit: int = 30) -> List[Dict]:
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": cmtitle,
        "cmtype": "file",
        "cmlimit": str(limit),
        "format": "json",
    }
    url = f"{COMMONS_API}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT_API})
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("query", {}).get("categorymembers", [])
    except Exception as e:
        print(f"[-] API error for {cmtitle}: {e}")
        return []


def download_thumb_image(title: str, width: int = 1280) -> Optional[Tuple[np.ndarray, str]]:
    params = {
        "action": "query",
        "titles": title,
        "prop": "imageinfo",
        "iiprop": "url",
        "iiurlwidth": str(width),
        "format": "json",
    }
    url = f"{COMMONS_API}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT_API})
    try:
        with urllib.request.urlopen(req, timeout=12) as r:
            data = json.loads(r.read().decode("utf-8"))
        for p in data.get("query", {}).get("pages", {}).values():
            infos = p.get("imageinfo", [])
            if infos:
                thumb_url = infos[0].get("thumburl") or infos[0].get("url")
                req2 = urllib.request.Request(thumb_url, headers=HEADERS_DOWNLOAD)
                with urllib.request.urlopen(req2, timeout=15) as r2:
                    content = r2.read()
                if len(content) < 15000:
                    return None
                img = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
                if img is not None and img.shape[0] >= 300 and img.shape[1] >= 400:
                    return img, thumb_url
    except Exception:
        pass
    return None


def harvest_type1b(quota: int = 10, existing_sources: set = None) -> List[Dict]:
    """Harvests 10 real Type 1B public transport images from Wikimedia Commons."""
    if existing_sources is None:
        existing_sources = set()

    categories = [
        "Category:LiAZ-5292 in Moscow",
        "Category:Buses in Moscow",
        "Category:PAZ buses in Russia",
        "Category:Yandex.Taxi in Moscow",
        "Category:Taxis in Moscow",
        "Category:Taxis in Russia",
    ]
    results = []
    seen_titles = set()

    for cat in categories:
        if len(results) >= quota:
            break
        members = query_commons(cat, limit=30)
        time.sleep(0.3)
        for m in members:
            if len(results) >= quota:
                break
            title = m.get("title", "")
            suffix = Path(title).suffix.lower()
            if suffix not in (".jpg", ".jpeg", ".png"):
                continue
            if title in seen_titles:
                continue
            seen_titles.add(title)

            res = download_thumb_image(title, width=1280)
            time.sleep(0.2)
            if not res:
                continue
            img, thumb_url = res

            # Check if url already in meta
            if thumb_url in existing_sources or title in existing_sources:
                continue

            results.append({
                "target_type": "type1b",
                "source": thumb_url,
                "title": title,
                "gt_hint": "",
                "image": img,
            })
            print(f"  [Type 1B] #{len(results)}: {title[:50]} ({img.shape[1]}x{img.shape[0]})")

    print(f"[+] Harvested {len(results)} Type 1B images from Wikimedia.")
    return results


def harvest_other(quota: int = 10, existing_sources: set = None) -> List[Dict]:
    """Harvests 10 candidate scenes of trailers, special vehicles, and negatives."""
    if existing_sources is None:
        existing_sources = set()

    results = []
    
    # 1. First take available unseen from candidates_review/type2/
    manifest_csv = CAND_T2_DIR.parent / "candidates_manifest.csv"
    hints = {}
    if manifest_csv.exists():
        with open(manifest_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=";")
            for r in reader:
                hints[Path(r["filename"]).name] = (r.get("plate_num", ""), r.get("source", ""))

    files = sorted(list(CAND_T2_DIR.glob("*.jpg")))
    for p in files:
        if len(results) >= quota:
            break
        hint_plate, src = hints.get(p.name, ("", f"type2:{p.name}"))
        if src in existing_sources:
            continue
        img = cv2.imread(str(p))
        if img is None or img.shape[0] < 200:
            continue

        results.append({
            "target_type": "other",
            "source": src,
            "title": f"Candidate Trailer: {hint_plate} ({p.name})",
            "gt_hint": hint_plate,
            "image": img,
        })
        print(f"  [Other] #{len(results)}: from candidates_review ({p.name})")

    # 2. Complete remaining quota from Wikimedia Commons
    categories = [
        "Category:Trailers in Russia",
        "Category:Tractors in Russia",
        "Category:Motorcycles in Russia",
        "Category:Semi-trailers in Russia",
        "Category:Street scenes in Russia",
    ]
    seen_titles = set()
    for cat in categories:
        if len(results) >= quota:
            break
        members = query_commons(cat, limit=25)
        time.sleep(0.3)
        for m in members:
            if len(results) >= quota:
                break
            title = m.get("title", "")
            suffix = Path(title).suffix.lower()
            if suffix not in (".jpg", ".jpeg", ".png"):
                continue
            if title in seen_titles:
                continue
            seen_titles.add(title)

            res = download_thumb_image(title, width=1280)
            time.sleep(0.2)
            if not res:
                continue
            img, thumb_url = res

            results.append({
                "target_type": "other",
                "source": thumb_url,
                "title": title,
                "gt_hint": "",
                "image": img,
            })
            print(f"  [Other] #{len(results)}: from Wikimedia ({title[:50]})")

    print(f"[+] Harvested {len(results)} Other/Trailer candidate images.")
    return results


def render_contact_sheet(images_dir: Path, crops_dir: Path, items: List[Dict], sheet_num: int) -> np.ndarray:
    cell_w = 700
    cell_h = 320
    rows = 5
    cols = 2
    sheet = np.full((cell_h * rows + 70, cell_w * cols + 40, 3), 245, dtype=np.uint8)

    title_text = f"OmniPlate-RU Blind Test 50 - Sheet #{sheet_num:02d} (Items {items[0]['idx']} to {items[-1]['idx']})"
    cv2.putText(sheet, title_text, (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2, cv2.LINE_AA)

    for i, item in enumerate(items):
        r = i // cols
        c = i % cols
        x0 = 20 + c * (cell_w + 10)
        y0 = 70 + r * cell_h

        # Cell background
        cv2.rectangle(sheet, (x0, y0), (x0 + cell_w, y0 + cell_h - 10), (225, 225, 225), -1)
        cv2.rectangle(sheet, (x0, y0), (x0 + cell_w, y0 + cell_h - 10), (160, 160, 160), 1)

        # Full scene thumbnail
        img_p = images_dir / item["filename"]
        img = cv2.imread(str(img_p))
        if img is not None:
            th_h, th_w = 220, 330
            h, w = img.shape[:2]
            scale = min(th_w / w, th_h / h)
            nw, nh = int(w * scale), int(h * scale)
            thumb = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
            pad_x = x0 + 10 + (th_w - nw) // 2
            pad_y = y0 + 10 + (th_h - nh) // 2
            sheet[pad_y : pad_y + nh, pad_x : pad_x + nw] = thumb

        # Crop preview
        crop_p = crops_dir / item["crop_file"]
        crop = cv2.imread(str(crop_p))
        if crop is not None and crop.size > 0:
            cr_h, cr_w = 110, 330
            ch, cw = crop.shape[:2]
            scale_cr = min(cr_w / cw, cr_h / ch)
            ncw, nch = int(cw * scale_cr), int(ch * scale_cr)
            thumb_crop = cv2.resize(crop, (ncw, nch), interpolation=cv2.INTER_AREA)
            cp_x = x0 + 355 + (cr_w - ncw) // 2
            cp_y = y0 + 35 + (cr_h - nch) // 2
            sheet[cp_y : cp_y + nch, cp_x : cp_x + ncw] = thumb_crop
            cv2.rectangle(sheet, (cp_x - 1, cp_y - 1), (cp_x + ncw + 1, cp_y + nch + 1), (0, 0, 200), 1)

        # Meta & Prediction text
        info_x = x0 + 10
        info_y = y0 + 250
        cv2.putText(sheet, f"#{item['idx']:02d} [{item['target_type'].upper()}] {item['filename']}",
                    (info_x, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (120, 40, 0), 2)
        
        pred_color = (0, 120, 0) if item['pred_text'] else (0, 0, 180)
        cv2.putText(sheet, f"Pred: {item['pred_type'].upper()} | '{item['pred_text']}' | Conf: {item['pred_conf']} | {item['latency_ms']} ms",
                    (info_x, info_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, pred_color, 2)
        hint_str = f"Hint: {item.get('gt_hint', '')} | {item['title'][:40]}"
        cv2.putText(sheet, hint_str, (info_x, info_y + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (80, 80, 80), 1)

    return sheet


def main():
    print("=" * 70)
    print("  OmniPlate-RU: Сборка и прогон слепого теста на 50 изображениях")
    print("=" * 70)

    images_dir = OUT_DIR / "images"
    crops_dir = OUT_DIR / "crops"
    sheets_dir = OUT_DIR / "contact_sheets"
    images_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)
    sheets_dir.mkdir(parents=True, exist_ok=True)

    meta_sources = get_meta_sources()
    print(f"[*] Total existing sources in meta.csv: {len(meta_sources)}")

    # 1. Harvest candidates
    t1_items = harvest_type1(quota=15, existing_sources=meta_sources)
    t1a_items = harvest_type1a(quota=15, existing_sources=meta_sources)
    t1b_items = harvest_type1b(quota=10, existing_sources=meta_sources)
    other_items = harvest_other(quota=10, existing_sources=meta_sources)

    all_harvested = t1_items + t1a_items + t1b_items + other_items
    print(f"\n[*] Total harvested images: {len(all_harvested)} / 50")
    if len(all_harvested) != 50:
        raise ValueError(f"Expected 50 images, got {len(all_harvested)}")

    # 2. De-identification via YuNet FaceBlurrer
    print("\n[*] Running FaceBlurrer (YuNet ONNX) for 100% privacy compliance...")
    blurrer = FaceBlurrer()
    total_faces_blurred = 0

    items_processed = []
    for idx, item in enumerate(all_harvested, start=1):
        fn = f"blind_{idx:02d}_{item['target_type']}.jpg"
        dest_path = images_dir / fn

        img = item["image"]
        blurred_img, stats = blurrer.process_image(img)
        faces_detected = stats.get("faces_detected", 0)
        total_faces_blurred += faces_detected

        cv2.imwrite(str(dest_path), blurred_img)
        item["filename"] = fn
        item["width"] = blurred_img.shape[1]
        item["height"] = blurred_img.shape[0]
        item["faces_blurred"] = faces_detected
        items_processed.append(item)

    print(f"[+] Face blurring complete. Total faces anonymized across 50 images: {total_faces_blurred}")

    # 3. Pipeline Inference (MoE OCR on CUDA)
    print("\n[*] Initializing OmniPlatePipeline with MoE OCR on CUDA...")
    pipeline = OmniPlatePipeline(device="cuda", ocr_version="moe", conf_threshold=0.10)
    pipeline.warmup(iterations=2)

    predictions = []
    print("\n[*] Running inference and extracting crops...")
    for idx, item in enumerate(items_processed, start=1):
        fn = item["filename"]
        img_p = images_dir / fn
        img = cv2.imread(str(img_p))

        t0 = time.perf_counter()
        dets = pipeline.predict(img)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        best_det = None
        if dets:
            target_dets = [d for d in dets if d.plate_type == item["target_type"]]
            best_det = target_dets[0] if target_dets else dets[0]

        crop_name = f"crop_{fn}"
        crop_path = crops_dir / crop_name

        if best_det and best_det.plate_type != "other":
            bx, by, bw, bh = best_det.bbox
            bx = max(0, bx)
            by = max(0, by)
            bw = min(img.shape[1] - bx, bw)
            bh = min(img.shape[0] - by, bh)
            crop = img[by : by + bh, bx : bx + bw]
            if crop.size > 0:
                cv2.imwrite(str(crop_path), crop)
            else:
                crop = np.zeros((60, 160, 3), dtype=np.uint8)
                cv2.imwrite(str(crop_path), crop)
            pred_text = best_det.text or ""
            pred_type = best_det.plate_type
            pred_conf = round(best_det.confidence, 3)
            ocr_conf = round(best_det.ocr_confidence, 3)
            bbox = best_det.bbox
            quad = best_det.quad
        else:
            crop = np.zeros((60, 160, 3), dtype=np.uint8)
            cv2.putText(crop, "NO DETECT", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imwrite(str(crop_path), crop)
            pred_text = ""
            pred_type = "other"
            pred_conf = 0.0
            ocr_conf = 0.0
            bbox = [0, 0, 0, 0]
            quad = [0, 0, 0, 0, 0, 0, 0, 0]

        predictions.append({
            "idx": idx,
            "filename": fn,
            "target_type": item["target_type"],
            "source": item["source"],
            "title": item["title"],
            "gt_hint": item.get("gt_hint", ""),
            "pred_text": pred_text,
            "pred_type": pred_type,
            "pred_conf": pred_conf,
            "ocr_conf": ocr_conf,
            "latency_ms": round(dt_ms, 1),
            "crop_file": crop_name,
            "bbox": bbox,
            "quad": quad,
            "faces_blurred": item["faces_blurred"],
        })
        print(f"[{idx:>2}/50] {fn:<25} -> Pred: {pred_type:<7} '{pred_text:<10}' (Conf: {pred_conf:.2f}, {dt_ms:5.1f} ms)")

    # Save manifest and predictions
    with open(OUT_DIR / "manifest_50.json", "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    print(f"\n[+] Saved manifest and predictions to {OUT_DIR / 'manifest_50.json'}")

    # 4. Generate 5 contact sheets (10 images each)
    print("\n[*] Generating 5 visual contact sheets (10 images each)...")
    chunk_size = 10
    for s_idx in range(0, len(predictions), chunk_size):
        chunk = predictions[s_idx : s_idx + chunk_size]
        sheet_num = s_idx // chunk_size + 1
        sheet = render_contact_sheet(images_dir, crops_dir, chunk, sheet_num)
        sheet_p = sheets_dir / f"sheet_{sheet_num:02d}.jpg"
        cv2.imwrite(str(sheet_p), sheet)
        print(f"  [+] Saved Sheet #{sheet_num:02d}: {sheet_p.name}")

    print("\n[SUCCESS] Blind Test 50 dataset, inference, and contact sheets generated successfully!")


if __name__ == "__main__":
    main()
