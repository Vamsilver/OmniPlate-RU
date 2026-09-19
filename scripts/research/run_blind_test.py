#!/usr/bin/env python3
"""
scripts/research/run_blind_test.py
Harvests 50 fresh real-world vehicle images from Wikimedia Commons across all categories:
- Type 1 (Civilian 1-line): 15 images
- Type 1A (Square 2-line): 15 images
- Type 1B (Yellow buses/taxis): 10 images
- Other (Trailers, motorcycles, special, background): 10 images

Performs Face De-identification via OpenCV YuNet ONNX.
Runs OmniPlatePipeline (MoE) to detect and crop plates.
Creates visual contact sheets for AI vision inspection and Ground Truth labeling.
"""

import csv
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from dataset.privacy.face_blur import FaceBlurrer
from src.pipeline.pipeline import OmniPlatePipeline, PlateDetection

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT_API = "VolgaITBlindTest/1.0 (https://github.com/Vamsilver/OmniPlate-RU; test@example.com)"
HEADERS_DOWNLOAD = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Referer": "https://commons.wikimedia.org/",
    "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
}

CATEGORIES = {
    "type1": [
        "Category:Automobiles in Moscow",
        "Category:Automobiles in Saint Petersburg",
        "Category:Automobiles in Nizhny Novgorod",
        "Category:Automobiles in Novosibirsk",
        "Category:Automobiles in Kazan",
        "Category:Automobiles in Samara",
        "Category:Lada Vesta in Russia",
        "Category:Ford Focus in Russia",
        "Category:Kia Rio in Russia",
    ],
    "type1a": [
        "Category:Automobiles in Vladivostok",
        "Category:Automobiles in Primorsky Krai",
        "Category:Automobiles with license plates of Primorsky Krai",
        "Category:Automobiles in Sakhalin Oblast",
        "Category:Automobiles in Petropavlovsk-Kamchatsky",
        "Category:Toyota Mark II in Russia",
        "Category:Toyota Crown in Russia",
        "Category:Nissan Leaf in Russia",
        "Category:Subaru Forester in Russia",
    ],
    "type1b": [
        "Category:LiAZ-5292 in Moscow",
        "Category:Buses in Moscow",
        "Category:PAZ buses in Russia",
        "Category:Marshrutkas in Russia",
        "Category:Taxis in Moscow",
        "Category:Taxis in Saint Petersburg",
        "Category:Taxis in Russia",
    ],
    "other": [
        "Category:Trailers in Russia",
        "Category:Semi-trailers in Russia",
        "Category:Motorcycles in Russia",
        "Category:Tractors in Russia",
        "Category:Agricultural machinery in Russia",
        "Category:Military vehicles of Russia",
        "Category:Street scenes in Russia",
    ],
}

QUOTAS = {
    "type1": 15,
    "type1a": 15,
    "type1b": 10,
    "other": 10,
}


def query_commons(cmtitle: str, limit: int = 50) -> List[Dict]:
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


def get_image_info(title: str) -> Optional[Dict]:
    params = {
        "action": "query",
        "titles": title,
        "prop": "imageinfo",
        "iiprop": "url|size|extmetadata",
        "format": "json",
    }
    url = f"{COMMONS_API}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT_API})
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            pages = data.get("query", {}).get("pages", {})
            for page in pages.values():
                infos = page.get("imageinfo", [])
                if infos:
                    info = infos[0]
                    w = info.get("width", 0)
                    h = info.get("height", 0)
                    if w >= 600 and h >= 400:
                        return info
    except Exception:
        pass
    return None


def download_file(url: str, dest: Path) -> bool:
    clean_url = url.split("?")[0]
    req = urllib.request.Request(clean_url, headers=HEADERS_DOWNLOAD)
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            content = resp.read()
            if len(content) < 15000:
                return False
            with open(dest, "wb") as f:
                f.write(content)
            return True
    except Exception as e:
        return False


def harvest_50_images(output_dir: Path) -> List[Dict]:
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    seen_titles = set()
    face_blurrer = FaceBlurrer()

    print("[*] Harvesting 50 blind test candidates from Wikimedia Commons...")
    for target_type, quota in QUOTAS.items():
        collected = 0
        categories = CATEGORIES.get(target_type, [])
        print(f"\n--- Harvesting target '{target_type}' (Quota: {quota}) ---")
        for cat in categories:
            if collected >= quota:
                break
            members = query_commons(cat, limit=60)
            time.sleep(0.3)
            for m in members:
                if collected >= quota:
                    break
                title = m.get("title", "")
                if not title or title in seen_titles:
                    continue
                suffix = Path(title).suffix.lower()
                if suffix not in (".jpg", ".jpeg", ".png"):
                    continue

                seen_titles.add(title)
                info = get_image_info(title)
                time.sleep(0.2)
                if not info:
                    continue

                img_url = info.get("url")
                if not img_url:
                    continue

                dest_name = f"blind_{target_type}_{collected+1:02d}{suffix}"
                dest_path = images_dir / dest_name
                print(f"  [{target_type}] #{collected+1:02d}: {title[:45]}...", end="", flush=True)
                ok = download_file(img_url, dest_path)
                if not ok:
                    print(" FAIL")
                    continue

                img = cv2.imread(str(dest_path))
                if img is None or img.shape[0] < 300 or img.shape[1] < 400:
                    print(" INVALID")
                    dest_path.unlink(missing_ok=True)
                    continue

                # Resize if monstrous (> 2500px) to conserve memory and processing time
                h, w = img.shape[:2]
                if max(h, w) > 2000:
                    scale = 2000.0 / max(h, w)
                    img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

                # De-identify faces
                blurred_img, stats = face_blurrer.process_image(img)
                faces_found = stats.get("face_count", 0)
                cv2.imwrite(str(dest_path), blurred_img)

                meta_entry = {
                    "filename": dest_name,
                    "target_type": target_type,
                    "title": title,
                    "url": img_url,
                    "width": blurred_img.shape[1],
                    "height": blurred_img.shape[0],
                    "faces_blurred": faces_found,
                }
                manifest.append(meta_entry)
                collected += 1
                print(f" OK ({blurred_img.shape[1]}x{blurred_img.shape[0]}, Faces: {faces_found})")

    print(f"\n[+] Harvesting complete! Total collected: {len(manifest)} images.")
    with open(output_dir / "harvest_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest


def run_pipeline_and_generate_sheets(output_dir: Path, manifest: List[Dict]):
    images_dir = output_dir / "images"
    crops_dir = output_dir / "crops"
    sheets_dir = output_dir / "contact_sheets"
    crops_dir.mkdir(parents=True, exist_ok=True)
    sheets_dir.mkdir(parents=True, exist_ok=True)

    print("\n[*] Initializing OmniPlatePipeline with MoE OCR on CUDA...")
    pipeline = OmniPlatePipeline(device="cuda", ocr_version="moe", conf_threshold=0.12)
    pipeline.warmup(iterations=2)

    results = []
    print("\n[*] Running inference and extracting crops for AI Vision inspection...")
    for idx, item in enumerate(manifest, start=1):
        fn = item["filename"]
        target_type = item["target_type"]
        img_path = images_dir / fn
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        t0 = time.perf_counter()
        dets = pipeline.predict(img)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        # Extract best detection or none
        best_det = None
        if dets:
            target_dets = [d for d in dets if d.plate_type in ("type1", "type1a", "type1b")]
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
            pred_conf = best_det.confidence
            ocr_conf = best_det.ocr_confidence
        else:
            # Empty black placeholder for non-detections
            crop = np.zeros((60, 160, 3), dtype=np.uint8)
            cv2.putText(crop, "NO DETECT", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imwrite(str(crop_path), crop)
            pred_text = ""
            pred_type = "other"
            pred_conf = 0.0
            ocr_conf = 0.0

        results.append({
            "idx": idx,
            "filename": fn,
            "target_type": target_type,
            "title": item["title"],
            "pred_text": pred_text,
            "pred_type": pred_type,
            "pred_conf": round(pred_conf, 3),
            "ocr_conf": round(ocr_conf, 3),
            "latency_ms": round(dt_ms, 1),
            "crop_file": crop_name,
        })
        print(f"[{idx:>2}/50] {fn:<22} -> Pred: {pred_type:<7} '{pred_text:<10}' ({dt_ms:5.1f} ms)")

    with open(output_dir / "predictions.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Generate visual contact sheets (10 items per sheet, 5 sheets total)
    print("\n[*] Generating 5 visual contact sheets (10 images each) for AI Vision inspection...")
    chunk_size = 10
    for sheet_idx in range(0, len(results), chunk_size):
        chunk = results[sheet_idx : sheet_idx + chunk_size]
        sheet_img = render_contact_sheet(images_dir, crops_dir, chunk, sheet_idx // chunk_size + 1)
        sheet_path = sheets_dir / f"sheet_{sheet_idx // chunk_size + 1:02d}.jpg"
        cv2.imwrite(str(sheet_path), sheet_img)
        print(f"  [+] Saved Contact Sheet: {sheet_path.name}")

    print("[SUCCESS] Pipeline inference and contact sheets ready!")


def render_contact_sheet(images_dir: Path, crops_dir: Path, items: List[Dict], sheet_num: int) -> np.ndarray:
    """Renders a 2-column x 5-row or 5-row x 2-panel contact sheet (width=1400, height=1750)."""
    cell_w = 680
    cell_h = 320
    rows = 5
    cols = 2
    sheet = np.full((cell_h * rows + 70, cell_w * cols + 40, 3), 245, dtype=np.uint8)

    # Title header
    cv2.putText(sheet, f"OmniPlate-RU Blind Test 50 — Sheet #{sheet_num:02d} (Items {items[0]['idx']} to {items[-1]['idx']})",
                (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (20, 20, 20), 2, cv2.LINE_AA)

    for i, item in enumerate(items):
        r = i // cols
        c = i % cols
        x0 = 20 + c * (cell_w + 10)
        y0 = 70 + r * cell_h

        # Bounding box for cell
        cv2.rectangle(sheet, (x0, y0), (x0 + cell_w, y0 + cell_h - 10), (220, 220, 220), -1)
        cv2.rectangle(sheet, (x0, y0), (x0 + cell_w, y0 + cell_h - 10), (160, 160, 160), 1)

        # Full image thumbnail (left)
        img_p = images_dir / item["filename"]
        img = cv2.imread(str(img_p))
        if img is not None:
            # Aspect-preserve scale to 320x220
            th_h, th_w = 210, 320
            h, w = img.shape[:2]
            scale = min(th_w / w, th_h / h)
            nw, nh = int(w * scale), int(h * scale)
            thumb = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
            pad_x = x0 + 10 + (th_w - nw) // 2
            pad_y = y0 + 10 + (th_h - nh) // 2
            sheet[pad_y : pad_y + nh, pad_x : pad_x + nw] = thumb

        # Crop preview (right)
        crop_p = crops_dir / item["crop_file"]
        crop = cv2.imread(str(crop_p))
        if crop is not None and crop.size > 0:
            cr_h, cr_w = 90, 310
            ch, cw = crop.shape[:2]
            scale_cr = min(cr_w / cw, cr_h / ch)
            ncw, nch = int(cw * scale_cr), int(ch * scale_cr)
            thumb_crop = cv2.resize(crop, (ncw, nch), interpolation=cv2.INTER_AREA)
            cp_x = x0 + 345 + (cr_w - ncw) // 2
            cp_y = y0 + 35 + (cr_h - nch) // 2
            sheet[cp_y : cp_y + nch, cp_x : cp_x + ncw] = thumb_crop
            cv2.rectangle(sheet, (cp_x - 1, cp_y - 1), (cp_x + ncw + 1, cp_y + nch + 1), (0, 0, 255), 1)

        # Labels & metadata
        info_x = x0 + 10
        info_y = y0 + 250
        cv2.putText(sheet, f"#{item['idx']:02d} [{item['target_type'].upper()}] {item['filename']}",
                    (info_x, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 50, 150), 2)
        cv2.putText(sheet, f"Pred: {item['pred_type'].upper()} | Text: '{item['pred_text']}' | Conf: {item['pred_conf']} | {item['latency_ms']} ms",
                    (info_x, info_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 100, 0) if item['pred_text'] else (0, 0, 180), 2)
        cv2.putText(sheet, f"Title: {item['title'][:45]}",
                    (info_x, info_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 100, 100), 1)

    return sheet


if __name__ == "__main__":
    out_dir = ROOT_DIR / "test_output" / "blind_test_50"
    manifest = harvest_50_images(out_dir)
    if manifest:
        run_pipeline_and_generate_sheets(out_dir, manifest)
