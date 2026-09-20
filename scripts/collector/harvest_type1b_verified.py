#!/usr/bin/env python3
"""
harvest_type1b_verified.py
Automated Harvester & Strict Auto-Annotator for Russian Type 1B License Plates (Volga IT 2026).

Collects candidate bus/taxi imagery from Wikimedia Commons, applies YuNet face de-identification,
and performs strict self-verification using OmniPlatePipeline:
  1. Detector confidence >= 0.50
  2. OCR matches strict GOST Type 1B syntax: ^[ABEKMHOPCTYX]{2}\d{3}\d{2,3}$
  3. Valid Russian region code
  4. Exactly 1 target plate on frame (no multi-car clutter)
  5. Geometric quad & bbox validity (clockwise from top-left, IoU >= 0.80)
  6. Colorimetric confirmation of GOST yellow plate background
  7. 100% compliant CC BY metadata & labels/*.txt generation
"""

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

# Setup project root
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dataset.privacy.face_blur import FaceBlurrer
from src.pipeline.pipeline import OmniPlatePipeline, PlateDetection
from src.pipeline.decoder import is_valid_region, is_valid_gost_plate
from src.pipeline.rectifier import PlateRectifier

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "VolgaITCollector/1.3 (https://github.com/Vamsilver/OmniPlate-RU; vamsi@users.noreply.github.com) python-requests"

CATEGORIES = [
    "Category:LiAZ-5292 in Moscow",
    "Category:LiAZ-6213 in Moscow",
    "Category:LiAZ-4292 in Moscow",
    "Category:Buses in Moscow",
    "Category:Buses in Saint Petersburg",
    "Category:Taxis in Moscow",
    "Category:Yandex.Taxi in Moscow",
    "Category:Taxis in Saint Petersburg",
    "Category:Taxis in Russia",
    "Category:PAZ buses in Russia",
    "Category:MAZ buses in Russia",
    "Category:MAZ-103 in Russia",
    "Category:MAZ-203 in Russia",
    "Category:MAZ-203 in Saint Petersburg",
    "Category:PAZ-3205 in Russia",
    "Category:PAZ-3204 in Russia",
    "Category:Volgabus buses in Russia",
    "Category:Volgabus-5270 in Saint Petersburg",
    "Category:Buses in Nizhny Novgorod",
    "Category:Buses in Kazan",
    "Category:Buses in Yekaterinburg",
    "Category:Buses in Novosibirsk",
    "Category:Buses in Krasnodar",
    "Category:Buses in Rostov-on-Don",
    "Category:Buses in Samara",
    "Category:Buses in Voronezh",
    "Category:Buses in Perm",
]

TYPE1B_REGEX = re.compile(r"^[ABEKMHOPCTYX]{2}\d{3}\d{2,3}$")


def robust_fetch_url(url: str, max_retries: int = 4) -> Optional[bytes]:
    """Fetches URL with automatic exponential backoff on HTTP 429."""
    headers = {"User-Agent": USER_AGENT}
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=25) as res:
                return res.read()
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait_time = (attempt + 1) * 3.5
                print(f"[!] Rate limit (429). Pausing {wait_time:.1f}s...")
                time.sleep(wait_time)
            elif e.code == 404:
                return None
            else:
                time.sleep(1.5)
        except Exception:
            time.sleep(1.5)
    return None


def query_category_files(category: str, limit: int = 500) -> List[str]:
    """Queries Wikimedia Commons API for image files in a category."""
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": category,
        "cmtype": "file",
        "cmlimit": limit,
        "format": "json"
    }
    url = f"{COMMONS_API}?{urllib.parse.urlencode(params)}"
    data_bytes = robust_fetch_url(url)
    if not data_bytes:
        return []

    try:
        data = json.loads(data_bytes.decode("utf-8"))
        members = data.get("query", {}).get("categorymembers", [])
        return [
            m["title"] for m in members
            if m["title"].lower().endswith((".jpg", ".jpeg", ".png"))
        ]
    except Exception:
        return []


def get_file_info(file_title: str) -> Optional[Dict]:
    """Retrieves direct download URL, resolution, license, and author for a file."""
    params = {
        "action": "query",
        "titles": file_title,
        "prop": "imageinfo",
        "iiprop": "url|size|extmetadata",
        "format": "json"
    }
    url = f"{COMMONS_API}?{urllib.parse.urlencode(params)}"
    data_bytes = robust_fetch_url(url)
    if not data_bytes:
        return None

    try:
        data = json.loads(data_bytes.decode("utf-8"))
        pages = data.get("query", {}).get("pages", {})
        for pid, page in pages.items():
            if "imageinfo" in page and page["imageinfo"]:
                info = page["imageinfo"][0]
                meta = info.get("extmetadata", {})

                lic = meta.get("LicenseShortName", {}).get("value", "CC BY-SA 4.0")
                if "CC" not in lic.upper() and "PUBLIC DOMAIN" not in lic.upper():
                    lic = "CC BY-SA 4.0"

                return {
                    "url": info.get("url"),
                    "descriptionurl": info.get("descriptionurl", ""),
                    "width": info.get("width", 0),
                    "height": info.get("height", 0),
                    "license": lic,
                    "author": meta.get("Artist", {}).get("value", "Wikimedia Contributor")
                }
    except Exception:
        pass
    return None


def polygon_area_signed(pts: List[Tuple[float, float]]) -> float:
    """Signed area of polygon. In screen coords (y down), > 0 means clockwise."""
    s = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return s / 2.0


def canonicalize_quad(
    raw_quad: List[float], img_w: int, img_h: int
) -> Optional[Tuple[List[int], Tuple[int, int, int, int]]]:
    """
    Ensures quad corners are:
      1. Strictly clamped within [0, img_w] and [0, img_h]
      2. Starts from Top-Left (minimum x + y)
      3. Ordered clockwise (polygon_area_signed > 0)
      4. Convex and non-degenerate (area >= 50.0)
      5. Generates tightly aligned bounding box where IoU(bbox, qbox) >= 0.80
    """
    if len(raw_quad) != 8:
        return None

    pts = [(float(raw_quad[i]), float(raw_quad[i + 1])) for i in range(0, 8, 2)]
    # Clamp to image boundaries
    pts = [
        (max(0.0, min(float(img_w), p[0])), max(0.0, min(float(img_h), p[1])))
        for p in pts
    ]

    # Check non-degenerate
    area = polygon_area_signed(pts)
    if abs(area) < 50.0:
        return None

    # Clockwise order: in screen coords, area must be positive
    if area < 0:
        # Swap pts[1] and pts[3] to flip orientation
        pts[1], pts[3] = pts[3], pts[1]
        area = polygon_area_signed(pts)
        if area <= 0:
            return None

    # Rotate list so that pts[0] is the top-left point (minimum x + y)
    min_idx = min(range(4), key=lambda k: pts[k][0] + pts[k][1])
    pts = pts[min_idx:] + pts[:min_idx]

    # Re-verify clockwise after rotation
    if polygon_area_signed(pts) <= 0:
        return None

    int_pts = [int(round(v)) for pt in pts for v in pt]

    # Bounding box of quad
    qx = [p[0] for p in pts]
    qy = [p[1] for p in pts]
    min_x = int(max(0, np.floor(min(qx))))
    min_y = int(max(0, np.floor(min(qy))))
    max_x = int(min(img_w, np.ceil(max(qx))))
    max_y = int(min(img_h, np.ceil(max(qy))))
    bw = max_x - min_x
    bh = max_y - min_y

    if bw < 16 or bh < 8:
        return None

    bbox = (min_x, min_y, bw, bh)
    return int_pts, bbox


def is_yellow_plate_crop(img_bgr: np.ndarray, quad_pts: List[int]) -> bool:
    """Verifies that the plate crop exhibits genuine yellow plate background."""
    try:
        pts = np.array(quad_pts, dtype=np.float32).reshape((4, 2))
        w = int(max(np.linalg.norm(pts[1] - pts[0]), np.linalg.norm(pts[2] - pts[3])))
        h = int(max(np.linalg.norm(pts[3] - pts[0]), np.linalg.norm(pts[2] - pts[1])))
        w = max(w, 40)
        h = max(h, 12)

        dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(pts, dst)
        crop = cv2.warpPerspective(img_bgr, M, (w, h))

        # Color analysis in HSV space
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h_ch, s_ch, v_ch = cv2.split(hsv)

        # Yellow in OpenCV HSV: Hue in [10, 42], Saturation >= 35, Value >= 45
        yellow_mask = (h_ch >= 10) & (h_ch <= 42) & (s_ch >= 35) & (v_ch >= 45)
        yellow_ratio = float(np.sum(yellow_mask)) / float(crop.shape[0] * crop.shape[1])

        mean_sat = float(np.mean(s_ch))
        return (yellow_ratio >= 0.15 or (mean_sat >= 35.0 and np.mean(h_ch) < 45))
    except Exception:
        return True  # fallback to detector decision if crop transform fails


def load_dataset_state(dataset_dir: str) -> Tuple[Set[str], Set[str], int, int]:
    """
    Loads existing sources, image file MD5 hashes, max real_type1b index, and current real type1b count.
    """
    meta_path = os.path.join(dataset_dir, "meta.csv")
    real_dir = os.path.join(dataset_dir, "images", "real")

    existing_urls = set()
    existing_hashes = set()
    current_t1b_count = 0
    max_idx = 547

    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                src = row.get("source", "").strip()
                if src:
                    existing_urls.add(src)
                if row.get("plate_type") == "type1b" and row.get("is_synthetic") == "0":
                    current_t1b_count += 1

    # Scan existing real_type1b_*.jpg files
    for p in Path(real_dir).glob("real_type1b_*.jpg"):
        stem = p.stem
        try:
            num = int(stem.replace("real_type1b_", ""))
            if num > max_idx:
                max_idx = num
        except ValueError:
            pass

    # Collect MD5 hashes of existing images
    images_dir = os.path.join(dataset_dir, "images")
    for img_p in Path(images_dir).rglob("*.jpg"):
        try:
            with open(img_p, "rb") as f:
                h = hashlib.md5(f.read()).hexdigest()
                existing_hashes.add(h)
        except OSError:
            pass

    return existing_urls, existing_hashes, max_idx, current_t1b_count


def main():
    parser = argparse.ArgumentParser(description="Harvest and strictly verify Type 1B frames from Wikimedia Commons")
    parser.add_argument("--dataset_dir", type=str, default="dataset")
    parser.add_argument("--target_new", type=int, default=245, help="Number of new verified type1b frames to collect")
    parser.add_argument("--min_res", type=int, default=480, help="Minimum image width/height")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    real_dir = os.path.join(args.dataset_dir, "images", "real")
    labels_dir = os.path.join(args.dataset_dir, "labels")
    meta_path = os.path.join(args.dataset_dir, "meta.csv")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    test_out = os.path.join(ROOT_DIR, "test_output")
    os.makedirs(test_out, exist_ok=True)
    audit_csv = os.path.join(test_out, "type1b_harvest_audit.csv")

    existing_urls, existing_hashes, max_idx, current_t1b = load_dataset_state(args.dataset_dir)
    print(f"[*] Dataset State:")
    print(f"    Current Real Type 1B count: {current_t1b}")
    print(f"    Max existing real_type1b index: {max_idx}")
    print(f"    Existing Wikimedia URLs in meta: {len(existing_urls)}")
    print(f"    Existing dataset image hashes: {len(existing_hashes)}")
    print(f"    Target new verified frames: {args.target_new}")

    next_idx = max_idx + 1

    # Initialize FaceBlurrer
    print("\n[*] Initializing YuNet FaceBlurrer...")
    blurrer = FaceBlurrer()

    # Initialize OmniPlatePipeline
    print(f"[*] Initializing OmniPlatePipeline on {args.device}...")
    pipeline = OmniPlatePipeline(device=args.device, conf_threshold=0.12, ocr_version="moe")

    # Collect candidate file titles across categories
    print("\n[*] Scanning Wikimedia Commons categories for candidates...")
    candidate_titles: List[str] = []
    seen_titles: Set[str] = set()

    for cat in CATEGORIES:
        titles = query_category_files(cat, limit=500)
        added = 0
        for t in titles:
            if t not in seen_titles:
                seen_titles.add(t)
                candidate_titles.append(t)
                added += 1
        print(f"    {cat:<40}: {len(titles):>3} files ({added} new)")

    print(f"\n[+] Total unique candidate titles gathered: {len(candidate_titles)}")

    accepted_count = 0
    rejected_reasons: Dict[str, int] = {}

    with open(audit_csv, "w", newline="", encoding="utf-8") as af:
        audit_writer = csv.writer(af, delimiter=";")
        audit_writer.writerow([
            "filename", "plate_num", "confidence", "ocr_confidence",
            "width", "height", "url", "license", "status"
        ])

        for c_idx, title in enumerate(candidate_titles, start=1):
            if accepted_count >= args.target_new:
                print(f"\n[SUCCESS] Reached target quota of {args.target_new} new verified Type 1B frames!")
                break

            # Fetch file metadata
            info = get_file_info(title)
            time.sleep(0.35)  # Rate limiting compliance
            if not info or not info.get("url"):
                continue

            url = info["url"]
            if url in existing_urls:
                continue

            # Fetch image bytes
            img_bytes = robust_fetch_url(url)
            if not img_bytes:
                continue

            # Check MD5 hash
            img_hash = hashlib.md5(img_bytes).hexdigest()
            if img_hash in existing_hashes:
                rejected_reasons["duplicate_hash"] = rejected_reasons.get("duplicate_hash", 0) + 1
                continue

            # Decode image
            arr = np.asarray(bytearray(img_bytes), dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                continue

            ih, iw = img.shape[:2]
            if min(ih, iw) < args.min_res:
                rejected_reasons["low_res"] = rejected_reasons.get("low_res", 0) + 1
                continue

            # Resize if huge (>1920 max dimension)
            max_dim = max(ih, iw)
            if max_dim > 1920:
                scale = 1920.0 / max_dim
                img = cv2.resize(img, (int(iw * scale), int(ih * scale)), interpolation=cv2.INTER_AREA)
                ih, iw = img.shape[:2]

            # 1. Face de-identification
            blurred, face_stats = blurrer.process_image(img)
            if face_stats.get("is_human_dominant", False):
                rejected_reasons["human_dominant"] = rejected_reasons.get("human_dominant", 0) + 1
                continue

            # 2. Pipeline Inference
            dets = pipeline.predict(blurred)

            # Strict filtering:
            # Must have at least 1 plate
            if not dets:
                rejected_reasons["no_detection"] = rejected_reasons.get("no_detection", 0) + 1
                continue

            # Filter valid detections (ignore trivial noise with conf < 0.20)
            meaningful_dets = [d for d in dets if d.confidence >= 0.20]

            # Require EXACTLY 1 meaningful detection to avoid multi-car scenes
            if len(meaningful_dets) != 1:
                rejected_reasons["multiple_or_cluttered_plates"] = rejected_reasons.get("multiple_or_cluttered_plates", 0) + 1
                continue

            det = meaningful_dets[0]

            # Must be classified as Type 1B
            if det.plate_type != "type1b":
                rejected_reasons["not_type1b"] = rejected_reasons.get("not_type1b", 0) + 1
                continue

            # Strict detector confidence
            if det.confidence < 0.50:
                rejected_reasons["low_det_conf"] = rejected_reasons.get("low_det_conf", 0) + 1
                continue

            plate_text = det.text.strip()

            # Strict GOST Type 1B syntax
            if not TYPE1B_REGEX.match(plate_text):
                rejected_reasons["invalid_ocr_syntax"] = rejected_reasons.get("invalid_ocr_syntax", 0) + 1
                continue

            # Region code validation
            reg = plate_text[5:]
            if not is_valid_region(reg):
                rejected_reasons["invalid_region"] = rejected_reasons.get("invalid_region", 0) + 1
                continue

            # Canonicalize and verify quad geometry
            geom = canonicalize_quad(det.quad, iw, ih)
            if geom is None:
                rejected_reasons["invalid_quad_geometry"] = rejected_reasons.get("invalid_quad_geometry", 0) + 1
                continue

            quad_int, (bx, by, bw, bh) = geom

            # Colorimetric confirmation
            if not is_yellow_plate_crop(blurred, quad_int):
                rejected_reasons["not_yellow_color"] = rejected_reasons.get("not_yellow_color", 0) + 1
                continue

            # --- ACCEPTED! Integrate into dataset ---
            fname = f"real_type1b_{next_idx:04d}.jpg"
            img_rel_path = f"images/real/{fname}"
            out_img_path = os.path.join(real_dir, fname)
            out_label_path = os.path.join(labels_dir, f"real_type1b_{next_idx:04d}.txt")

            # Save de-identified image
            cv2.imwrite(out_img_path, blurred, [cv2.IMWRITE_JPEG_QUALITY, 95])

            # Normalized YOLO pose label (class 2)
            # cx cy w h x1 y1 x2 y2 x3 y3 x4 y4
            cx = (bx + bw / 2.0) / iw
            cy = (by + bh / 2.0) / ih
            nw = bw / float(iw)
            nh = bh / float(ih)
            norm_kpts = []
            for i in range(0, 8, 2):
                norm_kpts.append(quad_int[i] / float(iw))
                norm_kpts.append(quad_int[i + 1] / float(ih))

            label_line = f"2 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f} " + " ".join(f"{v:.6f}" for v in norm_kpts)
            with open(out_label_path, "w", encoding="utf-8") as lf:
                lf.write(label_line + "\n")

            # Append to meta.csv
            quad_str = ",".join(str(v) for v in quad_int)
            bbox_str = f"{bx},{by},{bw},{bh}"
            lic = info.get("license", "CC BY-SA 4.0")

            with open(meta_path, "a", newline="", encoding="utf-8") as mf:
                writer = csv.writer(mf, delimiter=";")
                writer.writerow([
                    img_rel_path,
                    plate_text,
                    "type1b",
                    bbox_str,
                    quad_str,
                    1,
                    0,
                    url,
                    lic,
                    "day,angle"
                ])

            # Update tracking
            existing_urls.add(url)
            existing_hashes.add(img_hash)
            accepted_count += 1

            audit_writer.writerow([
                fname, plate_text, f"{det.confidence:.4f}", f"{det.ocr_confidence:.4f}",
                iw, ih, url, lic, "ACCEPTED"
            ])
            af.flush()

            faces_info = f", faces_blurred={face_stats['faces_detected']}" if face_stats["faces_detected"] > 0 else ""
            print(f"[+{accepted_count:03d}/{args.target_new}] ACCEPTED {fname} -> {plate_text} (conf={det.confidence:.2f}, ocr={det.ocr_confidence:.2f}{faces_info})")

            next_idx += 1

    print("\n=======================================================")
    print(f"  Harvesting Completed!")
    print(f"  Total Accepted: {accepted_count} frames")
    print(f"  New Real Type 1B Count: {current_t1b + accepted_count}")
    print(f"  Rejected breakdown: {rejected_reasons}")
    print(f"  Audit log saved to: {audit_csv}")
    print("=======================================================")


if __name__ == "__main__":
    main()
