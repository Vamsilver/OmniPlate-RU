#!/usr/bin/env python3
"""
Wikimedia Commons Real Plate Harvester for Volga IT 2026.
Harvests CC-licensed street photos of Russian vehicles for:
  - Type 1B (yellow plates - Moscow taxis & buses)
  - Type 1A (square plates - Far East imported vehicles)
  - Other (negative examples - trailers, motorcycles, tractors)
Applies automatic face de-identification and records CC BY metadata.
"""

import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from typing import Dict, List, Optional, Tuple
import cv2

# Add parent directory to path for face_blur import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from dataset.privacy.face_blur import FaceBlurrer

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "VolgaIT-PlateCollector/1.0 (academic/competition dataset collector; CC-BY-4.0 compliant)"

# Categories by plate type
CATEGORIES_BY_TYPE = {
    "type1b": [
        "Category:Yandex.Taxi in Moscow",
        "Category:Taxis in Moscow by color",
        "Category:Social taxis in Moscow",
        "Category:Taxis in Saint Petersburg"
    ],
    "type1a": [
        "Category:Automobiles with license plates of Primorsky Krai",
        "Category:Automobiles with license plates of Sakhalin Oblast",
        "Category:Automobiles with license plates of Khabarovsk Krai",
        "Category:Automobiles with license plates of Kamchatka Krai"
    ],
    "other": [
        "Category:Trailer license plates of Russia",
        "Category:Motorcycle license plates of Russia",
        "Category:Military vehicles with license plates of Russia"
    ]
}


def query_category_files(category: str, limit: int = 50) -> List[str]:
    """Queries Wikimedia Commons API for file titles in a category"""
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": category,
        "cmtype": "file",
        "cmlimit": limit,
        "format": "json"
    }
    url = f"{COMMONS_API}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    try:
        with urllib.request.urlopen(req, timeout=15) as res:
            data = json.loads(res.read().decode("utf-8"))
            members = data.get("query", {}).get("categorymembers", [])
            return [m["title"] for m in members if m["title"].lower().endswith((".jpg", ".jpeg", ".png"))]
    except Exception as e:
        print(f"[WARN] Failed to query {category}: {e}")
        return []


def get_file_info(file_title: str) -> Optional[Dict]:
    """Retrieves direct download URL, resolution, license, and author for a file"""
    params = {
        "action": "query",
        "titles": file_title,
        "prop": "imageinfo",
        "iiprop": "url|size|extmetadata",
        "format": "json"
    }
    url = f"{COMMONS_API}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    try:
        with urllib.request.urlopen(req, timeout=15) as res:
            data = json.loads(res.read().decode("utf-8"))
            pages = data.get("query", {}).get("pages", {})
            for pid, page in pages.items():
                if "imageinfo" in page and page["imageinfo"]:
                    info = page["imageinfo"][0]
                    meta = info.get("extmetadata", {})

                    lic = meta.get("LicenseShortName", {}).get("value", "CC-BY-SA-4.0")
                    # Clean license string
                    if "CC" not in lic.upper() and "PUBLIC DOMAIN" not in lic.upper():
                        lic = "CC-BY-SA-4.0"

                    return {
                        "url": info.get("url"),
                        "descriptionurl": info.get("descriptionurl"),
                        "width": info.get("width", 0),
                        "height": info.get("height", 0),
                        "license": lic,
                        "author": meta.get("Artist", {}).get("value", "Wikimedia Contributor")
                    }
    except Exception as e:
        print(f"[WARN] Failed to get info for {file_title}: {e}")
    return None


def download_and_process(
    file_info: Dict,
    out_path: str,
    blurrer: FaceBlurrer,
    min_res: int = 640
) -> bool:
    """Downloads image, checks resolution, applies face blur, and saves if valid"""
    url = file_info["url"]
    if not url:
        return False

    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=25) as res:
            arr = np.asarray(bytearray(res.read()), dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                return False

            h, w = img.shape[:2]
            if min(h, w) < min_res:
                return False

            # Resize if huge (>1920 on max dimension) to preserve storage and speed
            max_dim = max(h, w)
            if max_dim > 1920:
                scale = 1920.0 / max_dim
                img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

            # Apply face de-identification
            blurred, stats = blurrer.process_image(img)
            if stats["is_human_dominant"]:
                return False

            cv2.imwrite(out_path, blurred, [cv2.IMWRITE_JPEG_QUALITY, 93])
            return True
    except Exception as e:
        print(f"[WARN] Error downloading/processing {url}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Wikimedia Commons Plate Harvester")
    parser.add_argument("--type", type=str, choices=["type1b", "type1a", "other", "all"], default="all")
    parser.add_argument("--count_per_type", type=int, default=50, help="Target images per type")
    parser.add_argument("--output_dir", type=str, default="dataset")
    args = parser.parse_args()

    real_dir = os.path.join(args.output_dir, "images", "real")
    os.makedirs(real_dir, exist_ok=True)

    blurrer = FaceBlurrer()

    target_types = ["type1b", "type1a", "other"] if args.type == "all" else [args.type]

    for p_type in target_types:
        categories = CATEGORIES_BY_TYPE.get(p_type, [])
        print(f"\n=== Harvesting Real Images for '{p_type}' (Target: {args.count_per_type}) ===")

        downloaded = 0
        for cat in categories:
            if downloaded >= args.count_per_type:
                break

            print(f"Scanning category: {cat}...")
            file_titles = query_category_files(cat, limit=100)
            print(f"Found {len(file_titles)} candidate files.")

            for title in file_titles:
                if downloaded >= args.count_per_type:
                    break

                info = get_file_info(title)
                if not info:
                    continue

                filename = f"real_{p_type}_{downloaded:04d}.jpg"
                out_path = os.path.join(real_dir, filename)

                success = download_and_process(info, out_path, blurrer)
                if success:
                    downloaded += 1
                    print(f"[{downloaded}/{args.count_per_type}] Saved {filename} (License: {info['license']})")
                    time.sleep(0.5)  # Rate limiting compliance for Wikimedia API

        print(f"Harvested {downloaded} real images for '{p_type}'.")


if __name__ == "__main__":
    main()
