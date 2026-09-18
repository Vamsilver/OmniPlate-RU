import csv
import os
import cv2
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(r"D:\AIProjects\VolgaIT")
sys.path.insert(0, str(PROJECT_ROOT))
from src.pipeline.rectifier import PlateRectifier

rectifier = PlateRectifier()

META_PATH = PROJECT_ROOT / "dataset" / "meta.csv"
OUT_DIR = PROJECT_ROOT / "test_output" / "type1a_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

rows = []
with open(META_PATH, "r", encoding="utf-8") as f:
    reader = csv.reader(f, delimiter=";")
    header = next(reader)
    for idx, r in enumerate(reader, start=2):
        if len(r) > 6 and r[2] == "type1a" and r[6] == "0":
            rows.append((idx, r))

print(f"Loaded {len(rows)} real type1a rows from meta.csv")

results = []

for line_idx, r in rows:
    rel_path = r[0]
    plate_num = r[1]
    bbox_str = r[3]
    quad_str = r[4]
    
    img_path = PROJECT_ROOT / "dataset" / rel_path
    if not img_path.exists():
        results.append({
            "line_idx": line_idx,
            "rel_path": rel_path,
            "plate_num": plate_num,
            "status": "MISSING_IMAGE",
            "score": 0.0,
            "reason": "File does not exist"
        })
        continue
        
    img = cv2.imread(str(img_path))
    if img is None:
        results.append({
            "line_idx": line_idx,
            "rel_path": rel_path,
            "plate_num": plate_num,
            "status": "CORRUPT_IMAGE",
            "score": 0.0,
            "reason": "cv2.imread returned None"
        })
        continue
        
    pts = rectifier.parse_quad(quad_str)
    warped = rectifier.rectify(img, pts, "type1a") # (96, 160, 3)
    top_l, bot_l = rectifier.split_type1a(warped, adaptive_seam=True)
    
    # Analyze edge / contrast density in top vs bottom
    gray_top = cv2.cvtColor(top_l, cv2.COLOR_BGR2GRAY)
    gray_bot = cv2.cvtColor(bot_l, cv2.COLOR_BGR2GRAY)
    
    edges_top = cv2.Canny(gray_top, 50, 150)
    edges_bot = cv2.Canny(gray_bot, 50, 150)
    
    density_top = np.sum(edges_top > 0) / edges_top.size
    density_bot = np.sum(edges_bot > 0) / edges_bot.size
    
    # Physical aspect ratio of original polygon:
    # width / height in original scene
    w_top = np.linalg.norm(pts[1] - pts[0])
    w_bot = np.linalg.norm(pts[2] - pts[3])
    h_left = np.linalg.norm(pts[3] - pts[0])
    h_right = np.linalg.norm(pts[2] - pts[1])
    avg_w = (w_top + w_bot) / 2.0
    avg_h = (h_left + h_right) / 2.0
    aspect = avg_w / avg_h if avg_h > 0 else 0
    
    # Save composite debug crop: [original_crop, warped, top_line, bottom_line]
    # For quick visual review
    b_parts = [int(x) for x in bbox_str.split(",")]
    bx, by, bw, bh = b_parts
    h_im, w_im = img.shape[:2]
    crop = img[max(0, by):min(h_im, by+bh), max(0, bx):min(w_im, bx+bw)]
    if crop.size > 0:
        crop_resized = cv2.resize(crop, (160, 96))
    else:
        crop_resized = np.zeros((96, 160, 3), dtype=np.uint8)
        
    split_vis = np.vstack([top_l, bot_l]) # 96x160
    # draw red dividing line
    cv2.line(split_vis, (0, 48), (160, 48), (0, 0, 255), 1)
    
    combo = np.hstack([crop_resized, warped, split_vis])
    cv2.putText(combo, f"{plate_num} AR:{aspect:.2f}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    crop_fname = f"{Path(rel_path).stem}_{plate_num}.jpg"
    cv2.imwrite(str(OUT_DIR / crop_fname), combo)
    
    results.append({
        "line_idx": line_idx,
        "rel_path": rel_path,
        "plate_num": plate_num,
        "aspect": aspect,
        "density_top": density_top,
        "density_bot": density_bot,
        "crop_fname": crop_fname
    })

print(f"Processed {len(results)} rows and saved audit crops to {OUT_DIR}")
