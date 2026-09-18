import csv
import json
import os
import sys
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(r"D:\AIProjects\VolgaIT")
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.rectifier import PlateRectifier
from src.pipeline.ocr import PlateOCR

ocr = PlateOCR(model_path="models/ocr_lprnet_best.pt", device="cuda")
rectifier = PlateRectifier()

META_PATH = PROJECT_ROOT / "dataset" / "meta.csv"

# Read all real type1a rows
rows = []
with open(META_PATH, "r", encoding="utf-8") as f:
    reader = csv.reader(f, delimiter=";")
    header = next(reader)
    for idx, r in enumerate(reader, start=2):
        if len(r) > 6 and r[2] == "type1a" and r[6] == "0":
            rows.append((idx, r))

print(f"Total real type1a rows in meta.csv: {len(rows)}")

audit_results = []
OUT_DIR = PROJECT_ROOT / "test_output" / "deep_audit_type1a"
OUT_DIR.mkdir(parents=True, exist_ok=True)

for line_idx, r in rows:
    rel_path = r[0]
    gt_num = r[1]
    bbox_str = r[3]
    quad_str = r[4]
    source = r[7]
    
    img_path = PROJECT_ROOT / "dataset" / rel_path
    if not img_path.exists():
        audit_results.append({
            "rel_path": rel_path,
            "status": "MISSING_FILE",
            "gt": gt_num
        })
        continue
        
    img = cv2.imread(str(img_path))
    if img is None:
        audit_results.append({
            "rel_path": rel_path,
            "status": "CORRUPT_IMAGE",
            "gt": gt_num
        })
        continue
        
    h_im, w_im = img.shape[:2]
    pts = rectifier.parse_quad(quad_str)
    
    # Aspect ratio
    w_top = np.linalg.norm(pts[1] - pts[0])
    w_bot = np.linalg.norm(pts[2] - pts[3])
    h_left = np.linalg.norm(pts[3] - pts[0])
    h_right = np.linalg.norm(pts[2] - pts[1])
    aspect = ((w_top + w_bot) / 2.0) / max(1.0, (h_left + h_right) / 2.0)
    
    # Rectify 1A (96x160)
    rect_1a = rectifier.rectify(img, pts, plate_type="type1a")
    top_l, bot_l = rectifier.split_type1a(rect_1a, adaptive_seam=True)
    stitched_1a = rectifier.stitch_type1a_horizontal(top_l, bot_l, target_size=(160, 36))
    txt_1a, conf_1a = ocr.predict_single(stitched_1a, plate_type="type1a")
    
    # Rectify 1 (36x160)
    rect_1 = rectifier.rectify(img, pts, plate_type="type1")
    txt_1, conf_1 = ocr.predict_single(rect_1, plate_type="type1")[:2]
    
    # Text energy distribution
    gray = cv2.cvtColor(rect_1a, cv2.COLOR_BGR2GRAY)
    sob = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    row_means = np.mean(sob, axis=1) # 96
    
    mid_energy = float(np.mean(row_means[42:54]))
    top_energy = float(np.mean(row_means[15:35]))
    bot_energy = float(np.mean(row_means[60:80]))
    
    # Check if single line cuts through center
    # In single-line plate, row 48 has dense strokes, whereas in 2-line it is the border between lines
    # Also evaluate whether txt_1 matches GT better than txt_1a
    match_1a = (txt_1a == gt_num)
    match_1 = (txt_1 == gt_num)
    
    # Suspicious flag:
    # 1. aspect >= 2.15
    # 2. txt_1 matches GT while txt_1a fails
    # 3. conf_1 >= 0.85 while conf_1a <= 0.40
    is_suspicious_1line = False
    reason = "PASS"
    
    if match_1 and not match_1a:
        is_suspicious_1line = True
        reason = f"Type 1 OCR matches GT ({txt_1}), 1A failed ({txt_1a})"
    elif conf_1 >= 0.85 and conf_1a < 0.45:
        is_suspicious_1line = True
        reason = f"Type 1 confidence high ({conf_1:.2f}) vs 1A low ({conf_1a:.2f})"
    elif aspect >= 2.20:
        is_suspicious_1line = True
        reason = f"Aspect ratio {aspect:.2f} >= 2.20"
        
    audit_results.append({
        "rel_path": rel_path,
        "gt": gt_num,
        "aspect": float(aspect),
        "txt_1a": txt_1a,
        "conf_1a": float(conf_1a),
        "txt_1": txt_1,
        "conf_1": float(conf_1),
        "is_suspicious_1line": is_suspicious_1line,
        "reason": reason,
        "source": source
    })

# Summary
suspicious = [r for r in audit_results if r["is_suspicious_1line"]]
pure_1a = [r for r in audit_results if not r["is_suspicious_1line"]]

print("\n" + "=" * 60)
print(f"DEEP AUDIT RESULTS:")
print(f"  • Total Type 1A audited: {len(audit_results)}")
print(f"  • Confirmed True 2-row Type 1A: {len(pure_1a)}")
print(f"  • Suspicious Single-line Type 1: {len(suspicious)}")
print("=" * 60)

if suspicious:
    print("\nSuspicious cases sample (first 15):")
    for s in suspicious[:15]:
        print(f"  {s['rel_path']} | GT: {s['gt']} | AR: {s['aspect']:.2f} | 1A: {s['txt_1a']} ({s['conf_1a']:.2f}) vs 1: {s['txt_1']} ({s['conf_1']:.2f}) | {s['reason']}")

with open(PROJECT_ROOT / "test_output" / "deep_audit_type1a.json", "w", encoding="utf-8") as f:
    json.dump(audit_results, f, indent=2, ensure_ascii=False)
