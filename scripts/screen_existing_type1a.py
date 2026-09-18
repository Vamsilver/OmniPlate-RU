import csv
import sys
import os
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

rows = []
with open(META_PATH, "r", encoding="utf-8") as f:
    reader = csv.reader(f, delimiter=";")
    header = next(reader)
    for idx, r in enumerate(reader, start=2):
        if len(r) > 6 and r[2] == "type1a" and r[6] == "0":
            rows.append((idx, r))

print(f"Screening {len(rows)} real type1a rows...")

audit_records = []

for line_idx, r in rows:
    rel_p = r[0]
    gt_num = r[1]
    bbox_str = r[3]
    quad_str = r[4]
    
    img_p = PROJECT_ROOT / "dataset" / rel_p
    img = cv2.imread(str(img_p))
    if img is None:
        audit_records.append({
            "line_idx": line_idx,
            "rel_path": rel_p,
            "gt": gt_num,
            "classification": "CORRUPT",
            "reason": "Missing or corrupt image"
        })
        continue
        
    pts = rectifier.parse_quad(quad_str)
    
    # 1. Physical aspect ratio of annotated quad
    w_top = np.linalg.norm(pts[1] - pts[0])
    w_bot = np.linalg.norm(pts[2] - pts[3])
    h_left = np.linalg.norm(pts[3] - pts[0])
    h_right = np.linalg.norm(pts[2] - pts[1])
    aspect = ((w_top + w_bot) / 2.0) / (max(1.0, (h_left + h_right) / 2.0))
    
    # 2. Hypothesis A: Type 1A 2-line split
    rect_1a = rectifier.rectify(img, pts, plate_type="type1a", margin=(0.0, 0.0))
    top_l, bot_l = rectifier.split_type1a(rect_1a, adaptive_seam=True)
    stitched_1a = rectifier.stitch_type1a_horizontal(top_l, bot_l, target_size=(160, 36))
    text_1a, conf_1a = ocr.predict_single(stitched_1a, plate_type="type1a")
    
    # 3. Hypothesis B: Type 1 direct single-line
    rect_1 = rectifier.rectify(img, pts, plate_type="type1", margin=(0.010, 0.010))
    ocr_res_1 = ocr.predict_single(rect_1, plate_type="type1", return_type=True)
    if len(ocr_res_1) == 3:
        text_1, conf_1, _ = ocr_res_1
    else:
        text_1, conf_1 = ocr_res_1[:2]
        
    # Check vertical center intensity vs top/bot
    gray = cv2.cvtColor(rect_1a, cv2.COLOR_BGR2GRAY)
    sob = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    row_means = np.mean(sob, axis=1) # 96 values
    
    # In true 1A, top tier is ~10..40, bottom tier is ~55..85
    # In single line plate, characters are right across y=30..65
    mid_energy = np.mean(row_means[43:53])
    top_energy = np.mean(row_means[15:35])
    bot_energy = np.mean(row_means[60:80])
    
    # Match analysis with GT
    match_1a = (text_1a == gt_num)
    match_1 = (text_1 == gt_num)
    
    # Score hypotheses
    score_1a = conf_1a * 10.0 - (text_1a.count("#") * 2.5) + (5.0 if match_1a else 0.0)
    score_1 = conf_1 * 10.0 - (text_1.count("#") * 2.5) + (5.0 if match_1 else 0.0)
    
    # Final classification logic:
    # A plate is definitively Type 1 (single-line) if:
    # - text_1 matches GT or has very high confidence while 1A is garbled
    # - OR aspect ratio >= 2.1 and score_1 > score_1a
    # - OR score_1 - score_1a > 4.0
    is_false_1a = False
    decision_reason = ""
    
    if match_1 and not match_1a:
        is_false_1a = True
        decision_reason = f"Exact match with Type 1 direct ({text_1}), 1A failed ({text_1a})"
    elif match_1a and not match_1:
        is_false_1a = False
        decision_reason = f"Exact match with Type 1A 2-line ({text_1a}), 1 direct failed ({text_1})"
    elif conf_1 >= 0.85 and conf_1a < 0.50:
        is_false_1a = True
        decision_reason = f"Type 1 OCR confidence high ({conf_1:.2f}) vs 1A low ({conf_1a:.2f})"
    elif conf_1a >= 0.85 and conf_1 < 0.50:
        is_false_1a = False
        decision_reason = f"Type 1A OCR confidence high ({conf_1a:.2f}) vs 1 low ({conf_1:.2f})"
    elif aspect >= 2.20:
        is_false_1a = True
        decision_reason = f"Physical aspect ratio {aspect:.2f} >= 2.20 (elongated strip)"
    elif aspect <= 1.45:
        is_false_1a = False
        decision_reason = f"Physical aspect ratio {aspect:.2f} <= 1.45 (square)"
    else:
        # Borderline: use relative scores
        if score_1 > score_1a + 2.0:
            is_false_1a = True
            decision_reason = f"Borderline AR {aspect:.2f}, Type 1 favored (score1={score_1:.1f} vs score1a={score_1a:.1f})"
        else:
            is_false_1a = False
            decision_reason = f"Borderline AR {aspect:.2f}, Type 1A favored (score1a={score_1a:.1f} vs score1={score_1:.1f})"
            
    audit_records.append({
        "line_idx": line_idx,
        "rel_path": rel_p,
        "gt": gt_num,
        "aspect": float(aspect),
        "is_false_1a": bool(is_false_1a),
        "decision": "TYPE1_SINGLE_LINE" if is_false_1a else "TYPE1A_TRUE_2ROW",
        "reason": decision_reason,
        "text_1a": text_1a,
        "conf_1a": float(conf_1a),
        "text_1": text_1,
        "conf_1": float(conf_1),
        "row_data": r
    })

false_count = sum(1 for a in audit_records if a["is_false_1a"])
true_count = sum(1 for a in audit_records if not a["is_false_1a"])

print(f"\nAudit complete across {len(audit_records)} frames:")
print(f"  • True 2-row Type 1A: {true_count}")
print(f"  • False Type 1A (single-line): {false_count}")

# Check breakdown in 0280..0401
records_280 = [a for a in audit_records if "real_type1a_028" in a["rel_path"] or "real_type1a_029" in a["rel_path"] or "real_type1a_03" in a["rel_path"] or "real_type1a_040" in a["rel_path"]]
false_280 = sum(1 for a in records_280 if a["is_false_1a"])
true_280 = sum(1 for a in records_280 if not a["is_false_1a"])
print(f"\nIn subset 0280..0401 ({len(records_280)} frames):")
print(f"  • True 2-row: {true_280}")
print(f"  • False 1-row: {false_280}")

# Save detailed JSON report
import json
with open(PROJECT_ROOT / "test_output" / "type1a_audit_decisions.json", "w", encoding="utf-8") as f:
    json.dump(audit_records, f, indent=2, ensure_ascii=False)
