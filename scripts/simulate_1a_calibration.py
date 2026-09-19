import csv
import json
import os
import sys
from pathlib import Path
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.pipeline import OmniPlatePipeline, is_valid_gost_plate
from scripts.audit_type1a_errors import normalize_plate, is_seq_match, levenshtein_dist

meta_path = PROJECT_ROOT / "dataset" / "meta.csv"
rows_1a = []
with open(meta_path, "r", encoding="utf-8") as f:
    for r in csv.DictReader(f, delimiter=";"):
        if r.get("is_synthetic", "0").strip() == "0" and r.get("plate_type", "").strip() == "type1a":
            rows_1a.append(r)

print(f"Testing {len(rows_1a)} Type 1A images with calibrated guard...")

pipe = OmniPlatePipeline(device="cuda", conf_threshold=0.12, ocr_version="moe")
pipe.warmup(2)

# Check how many were recovered
recovered = []
for idx, r in enumerate(rows_1a):
    img_path = PROJECT_ROOT / "dataset" / r["image"]
    img = cv2.imread(str(img_path))
    if img is None:
        continue
    
    dets = pipe.detect(img)
    if not dets:
        continue
    
    det = dets[0]
    gt_num = normalize_plate(r["plate_num"])
    
    # Run recognize_single
    pipe.recognize_single(img, det)
    
    # If it was marked other, test if calibrated guard would rescue it
    if det.plate_type == "other":
        rect_1a = pipe.rectifier.rectify(img, det.quad, plate_type="type1a", margin=(0.020, 0.015), refine_corners=True)
        txt_1a, conf_1a = pipe.ocr.predict_type1a_native(rect_1a)
        is_gost = is_valid_gost_plate(txt_1a, "type1a", allow_wildcards=False)
        match = is_seq_match(txt_1a, gt_num)
        
        # Test rescue conditions
        is_high = (
            is_gost
            and conf_1a >= 0.85
            and det.confidence >= 0.20
            and (det.confidence * conf_1a) >= 0.18
        )
        if is_high:
            recovered.append({
                "image": r["image"],
                "gt": gt_num,
                "txt": txt_1a,
                "conf_1a": round(conf_1a, 3),
                "det_conf": round(det.confidence, 3),
                "is_match": match
            })

print(f"\nRescued Type 1A candidates: {len(recovered)}")
for item in recovered:
    print(item)
