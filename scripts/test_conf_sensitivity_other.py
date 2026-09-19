import csv
import os
import sys
from pathlib import Path
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.pipeline import OmniPlatePipeline

meta_path = PROJECT_ROOT / "dataset" / "meta.csv"
other_images = []
with open(meta_path, "r", encoding="utf-8") as f:
    for r in csv.DictReader(f, delimiter=";"):
        if r.get("is_synthetic", "0").strip() == "0" and r.get("plate_type", "").strip() == "other":
            other_images.append(r["image"].strip())

print(f"Loaded {len(other_images)} negative images for sensitivity test.")

thresholds = [0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]
results = {}

for conf in thresholds:
    pipe = OmniPlatePipeline(device="cuda", conf_threshold=conf, ocr_version="moe")
    fp_count = 0
    fp_details = []
    
    for img_rel in other_images:
        img_path = PROJECT_ROOT / "dataset" / img_rel
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        dets = pipe.predict(img)
        # Any valid detected plate on other is a Fatal Penalty!
        valid_dets = [d for d in dets if d.plate_type in ("type1", "type1a", "type1b") and d.text]
        if valid_dets:
            fp_count += len(valid_dets)
            fp_details.append((img_rel, [(d.plate_type, d.text, d.confidence, d.ocr_confidence) for d in valid_dets]))
            
    tn_rate = ((len(other_images) - len(fp_details)) / len(other_images)) * 100.0
    results[conf] = {
        "fps": fp_count,
        "tn_rate": round(tn_rate, 2),
        "fp_details": fp_details
    }
    print(f"Conf={conf:4.2f} -> FP={fp_count}, TN Rate={tn_rate:.2f}% (Fatal Penalties: {fp_count})")

print("\nSensitivity Test Results:")
for conf, res in results.items():
    print(f"  Threshold {conf:4.2f}: Fatal Penalties = {res['fps']} (TN Rate: {res['tn_rate']}%)")
