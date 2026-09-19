import csv
import json
import os
import sys
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.pipeline import OmniPlatePipeline, is_valid_gost_plate
from scripts.audit_type1a_errors import normalize_plate, is_seq_match, levenshtein_dist

meta_path = PROJECT_ROOT / "dataset" / "meta.csv"
real_rows = []
with open(meta_path, "r", encoding="utf-8") as f:
    for r in csv.DictReader(f, delimiter=";"):
        if r.get("is_synthetic", "0").strip() == "0":
            real_rows.append(r)

print(f"Total real images: {len(real_rows)}")

# Test current pipeline vs patched pipeline behavior
# Let's inspect other_0099 and other_0179 first
pipe = OmniPlatePipeline(device="cuda", conf_threshold=0.12, ocr_version="moe")
for img_name in ["real_other_0099.jpg", "real_other_0179.jpg"]:
    img = cv2.imread(str(PROJECT_ROOT / "dataset" / "images" / "real" / img_name))
    if img is not None:
        dets = pipe.predict(img)
        print(img_name, [(d.plate_type, d.text, d.confidence, d.ocr_confidence) for d in dets])
