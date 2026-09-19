import cv2
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.pipeline import OmniPlatePipeline, is_valid_gost_plate

with open("test_output/audit_type1a_75_errors.json", "r", encoding="utf-8") as f:
    data = json.load(f)

other_errors = [e for e in data["errors"] if e.get("category") == "MISCLASSIFIED_AS_OTHER"]
print(f"Analyzing {len(other_errors)} images misclassified as OTHER:")

pipe = OmniPlatePipeline(device="cuda", conf_threshold=0.12, ocr_version="moe")
pipe.warmup(1)

for e in other_errors:
    img_path = PROJECT_ROOT / "dataset" / e["image"]
    img = cv2.imread(str(img_path))
    dets = pipe.detect(img)
    if not dets:
        print(f"{e['image']} | NO DETECTION")
        continue
    det = dets[0]
    rect_1a = pipe.rectifier.rectify(img, det.quad, plate_type="type1a", margin=(0.020, 0.015), refine_corners=True)
    raw_txt, raw_conf = pipe.ocr.predict_type1a_native(rect_1a)
    is_gost = is_valid_gost_plate(raw_txt, "type1a")
    p_plate, p_score = pipe.verifier.verify_single(rect_1a) if pipe.verifier else (True, 1.0)
    print(f"{e['image']} | GT: {e['gt_num']:<9} | det_conf: {det.confidence:.3f} | raw: {raw_txt:<9} (c={raw_conf:.3f}) | GOST: {is_gost} | verifier: ({p_plate}, {p_score:.3f})")
