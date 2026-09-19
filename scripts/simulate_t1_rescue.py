import csv, os, sys
from pathlib import Path
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.pipeline import OmniPlatePipeline, is_valid_gost_plate
from scripts.audit_type1a_errors import normalize_plate, is_seq_match

pipe = OmniPlatePipeline(device="cuda", conf_threshold=0.12, ocr_version="moe")
pipe.warmup(1)

# 1. Test on other
other_rows = [r for r in csv.DictReader(open(PROJECT_ROOT / "dataset" / "meta.csv", encoding="utf-8"), delimiter=";") if r.get("is_synthetic", "0").strip() == "0" and r.get("plate_type", "").strip() == "other"]

other_fps = []
for r in other_rows:
    img = cv2.imread(str(PROJECT_ROOT / "dataset" / r["image"]))
    if img is None: continue
    dets = pipe.detect(img)
    for det in dets:
        rect = pipe.rectifier.rectify(img, det.quad, plate_type="type1")
        txt, ocr_c = pipe.ocr.predict_single(rect, plate_type="type1")[:2]
        is_gost = is_valid_gost_plate(txt, "type1", allow_wildcards=False)
        is_p, p_sc = pipe.verifier.verify_single(rect)
        if is_gost and ocr_c >= 0.95 and det.confidence >= 0.70 and p_sc >= 0.005:
            other_fps.append((r["image"], txt, ocr_c, det.confidence, p_sc))

print(f"False Positives on OTHER with calibrated threshold: {len(other_fps)}")
for fp in other_fps:
    print(" ", fp)

# 2. Test rescues on clean Type 1
t1_rows = [r for r in csv.DictReader(open(PROJECT_ROOT / "dataset" / "meta.csv", encoding="utf-8"), delimiter=";") if r.get("is_synthetic", "0").strip() == "0" and r.get("plate_type", "").strip() == "type1" and "#" not in r["plate_num"]]

rescued = []
for r in t1_rows:
    img = cv2.imread(str(PROJECT_ROOT / "dataset" / r["image"]))
    if img is None: continue
    dets = pipe.predict(img)
    gt = normalize_plate(r["plate_num"])
    
    # Check if currently annulled
    if dets and dets[0].text == "":
        det = pipe.detect(img)[0]
        rect = pipe.rectifier.rectify(img, det.quad, plate_type="type1")
        txt, ocr_c = pipe.ocr.predict_single(rect, plate_type="type1")[:2]
        is_gost = is_valid_gost_plate(txt, "type1", allow_wildcards=False)
        is_p, p_sc = pipe.verifier.verify_single(rect)
        if is_gost and ocr_c >= 0.95 and det.confidence >= 0.70 and p_sc >= 0.005:
            match = is_seq_match(txt, gt)
            rescued.append((r["image"], gt, txt, ocr_c, det.confidence, p_sc, match))

print(f"\nRescued Clean Type 1: {len(rescued)}")
for item in rescued:
    print(" ", item)
