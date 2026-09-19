import cv2
import os
import csv
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.pipeline import OmniPlatePipeline, is_valid_gost_plate

pipe = OmniPlatePipeline(device='cuda', conf_threshold=0.12, ocr_version='moe')
pipe.warmup(1)

meta_path = PROJECT_ROOT / 'dataset' / 'meta.csv'
other_rows = [r for r in csv.DictReader(open(meta_path, encoding='utf-8'), delimiter=';') if r.get('is_synthetic', '0').strip() == '0' and r.get('plate_type', '').strip() == 'other']

print(f"Total other rows: {len(other_rows)}")
high_signals = []

for r in other_rows:
    img = cv2.imread(str(PROJECT_ROOT / 'dataset' / r['image']))
    if img is None:
        continue
    dets = pipe.detect(img)
    for det in dets:
        rect = pipe.rectifier.rectify(img, det.quad, plate_type='type1')
        txt, ocr_c = pipe.ocr.predict_single(rect, plate_type='type1')[:2]
        is_gost = is_valid_gost_plate(txt, 'type1')
        is_p, p_sc = pipe.verifier.verify_single(rect) if pipe.verifier else (False, 0.0)
        
        # Also test 1A
        rect_1a = pipe.rectifier.rectify(img, det.quad, plate_type='type1a')
        txt_1a, ocr_1a_c = pipe.ocr.predict_type1a_native(rect_1a)
        is_gost_1a = is_valid_gost_plate(txt_1a, 'type1a')
        
        if is_gost or is_gost_1a or ocr_c > 0.40 or ocr_1a_c > 0.40:
            high_signals.append({
                "image": r['image'],
                "det_c": round(det.confidence, 3),
                "txt_1": txt, "ocr_1": round(ocr_c, 3), "gost_1": is_gost, "p_sc": round(p_sc, 3),
                "txt_1a": txt_1a, "ocr_1a": round(ocr_1a_c, 3), "gost_1a": is_gost_1a,
            })

print(f"High signal candidates on other: {len(high_signals)}")
for s in high_signals:
    print(s)
