import cv2
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.pipeline import OmniPlatePipeline
from src.pipeline.decoder import is_valid_gost_plate

pipeline = OmniPlatePipeline(device="cuda")
pipeline.warmup(1)

img_dir = PROJECT_ROOT / "dataset" / "images" / "real"
other_imgs = sorted(list(img_dir.glob("real_other_*.jpg")))

print(f"Scanning all {len(other_imgs)} images with low-threshold detector and OCR...")

interesting = []

for idx, p in enumerate(other_imgs, start=1):
    img = cv2.imread(str(p))
    if img is None:
        continue
    
    # 1. Standard pipeline
    dets = pipeline.predict(img)
    
    # 2. Raw YOLO detections with low conf (0.05)
    results = pipeline.detector.predict(source=img, conf=0.05, verbose=False)[0]
    
    raw_candidates = []
    if results.boxes is not None and len(results.boxes) > 0:
        for b_idx in range(len(results.boxes)):
            conf = float(results.boxes.conf[b_idx].item())
            cls_id = int(results.boxes.cls[b_idx].item())
            xyxy = results.boxes.xyxy[b_idx].cpu().numpy().astype(int)
            raw_candidates.append((cls_id, conf, xyxy))
            
    # Check if any det has non-other or valid text
    has_gost = False
    for d in dets:
        if d.plate_type in ("type1", "type1a", "type1b"):
            has_gost = True
            interesting.append((idx, p.name, "pipeline", d.plate_type, d.text, d.confidence, d.ocr_confidence))
            break
            
    if not has_gost and raw_candidates:
        for cls_id, conf, xyxy in raw_candidates:
            if cls_id in (0, 1, 2) and conf > 0.2:
                # Run OCR on this crop
                x1, y1, x2, y2 = xyxy
                crop = img[max(0, y1):min(img.shape[0], y2), max(0, x1):min(img.shape[1], x2)]
                if crop.size > 0:
                    t_name = {0: "type1", 1: "type1a", 2: "type1b"}[cls_id]
                    t_text, t_ocr_c = pipeline.ocr.predict_single(crop, t_name)
                    if is_valid_gost_plate(t_text, t_name):
                        interesting.append((idx, p.name, "raw_box", t_name, t_text, conf, t_ocr_c))

print(f"\nFound {len(interesting)} interesting detections:")
for item in interesting:
    print(item)
