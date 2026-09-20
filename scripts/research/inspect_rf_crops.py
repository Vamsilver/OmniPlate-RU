import cv2
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.pipeline import OmniPlatePipeline
from src.pipeline.decoder import is_valid_gost_plate

pipeline = OmniPlatePipeline(device="cuda")
pipeline.warmup(1)

out_crops = PROJECT_ROOT / "test_output" / "visual_audit" / "crops"
out_crops.mkdir(parents=True, exist_ok=True)

# List of roboflow images in batch 9: real_other_0252 to 0272
rf_imgs = sorted(list((PROJECT_ROOT / "dataset" / "images" / "real").glob("real_other_025*.jpg"))) + \
          sorted(list((PROJECT_ROOT / "dataset" / "images" / "real").glob("real_other_026*.jpg"))) + \
          sorted(list((PROJECT_ROOT / "dataset" / "images" / "real").glob("real_other_027*.jpg")))

for p in rf_imgs:
    img = cv2.imread(str(p))
    if img is None:
        continue
    dets = pipeline.predict(img)
    print(f"--- {p.name} ---")
    if not dets:
        print("  No detection")
    for i, d in enumerate(dets):
        print(f"  Det {i}: type={d.plate_type}, text='{d.text}', conf={d.confidence:.2f}, ocr_conf={d.ocr_confidence:.2f}")
        if d.rectified_crop is not None:
            crop_path = out_crops / f"{p.stem}_crop_{i}_{d.plate_type}_{d.text}.jpg"
            cv2.imwrite(str(crop_path), d.rectified_crop)
