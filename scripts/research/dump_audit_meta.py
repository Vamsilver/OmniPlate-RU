"""
Dump detailed metadata and pipeline predictions for each batch of real_other images.
"""
import csv
import sys
from pathlib import Path
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.pipeline import OmniPlatePipeline
from src.pipeline.decoder import is_valid_gost_plate

def run_metadata_dump():
    pipeline = OmniPlatePipeline(device="cuda")
    pipeline.warmup(2)

    with open(PROJECT_ROOT / "dataset" / "meta.csv", "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f, delimiter=";"))

    meta_map = {r["image"]: r for r in reader}
    img_dir = PROJECT_ROOT / "dataset" / "images" / "real"
    other_imgs = sorted(list(img_dir.glob("real_other_*.jpg")))

    print(f"Total: {len(other_imgs)}")

    out_file = PROJECT_ROOT / "test_output" / "visual_audit" / "audit_meta_report.txt"
    with open(out_file, "w", encoding="utf-8") as out:
        for idx, img_path in enumerate(other_imgs, start=1):
            rel_path = f"images/real/{img_path.name}"
            meta = meta_map.get(rel_path, {})
            source = meta.get("source", "")
            orig_plate = meta.get("plate_num", "")
            orig_type = meta.get("plate_type", "")
            
            img = cv2.imread(str(img_path))
            h, w = img.shape[:2] if img is not None else (0, 0)
            ar = w / float(h) if h > 0 else 0

            # Run detection
            dets = pipeline.predict(img) if img is not None else []
            det_info = "NONE"
            if dets:
                d = dets[0]
                det_info = f"type={d.plate_type}, text='{d.text}', conf={d.confidence:.2f}, ocr_conf={d.ocr_confidence:.2f}, valid={is_valid_gost_plate(d.text, d.plate_type)}"

            line = f"#{idx:03d} | {img_path.name} | {w}x{h} (ar={ar:.2f}) | source: {source} | pipeline: {det_info}"
            out.write(line + "\n")
            print(line)

if __name__ == "__main__":
    run_metadata_dump()
