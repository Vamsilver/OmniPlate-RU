import csv
import sys
from pathlib import Path
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.pipeline import OmniPlatePipeline
from src.pipeline.decoder import is_valid_gost_plate

with open(PROJECT_ROOT / "dataset" / "meta.csv", "r", encoding="utf-8") as f:
    reader = list(csv.DictReader(f, delimiter=";"))

real_other_rows = {r["image"]: r for r in reader if "real_other_" in r["image"]}

pipeline = OmniPlatePipeline(device="cuda")
pipeline.warmup(2)

rows_to_check = []
for rel_img, r in real_other_rows.items():
    img_path = PROJECT_ROOT / "dataset" / rel_img
    img = cv2.imread(str(img_path))
    if img is None:
        continue
    h, w = img.shape[:2]

    # Pipeline detect
    dets = pipeline.predict(img)
    best_det = None
    if dets:
        def sort_key(d):
            is_target = 1 if d.plate_type in ("type1", "type1a", "type1b") else 0
            has_text = 1 if d.text and "#" not in d.text else 0
            quality = (d.confidence ** 0.5) * (d.ocr_confidence ** 2)
            return (is_target, has_text, quality, d.confidence)
        dets = sorted(dets, key=sort_key, reverse=True)
        best_det = dets[0]

    # OCR direct on crop or image
    t1_txt, t1_c = pipeline.ocr.predict_single(img, "type1")
    t1a_txt, t1a_c = pipeline.ocr.predict_single(img, "type1a")
    t1b_txt, t1b_c = pipeline.ocr.predict_single(img, "type1b")

    meta_num = r["plate_num"]
    meta_type = r["plate_type"]

    is_gost_det = best_det and best_det.plate_type in ("type1", "type1a", "type1b") and is_valid_gost_plate(best_det.text, best_det.plate_type)
    is_gost_crop = is_valid_gost_plate(t1_txt, "type1") and t1_c > 0.85
    is_gost_meta = is_valid_gost_plate(meta_num, "type1")

    if is_gost_det or is_gost_crop or is_gost_meta or (meta_num != "###" and meta_num != ""):
        rows_to_check.append({
            "image": rel_img,
            "shape": (w, h),
            "meta_num": meta_num,
            "meta_type": meta_type,
            "meta_bbox": r["bbox"],
            "meta_quad": r["quad"],
            "source": r["source"],
            "best_det": best_det,
            "t1": (t1_txt, t1_c),
            "t1a": (t1a_txt, t1a_c),
            "t1b": (t1b_txt, t1b_c),
            "is_gost_det": is_gost_det,
            "is_gost_crop": is_gost_crop,
            "is_gost_meta": is_gost_meta,
        })

print(f"Total rows to check: {len(rows_to_check)}")
for item in rows_to_check:
    det_s = f"{item['best_det'].plate_type}:{item['best_det'].text}(det={item['best_det'].confidence:.2f},ocr={item['best_det'].ocr_confidence:.2f})" if item["best_det"] else "NO_DET"
    print(f"{item['image']} | shape={item['shape']} | meta={item['meta_num']} | det={det_s} | t1_ocr={item['t1'][0]}({item['t1'][1]:.2f}) | src={item['source'][:50]}")
