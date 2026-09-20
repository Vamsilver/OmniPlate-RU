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

other_rows = [r for r in reader if "real_other_" in r["image"]]
pipeline = OmniPlatePipeline(device="cuda")
pipeline.warmup(2)

results = []

for r in other_rows:
    img_rel = r["image"]
    img_path = PROJECT_ROOT / "dataset" / img_rel
    img = cv2.imread(str(img_path))
    if img is None:
        results.append({
            "image": img_rel, "status": "MISSING", "meta_num": r["plate_num"],
            "meta_type": r["plate_type"], "source": r["source"]
        })
        continue

    h, w = img.shape[:2]

    # 1. Pipeline predict
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

    # 2. Raw detector at low threshold if no det
    raw_box = None
    if not best_det:
        res = pipeline.detector.predict(source=img, conf=0.01, verbose=False)[0]
        if res.boxes is not None and len(res.boxes) > 0:
            b_conf = float(res.boxes.conf[0].item())
            b_cls = int(res.boxes.cls[0].item())
            b_xyxy = res.boxes.xyxy[0].cpu().numpy()
            raw_box = (b_cls, b_conf, b_xyxy)

    # 3. Direct OCR on image (if image aspect ratio looks like a plate or tight crop)
    img_ar = w / float(h)
    direct_ocr_t1 = None
    direct_ocr_t1b = None
    if (img_ar > 2.0 and h < 300) or w < 500:
        t1_txt, t1_c = pipeline.ocr.predict_single(img, "type1")
        t1b_txt, t1b_c = pipeline.ocr.predict_single(img, "type1b")
        direct_ocr_t1 = (t1_txt, t1_c)
        direct_ocr_t1b = (t1b_txt, t1b_c)

    results.append({
        "image": img_rel,
        "shape": (w, h),
        "meta_num": r["plate_num"],
        "meta_type": r["plate_type"],
        "source": r["source"],
        "det": best_det,
        "raw_box": raw_box,
        "direct_ocr_t1": direct_ocr_t1,
        "direct_ocr_t1b": direct_ocr_t1b,
    })

# Print summary
candidates = []
for res in results:
    img = res["image"]
    mnum = res["meta_num"]
    det = res["det"]
    doc_t1 = res["direct_ocr_t1"]
    
    # Check if pipeline detected valid gost
    found_gost = False
    gost_type = None
    gost_text = None
    reason = ""

    if det and det.plate_type in ("type1", "type1a", "type1b") and is_valid_gost_plate(det.text, det.plate_type):
        found_gost = True
        gost_type = det.plate_type
        gost_text = det.text
        reason = f"pipeline_det(conf={det.confidence:.2f},ocr={det.ocr_confidence:.2f})"
    elif doc_t1 and doc_t1[1] > 0.85 and is_valid_gost_plate(doc_t1[0], "type1"):
        found_gost = True
        gost_type = "type1"
        gost_text = doc_t1[0]
        reason = f"direct_crop_ocr(conf={doc_t1[1]:.2f})"
    elif is_valid_gost_plate(mnum, "type1"):
        found_gost = True
        gost_type = "type1"
        gost_text = mnum
        reason = f"meta_csv_type1"

    if found_gost:
        candidates.append((img, gost_type, gost_text, reason, res))

print(f"\n==========================================")
print(f"TOTAL CANDIDATES FOR RECLASSIFICATION: {len(candidates)}")
print(f"==========================================")
for c in candidates:
    print(f"{c[0]}: target={c[1]} text={c[2]} reason={c[3]} src={c[4]['source'][:60]}")

remaining_other = len(other_rows) - len(candidates)
print(f"\nRemaining true other frames: {remaining_other} (Quota min: 50)")
