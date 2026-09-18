import sys
from pathlib import Path
import pyarrow.parquet as pq
import cv2
import numpy as np
import re

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.pipeline import OmniPlatePipeline

pipeline = OmniPlatePipeline(device="cuda")
PLATE_REGEX = re.compile(r"^[ABEKMHOPCTYX][\d]{3}[ABEKMHOPCTYX]{2}(?:\d{2}|[127]\d{2})$")

def evaluate_seam_ratio(crop):
    if crop is None or crop.size == 0 or crop.shape[0] < 16 or crop.shape[1] < 16:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sob_y = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    norm_sob = cv2.resize(sob_y, (100, 100))
    p100 = np.mean(norm_sob, axis=1)
    mid_e = np.mean(p100[42:54])
    top_e = np.mean(p100[18:36])
    bot_e = np.mean(p100[60:78])
    return float((top_e + bot_e) / (2.0 * max(1e-3, mid_e)))

pq_p = r"C:\Users\Vamsi\Downloads\train-00000-of-00001.parquet"
table = pq.read_table(pq_p)
rows = table.to_pylist()
print("Total rows in train.parquet:", len(rows))

true_1a = []
for r in rows[:1500]:
    im_bytes = r["image"]["bytes"]
    arr = np.asarray(bytearray(im_bytes), dtype=np.uint8)
    im = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if im is None:
        continue
    h, w = im.shape[:2]
    res = pipeline.detector.predict(source=im, imgsz=640, conf=0.10, device='cuda', verbose=False)
    parsed = pipeline._parse_results(res[0], w, h)
    for det in parsed:
        bx, by, bw, bh = det.bbox
        ar = bw / max(1.0, bh)
        if not (1.08 <= ar <= 1.85):
            continue
        crop = im[max(0, by):min(h, by+bh), max(0, bx):min(w, bx+bw)]
        seam = evaluate_seam_ratio(crop)
        if seam < 1.05:
            continue
        rect_1a = pipeline.rectifier.rectify(im, det.quad, plate_type="type1a")
        top_l, bot_l = pipeline.rectifier.split_type1a(rect_1a)
        stitched_1a = pipeline.rectifier.stitch_type1a_horizontal(top_l, bot_l)
        txt_1a, conf_1a = pipeline.ocr.predict_single(stitched_1a, plate_type="type1a")
        rect_1 = pipeline.rectifier.rectify(im, det.quad, plate_type="type1")
        txt_1, conf_1 = pipeline.ocr.predict_single(rect_1, plate_type="type1")[:2]
        txt_clean = txt_1a.strip().upper()
        if not PLATE_REGEX.match(txt_clean):
            continue
        if conf_1a >= 0.55 and conf_1a > conf_1:
            true_1a.append((txt_clean, conf_1a, conf_1, ar, seam))
            print(f"Found True 1A: {txt_clean} | 1A:{conf_1a:.2f} > 1:{conf_1:.2f} | AR:{ar:.2f} | Seam:{seam:.2f}")

print("Found true 1A in sample:", len(true_1a))
