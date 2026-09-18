import pyarrow.parquet as pq
import cv2
import numpy as np
import csv
import re
from pathlib import Path
import sys

PROJECT_ROOT = Path(r"D:\AIProjects\VolgaIT")
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.rectifier import PlateRectifier
from src.pipeline.ocr import PlateOCR

ocr = PlateOCR(model_path="models/ocr_lprnet_best.pt", device="cuda")
rectifier = PlateRectifier()

with open(PROJECT_ROOT / "dataset" / "meta.csv", "r", encoding="utf-8") as f:
    existing_sources = set(r[7] for r in csv.reader(f, delimiter=";") if len(r) > 7)

pq_files = [
    r"C:\Users\Vamsi\Downloads\train-00000-of-00001.parquet",
    r"C:\Users\Vamsi\Downloads\validation-00000-of-00001.parquet",
    r"C:\Users\Vamsi\Downloads\test-00000-of-00001.parquet"
]

fresh_pure_1a = []
for pq_p in pq_files:
    table = pq.read_table(pq_p)
    fn = Path(pq_p).name
    for r in table.to_pylist():
        p = r["image"]["path"]
        src = f"{fn}:{p}"
        if src in existing_sources:
            continue
        objs = r.get("objects", {})
        bboxes = objs.get("bbox", [])
        if not bboxes:
            continue
        bx, by, bw, bh = bboxes[0]
        ar = bw / max(1.0, bh)
        if 0.8 <= ar <= 1.80:
            im_bytes = r["image"]["bytes"]
            im = cv2.imdecode(np.asarray(bytearray(im_bytes), dtype=np.uint8), cv2.IMREAD_COLOR)
            if im is None:
                continue
            h_img, w_img = im.shape[:2]
            bx = max(0, min(w_img - 1, int(bx)))
            by = max(0, min(h_img - 1, int(by)))
            bw = max(1, min(w_img - bx, int(bw)))
            bh = max(1, min(h_img - by, int(bh)))
            pts = np.array([[bx, by], [bx+bw, by], [bx+bw, by+bh], [bx, by+bh]], dtype=np.float32)

            rect_1a = rectifier.rectify(im, pts, plate_type="type1a")
            top_l, bot_l = rectifier.split_type1a(rect_1a, adaptive_seam=True)
            stitched = rectifier.stitch_type1a_horizontal(top_l, bot_l)
            txt_1a, conf_1a = ocr.predict_single(stitched, plate_type="type1a")

            rect_1 = rectifier.rectify(im, pts, plate_type="type1")
            txt_1, conf_1 = ocr.predict_single(rect_1, plate_type="type1")[:2]

            if conf_1a >= 0.40 and conf_1a >= conf_1 - 0.10 and len(txt_1a) in (8, 9) and txt_1a.count('#') <= 2:
                fresh_pure_1a.append({
                    "src": src,
                    "path": p,
                    "txt_1a": txt_1a,
                    "conf_1a": float(conf_1a),
                    "txt_1": txt_1,
                    "conf_1": float(conf_1),
                    "ar": float(ar),
                    "bbox": [bx, by, bw, bh]
                })

print(f"Fresh pure 1A candidates found in Parquet: {len(fresh_pure_1a)}")
for f in fresh_pure_1a[:15]:
    print(f"{f['path'][:35]} | {f['txt_1a']} | 1A={f['conf_1a']:.2f} | 1={f['conf_1']:.2f} | AR={f['ar']:.2f}")
