import sys
import re
import requests
import cv2
import numpy as np
from pathlib import Path

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

r = requests.get('https://auto.drom.ru/vladivostok/toyota/succeed/', headers={'User-Agent': 'Mozilla/5.0'})
links = set(re.findall(r'href="(https://auto\.drom\.ru/vladivostok/toyota/succeed/\d+\.html)"', r.text))
print("Found Succeed cars on Drom:", len(links))

total_imgs = 0
found_1a = []
for car_url in list(links)[:4]:
    rc = requests.get(car_url, headers={'User-Agent': 'Mozilla/5.0'})
    photo_dirs = set(re.findall(r'https://[a-zA-Z0-9.-]+\.drom\.ru/photo/v2/[a-zA-Z0-9_-]+/', rc.text))
    print(f"Car {car_url} -> {len(photo_dirs)} photos")
    for pd in photo_dirs:
        img_url = pd + "gen1200.jpg"
        ri = requests.get(img_url, timeout=5)
        if ri.status_code == 200:
            total_imgs += 1
            im = cv2.imdecode(np.asarray(bytearray(ri.content), dtype=np.uint8), cv2.IMREAD_COLOR)
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
                if PLATE_REGEX.match(txt_clean) and conf_1a >= 0.55 and conf_1a > conf_1:
                    print(f"  [+] Found Drom 1A: {txt_clean} | 1A:{conf_1a:.2f} > 1:{conf_1:.2f} | AR:{ar:.2f} | Seam:{seam:.2f}")
                    found_1a.append(txt_clean)

print(f"Total checked: {total_imgs}, Found 1A: {len(found_1a)}")
