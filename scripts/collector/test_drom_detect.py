import sys
from pathlib import Path
import requests
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.pipeline import OmniPlatePipeline

pipeline = OmniPlatePipeline(device="cuda")

urls = [
    'https://s31.auto.drom.ru/photo/v2/eQL1mlRxEfbFSzrSZNsBqP4ppPs53gTJdrut4Jy8yNo1eNLLa97-yX_ukUtPX_pSunY4uvFjadivIZuY/gen1200.jpg',
    'https://s31.auto.drom.ru/photo/v2/_qLVATTEV41OALYROfnDZepHwi1mpR4OcJAFC5mmT__LkJLO9o4JSVJjHdaENsEd_0VQO_tTBop00PGX/gen1200.jpg'
]

for u in urls:
    r = requests.get(u)
    arr = np.asarray(bytearray(r.content), dtype=np.uint8)
    im = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    print("Downloaded shape:", im.shape)
    h, w = im.shape[:2]
    res = pipeline.detector.predict(source=im, imgsz=640, conf=0.10, device='cuda', verbose=False)
    parsed = pipeline._parse_results(res[0], w, h)
    print("Detections found:", len(parsed))
    for d in parsed:
        print("  det:", d.bbox, d.plate_type, d.confidence)
