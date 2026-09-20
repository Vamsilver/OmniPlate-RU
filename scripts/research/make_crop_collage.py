import cv2
import numpy as np
from pathlib import Path

crop_dir = Path("test_output/visual_audit/crops_rf")
crops = sorted(list(crop_dir.glob("*.jpg")))

cell_w, cell_h = 300, 100
cols = 4
rows = (len(crops) + cols - 1) // cols

collage = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)

for idx, cp in enumerate(crops):
    r = idx // cols
    c = idx % cols
    y0 = r * cell_h
    x0 = c * cell_w

    img = cv2.imread(str(cp))
    if img is None: continue
    h, w = img.shape[:2]
    scale = min(cell_w / w, (cell_h - 25) / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    res = cv2.resize(img, (nw, nh))

    oy = y0 + 25 + (cell_h - 25 - nh) // 2
    ox = x0 + (cell_w - nw) // 2
    collage[oy:oy+nh, ox:ox+nw] = res

    cv2.putText(collage, cp.stem.replace("_box_0", ""), (x0 + 5, y0 + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(collage, (x0, y0), (x0 + cell_w, y0 + cell_h), (100, 100, 100), 1)

cv2.imwrite("test_output/visual_audit/crops_rf_collage.jpg", collage)
print("Saved crops_rf_collage.jpg")
