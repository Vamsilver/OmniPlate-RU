import cv2
import numpy as np
from pathlib import Path

cases = [
    ('real_type1b_0067.jpg', '0067'),
    ('real_type1b_0068.jpg', '0068'),
    ('real_type1b_0084.jpg', '0084'),
    ('real_type1b_0096.jpg', '0096'),
    ('real_type1b_0119.jpg', '0119'),
    ('real_type1b_0123.jpg', '0123'),
    ('real_type1b_0126.jpg', '0126'),
    ('real_type1b_0139.jpg', '0139'),
    ('real_type1b_0141.jpg', '0141'),
    ('real_type1b_0167.jpg', '0167'),
    ('real_type1b_0181.jpg', '0181'),
    ('real_type1b_0183.jpg', '0183'),
    ('real_type1b_0191.jpg', '0191'),
    ('real_type1b_0198.jpg', '0198'),
    ('real_type1b_0208.jpg', '0208'),
    ('real_type1b_0213.jpg', '0213'),
    ('real_type1b_0214.jpg', '0214'),
    ('real_type1b_0232.jpg', '0232'),
    ('real_type1b_0235.jpg', '0235'),
    ('real_type1b_0236.jpg', '0236'),
    ('real_type1b_0237.jpg', '0237'),
    ('real_type1b_0251.jpg', '0251'),
    ('real_type1b_0252.jpg', '0252'),
    ('real_type1b_0266.jpg', '0266'),
    ('real_type1b_0267.jpg', '0267'),
    ('real_type1b_0280.jpg', '0280'),
    ('real_type1b_0284.jpg', '0284'),
    ('real_type1b_0286.jpg', '0286'),
    ('real_type1b_0300.jpg', '0300'),
    ('real_type1b_0306.jpg', '0306'),
    ('real_type1b_0307.jpg', '0307'),
    ('real_type1b_0308.jpg', '0308'),
    ('real_type1b_0309.jpg', '0309'),
    ('real_type1b_0310.jpg', '0310'),
]

labels_dir = Path('dataset/labels')
images_dir = Path('dataset/images/real')

crops = []
for fname, title in cases:
    img = cv2.imread(str(images_dir / fname))
    lbl_file = labels_dir / f"{Path(fname).stem}.txt"
    if not lbl_file.exists():
        continue
    txt = lbl_file.read_text().strip()
    parts = [float(x) for x in txt.split()]
    cx, cy, bw, bh = parts[1:5]
    h, w = img.shape[:2]
    bx1 = max(0, int((cx - bw/2)*w))
    by1 = max(0, int((cy - bh/2)*h))
    bx2 = min(w, int((cx + bw/2)*w))
    by2 = min(h, int((cy + bh/2)*h))
    
    # margin
    mw = int((bx2 - bx1)*0.2)
    mh = int((by2 - by1)*0.2)
    cbx1 = max(0, bx1 - mw)
    cby1 = max(0, by1 - mh)
    cbx2 = min(w, bx2 + mw)
    cby2 = min(h, by2 + mh)
    crop = img[cby1:cby2, cbx1:cbx2]
    
    # resize to fixed width 320, height 90
    crop_res = cv2.resize(crop, (320, 90))
    # banner
    banner = np.zeros((30, 320, 3), dtype=np.uint8)
    cv2.putText(banner, f"{title}: {fname}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cell = np.vstack([banner, crop_res])
    crops.append(cell)

# Arrange into grid of 4 columns
cols = 4
rows = (len(crops) + cols - 1) // cols
grid_h = rows * 120
grid_w = cols * 320
grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

for idx, c in enumerate(crops):
    r = idx // cols
    col = idx % cols
    grid[r*120:(r+1)*120, col*320:(col+1)*320] = c

cv2.imwrite('test_output/type1b_corrections_grid.jpg', grid)
print("Saved test_output/type1b_corrections_grid.jpg")
