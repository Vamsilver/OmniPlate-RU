import pandas as pd
from PIL import Image, ImageDraw
import os
import math

df = pd.read_csv('dataset/meta.csv', sep=';')
real_t1 = df[(df['is_synthetic'] == 0) & (df['plate_type'] == 'type1')].copy()

suspicious = []
for idx, r in real_t1.iterrows():
    img_path = os.path.join('dataset', r['image'])
    bbox = [float(x) for x in str(r['bbox']).split(',')]
    x, y, w, h = bbox
    plate_num = str(r['plate_num'])
    source = str(r['source'])
    try:
        with Image.open(img_path) as im:
            iw, ih = im.size
        area_ratio = (w * h) / (iw * ih)
        ar = w / max(h, 1e-5)
    except:
        continue
        
    is_sus = False
    reason = []
    if '000' in plate_num:
        is_sus = True
        reason.append('000')
    if area_ratio > 0.80:
        is_sus = True
        reason.append(f'crop>{round(area_ratio,2)}')
    if ar < 1.8 and ('parquet' in source or 'roboflow' in source):
        is_sus = True
        reason.append(f'AR={round(ar,2)}')
        
    if is_sus:
        suspicious.append({
            'image': r['image'],
            'plate_num': plate_num,
            'reason': ','.join(reason),
            'bbox': bbox,
            'source': source
        })

print(f'Collected {len(suspicious)} suspicious items')

os.makedirs('test_output/audit_type1_suspicious', exist_ok=True)

cell_w, cell_h = 360, 260
cols, rows = 5, 5
per_sheet = cols * rows
num_sheets = math.ceil(len(suspicious) / per_sheet)

for s in range(num_sheets):
    sheet_img = Image.new('RGB', (cols * cell_w, rows * cell_h), color=(30, 30, 30))
    draw = ImageDraw.Draw(sheet_img)
    items = suspicious[s * per_sheet : (s + 1) * per_sheet]
    
    for i, item in enumerate(items):
        r_idx = i // cols
        c_idx = i % cols
        x0 = c_idx * cell_w
        y0 = r_idx * cell_h
        
        im_p = os.path.join('dataset', item['image'])
        try:
            with Image.open(im_p) as im:
                im_rgb = im.convert('RGB')
                im_rgb.thumbnail((cell_w - 20, cell_h - 60))
                sheet_img.paste(im_rgb, (x0 + 10, y0 + 35))
        except Exception as e:
            pass
            
        base = os.path.basename(item['image'])
        text = f"{base} | {item['plate_num']}"
        draw.text((x0 + 10, y0 + 5), text, fill=(255, 255, 0))
        draw.text((x0 + 10, y0 + 20), item['reason'][:35], fill=(255, 100, 100))
        draw.rectangle([x0, y0, x0 + cell_w - 1, y0 + cell_h - 1], outline=(60, 60, 60), width=1)
        
    out_path = f'test_output/audit_type1_suspicious/sheet_{s+1:02d}.jpg'
    sheet_img.save(out_path, quality=85)
    print(f'Saved {out_path}')
