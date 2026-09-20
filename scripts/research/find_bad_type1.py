import pandas as pd
from PIL import Image
import os
import re

df = pd.read_csv('dataset/meta.csv', sep=';')
real_t1 = df[(df['is_synthetic'] == 0) & (df['plate_type'] == 'type1')].copy()

# Specific bad files identified during visual audit
bad_images = set()

# 1. Obvious stock / posters / drawings / blueprints / souvenirs
known_bad_patterns = [
    # Stock cliparts with illegal 000 or fake numbers
    'A000AA78', 'A111AA77', 'A888AA177', 'H123PC456', 'OA323HT777', 
    'A389BT10', # isolated on white background
]

for idx, r in real_t1.iterrows():
    plate = str(r['plate_num'])
    img = r['image']
    base = os.path.basename(img)
    
    # 000 illegal
    if '000' in plate:
        bad_images.add(img)
        
    # Blueprints and infographics
    if base in ['real_type1_0187.jpg', 'real_type1_0392.jpg', 'real_type1_0412.jpg', 
                'real_type1_0418.jpg', 'real_type1_0419.jpg', 'real_type1_0420.jpg',
                'real_type1_0395.jpg', 'real_type1_0423.jpg', 'real_type1_0413.jpg',
                'real_type1_0414.jpg', 'real_type1_0382.jpg', 'real_type1_0383.jpg',
                'real_type1_0385.jpg', 'real_type1_0397.jpg', 'real_type1_0199.jpg',
                'real_type1_0739.jpg', 'real_type1_0192.jpg', 'real_type1_0193.jpg',
                'real_type1_0243.jpg']: # collage
        bad_images.add(img)
        
    # People holding plates
    if base in ['real_type1_0194.jpg', 'real_type1_0195.jpg', 'real_type1_0196.jpg',
                'real_type1_0215.jpg', 'real_type1_0216.jpg', 'real_type1_0217.jpg']:
        bad_images.add(img)
        
    # Rotated vertical crops from roboflow
    if base in ['real_type1_0204.jpg', 'real_type1_0205.jpg', 'real_type1_0206.jpg',
                'real_type1_0207.jpg', 'real_type1_0208.jpg', 'real_type1_0209.jpg',
                'real_type1_0218.jpg', 'real_type1_0219.jpg', 'real_type1_0220.jpg',
                'real_type1_0230.jpg', 'real_type1_0231.jpg', 'real_type1_0277.jpg',
                'real_type1_0278.jpg', 'real_type1_0279.jpg', 'real_type1_0373.jpg',
                'real_type1_0374.jpg', 'real_type1_0375.jpg', 'real_type1_0376.jpg',
                'real_type1_0377.jpg', 'real_type1_0378.jpg', 'real_type1_0379.jpg',
                'real_type1_0380.jpg', 'real_type1_0381.jpg', 'real_type1_0387.jpg',
                'real_type1_0394.jpg', 'real_type1_0401.jpg', 'real_type1_0410.jpg',
                'real_type1_0411.jpg']:
        bad_images.add(img)
        
    # Pure plate crops without car (aspect ratio > 0.85 of image)
    img_path = os.path.join('dataset', img)
    bbox = [float(x) for x in str(r['bbox']).split(',')]
    x, y, w, h = bbox
    try:
        with Image.open(img_path) as im:
            iw, ih = im.size
        area_ratio = (w * h) / (iw * ih)
        # If it's a tight crop taking almost whole image
        if area_ratio > 0.80 and (w / h > 3.0 or h / w > 0.8):
            # Check if it's one of the known isolated plate crops
            if ('nomeroff' in str(r['source']).lower() or 'parquet' in str(r['source']).lower() or 'roboflow' in str(r['source']).lower()):
                bad_images.add(img)
    except:
        pass

print(f'Total bad images to remove: {len(bad_images)}')
print(f'Remaining Type 1 count: {len(real_t1) - len(bad_images)}')
