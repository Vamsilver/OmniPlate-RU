import os
import cv2
import shutil
import random
import numpy as np
import pandas as pd
from PIL import Image

import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'dataset/generator')

# Import synthetic generator components
from plate_renderer import PlateRenderer
from augmentations import PlateAugmentor
from perspective import PerspectiveTransformer
from generate_synthetic import generate_procedural_background, CLASS_MAP

# 1. Regenerate 2 synthetic samples that had '000'
renderer = PlateRenderer()
augmentor = PlateAugmentor()
transformer = PerspectiveTransformer()

# synth_01919 is type1b
# synth_02665 is type1a
synth_fixes = {
    'images/synthetic/synth_01919.jpg': ('type1b', 191901),
    'images/synthetic/synth_02665.jpg': ('type1a', 266501),
}

df = pd.read_csv('dataset/meta.csv', sep=';')

for rel_path, (ptype, seed) in synth_fixes.items():
    random.seed(seed)
    np.random.seed(seed)
    
    clean_plate, plate_num = renderer.render(ptype)
    aug_plate = augmentor.augment_plate(clean_plate)
    plate_arr = np.array(aug_plate)
    
    scene_w, scene_h = 1280, 720
    bg = generate_procedural_background(scene_w, scene_h)
    scene, bbox, quad = transformer.project_plate_onto_background(plate_arr, bg)
    
    # save image
    full_img_path = os.path.join('dataset', rel_path)
    cv2_img = cv2.cvtColor(scene, cv2.COLOR_RGB2BGR)
    import cv2
    cv2.imwrite(full_img_path, cv2_img)
    
    # save yolo label
    base_name = os.path.splitext(os.path.basename(rel_path))[0]
    txt_path = os.path.join('dataset', 'labels', f'{base_name}.txt')
    
    bx, by, bw, bh = bbox
    xc, yc = (bx + bw / 2.0) / scene_w, (by + bh / 2.0) / scene_h
    wn, hn = bw / scene_w, bh / scene_h
    
    cid = CLASS_MAP[ptype]
    norm_quad = []
    for i in range(4):
        norm_quad.extend([quad[i * 2] / scene_w, quad[i * 2 + 1] / scene_h])
        
    line_parts = [str(cid), f"{xc:.6f}", f"{yc:.6f}", f"{wn:.6f}", f"{hn:.6f}"]
    for coord in norm_quad:
        line_parts.append(f"{coord:.6f}")
    with open(txt_path, 'w') as f:
        f.write(" ".join(line_parts) + "\n")
        
    # update df row
    bbox_str = f"{bx},{by},{bw},{bh}"
    quad_str = ",".join(map(str, quad))
    
    row_mask = df['image'] == rel_path
    df.loc[row_mask, 'plate_num'] = plate_num
    df.loc[row_mask, 'bbox'] = bbox_str
    df.loc[row_mask, 'quad'] = quad_str
    print(f'Regenerated {rel_path}: new plate_num={plate_num}')

# 2. Identify bad Type 1 files
real_t1 = df[(df['is_synthetic'] == 0) & (df['plate_type'] == 'type1')].copy()

known_bad_bases = {
    'real_type1_0185.jpg', 'real_type1_0187.jpg', 'real_type1_0188.jpg', 'real_type1_0189.jpg',
    'real_type1_0192.jpg', 'real_type1_0193.jpg', 'real_type1_0194.jpg', 'real_type1_0195.jpg',
    'real_type1_0196.jpg', 'real_type1_0199.jpg', 'real_type1_0739.jpg', 'real_type1_0243.jpg',
    'real_type1_0382.jpg', 'real_type1_0383.jpg', 'real_type1_0385.jpg', 'real_type1_0392.jpg',
    'real_type1_0395.jpg', 'real_type1_0397.jpg', 'real_type1_0412.jpg', 'real_type1_0413.jpg',
    'real_type1_0414.jpg', 'real_type1_0418.jpg', 'real_type1_0419.jpg', 'real_type1_0420.jpg',
    'real_type1_0423.jpg',
    # people holding
    'real_type1_0215.jpg', 'real_type1_0216.jpg', 'real_type1_0217.jpg',
    # rotated 90 deg roboflow
    'real_type1_0204.jpg', 'real_type1_0205.jpg', 'real_type1_0206.jpg', 'real_type1_0207.jpg',
    'real_type1_0208.jpg', 'real_type1_0209.jpg', 'real_type1_0218.jpg', 'real_type1_0219.jpg',
    'real_type1_0220.jpg', 'real_type1_0230.jpg', 'real_type1_0231.jpg', 'real_type1_0277.jpg',
    'real_type1_0278.jpg', 'real_type1_0279.jpg', 'real_type1_0373.jpg', 'real_type1_0374.jpg',
    'real_type1_0375.jpg', 'real_type1_0376.jpg', 'real_type1_0377.jpg', 'real_type1_0378.jpg',
    'real_type1_0379.jpg', 'real_type1_0380.jpg', 'real_type1_0381.jpg', 'real_type1_0387.jpg',
    'real_type1_0394.jpg', 'real_type1_0401.jpg', 'real_type1_0410.jpg', 'real_type1_0411.jpg'
}

to_remove = set()
for idx, r in real_t1.iterrows():
    img = r['image']
    base = os.path.basename(img)
    plate = str(r['plate_num'])
    
    if base in known_bad_bases or '000' in plate:
        to_remove.add(img)
        continue
        
    img_path = os.path.join('dataset', img)
    bbox = [float(x) for x in str(r['bbox']).split(',')]
    x, y, w, h = bbox
    try:
        with Image.open(img_path) as im:
            iw, ih = im.size
        ratio = (w * h) / (iw * ih)
        if ratio > 0.75:
            to_remove.add(img)
    except:
        pass

print(f'Total bad Type 1 files to delete: {len(to_remove)}')

# Delete bad files from disk
for img in to_remove:
    img_p = os.path.join('dataset', img)
    base_no_ext = os.path.splitext(os.path.basename(img))[0]
    txt_p = os.path.join('dataset', 'labels', f'{base_no_ext}.txt')
    
    if os.path.exists(img_p):
        os.remove(img_p)
    if os.path.exists(txt_p):
        os.remove(txt_p)

# Filter df: keep only non-removed
df = df[~df['image'].isin(to_remove)].copy()
print(f'Total rows in df after removal: {len(df)}')

# 3. Renumber remaining real_type1 strictly 1..N
# We must temporarily rename to tmp_ to avoid collisions
real_t1_rem = df[(df['is_synthetic'] == 0) & (df['plate_type'] == 'type1')].copy()
real_t1_sorted = real_t1_rem.sort_values(by='image').reset_index(drop=True)

print(f'Renumbering {len(real_t1_sorted)} Type 1 files...')

# Phase A: rename to tmp
tmp_map = {}
for i, r in real_t1_sorted.iterrows():
    old_rel = r['image']
    old_base = os.path.basename(old_rel)
    old_stem = os.path.splitext(old_base)[0]
    
    old_img_p = os.path.join('dataset', old_rel)
    old_txt_p = os.path.join('dataset', 'labels', f'{old_stem}.txt')
    
    tmp_stem = f'tmp_type1_{i+1:04d}'
    tmp_img_p = os.path.join('dataset', 'images', 'real', f'{tmp_stem}.jpg')
    tmp_txt_p = os.path.join('dataset', 'labels', f'{tmp_stem}.txt')
    
    if os.path.exists(old_img_p):
        os.rename(old_img_p, tmp_img_p)
    if os.path.exists(old_txt_p):
        os.rename(old_txt_p, tmp_txt_p)
        
    tmp_map[old_rel] = (tmp_img_p, tmp_txt_p, i+1)

# Phase B: rename from tmp to real_type1_0001..
final_map = {}
for old_rel, (tmp_img_p, tmp_txt_p, idx_num) in tmp_map.items():
    new_stem = f'real_type1_{idx_num:04d}'
    new_rel = f'images/real/{new_stem}.jpg'
    new_img_p = os.path.join('dataset', 'images', 'real', f'{new_stem}.jpg')
    new_txt_p = os.path.join('dataset', 'labels', f'{new_stem}.txt')
    
    if os.path.exists(tmp_img_p):
        os.rename(tmp_img_p, new_img_p)
    if os.path.exists(tmp_txt_p):
        os.rename(tmp_txt_p, new_txt_p)
        
    final_map[old_rel] = new_rel

# Update meta.csv
df['image'] = df['image'].apply(lambda x: final_map.get(x, x))

# Save meta.csv
df.to_csv('dataset/meta.csv', sep=';', index=False)
print('Saved updated dataset/meta.csv')

# Verify distribution
print('\nFinal plate_type distribution:')
print(df['plate_type'].value_counts())
print('\nReal vs Synthetic:')
print(df['is_synthetic'].value_counts())
real_sub = df[df['is_synthetic'] == 0]
print('\nReal plate_type distribution:')
print(real_sub['plate_type'].value_counts())
