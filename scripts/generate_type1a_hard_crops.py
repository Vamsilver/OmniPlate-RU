#!/usr/bin/env python3
import csv, os, random, sys
from pathlib import Path
from typing import Tuple
import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path('.').resolve()
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'dataset' / 'generator'))

from plate_renderer import PlateRenderer, ALLOWED_LETTERS, DIGITS, POPULAR_REGIONS
from src.pipeline.rectifier import PlateRectifier

OUTPUT_DIR = PROJECT_ROOT / 'dataset' / 'verified_crops'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_PATH = OUTPUT_DIR / 'manifest.csv'

CONFUSION_STRATEGIES = [
    'm_b', 'h_a', 'o_c', 'k_h', 'y_t', 'digits_80', 'digits_56', 'jdm_mixed',
    'digits_37', 'digits_71', 'digits_24', 'letters_oc_op', 'letters_mh_bh',
    'letters_yx_tk', 'regions_3digit'
]

HARD_3DIGIT_REGIONS = ['154', '750', '777', '199', '799', '116', '197', '174', '125', '138', '198', '196', '123', '716', '761', '790']

def generate_targeted_1a_text(strategy: str) -> Tuple[str, str, str, str, str]:
    reg = random.choice(HARD_3DIGIT_REGIONS if random.random() < 0.65 else POPULAR_REGIONS)
    if strategy == 'm_b':
        l1 = random.choice(['M', 'B'])
        d3 = ''.join(random.choices(DIGITS, k=3))
        l2 = random.choice(['MB', 'BM', 'BB', 'MM', 'BC', 'MC'])
    elif strategy == 'h_a':
        l1 = random.choice(['H', 'A'])
        d3 = ''.join(random.choices(DIGITS, k=3))
        l2 = random.choice(['HA', 'AH', 'AA', 'HH', 'AK', 'HK'])
    elif strategy == 'o_c':
        l1 = random.choice(['O', 'C'])
        d3 = ''.join(random.choices(DIGITS, k=3))
        l2 = random.choice(['OC', 'CO', 'OO', 'CC', 'OP', 'CP'])
    elif strategy == 'k_h':
        l1 = random.choice(['K', 'H'])
        d3 = ''.join(random.choices(DIGITS, k=3))
        l2 = random.choice(['KH', 'HK', 'KK', 'HH', 'KM', 'HM'])
    elif strategy == 'y_t':
        l1 = random.choice(['Y', 'T'])
        d3 = ''.join(random.choices(DIGITS, k=3))
        l2 = random.choice(['YT', 'TY', 'YY', 'TT', 'TO', 'YO'])
    elif strategy == 'digits_80':
        l1 = random.choice(ALLOWED_LETTERS)
        d3 = random.choice(['808', '080', '880', '008', '800', '088'])
        l2 = ''.join(random.choices(ALLOWED_LETTERS, k=2))
    elif strategy == 'digits_56':
        l1 = random.choice(ALLOWED_LETTERS)
        d3 = random.choice(['565', '656', '556', '665', '506', '605'])
        l2 = ''.join(random.choices(ALLOWED_LETTERS, k=2))
    elif strategy == 'digits_37':
        l1 = random.choice(ALLOWED_LETTERS)
        d3 = random.choice(['137', '371', '173', '731', '373', '737', '337', '773', '177', '717', '139', '179'])
        l2 = ''.join(random.choices(ALLOWED_LETTERS, k=2))
    elif strategy == 'digits_71':
        l1 = random.choice(ALLOWED_LETTERS)
        d3 = random.choice(['917', '971', '117', '711', '771', '171', '317', '378', '318', '718', '141', '174'])
        l2 = ''.join(random.choices(ALLOWED_LETTERS, k=2))
    elif strategy == 'digits_24':
        l1 = random.choice(ALLOWED_LETTERS)
        d3 = random.choice(['209', '409', '242', '424', '247', '427', '224', '442', '437', '237', '402', '202'])
        l2 = ''.join(random.choices(ALLOWED_LETTERS, k=2))
    elif strategy == 'letters_oc_op':
        l1 = random.choice(['O', 'C', 'P', 'E'])
        d3 = ''.join(random.choices(DIGITS, k=3))
        l2 = random.choice(['OC', 'CO', 'OP', 'PO', 'CP', 'PC', 'OE', 'EO', 'CE', 'EC'])
    elif strategy == 'letters_mh_bh':
        l1 = random.choice(['M', 'H', 'B', 'P'])
        d3 = ''.join(random.choices(DIGITS, k=3))
        l2 = random.choice(['MH', 'HM', 'BH', 'HB', 'MB', 'BM', 'HP', 'PH', 'BP', 'PB'])
    elif strategy == 'letters_yx_tk':
        l1 = random.choice(['Y', 'X', 'T', 'K'])
        d3 = ''.join(random.choices(DIGITS, k=3))
        l2 = random.choice(['YX', 'XY', 'TK', 'KT', 'XT', 'YT', 'YK', 'KY', 'KX', 'XK'])
    elif strategy == 'regions_3digit':
        l1 = random.choice(['K', 'M', 'H', 'P', 'O', 'T', 'C', 'B', 'A'])
        d3 = random.choice(['131', '331', '437', '141', '378', '139', '022', '346', '947', '777', '309'])
        l2 = random.choice(['TB', 'HB', 'CO', 'YB', 'EO', 'BP', 'AO', 'BC', 'OO', 'BB', 'PP', 'ME', 'AP'])
        reg = random.choice(['199', '799', '154', '750', '777', '197', '174', '125', '138', '198', '196', '123'])
    else:
        l1 = random.choice(ALLOWED_LETTERS)
        d3 = ''.join(random.choices(DIGITS, k=3))
        l2 = ''.join(random.choices(ALLOWED_LETTERS, k=2))
    full_str = f'{l1}{d3}{l2}{reg}'
    return full_str, l1, d3, l2, reg

def apply_1a_augmentations(img_bgr: np.ndarray) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    if random.random() < 0.60:
        dx = random.uniform(-3, 3)
        dy = random.uniform(-2, 2)
        pts1 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        pts2 = np.float32([[dx, dy], [w - dx, -dy], [w - dx, h + dy], [dx, h - dy]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        img_bgr = cv2.warpPerspective(img_bgr, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if random.random() < 0.50:
        k = random.choice([3, 5])
        img_bgr = cv2.GaussianBlur(img_bgr, (k, k), 0)
    alpha = random.uniform(0.75, 1.20)
    beta = random.uniform(-20, 20)
    img_bgr = np.clip(alpha * img_bgr + beta, 0, 255).astype(np.uint8)
    if random.random() < 0.40:
        noise = np.random.normal(0, random.uniform(3, 7), img_bgr.shape).astype(np.float32)
        img_bgr = np.clip(img_bgr.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img_bgr

def main():
    print('[*] Initializing targeted Type 1A generator...')
    renderer = PlateRenderer()
    rectifier = PlateRectifier()
    start_idx = 401
    end_idx = 2000
    num_samples = end_idx - start_idx + 1
    generated_rows = []
    for idx in range(start_idx, end_idx + 1):
        strat = random.choice(CONFUSION_STRATEGIES)
        plate_num, _, _, _, _ = generate_targeted_1a_text(strat)
        pil_im, _ = renderer.render_type1a(plate_num)
        arr_96 = cv2.resize(np.array(pil_im), (160, 96), interpolation=cv2.INTER_LINEAR)
        bgr_96 = cv2.cvtColor(arr_96, cv2.COLOR_RGB2BGR)
        aug_96 = apply_1a_augmentations(bgr_96)
        top_l, bot_l = rectifier.split_type1a(aug_96, adaptive_seam=True, vertical_margin=0)
        stitched_36 = rectifier.stitch_type1a_horizontal(top_l, bot_l, target_size=(160, 36))
        fname = f'synth_hard_1a_{idx:04d}.jpg'
        out_path = OUTPUT_DIR / fname
        cv2.imwrite(str(out_path), stitched_36)
        generated_rows.append([fname, plate_num, 'type1a', '160x36', 'synth_hard_1a'])
    with open(MANIFEST_PATH, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=';')
        for row in generated_rows:
            writer.writerow(row)
    print(f'[SUCCESS] Generated {num_samples} Type 1A crops ({start_idx:04d}..{end_idx:04d}) into {OUTPUT_DIR}')

if __name__ == '__main__':
    main()
