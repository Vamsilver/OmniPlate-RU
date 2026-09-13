#!/usr/bin/env python3
"""
Procedural Synthetic License Plate Generator for Volga IT 2026.
Generates realistic Russian license plates (Types 1, 1A, 1B) with 3D perspective projection,
exact ground truth quadrilateral corners (quad), and YOLO-pose labels.
Optimized with SIMD noise generation and multi-core parallel processing.
"""

import argparse
import concurrent.futures
import csv
import multiprocessing
import os
import random
import sys
import time
from typing import List, Tuple
import cv2
import numpy as np
from PIL import Image

# Ensure generator package can import its sibling modules
sys.path.insert(0, os.path.dirname(__file__))

from plate_renderer import PlateRenderer
from augmentations import PlateAugmentor
from perspective import PerspectiveTransformer

# Class mappings: 0: type1, 1: type1a, 2: type1b, 3: other
CLASS_MAP = {
    "type1": 0,
    "type1a": 1,
    "type1b": 2,
    "other": 3
}

# Car body colors (realistic automotive paints)
CAR_COLORS = [
    (240, 240, 242),  # White metallic
    (30, 30, 32),     # Black
    (90, 95, 100),    # Charcoal gray
    (160, 165, 170),  # Silver
    (140, 25, 25),    # Deep Red
    (20, 45, 90),     # Dark Blue
    (40, 60, 50),     # Forest Green
    (180, 140, 90),   # Champagne / Bronze
    (120, 120, 125),  # Slate
    (55, 55, 60)      # Asphalt
]

# Process worker state
_worker_renderer = None
_worker_augmentor = None
_worker_transformer = None


def init_worker():
    global _worker_renderer, _worker_augmentor, _worker_transformer
    _worker_renderer = PlateRenderer()
    _worker_augmentor = PlateAugmentor()
    _worker_transformer = PerspectiveTransformer()


def generate_procedural_background(width: int = 1280, height: int = 720) -> np.ndarray:
    """Generates realistic vehicle bumper / trunk background with metallic gradients and noise"""
    base_color = random.choice(CAR_COLORS)
    bg = np.zeros((height, width, 3), dtype=np.float32)

    # Base solid color with vertical gradient (light reflecting from car curves)
    grad = np.linspace(random.uniform(0.7, 0.9), random.uniform(1.0, 1.3), height)[:, np.newaxis, np.newaxis]
    for c in range(3):
        bg[:, :, c] = base_color[c] * grad[:, :, 0]

    # Add bumper / trunk horizontal line
    bumper_y = random.randint(int(height * 0.4), int(height * 0.6))
    cv2.line(bg, (0, bumper_y), (width, bumper_y), (base_color[0] * 0.6, base_color[1] * 0.6, base_color[2] * 0.6), 3)

    # Fast vectorized metallic flake noise via cv2.randn
    noise = np.empty((height, width, 3), dtype=np.float32)
    cv2.randn(noise, 0, random.uniform(3, 8))
    bg = np.clip(bg + noise, 0, 255).astype(np.uint8)

    return bg


def generate_single_sample(args_tuple: Tuple) -> Tuple[int, List]:
    """Worker task to generate 1 plate image, label, and meta row"""
    idx, plate_type, images_dir, labels_dir, sample_seed = args_tuple

    # Ensure deterministic reproduction per sample
    random.seed(sample_seed)
    np.random.seed(sample_seed)

    global _worker_renderer, _worker_augmentor, _worker_transformer
    if _worker_renderer is None:
        init_worker()

    # 1. Render base plate
    clean_plate, plate_num = _worker_renderer.render(plate_type)

    # 2. Augment plate
    aug_plate = _worker_augmentor.augment_plate(clean_plate)
    plate_arr = np.array(aug_plate)

    # 3. Create background scene
    scene_w, scene_h = 1280, 720
    bg = generate_procedural_background(scene_w, scene_h)

    # 4. Project plate with 3D homography
    scene, bbox, quad = _worker_transformer.project_plate_onto_background(plate_arr, bg)

    # 5. Conditions tags
    conds = ["day", "angle"]
    if random.random() < 0.4:
        conds.append("glare")
    if random.random() < 0.35:
        conds.append("dirt")
    if random.random() < 0.2:
        conds.append("motion_blur")
    conditions_str = ",".join(conds)

    # 6. Save image
    img_filename = f"synth_{idx:05d}.jpg"
    img_rel_path = f"images/synthetic/{img_filename}"
    img_full_path = os.path.join(images_dir, img_filename)
    cv2.imwrite(img_full_path, cv2.cvtColor(scene, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 93])

    # 7. Save YOLO/Pose label (.txt)
    label_filename = f"synth_{idx:05d}.txt"
    label_full_path = os.path.join(labels_dir, label_filename)

    class_id = CLASS_MAP[plate_type]
    bx, by, bw, bh = bbox
    x_center = (bx + bw / 2.0) / scene_w
    y_center = (by + bh / 2.0) / scene_h
    norm_w = bw / scene_w
    norm_h = bh / scene_h

    norm_quad = []
    for i in range(4):
        norm_quad.extend([quad[i * 2] / scene_w, quad[i * 2 + 1] / scene_h])

    quad_str = " ".join(f"{v:.6f}" for v in norm_quad)
    with open(label_full_path, "w", encoding="utf-8") as lf:
        lf.write(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f} {quad_str} {plate_num}\n")

    # 8. Return metadata row
    bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
    quad_csv_str = ",".join(str(v) for v in quad)
    row = [
        img_rel_path,
        plate_num,
        plate_type,
        bbox_str,
        quad_csv_str,
        1,  # is_vehicle
        1,  # is_synthetic
        "generator_v1",
        "CC-BY-4.0",
        conditions_str
    ]

    return idx, row


def main():
    parser = argparse.ArgumentParser(description="Procedural Russian License Plate Synthetic Generator")
    parser.add_argument("--count", type=int, default=5000, help="Number of images to generate (default: 5000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for 100% reproducibility")
    parser.add_argument("--output_dir", type=str, default="dataset", help="Output directory (root of dataset)")
    parser.add_argument("--type1_ratio", type=float, default=0.20, help="Ratio of Type 1 plates (default: 0.20)")
    parser.add_argument("--type1a_ratio", type=float, default=0.40, help="Ratio of Type 1A square plates (default: 0.40)")
    parser.add_argument("--type1b_ratio", type=float, default=0.40, help="Ratio of Type 1B yellow plates (default: 0.40)")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2), help="Number of worker processes")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing meta.csv and synthetic images")
    args = parser.parse_args()

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Prepare directories
    images_dir = os.path.join(args.output_dir, "images", "synthetic")
    labels_dir = os.path.join(args.output_dir, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    meta_path = os.path.join(args.output_dir, "meta.csv")
    meta_exists = os.path.exists(meta_path)

    print(f"=== Starting Multi-Process Synthetic Generation ===")
    print(f"Target count: {args.count}")
    print(f"Random seed: {args.seed}")
    print(f"Workers: {args.workers}")
    print(f"Ratios: type1={args.type1_ratio}, type1a={args.type1a_ratio}, type1b={args.type1b_ratio}")
    print(f"Output: {args.output_dir}\n", flush=True)

    # Determine type sequence based on ratios
    types_pool = []
    types_pool.extend(["type1"] * int(args.count * args.type1_ratio))
    types_pool.extend(["type1a"] * int(args.count * args.type1a_ratio))
    types_pool.extend(["type1b"] * (args.count - len(types_pool)))
    random.shuffle(types_pool)

    # Prepare tasks list with deterministic seeds
    tasks = []
    for idx in range(args.count):
        sample_seed = (args.seed * 10007 + idx * 7919) & 0x7FFFFFFF
        tasks.append((idx, types_pool[idx], images_dir, labels_dir, sample_seed))

    start_time = time.time()
    results = {}
    completed = 0

    if args.workers <= 1:
        init_worker()
        for t in tasks:
            idx, row = generate_single_sample(t)
            results[idx] = row
            completed += 1
            if completed % 500 == 0 or completed == args.count:
                elapsed = time.time() - start_time
                fps = completed / max(0.1, elapsed)
                print(f"Generated [{completed}/{args.count}] ({fps:.1f} img/s)...", flush=True)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker) as executor:
            future_to_idx = {executor.submit(generate_single_sample, t): t[0] for t in tasks}
            for future in concurrent.futures.as_completed(future_to_idx):
                idx, row = future.result()
                results[idx] = row
                completed += 1
                if completed % 500 == 0 or completed == args.count:
                    elapsed = time.time() - start_time
                    fps = completed / max(0.1, elapsed)
                    print(f"Generated [{completed}/{args.count}] ({fps:.1f} img/s, elapsed: {elapsed:.1f}s)...", flush=True)

    # Write meta.csv in deterministic sorted order
    mode = "w" if (args.overwrite or not meta_exists) else "a"
    with open(meta_path, mode, newline="", encoding="utf-8") as meta_f:
        writer = csv.writer(meta_f, delimiter=";")
        if mode == "w":
            writer.writerow([
                "image", "plate_num", "plate_type", "bbox", "quad",
                "is_vehicle", "is_synthetic", "source", "license", "conditions"
            ])
        for idx in range(args.count):
            if idx in results:
                writer.writerow(results[idx])

    total_time = time.time() - start_time
    print(f"\n[SUCCESS] Generated {len(results)} synthetic images in {total_time:.1f}s ({len(results)/max(0.1, total_time):.1f} img/s)!")
    print(f"Images: {images_dir}")
    print(f"Labels: {labels_dir}")
    print(f"Meta: {meta_path}\n", flush=True)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
