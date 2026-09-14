#!/usr/bin/env python3
"""
Zero-Copy YOLO-Pose Dataset Preparation for Volga IT 2026.
Reads dataset annotations, prepares labels in dataset/labels/synthetic and dataset/labels/real,
creates stratified train/val splits across both synthetic and real images,
and generates ultralytics YOLO-pose data.yaml.
"""

import argparse
import csv
import glob
import os
import random
import sys
from typing import Dict, List
from PIL import Image


def prepare_yolo_pose(dataset_dir: str, val_ratio: float = 0.15, seed: int = 42):
    random.seed(seed)

    meta_path = os.path.join(dataset_dir, "meta.csv")
    if not os.path.exists(meta_path):
        print(f"[ERROR] meta.csv not found at {meta_path}")
        sys.exit(1)

    pose_dir = os.path.join(dataset_dir, "yolo_pose")
    os.makedirs(pose_dir, exist_ok=True)

    # Clean existing Ultralytics .cache files so it re-indexes freshly
    for cache_f in glob.glob(os.path.join(dataset_dir, "**", "*.cache"), recursive=True):
        try:
            os.remove(cache_f)
            print(f"[*] Removed stale cache: {cache_f}")
        except Exception:
            pass

    class_map = {"type1": 0, "type1a": 1, "type1b": 2, "other": 3}

    # Group samples by (plate_type, is_synthetic) for balanced stratification
    strata: Dict[str, List[Dict]] = {}

    with open(meta_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            p_type = row.get("plate_type")
            if p_type not in class_map:
                continue
            is_syn = row.get("is_synthetic", "0")
            key = f"{p_type}_{is_syn}"
            if key not in strata:
                strata[key] = []
            strata[key].append(row)

    train_rows = []
    val_rows = []

    print("\nDataset Stratification:")
    for stratum_key, rows in sorted(strata.items()):
        random.shuffle(rows)
        n_val = max(1, int(len(rows) * val_ratio)) if len(rows) > 5 else 0
        val_subset = rows[:n_val]
        train_subset = rows[n_val:]
        val_rows.extend(val_subset)
        train_rows.extend(train_subset)
        print(f"  - {stratum_key:<12}: Total={len(rows):<5} -> Train={len(train_subset):<5} | Val={len(val_subset):<4}")

    print(f"\nTotal samples: {len(train_rows) + len(val_rows)}")
    print(f"Train split:   {len(train_rows)}")
    print(f"Val split:     {len(val_rows)}")

    # Generate standard YOLO-pose labels for each image
    count = 0
    real_count = 0
    synth_count = 0

    for row in train_rows + val_rows:
        img_rel = row["image"]
        base_name = os.path.splitext(os.path.basename(img_rel))[0]
        p_type = row["plate_type"]
        cls_id = class_map[p_type]
        is_syn = row.get("is_synthetic", "0")

        # Determine label path: images/real/xxx.jpg -> labels/real/xxx.txt
        subdir = os.path.dirname(img_rel).replace("images", "labels")
        target_dir = os.path.join(dataset_dir, subdir)
        os.makedirs(target_dir, exist_ok=True)
        target_label_path = os.path.join(target_dir, f"{base_name}.txt")

        # Negative sample (other) -> empty label file teaches YOLO zero false positives
        if p_type == "other":
            with open(target_label_path, "w", encoding="utf-8") as wf:
                pass
            count += 1
            if is_syn == "1":
                synth_count += 1
            else:
                real_count += 1
            continue

        # Check if precomputed raw label exists (for synthetic)
        raw_label_path = os.path.join(dataset_dir, "labels", f"{base_name}.txt")
        if os.path.exists(raw_label_path) and is_syn == "1":
            with open(raw_label_path, "r", encoding="utf-8") as rf:
                line = rf.readline().strip()
                tokens = line.split()
                if len(tokens) >= 13:
                    yolo_pose_str = " ".join(tokens[:13])
                    with open(target_label_path, "w", encoding="utf-8") as wf:
                        wf.write(yolo_pose_str + "\n")
            count += 1
            synth_count += 1
            continue

        # Parse bbox and quad from meta.csv
        full_img_path = os.path.join(dataset_dir, img_rel)
        if not os.path.exists(full_img_path):
            continue

        try:
            with Image.open(full_img_path) as img:
                img_w, img_h = img.size
        except Exception:
            continue

        bbox_parts = [float(v.strip()) for v in row["bbox"].split(",")]
        quad_parts = [float(v.strip()) for v in row["quad"].split(",")]

        if len(bbox_parts) == 4 and len(quad_parts) == 8 and img_w > 0 and img_h > 0:
            bx, by, bw, bh = bbox_parts
            xc = min(1.0, max(0.0, (bx + bw / 2.0) / img_w))
            yc = min(1.0, max(0.0, (by + bh / 2.0) / img_h))
            wn = min(1.0, max(0.0, bw / img_w))
            hn = min(1.0, max(0.0, bh / img_h))

            # Quad 4 corners (tl, tr, br, bl)
            kpts = []
            for k in range(4):
                px = min(1.0, max(0.0, quad_parts[k * 2] / img_w))
                py = min(1.0, max(0.0, quad_parts[k * 2 + 1] / img_h))
                kpts.extend([f"{px:.6f}", f"{py:.6f}"])

            yolo_pose_str = f"{cls_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f} " + " ".join(kpts)
            with open(target_label_path, "w", encoding="utf-8") as wf:
                wf.write(yolo_pose_str + "\n")
            count += 1
            if is_syn == "1":
                synth_count += 1
            else:
                real_count += 1

    # Shuffle train and val rows
    random.shuffle(train_rows)
    random.shuffle(val_rows)

    # Write train.txt and val.txt with forward-slashed absolute paths
    train_txt_path = os.path.join(pose_dir, "train.txt")
    val_txt_path = os.path.join(pose_dir, "val.txt")

    with open(train_txt_path, "w", encoding="utf-8") as tf:
        for r in train_rows:
            tf.write(os.path.abspath(os.path.join(dataset_dir, r["image"])).replace("\\", "/") + "\n")

    with open(val_txt_path, "w", encoding="utf-8") as vf:
        for r in val_rows:
            vf.write(os.path.abspath(os.path.join(dataset_dir, r["image"])).replace("\\", "/") + "\n")

    # Generate data.yaml
    data_yaml_path = os.path.join(pose_dir, "data.yaml")
    yaml_content = f"""# YOLO-Pose Configuration for Russian License Plates (Volga IT 2026)
train: {os.path.abspath(train_txt_path).replace(chr(92), '/')}
val: {os.path.abspath(val_txt_path).replace(chr(92), '/')}

# 4 quadrilateral corner keypoints: (tl, tr, br, bl)
kpt_shape: [4, 2]

names:
  0: type1
  1: type1a
  2: type1b
  3: other
"""
    with open(data_yaml_path, "w", encoding="utf-8") as yf:
        yf.write(yaml_content)

    print(f"\n[SUCCESS] Prepared {count} YOLO-pose labels ({synth_count} synthetic, {real_count} real)")
    print(f"Manifest: {data_yaml_path}")
    print(f"Train:    {train_txt_path} ({len(train_rows)} samples)")
    print(f"Val:      {val_txt_path} ({len(val_rows)} samples)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare YOLO-Pose Dataset")
    parser.add_argument("--dataset_dir", type=str, default="dataset")
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prepare_yolo_pose(args.dataset_dir, args.val_ratio, args.seed)
