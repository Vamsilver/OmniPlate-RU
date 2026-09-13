#!/usr/bin/env python3
"""
Zero-Copy YOLO-Pose Dataset Preparation for Volga IT 2026.
Reads dataset annotations, extracts 4 quad keypoints, creates train/val splits,
and generates ultralytics YOLO-pose data.yaml without duplicating image files.
"""

import argparse
import csv
import os
import random
import sys
from typing import Dict, List


def prepare_yolo_pose(dataset_dir: str, val_ratio: float = 0.2, seed: int = 42):
    random.seed(seed)

    meta_path = os.path.join(dataset_dir, "meta.csv")
    if not os.path.exists(meta_path):
        print(f"[ERROR] meta.csv not found at {meta_path}")
        sys.exit(1)

    pose_dir = os.path.join(dataset_dir, "yolo_pose")
    pose_labels_dir = os.path.join(dataset_dir, "labels_pose")
    os.makedirs(pose_dir, exist_ok=True)
    os.makedirs(pose_labels_dir, exist_ok=True)

    class_map = {"type1": 0, "type1a": 1, "type1b": 2, "other": 3}

    # Group by class for stratified split
    samples_by_class: Dict[str, List[Dict]] = {k: [] for k in class_map.keys()}

    with open(meta_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            p_type = row.get("plate_type")
            if p_type in samples_by_class:
                samples_by_class[p_type].append(row)

    train_rows = []
    val_rows = []

    for p_type, rows in samples_by_class.items():
        random.shuffle(rows)
        n_val = int(len(rows) * val_ratio)
        val_rows.extend(rows[:n_val])
        train_rows.extend(rows[n_val:])

    print(f"Total samples: {len(train_rows) + len(val_rows)}")
    print(f"Train split:   {len(train_rows)}")
    print(f"Val split:     {len(val_rows)}")

    # Process each sample to generate standard YOLO-pose label
    for row in train_rows + val_rows:
        img_rel = row["image"]
        base_name = os.path.splitext(os.path.basename(img_rel))[0]
        p_type = row["plate_type"]
        class_id = class_map[p_type]

        # Read original label if exists, or compute from bbox & quad
        raw_label_path = os.path.join(dataset_dir, "labels", f"{base_name}.txt")
        pose_label_path = os.path.join(pose_labels_dir, f"{base_name}.txt")

        if os.path.exists(raw_label_path):
            with open(raw_label_path, "r", encoding="utf-8") as rf:
                line = rf.readline().strip()
                tokens = line.split()
                # Tokens: [class, xc, yc, w, h, x1, y1, x2, y2, x3, y3, x4, y4, plate_num]
                if len(tokens) >= 13:
                    yolo_pose_str = " ".join(tokens[:13])
                    with open(pose_label_path, "w", encoding="utf-8") as wf:
                        wf.write(yolo_pose_str + "\n")
        else:
            # Generate from meta.csv bbox and quad
            bbox = [int(v) for v in row["bbox"].split(",")]
            quad = [int(v) for v in row["quad"].split(",")]
            img_full = os.path.join(dataset_dir, img_rel)
            if os.path.exists(img_full):
                import cv2
                img = cv2.imread(img_full)
                if img is not None:
                    h, w = img.shape[:2]
                    bx, by, bw, bh = bbox
                    xc = (bx + bw / 2.0) / w
                    yc = (by + bh / 2.0) / h
                    nw = bw / w
                    nh = bh / h
                    norm_quad = []
                    for i in range(4):
                        norm_quad.extend([quad[i * 2] / w, quad[i * 2 + 1] / h])
                    q_str = " ".join(f"{v:.6f}" for v in norm_quad)
                    with open(pose_label_path, "w", encoding="utf-8") as wf:
                        wf.write(f"{class_id} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f} {q_str}\n")

    # Write train.txt and val.txt with absolute or relative paths
    train_txt_path = os.path.join(pose_dir, "train.txt")
    val_txt_path = os.path.join(pose_dir, "val.txt")

    with open(train_txt_path, "w", encoding="utf-8") as tf:
        for r in train_rows:
            tf.write(os.path.abspath(os.path.join(dataset_dir, r["image"])) + "\n")

    with open(val_txt_path, "w", encoding="utf-8") as vf:
        for r in val_rows:
            vf.write(os.path.abspath(os.path.join(dataset_dir, r["image"])) + "\n")

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

    print(f"\n[SUCCESS] YOLO-Pose dataset manifest created!")
    print(f"Manifest: {data_yaml_path}")
    print(f"Labels:   {pose_labels_dir}")
    print(f"Train:    {train_txt_path}")
    print(f"Val:      {val_txt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare YOLO-Pose Dataset")
    parser.add_argument("--dataset_dir", type=str, default="dataset")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prepare_yolo_pose(args.dataset_dir, args.val_ratio, args.seed)
