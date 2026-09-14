#!/usr/bin/env python3
"""
Zero-Copy YOLO-Pose Dataset Preparation for Volga IT 2026.
Reads dataset annotations, prepares labels in dataset/labels/synthetic,
creates stratified train/val splits, and generates ultralytics YOLO-pose data.yaml.
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
    pose_synth_labels_dir = os.path.join(dataset_dir, "labels", "synthetic")
    os.makedirs(pose_dir, exist_ok=True)
    os.makedirs(pose_synth_labels_dir, exist_ok=True)

    class_map = {"type1": 0, "type1a": 1, "type1b": 2, "other": 3}

    samples_by_class: Dict[str, List[Dict]] = {k: [] for k in class_map.keys()}

    with open(meta_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            p_type = row.get("plate_type")
            is_syn = row.get("is_synthetic", "0")
            # Strict synthetic-first strategy: train YOLO strictly on verified synthetic plates
            # + negative background samples to suppress false positives on background objects
            if is_syn == "1" and p_type in ("type1", "type1a", "type1b"):
                samples_by_class[p_type].append(row)
            elif p_type == "other":
                # Include negative background samples
                samples_by_class["other"].append(row)

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

    # Process each sample to generate standard YOLO-pose labels
    count = 0
    for row in train_rows + val_rows:
        img_rel = row["image"]
        base_name = os.path.splitext(os.path.basename(img_rel))[0]
        p_type = row["plate_type"]

        # Ensure label directory corresponds to image subdirectory (labels/synthetic or labels/real)
        subdir = os.path.dirname(img_rel).replace("images", "labels")
        target_dir = os.path.join(dataset_dir, subdir)
        os.makedirs(target_dir, exist_ok=True)
        target_label_path = os.path.join(target_dir, f"{base_name}.txt")

        if p_type == "other":
            # Negative background sample (no plate) -> empty file teaches YOLO zero false positives
            with open(target_label_path, "w", encoding="utf-8") as wf:
                pass
            count += 1
            continue

        raw_label_path = os.path.join(dataset_dir, "labels", f"{base_name}.txt")
        if os.path.exists(raw_label_path):
            with open(raw_label_path, "r", encoding="utf-8") as rf:
                line = rf.readline().strip()
                tokens = line.split()
                if len(tokens) >= 13:
                    yolo_pose_str = " ".join(tokens[:13])
                    with open(target_label_path, "w", encoding="utf-8") as wf:
                        wf.write(yolo_pose_str + "\n")
            count += 1

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

    print(f"\n[SUCCESS] Prepared {count} YOLO-pose labels in: {pose_synth_labels_dir}")
    print(f"Manifest: {data_yaml_path}")
    print(f"Train:    {train_txt_path}")
    print(f"Val:      {val_txt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare YOLO-Pose Dataset")
    parser.add_argument("--dataset_dir", type=str, default="dataset")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prepare_yolo_pose(args.dataset_dir, args.val_ratio, args.seed)
