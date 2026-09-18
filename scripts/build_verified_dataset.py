#!/usr/bin/env python3
"""
scripts/build_verified_dataset.py
Builds and verifies full YOLO-Pose dataset for Volga IT 2026.
Processes all 6,504 annotations from dataset/meta.csv:
- 5,000 synthetic (Type 1: 1000, Type 1A: 2000, Type 1B: 2000)
- 1,504 real (Type 1: 594, Type 1A: 302, Type 1B: 304, Other: 304)
Generates:
- dataset/labels/synthetic/*.txt
- dataset/labels/real/*.txt
- dataset/labels/*.txt (root sync)
- dataset/yolo_pose/train.txt
- dataset/yolo_pose/val.txt
- dataset/yolo_pose/data.yaml
Cleans stale Ultralytics .cache files.
"""

import argparse
import csv
import glob
import os
import random
import sys
from typing import Dict, List
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)


def build_dataset(dataset_dir: str = "dataset", val_ratio: float = 0.15, seed: int = 42):
    random.seed(seed)
    dataset_dir = os.path.abspath(dataset_dir)
    meta_path = os.path.join(dataset_dir, "meta.csv")
    if not os.path.exists(meta_path):
        print(f"[ERROR] meta.csv not found at: {meta_path}")
        sys.exit(1)

    pose_dir = os.path.join(dataset_dir, "yolo_pose")
    labels_synth_dir = os.path.join(dataset_dir, "labels", "synthetic")
    labels_real_dir = os.path.join(dataset_dir, "labels", "real")
    os.makedirs(pose_dir, exist_ok=True)
    os.makedirs(labels_synth_dir, exist_ok=True)
    os.makedirs(labels_real_dir, exist_ok=True)

    # 1. Purge stale YOLO cache files
    print("[*] Cleaning stale Ultralytics cache files...")
    cleaned_caches = 0
    for cache_f in glob.glob(os.path.join(dataset_dir, "**", "*.cache"), recursive=True):
        try:
            os.remove(cache_f)
            cleaned_caches += 1
        except Exception:
            pass
    print(f"[*] Cleaned {cleaned_caches} cache files.")

    # 2. Read meta.csv and group by strata: (plate_type, is_synthetic)
    class_map = {"type1": 0, "type1a": 1, "type1b": 2, "other": 3}
    strata: Dict[str, List[Dict[str, str]]] = {}
    total_meta_rows = 0

    with open(meta_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            total_meta_rows += 1
            p_type = row.get("plate_type", "").strip()
            if p_type not in class_map:
                continue
            is_syn = row.get("is_synthetic", "0").strip()
            key = f"{p_type}_{is_syn}"
            if key not in strata:
                strata[key] = []
            strata[key].append(row)

    print(f"\n[*] Total meta.csv annotations read: {total_meta_rows}")
    print("[*] Stratified distribution:")
    train_rows = []
    val_rows = []

    for stratum_key, rows in sorted(strata.items()):
        random.shuffle(rows)
        n_val = max(1, int(len(rows) * val_ratio)) if len(rows) > 5 else 0
        val_subset = rows[:n_val]
        train_subset = rows[n_val:]
        val_rows.extend(val_subset)
        train_rows.extend(train_subset)
        print(f"  • {stratum_key:<12}: Total={len(rows):<5} -> Train={len(train_subset):<5} | Val={len(val_subset):<4}")

    # 3. Generate YOLO-pose labels
    count_synth = 0
    count_real = 0
    count_other = 0

    all_rows = train_rows + val_rows
    print(f"\n[*] Generating YOLO-pose labels for {len(all_rows)} samples...")

    for row in all_rows:
        img_rel = row["image"].strip().replace("\\", "/")
        base_name = os.path.splitext(os.path.basename(img_rel))[0]
        p_type = row["plate_type"].strip()
        cls_id = class_map[p_type]
        is_syn = row.get("is_synthetic", "0").strip()

        # Target label files:
        # 1. Under labels/synthetic/ or labels/real/
        sub = "synthetic" if is_syn == "1" else "real"
        primary_label_path = os.path.join(dataset_dir, "labels", sub, f"{base_name}.txt")
        # 2. Also keep root labels/ in sync
        root_label_path = os.path.join(dataset_dir, "labels", f"{base_name}.txt")

        # Negative sample (other) -> empty label file teaches YOLO zero false positives
        if p_type == "other":
            with open(primary_label_path, "w", encoding="utf-8") as wf:
                pass
            with open(root_label_path, "w", encoding="utf-8") as wf:
                pass
            count_other += 1
            if is_syn == "1":
                count_synth += 1
            else:
                count_real += 1
            continue

        # If precomputed synthetic label exists in labels/{base_name}.txt, reuse keypoints
        precomputed_label = os.path.join(dataset_dir, "labels", f"{base_name}.txt")
        yolo_pose_str = None
        if os.path.exists(precomputed_label) and is_syn == "1":
            try:
                with open(precomputed_label, "r", encoding="utf-8") as rf:
                    line = rf.readline().strip()
                    tokens = line.split()
                    if len(tokens) >= 13:
                        # Ensure class id matches class_map
                        yolo_pose_str = f"{cls_id} " + " ".join(tokens[1:13])
            except Exception:
                yolo_pose_str = None

        if yolo_pose_str is None:
            full_img_path = os.path.join(dataset_dir, img_rel)
            if not os.path.exists(full_img_path):
                print(f"[WARN] Image file missing: {full_img_path}")
                continue

            try:
                with Image.open(full_img_path) as img:
                    img_w, img_h = img.size
            except Exception as e:
                print(f"[WARN] Failed to open image {full_img_path}: {e}")
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

        if yolo_pose_str:
            with open(primary_label_path, "w", encoding="utf-8") as wf:
                wf.write(yolo_pose_str + "\n")
            with open(root_label_path, "w", encoding="utf-8") as wf:
                wf.write(yolo_pose_str + "\n")

            if is_syn == "1":
                count_synth += 1
            else:
                count_real += 1

    # 4. Shuffle splits and write train.txt and val.txt
    random.shuffle(train_rows)
    random.shuffle(val_rows)

    train_txt_path = os.path.join(pose_dir, "train.txt")
    val_txt_path = os.path.join(pose_dir, "val.txt")

    with open(train_txt_path, "w", encoding="utf-8") as tf:
        for r in train_rows:
            p = os.path.abspath(os.path.join(dataset_dir, r["image"])).replace("\\", "/")
            tf.write(p + "\n")

    with open(val_txt_path, "w", encoding="utf-8") as vf:
        for r in val_rows:
            p = os.path.abspath(os.path.join(dataset_dir, r["image"])).replace("\\", "/")
            vf.write(p + "\n")

    # 5. Generate data.yaml
    data_yaml_path = os.path.join(pose_dir, "data.yaml")
    yaml_content = f"""# YOLO-Pose Configuration for Russian License Plates (Volga IT 2026)
train: {train_txt_path.replace(chr(92), '/')}
val: {val_txt_path.replace(chr(92), '/')}

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

    print("\n" + "=" * 60)
    print("✅ YOLO-Pose Dataset Preparation Completed Successfully!")
    print(f"  • Total labels generated: {count_synth + count_real}")
    print(f"    - Synthetic: {count_synth} / 5000")
    print(f"    - Real:      {count_real} / 1504 (including {count_other} other/negatives)")
    print(f"  • Train split: {len(train_rows)} samples -> {train_txt_path}")
    print(f"  • Val split:   {len(val_rows)} samples -> {val_txt_path}")
    print(f"  • Manifest:    {data_yaml_path}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Verified YOLO Dataset")
    parser.add_argument("--dataset_dir", type=str, default="dataset", help="Dataset root directory")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Validation ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    build_dataset(args.dataset_dir, args.val_ratio, args.seed)
