#!/usr/bin/env python3
"""
Quick evaluation of OmniPlate-RU pipeline using official Volga IT evaluate.py.
Tests 100 balanced real images (25 Type 1, 25 Type 1A, 25 Type 1B, 25 Other).
"""

import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SAMPLE_DIR = PROJECT_ROOT / "test_output" / "official_eval_sample"
IMG_DIR = SAMPLE_DIR / "images"
GT_CSV = SAMPLE_DIR / "gt.csv"
PRED_CSV = SAMPLE_DIR / "pred.csv"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import argparse
import random

def prepare_sample(seed=None, per_class=25):
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    if IMG_DIR.exists():
        shutil.rmtree(IMG_DIR)
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    meta_file = PROJECT_ROOT / "dataset" / "meta.csv"
    by_type = {"type1": [], "type1a": [], "type1b": [], "other": []}

    with open(meta_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for r in reader:
            if r.get("is_synthetic") == "0":  # Real only
                ptype = r.get("plate_type")
                if ptype in by_type:
                    img_path = PROJECT_ROOT / "dataset" / r["image"]
                    if img_path.exists():
                        by_type[ptype].append(r)

    if seed is not None:
        rng = random.Random(seed)
        for ptype in by_type:
            by_type[ptype] = rng.sample(by_type[ptype], min(per_class, len(by_type[ptype])))
    else:
        for ptype in by_type:
            by_type[ptype] = by_type[ptype][:per_class]

    gt_rows = []
    for ptype, rows in by_type.items():
        print(f"Selected {len(rows)} real images of {ptype} (seed={seed})")
        for r in rows:
            src_img = PROJECT_ROOT / "dataset" / r["image"]
            dst_img = IMG_DIR / src_img.name
            shutil.copy2(src_img, dst_img)
            gt_rows.append({
                "image": src_img.name,
                "plate_num": r["plate_num"],
                "plate_type": r["plate_type"],
                "is_vehicle": r["is_vehicle"],
            })

    with open(GT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "plate_num", "plate_type", "is_vehicle"], delimiter=";")
        writer.writeheader()
        for row in gt_rows:
            writer.writerow(row)

    print(f"Saved {len(gt_rows)} GT rows to {GT_CSV}")

def run_pipeline():
    python_bin = sys.executable
    cmd = [
        python_bin,
        str(PROJECT_ROOT / "run.py"),
        "--input", str(IMG_DIR),
        "--output", str(PRED_CSV),
        "--device", "cuda",
    ]
    print(f"Running pipeline: {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(res.stdout)
    if res.returncode != 0:
        print("Pipeline failed:", res.stderr)
        sys.exit(1)

def run_official_evaluate():
    python_bin = sys.executable
    eval_script = PROJECT_ROOT / "official_public" / "Полуфинал - public" / "evaluate.py"
    cmd = [
        python_bin,
        str(eval_script),
        "--gt", str(GT_CSV),
        "--pred", str(PRED_CSV),
        "--images", str(IMG_DIR),
        "--details", str(SAMPLE_DIR / "details.csv"),
    ]
    print(f"\nEvaluating with official evaluate.py...")
    res = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(res.stdout)
    if res.stderr:
        print("Stderr:", res.stderr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Official evaluate.py runner on balanced real subset")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for sampling (None = first N)")
    parser.add_argument("--per-class", type=int, default=25, help="Number of images per class (default: 25)")
    args = parser.parse_args()

    prepare_sample(seed=args.seed, per_class=args.per_class)
    run_pipeline()
    run_official_evaluate()
