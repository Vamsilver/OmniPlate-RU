#!/usr/bin/env python3
"""
Random evaluation of OmniPlate-RU pipeline using official Volga IT evaluate.py.
Randomly samples N images per category from valid subsets.
"""

import argparse
import csv
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SAMPLE_DIR = PROJECT_ROOT / "test_output" / "random_eval_sample"
IMG_DIR = SAMPLE_DIR / "images"
GT_CSV = SAMPLE_DIR / "gt.csv"
PRED_CSV = SAMPLE_DIR / "pred.csv"
DETAILS_CSV = SAMPLE_DIR / "details.csv"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def prepare_random_sample(seed: int = 42, per_class: int = 25):
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    if IMG_DIR.exists():
        shutil.rmtree(IMG_DIR)
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    meta_file = PROJECT_ROOT / "dataset" / "meta.csv"
    with open(meta_file, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f, delimiter=";"))

    real = [r for r in reader if r.get("is_synthetic") == "0"]

    # Filter pools
    # Type 1: clean real type1 without unreadable #
    pool_t1 = [r for r in real if r["plate_type"] == "type1" and "#" not in r["plate_num"]]
    # Type 1A: real type1a
    pool_t1a = [r for r in real if r["plate_type"] == "type1a" and "#" not in r["plate_num"]]
    # Type 1B: real yellow buses / taxis
    pool_t1b = [r for r in real if r["plate_type"] == "type1b" and "#" not in r["plate_num"]]
    # Other: real background / non-vehicle / foreign
    pool_other = [r for r in real if r["plate_type"] == "other"]

    rng = random.Random(seed)
    sampled_t1 = rng.sample(pool_t1, min(per_class, len(pool_t1)))
    sampled_t1a = rng.sample(pool_t1a, min(per_class, len(pool_t1a)))
    sampled_t1b = rng.sample(pool_t1b, min(per_class, len(pool_t1b)))
    sampled_other = rng.sample(pool_other, min(per_class, len(pool_other)))

    all_sampled = [
        ("type1", sampled_t1),
        ("type1a", sampled_t1a),
        ("type1b", sampled_t1b),
        ("other", sampled_other),
    ]

    pools = {
        "type1": pool_t1,
        "type1a": pool_t1a,
        "type1b": pool_t1b,
        "other": pool_other,
    }

    gt_rows = []
    for ptype, rows in all_sampled:
        pool_len = len(pools[ptype])
        print(f"Sampled {len(rows)} real images of {ptype} (pool size: {pool_len})")
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
    print(f"Running pipeline on {IMG_DIR}...")
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
        "--details", str(DETAILS_CSV),
    ]
    print(f"\nEvaluating with official evaluate.py...")
    res = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(res.stdout)
    if res.stderr:
        print("Stderr:", res.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Official evaluate.py on random real sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--per-class", type=int, default=25, help="Images per class (default: 25)")
    args = parser.parse_args()

    prepare_random_sample(seed=args.seed, per_class=args.per_class)
    run_pipeline()
    run_official_evaluate()
