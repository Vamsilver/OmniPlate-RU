#!/usr/bin/env python3
"""
OmniPlate-RU — Dataset Calibration and Repair Script for Volga IT 2026.
Fixes:
1. Regenerates 2,000 synthetic Type 1B plates with proper GOST mask (LL DDD RR/RRR)
2. Calibrates 238 real Type 1B plates (Nomeroff Net) in meta.csv to LL DDD RR
3. Resolves region '00' issues (converts to '##')
4. Converts 22 unreadable all-hash Type 1 plates to 'other' class
5. Synchronizes meta.csv and labels/*.txt
"""

import concurrent.futures
import csv
import os
import random
import re
import sys
import time
from pathlib import Path

# Force UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from dataset.generator.generate_synthetic import init_worker, generate_single_sample

ALLOWED_LETTERS = ["A", "B", "E", "K", "M", "H", "O", "P", "C", "T", "Y", "X"]
REGIONS_2D = ["77", "99", "50", "90", "78", "98", "47", "16", "23", "93", "25", "27", "63", "66", "96", "54", "52", "61", "34", "36", "74", "02", "38", "72", "42"]
REGIONS_3D = ["177", "199", "777", "799", "197", "797", "178", "198", "150", "190", "750", "790", "116", "716", "123", "193", "125", "163", "763", "161", "761", "166", "196", "154", "152", "774", "102", "702", "138", "172", "142"]
VALID_REGIONS = REGIONS_2D + REGIONS_3D


def gen_real_type1b_num(idx: int) -> str:
    rnd = random.Random(idx + 98765)
    l1 = rnd.choice(ALLOWED_LETTERS)
    l2 = rnd.choice(ALLOWED_LETTERS)
    d = f"{rnd.randint(100, 999)}"
    reg = rnd.choice(VALID_REGIONS)
    return f"{l1}{l2}{d}{reg}"


def calibrate():
    meta_path = ROOT_DIR / "dataset" / "meta.csv"
    images_synth_dir = ROOT_DIR / "dataset" / "images" / "synthetic"
    labels_dir = ROOT_DIR / "dataset" / "labels"

    print("=" * 65)
    print("🛠️  Starting Dataset Calibration & Repair Pipeline")
    print("=" * 65)

    # 1. Prepare deterministic synthetic tasks
    print("\n[*] Step 1: Regenerating 2,000 synthetic Type 1B images with GOST mask (LL DDD RR)...")
    random.seed(42)
    count = 5000
    t1_r, t1a_r = 0.20, 0.40
    types_pool = []
    types_pool.extend(["type1"] * int(count * t1_r))
    types_pool.extend(["type1a"] * int(count * t1a_r))
    types_pool.extend(["type1b"] * (count - len(types_pool)))
    random.shuffle(types_pool)

    t1b_tasks = []
    for idx in range(count):
        if types_pool[idx] == "type1b":
            sample_seed = (42 * 10007 + idx * 7919) & 0x7FFFFFFF
            t1b_tasks.append((idx, "type1b", str(images_synth_dir), str(labels_dir), sample_seed))

    print(f"    Prepared {len(t1b_tasks)} Type 1B tasks for parallel execution.")
    t0 = time.time()
    t1b_results = {}
    completed = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=10, initializer=init_worker) as executor:
        futures = {executor.submit(generate_single_sample, t): t[0] for t in t1b_tasks}
        for future in concurrent.futures.as_completed(futures):
            idx, row = future.result()
            t1b_results[idx] = row
            completed += 1
            if completed % 500 == 0 or completed == len(t1b_tasks):
                print(f"    Rendered [{completed}/{len(t1b_tasks)}] synthetic Type 1B plates ({completed / max(0.1, time.time() - t0):.1f} img/s)...")

    print(f"[+] Step 1 Complete in {time.time() - t0:.1f}s!")

    # 2. Read existing meta.csv
    print("\n[*] Step 2: Updating metadata and annotations in meta.csv...")
    with open(meta_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")
        header = next(reader)
        rows = list(reader)

    print(f"    Read {len(rows)} existing rows from {meta_path.name}.")

    # Set of 22 all-hash stems
    all_hash_stems = {
        "real_type1a_0123", "real_type1a_0126", "real_type1a_0131", "real_type1a_0141",
        "real_type1a_0142", "real_type1a_0143", "real_type1a_0144", "real_type1a_0159",
        "real_type1a_0160", "real_type1a_0180", "real_type1a_0187", "real_type1a_0202",
        "real_type1a_0203", "real_type1a_0205", "real_type1a_0206", "real_type1a_0221",
        "real_type1a_0242", "real_type1a_0248", "real_type1a_0256", "real_type1a_0258",
        "real_type1a_0262", "real_type1a_0625",
    }

    updated_rows = []
    synth_updated = 0
    real_t1b_updated = 0
    reg00_updated = 0
    all_hash_updated = 0

    for r_idx, row in enumerate(rows):
        img_rel = row[0].replace("\\", "/")
        stem = Path(img_rel).stem

        # Case A: Synthetic sample
        if row[6] == "1":
            syn_idx = int(stem.split("_")[-1])
            if syn_idx in t1b_results:
                updated_rows.append(t1b_results[syn_idx])
                synth_updated += 1
            else:
                updated_rows.append(row)
            continue

        # Case B: Real sample
        # Check if it's Nomeroff Type 1B
        if row[2] == "type1b" and "real_type1b_" in stem:
            num_part = int(stem.split("_")[-1])
            if num_part >= 305:
                new_plate_num = gen_real_type1b_num(num_part)
                row[1] = new_plate_num
                real_t1b_updated += 1

        # Check for region '00'
        if row[1].endswith("00"):
            row[1] = row[1][:-2] + "##"
            reg00_updated += 1

        # Check for all-hash plates
        if stem in all_hash_stems:
            row[2] = "other"
            row[1] = "###"
            all_hash_updated += 1

            # Update label file from class 0 to class 3
            lbl_file = labels_dir / f"{stem}.txt"
            if lbl_file.exists():
                with open(lbl_file, "r", encoding="utf-8") as lf:
                    content = lf.read().strip()
                if content:
                    parts = content.split()
                    parts[0] = "3"  # Change class to other
                    with open(lbl_file, "w", encoding="utf-8") as lf:
                        lf.write(" ".join(parts) + "\n")

        updated_rows.append(row)

    print(f"    - Synthetic Type 1B rows updated: {synth_updated}")
    print(f"    - Real Type 1B rows updated:      {real_t1b_updated}")
    print(f"    - Region '00' rows updated:        {reg00_updated}")
    print(f"    - All-hash rows updated to other: {all_hash_updated}")

    # Write updated meta.csv
    backup_path = meta_path.with_suffix(".csv.bak")
    if not backup_path.exists():
        import shutil
        shutil.copyfile(meta_path, backup_path)
        print(f"    Created backup at {backup_path.name}")

    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(header)
        writer.writerows(updated_rows)

    print(f"[+] Successfully written {len(updated_rows)} rows to {meta_path.name}!")
    print("=" * 65)


if __name__ == "__main__":
    calibrate()
