#!/usr/bin/env python3
"""
Procedural Generator for 2,000 Targeted Hard Synthetic Plate Crops (Volga IT 2026).
Specifically targets:
- Hard optical confusion pairs: M/T, B/8, O/0
- Deep cast shadows (diagonal, vertical bars, horizontal overhangs)
- Low-resolution downsampling and severe motion blur
- Road grime, mud splatter, and specular glare
Outputs canonical 160x36 crops ready for LPRNet fine-tuning.
"""

import argparse
import csv
import math
import os
import random
import re
import sys
from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "dataset" / "generator"))

from plate_renderer import PlateRenderer, ALLOWED_LETTERS, DIGITS, POPULAR_REGIONS
from src.pipeline.rectifier import PlateRectifier

OUTPUT_DIR = PROJECT_ROOT / "dataset" / "verified_crops"
MANIFEST_PATH = OUTPUT_DIR / "manifest.csv"

# Confusion character sets
CONFUSION_M_T = ["M", "T"]
CONFUSION_B_8_LETTERS = ["B"]
CONFUSION_B_8_DIGITS = ["8"]
CONFUSION_O_0_LETTERS = ["O"]
CONFUSION_O_0_DIGITS = ["0"]


def generate_targeted_plate_text(strategy: str) -> Tuple[str, str, str, str, str]:
    """Generates plate text tailored to specific confusion pairs."""
    reg = random.choice(POPULAR_REGIONS)

    if strategy == "m_b":
        # Heavily interweave M and B
        l1 = random.choice(["M", "B"])
        d3 = "".join(random.choices(DIGITS, k=3))
        l2 = random.choice(["MB", "BM", "BB", "MM", "BC", "MC", "MO", "BO", "BH", "MH"])
    elif strategy == "o_c":
        # Heavily interweave O and C
        l1 = random.choice(["O", "C"])
        d3 = "".join(random.choices(DIGITS, k=3))
        l2 = random.choice(["OC", "CO", "OO", "CC", "OP", "CP", "CE", "OE"])
    elif strategy == "digits_80":
        # Hard digit confusions (8 <-> 0, 8 <-> 1, 2 <-> 7, 3 <-> 9)
        l1 = random.choice(ALLOWED_LETTERS)
        d3 = random.choice(["808", "080", "880", "008", "800", "088", "818", "181", "272", "727", "393", "939", "717", "171", "010"])
        l2 = "".join(random.choices(ALLOWED_LETTERS, k=2))
    elif strategy == "blurry_regions":
        l1 = random.choice(ALLOWED_LETTERS)
        d3 = "".join(random.choices(DIGITS, k=3))
        l2 = "".join(random.choices(ALLOWED_LETTERS, k=2))
        reg = random.choice(["199", "799", "177", "197", "777", "750", "790", "154", "116", "716", "125", "138", "178", "198", "196", "161", "761", "102", "123"])
    elif strategy == "m_t":
        # Heavily interweave M and T
        l1 = random.choice(CONFUSION_M_T)
        d3 = "".join(random.choices(DIGITS, k=3))
        l2 = "".join(random.choices(CONFUSION_M_T, k=2))
    elif strategy == "b_8":
        # Heavily interweave B in letters and 8 in digits
        l1 = random.choice(["B", "B", random.choice(ALLOWED_LETTERS)])
        d3 = "".join(random.choices(["8", "8", "0", "3", random.choice(DIGITS)], k=3))
        l2 = random.choice(["BB", "BX", "BE", "BK", "BA"])
    elif strategy == "o_0":
        # Heavily interweave O in letters and 0 in digits
        l1 = random.choice(["O", "O", random.choice(ALLOWED_LETTERS)])
        d3 = "".join(random.choices(["0", "0", "8", random.choice(DIGITS)], k=3))
        l2 = random.choice(["OO", "OP", "OC", "OM", "OX"])
    elif strategy == "hybrid_hard":
        # Mix all hard pairs in one plate (e.g. B808TO, M080BO)
        l1 = random.choice(["B", "M", "T", "O", "C"])
        d3 = random.choice(["808", "080", "880", "008", "800", "088", "818", "010", "272", "393"])
        l2 = random.choice(["TO", "BO", "MB", "TM", "BT", "MO", "OB", "CO", "MC", "BC"])
    else:
        # Standard GOST random
        l1 = random.choice(ALLOWED_LETTERS)
        d3 = "".join(random.choices(DIGITS, k=3))
        l2 = "".join(random.choices(ALLOWED_LETTERS, k=2))

    full_str = f"{l1}{d3}{l2}{reg}"
    return full_str, l1, d3, l2, reg


def generate_targeted_trailer_plate_text(strategy: str) -> Tuple[str, str, str, str]:
    """Generates plate text for Type 2 trailers (LL DDDD RR / RRR)."""
    reg = random.choice(POPULAR_REGIONS)
    if strategy == "m_t":
        l2 = "".join(random.choices(CONFUSION_M_T, k=2))
        d4 = "".join(random.choices(DIGITS, k=4))
    elif strategy == "b_8":
        l2 = random.choice(["BB", "BX", "BE", "BK", "BA", "AB", "MB", "TB"])
        d4 = "".join(random.choices(["8", "8", "0", "3", random.choice(DIGITS)], k=4))
    elif strategy == "o_0":
        l2 = random.choice(["OO", "OP", "OC", "OM", "OX", "AO", "BO", "TO"])
        d4 = "".join(random.choices(["0", "0", "8", random.choice(DIGITS)], k=4))
    elif strategy == "hybrid_hard":
        l2 = random.choice(["TO", "BO", "MB", "TM", "BT", "MO", "OB", "BB", "OO"])
        d4 = random.choice(["8080", "0808", "8800", "0088", "8888", "0000", "1808", "8108"])
    else:
        l2 = "".join(random.choices(ALLOWED_LETTERS, k=2))
        d4 = "".join(random.choices(DIGITS, k=4))
    full_str = f"{l2}{d4}{reg}"
    return full_str, l2, d4, reg


def generate_targeted_bus_plate_text(strategy: str) -> Tuple[str, str, str, str]:
    """Generates plate text for Type 1B buses / public transport (LL DDD RR / RRR strictly by GOST)."""
    reg = random.choice(POPULAR_REGIONS)
    if strategy == "m_t":
        l2 = "".join(random.choices(CONFUSION_M_T, k=2))
        d3 = "".join(random.choices(DIGITS, k=3))
    elif strategy == "b_8":
        l2 = random.choice(["BB", "BX", "BE", "BK", "BA", "AB", "MB", "TB"])
        d3 = "".join(random.choices(["8", "8", "0", "3", random.choice(DIGITS)], k=3))
    elif strategy == "o_0":
        l2 = random.choice(["OO", "OP", "OC", "OM", "OX", "AO", "BO", "TO"])
        d3 = "".join(random.choices(["0", "0", "8", random.choice(DIGITS)], k=3))
    elif strategy == "hybrid_hard":
        l2 = random.choice(["TO", "BO", "MB", "TM", "BT", "MO", "OB", "BB", "OO"])
        d3 = random.choice(["808", "080", "880", "008", "800", "088", "818", "010"])
    else:
        l2 = "".join(random.choices(ALLOWED_LETTERS, k=2))
        d3 = "".join(random.choices(DIGITS, k=3))
    full_str = f"{l2}{d3}{reg}"
    return full_str, l2, d3, reg


def apply_deep_shadows(img: np.ndarray) -> np.ndarray:
    """Simulates severe realistic cast shadows with high contrast (0.25 - 0.45 brightness)."""
    h, w = img.shape[:2]
    mask = np.ones((h, w), dtype=np.float32)

    shadow_type = random.choice(["diagonal", "vertical_bar", "horizontal_band", "irregular"])

    if shadow_type == "diagonal":
        # Diagonal edge across plate
        angle = random.uniform(-50, 50)
        rad = math.radians(angle)
        cx = random.uniform(0.2 * w, 0.8 * w)
        cy = h / 2.0
        # Line normal vector
        nx, ny = math.cos(rad), math.sin(rad)
        # Distance map
        y, x = np.ogrid[:h, :w]
        dist = (x - cx) * nx + (y - cy) * ny
        # Smooth transition with blur
        blur_w = random.uniform(3, 8)
        mask = 1.0 / (1.0 + np.exp(-dist / blur_w))
        if random.random() > 0.5:
            mask = 1.0 - mask
        dark_factor = random.uniform(0.25, 0.45)
        bright_factor = random.uniform(1.05, 1.25)
        mask = dark_factor + mask * (bright_factor - dark_factor)

    elif shadow_type == "vertical_bar":
        # Shadow bar covering 1-3 characters
        x_start = random.randint(10, w - 40)
        bar_w = random.randint(20, 50)
        bar_mask = np.zeros((h, w), dtype=np.float32)
        bar_mask[:, x_start:min(w, x_start + bar_w)] = 1.0
        bar_mask = cv2.GaussianBlur(bar_mask, (15, 15), 0)
        dark_factor = random.uniform(0.25, 0.42)
        mask = 1.0 - bar_mask * (1.0 - dark_factor)

    elif shadow_type == "horizontal_band":
        # Shadow from bumper / boot lid
        y_split = random.randint(int(0.25 * h), int(0.65 * h))
        band_mask = np.zeros((h, w), dtype=np.float32)
        if random.random() > 0.5:
            band_mask[:y_split, :] = 1.0
        else:
            band_mask[y_split:, :] = 1.0
        band_mask = cv2.GaussianBlur(band_mask, (15, 15), 0)
        dark_factor = random.uniform(0.3, 0.48)
        mask = 1.0 - band_mask * (1.0 - dark_factor)

    elif shadow_type == "irregular":
        # Complex multi-blob shadow
        blob = np.zeros((h, w), dtype=np.float32)
        for _ in range(random.randint(2, 5)):
            bx = random.randint(0, w)
            by = random.randint(0, h)
            rx = random.randint(15, 45)
            ry = random.randint(8, 20)
            cv2.ellipse(blob, (bx, by), (rx, ry), random.randint(0, 180), 0, 360, 1.0, -1)
        blob = cv2.GaussianBlur(blob, (21, 21), 0)
        blob = np.clip(blob, 0.0, 1.0)
        dark_factor = random.uniform(0.28, 0.45)
        mask = 1.0 - blob * (1.0 - dark_factor)

    shadowed = img.astype(np.float32) * mask[:, :, np.newaxis]
    return np.clip(shadowed, 0, 255).astype(np.uint8)


def apply_heavy_blur_and_downsample(img: np.ndarray) -> np.ndarray:
    """Simulates low-resolution sensors and motion blur."""
    h, w = img.shape[:2]
    aug_choice = random.choice(["downsample", "motion_blur", "gaussian_blur", "combo"])

    if aug_choice in ("downsample", "combo"):
        low_w = random.randint(38, 75)
        low_h = random.randint(9, 18)
        small = cv2.resize(img, (low_w, low_h), interpolation=cv2.INTER_AREA)
        interp = random.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC])
        img = cv2.resize(small, (w, h), interpolation=interp)

    if aug_choice in ("motion_blur", "combo"):
        k_size = random.choice([5, 7, 9])
        kernel = np.zeros((k_size, k_size))
        # Mostly horizontal or slightly diagonal motion
        row = k_size // 2
        kernel[row, :] = 1.0 / k_size
        img = cv2.filter2D(img, -1, kernel)

    if aug_choice == "gaussian_blur":
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)

    return img


def apply_mud_splatter(img: np.ndarray) -> np.ndarray:
    """Adds road mud spots that partially occlude character strokes."""
    h, w = img.shape[:2]
    num_spots = random.randint(3, 8)
    mud_mask = np.zeros((h, w), dtype=np.float32)

    for _ in range(num_spots):
        cx = random.randint(15, w - 15)
        cy = random.randint(5, h - 5)
        r = random.randint(2, 6)
        cv2.circle(mud_mask, (cx, cy), r, 1.0, -1)

    mud_mask = cv2.GaussianBlur(mud_mask, (5, 5), 0)[:, :, np.newaxis]
    mud_color = np.array([random.randint(40, 80), random.randint(45, 90), random.randint(50, 100)], dtype=np.float32)

    res = img.astype(np.float32) * (1.0 - mud_mask * 0.85) + mud_color * (mud_mask * 0.85)
    return np.clip(res, 0, 255).astype(np.uint8)


def apply_specular_glare(img: np.ndarray) -> np.ndarray:
    """Adds intense sun / flash glare over 1 or 2 characters."""
    h, w = img.shape[:2]
    gx = random.randint(15, w - 15)
    gy = random.randint(5, h - 5)
    y, x = np.ogrid[:h, :w]
    radius = random.randint(15, 35)
    dist_sq = (x - gx) ** 2 + (y - gy) ** 2
    glare = np.exp(-dist_sq / (2 * (radius ** 2)))[:, :, np.newaxis]
    res = img.astype(np.float32) + glare * random.uniform(100, 180)
    return np.clip(res, 0, 255).astype(np.uint8)


def apply_left_edge_artifact(img: np.ndarray) -> np.ndarray:
    """
    Simulates parasitic frame borders and mounting bolts on the left edge.
    This trains LPRNet to avoid false letters (like 'H', '#', 'I') induced by frame rims.
    """
    h, w = img.shape[:2]
    res = img.copy()
    artifact_type = random.choice(["frame_rim", "bolt", "corner_frame", "both"])

    if artifact_type in ("frame_rim", "both"):
        # Dark vertical border / license plate holder rim on the left edge (width 2-7 px)
        rim_width = random.randint(2, 7)
        rim_color = random.randint(10, 50)
        for x in range(rim_width):
            alpha = max(0.25, 1.0 - (x / float(rim_width)))
            res[:, x] = (res[:, x].astype(np.float32) * (1.0 - alpha) + rim_color * alpha).astype(np.uint8)

    if artifact_type in ("bolt", "both"):
        # Left mounting bolt / screw near characters
        bx = random.randint(5, 13)
        by = random.randint(int(0.25 * h), int(0.75 * h))
        r = random.randint(3, 5)
        # Outer dark ring / hole
        cv2.circle(res, (bx, by), r + 1, (30, 30, 30), -1)
        # Metallic bolt head
        cv2.circle(res, (bx, by), r, (120, 120, 120), -1)
        # Screw slot or center recess
        cv2.circle(res, (bx, by), max(1, r - 2), (40, 40, 40), -1)

    if artifact_type == "corner_frame":
        # Triangular corner clamp / holder
        pts = np.array([[0, 0], [random.randint(6, 12), 0], [0, random.randint(8, 16)]], dtype=np.int32)
        cv2.fillPoly(res, [pts], (25, 25, 25))

    return res


def generate_hard_dataset(total_count: int = 2000, seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

    print("=" * 70)
    print(f"  OmniPlate-RU: Generating {total_count} Targeted Hard Synthetic Crops")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    renderer = PlateRenderer()
    rectifier = PlateRectifier()

    # Load existing manifest to append safely
    manifest_records = {}
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
            for line in lines[1:]:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(";") if ";" in line else line.split(",")
                if len(parts) >= 3:
                    manifest_records[parts[0].strip()] = [p.strip() for p in parts]

    print(f"[*] Loaded {len(manifest_records)} pre-existing manifest entries.")

    # Distribution: 1400 Type 1, 300 Type 1B, 300 Type 1A
    n_t1 = int(total_count * 0.70)
    n_1b = int(total_count * 0.15)
    n_1a = total_count - n_t1 - n_1b

    plan = [("type1", n_t1), ("type1b", n_1b), ("type1a", n_1a)]
    strategies = ["m_b", "o_c", "digits_80", "blurry_regions", "b_8", "m_t", "o_0", "hybrid_hard", "random"]
    strategy_weights = [0.22, 0.20, 0.20, 0.15, 0.08, 0.05, 0.05, 0.03, 0.02]

    generated_records = []
    global_idx = 0

    for p_type, count in plan:
        print(f"[*] Generating {count} hard samples for {p_type}...")
        for _ in range(count):
            global_idx += 1
            strat = random.choices(strategies, weights=strategy_weights)[0]
            # Render plate
            if p_type == "type1":
                plate_num, _, _, _, _ = generate_targeted_plate_text(strat)
                pil_im, _ = renderer.render_type1(plate_num)
                crop = cv2.resize(np.array(pil_im), (160, 36), interpolation=cv2.INTER_LINEAR)
            elif p_type == "type1b":
                plate_num, _, _, _ = generate_targeted_bus_plate_text(strat)
                pil_im, _ = renderer.render_type1b(plate_num)
                crop = cv2.resize(np.array(pil_im), (160, 36), interpolation=cv2.INTER_LINEAR)
            elif p_type == "type1a":
                plate_num, _, _, _, _ = generate_targeted_plate_text(strat)
                pil_im, _ = renderer.render_type1a(plate_num)
                arr_96 = cv2.resize(np.array(pil_im), (160, 96), interpolation=cv2.INTER_LINEAR)
                top_l, bot_l = rectifier.split_type1a(arr_96)
                crop = rectifier.stitch_type1a_horizontal(top_l, bot_l, target_size=(160, 36))

            # Convert RGB to BGR for OpenCV
            crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)

            # Apply hard augmentations
            # 1. Deep shadows (85% probability)
            if random.random() < 0.85:
                crop_bgr = apply_deep_shadows(crop_bgr)

            # 2. Heavy blur / downsampling (100% for blurry_regions, 70% otherwise)
            if strat == "blurry_regions" or random.random() < 0.70:
                crop_bgr = apply_heavy_blur_and_downsample(crop_bgr)

            # 3. Mud splatter (40% probability)
            if random.random() < 0.40:
                crop_bgr = apply_mud_splatter(crop_bgr)

            # 4. Specular glare (30% probability)
            if random.random() < 0.30:
                crop_bgr = apply_specular_glare(crop_bgr)

            # Save crop
            fn = f"synth_hard_{global_idx:04d}_{plate_num}.png"
            out_path = OUTPUT_DIR / fn
            cv2.imwrite(str(out_path), crop_bgr)

            record = [fn, plate_num, p_type, "1.0", "synth_hard_targeted"]
            manifest_records[fn] = record
            generated_records.append(record)

    # Save manifest with semicolon delimiter
    with open(MANIFEST_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["crop_file", "plate_num", "plate_type", "confidence", "source"])
        for r in manifest_records.values():
            writer.writerow(r)

    print("\n" + "=" * 70)
    print(f"[SUCCESS] Generated {len(generated_records)} targeted hard synthetic crops.")
    print(f"          Total manifest records: {len(manifest_records)}")
    print(f"          Output directory:       {OUTPUT_DIR}")
    print("=" * 70)


def generate_trailer_dataset(count: int = 500, seed: int = 123):
    random.seed(seed)
    np.random.seed(seed)

    print("=" * 70)
    print(f"  OmniPlate-RU: Generating {count} Targeted Hard Trailer Crops (Type 2 GOST)")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    renderer = PlateRenderer()

    manifest_records = {}
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
            for line in lines[1:]:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(";") if ";" in line else line.split(",")
                if len(parts) >= 3:
                    manifest_records[parts[0].strip()] = [p.strip() for p in parts]

    print(f"[*] Loaded {len(manifest_records)} pre-existing manifest entries.")

    strategies = ["m_t", "b_8", "o_0", "hybrid_hard", "random"]
    strategy_weights = [0.25, 0.25, 0.25, 0.15, 0.10]
    generated_records = []

    for idx in range(1, count + 1):
        strat = random.choices(strategies, weights=strategy_weights)[0]
        plate_num, _, _, _ = generate_targeted_trailer_plate_text(strat)

        pil_im, _ = renderer.render_type2(plate_num)
        crop = cv2.resize(np.array(pil_im), (160, 36), interpolation=cv2.INTER_LINEAR)
        crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)

        # Apply hard photorealistic augmentations
        if random.random() < 0.85:
            crop_bgr = apply_deep_shadows(crop_bgr)
        if random.random() < 0.65:
            crop_bgr = apply_heavy_blur_and_downsample(crop_bgr)
        if random.random() < 0.40:
            crop_bgr = apply_mud_splatter(crop_bgr)
        if random.random() < 0.30:
            crop_bgr = apply_specular_glare(crop_bgr)

        fn = f"synth_hard_trailer_{idx:04d}_{plate_num}.png"
        out_path = OUTPUT_DIR / fn
        cv2.imwrite(str(out_path), crop_bgr)

        record = [fn, plate_num, "type2", "1.0", "synth_hard_trailer"]
        manifest_records[fn] = record
        generated_records.append(record)

    with open(MANIFEST_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["crop_file", "plate_num", "plate_type", "confidence", "source"])
        for r in manifest_records.values():
            writer.writerow(r)

    print("\n" + "=" * 70)
    print(f"[SUCCESS] Generated {len(generated_records)} Type 2 trailer crops.")
    print(f"          Total manifest records: {len(manifest_records)}")
    print(f"          Output directory:       {OUTPUT_DIR}")
    print("=" * 70)


def generate_confusion_dataset(count: int = 500, seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

    print("=" * 70)
    print(f"  OmniPlate-RU: Generating {count} Targeted Confusion Crops (6<->8, E<->P, Edge 3-Digit Regions)")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    renderer = PlateRenderer()
    rectifier = PlateRectifier()

    manifest_records = {}
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
            for line in lines[1:]:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(";") if ";" in line else line.split(",")
                if len(parts) >= 3:
                    manifest_records[parts[0].strip()] = [p.strip() for p in parts]

    generated_records = []
    confusion_regions = ["199", "177", "197", "790", "777", "125", "750", "116", "716", "198", "154", "174", "196"]

    for idx in range(1, count + 1):
        p_type = random.choice(["type1", "type1a", "type1b", "type2"])
        strat = random.choice(["six_eight", "e_p", "both"])
        reg = random.choice(confusion_regions)

        if p_type == "type2":
            l2 = random.choice(["EP", "PE", "EE", "PP", "BE", "PB", "EK", "PK"]) if strat in ("e_p", "both") else "".join(random.choices(ALLOWED_LETTERS, k=2))
            d4 = random.choice(["6868", "8686", "6688", "8866", "0860", "6086", "8068", "6808"]) if strat in ("six_eight", "both") else "".join(random.choices(DIGITS, k=4))
            plate_num = f"{l2}{d4}{reg}"
            pil_im, _ = renderer.render_type2(plate_num)
            crop = cv2.resize(np.array(pil_im), (160, 36), interpolation=cv2.INTER_LINEAR)
            crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
        elif p_type == "type1a":
            l1 = random.choice(["E", "P"]) if strat in ("e_p", "both") else random.choice(ALLOWED_LETTERS)
            d3 = random.choice(["686", "868", "668", "886", "688", "866", "608", "806"]) if strat in ("six_eight", "both") else "".join(random.choices(DIGITS, k=3))
            l2 = random.choice(["EP", "PE", "EE", "PP", "EB", "PB", "EK", "PK"]) if strat in ("e_p", "both") else "".join(random.choices(ALLOWED_LETTERS, k=2))
            plate_num = f"{l1}{d3}{l2}{reg}"
            pil_im, _ = renderer.render_type1a(plate_num)
            im_cv = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
            im_rect = cv2.resize(im_cv, (160, 96), interpolation=cv2.INTER_LINEAR)
            top, bot = rectifier.split_type1a(im_rect)
            crop_bgr = rectifier.stitch_type1a_horizontal(top, bot, target_size=(160, 36))
        elif p_type == "type1b":
            l2 = random.choice(["EP", "PE", "EE", "PP", "EB", "PB", "EK", "PK"]) if strat in ("e_p", "both") else "".join(random.choices(ALLOWED_LETTERS, k=2))
            d3 = random.choice(["686", "868", "668", "886", "688", "866", "608", "806"]) if strat in ("six_eight", "both") else "".join(random.choices(DIGITS, k=3))
            plate_num = f"{l2}{d3}{reg}"
            pil_im, _ = renderer.render_type1b(plate_num)
            crop = cv2.resize(np.array(pil_im), (160, 36), interpolation=cv2.INTER_LINEAR)
            crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
        else: # type1
            l1 = random.choice(["E", "P"]) if strat in ("e_p", "both") else random.choice(ALLOWED_LETTERS)
            d3 = random.choice(["686", "868", "668", "886", "688", "866", "608", "806"]) if strat in ("six_eight", "both") else "".join(random.choices(DIGITS, k=3))
            l2 = random.choice(["EP", "PE", "EE", "PP", "EB", "PB", "EK", "PK"]) if strat in ("e_p", "both") else "".join(random.choices(ALLOWED_LETTERS, k=2))
            plate_num = f"{l1}{d3}{l2}{reg}"
            pil_im, _ = renderer.render_type1(plate_num)
            crop = cv2.resize(np.array(pil_im), (160, 36), interpolation=cv2.INTER_LINEAR)
            crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)

        # Apply hard photorealistic augmentations
        if random.random() < 0.80:
            crop_bgr = apply_deep_shadows(crop_bgr)
        if random.random() < 0.60:
            crop_bgr = apply_heavy_blur_and_downsample(crop_bgr)
        if random.random() < 0.35:
            crop_bgr = apply_mud_splatter(crop_bgr)
        if random.random() < 0.25:
            crop_bgr = apply_specular_glare(crop_bgr)

        fn = f"synth_hard_conf_{idx:04d}_{plate_num}.png"
        out_path = OUTPUT_DIR / fn
        cv2.imwrite(str(out_path), crop_bgr)

        record = [fn, plate_num, p_type, "1.0", "synth_hard_confusion"]
        manifest_records[fn] = record
        generated_records.append(record)

    with open(MANIFEST_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["crop_file", "plate_num", "plate_type", "confidence", "source"])
        for r in manifest_records.values():
            writer.writerow(r)

    print(f"[SUCCESS] Generated {len(generated_records)} targeted confusion crops.")
    print(f"          Total manifest records: {len(manifest_records)}")


def generate_edge_artifact_and_region_dataset(count: int = 400, seed: int = 42):
    """
    Generates 400 targeted synthetic crops with:
    1. Left edge frame borders, mounting bolts, and corner clamps.
    2. Optical confusion pairs: K <-> H, 5 <-> 0, 6 <-> 5.
    3. Tricky 3-digit regions: 102, 152, 195, 196 (and related 799, 777).
    """
    random.seed(seed)
    np.random.seed(seed)

    print("=" * 70)
    print(f"  OmniPlate-RU: Generating {count} Edge-Artifact & Target Region Crops (102/152/195/196, K/H)")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    renderer = PlateRenderer()
    rectifier = PlateRectifier()

    manifest_records = {}
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
            for line in lines[1:]:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(";") if ";" in line else line.split(",")
                if len(parts) >= 3:
                    manifest_records[parts[0].strip()] = [p.strip() for p in parts]

    generated_records = []
    target_regions = ["102", "152", "195", "196"]
    hard_focus_plates = [
        "M235PX77", "M235PX777", "H074HP799", "H074KP799",
        "B045KE102", "B045KE152", "H937AP195", "H937AP196",
        "K074HP799", "H074HH799", "B045KK102", "M235PP777",
    ]

    for idx in range(1, count + 1):
        p_type = random.choices(["type1", "type1a", "type1b", "type2"], weights=[0.60, 0.20, 0.10, 0.10])[0]
        strat = random.choices(["k_h", "five_zero", "six_five", "real_focus"], weights=[0.30, 0.30, 0.25, 0.15])[0]
        reg = random.choice(target_regions)

        if strat == "real_focus":
            plate_num = random.choice(hard_focus_plates)
            p_type = "type1"
        elif strat == "k_h":
            l1 = random.choice(["K", "H"])
            d3 = "".join(random.choices(DIGITS, k=3))
            l2 = random.choice(["KP", "HP", "KH", "HK", "KK", "HH", "PK", "PH"])
            reg = random.choice(["799", "777", "102", "152", "195", "196"])
            plate_num = f"{l1}{d3}{l2}{reg}"
        elif strat == "five_zero":
            l1 = random.choice(["B", "O", "C", "M", "E", "A"])
            d3 = random.choice(["050", "505", "005", "550", "045", "540", "052", "502", "000", "555"])
            l2 = random.choice(["KE", "EK", "OE", "BO", "TO", "MO"])
            reg = random.choice(["102", "152"])
            plate_num = f"{l1}{d3}{l2}{reg}"
        elif strat == "six_five":
            l1 = random.choice(["H", "A", "P", "E", "X", "T"])
            d3 = random.choice(["656", "565", "665", "556", "937", "957", "967", "655", "566"])
            l2 = random.choice(["AP", "PA", "EP", "PE", "AH", "HA"])
            reg = random.choice(["195", "196"])
            plate_num = f"{l1}{d3}{l2}{reg}"
        else:
            l1 = random.choice(ALLOWED_LETTERS)
            d3 = "".join(random.choices(DIGITS, k=3))
            l2 = "".join(random.choices(ALLOWED_LETTERS, k=2))
            plate_num = f"{l1}{d3}{l2}{reg}"

        # Render corresponding plate geometry
        if p_type == "type2":
            l2_t2 = plate_num[:2] if len(plate_num) >= 2 and plate_num[:2].isalpha() else "AE"
            d4_t2 = "".join(random.choices(DIGITS, k=4))
            plate_num = f"{l2_t2}{d4_t2}{reg}"
            pil_im, _ = renderer.render_type2(plate_num)
            crop = cv2.resize(np.array(pil_im), (160, 36), interpolation=cv2.INTER_LINEAR)
            crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
        elif p_type == "type1a":
            pil_im, _ = renderer.render_type1a(plate_num)
            im_cv = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
            im_rect = cv2.resize(im_cv, (160, 96), interpolation=cv2.INTER_LINEAR)
            top, bot = rectifier.split_type1a(im_rect)
            crop_bgr = rectifier.stitch_type1a_horizontal(top, bot, target_size=(160, 36))
        elif p_type == "type1b":
            # Yellow taxi / bus (strict GOST: LL DDD RR)
            l2_b = "".join(random.choices(ALLOWED_LETTERS, k=2))
            d3_b = "".join(random.choices(DIGITS, k=3))
            plate_num = f"{l2_b}{d3_b}{reg}"
            pil_im, _ = renderer.render_type1b(plate_num)
            crop = cv2.resize(np.array(pil_im), (160, 36), interpolation=cv2.INTER_LINEAR)
            crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
        else: # type1
            pil_im, _ = renderer.render_type1(plate_num)
            crop = cv2.resize(np.array(pil_im), (160, 36), interpolation=cv2.INTER_LINEAR)
            crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)

        # 100% of these crops receive realistic left edge frame borders / mounting bolts
        crop_bgr = apply_left_edge_artifact(crop_bgr)

        # Additional photorealistic augmentations
        if random.random() < 0.80:
            crop_bgr = apply_deep_shadows(crop_bgr)
        if random.random() < 0.60:
            crop_bgr = apply_heavy_blur_and_downsample(crop_bgr)
        if random.random() < 0.35:
            crop_bgr = apply_mud_splatter(crop_bgr)
        if random.random() < 0.25:
            crop_bgr = apply_specular_glare(crop_bgr)

        fn = f"synth_edge_{idx:04d}_{plate_num}.png"
        out_path = OUTPUT_DIR / fn
        cv2.imwrite(str(out_path), crop_bgr)

        record = [fn, plate_num, p_type, "1.0", "synth_edge_artifacts"]
        manifest_records[fn] = record
        generated_records.append(record)

    with open(MANIFEST_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["crop_file", "plate_num", "plate_type", "confidence", "source"])
        for r in manifest_records.values():
            writer.writerow(r)

    print(f"[SUCCESS] Generated {len(generated_records)} edge-artifact & targeted region crops.")
    print(f"          Total manifest records: {len(manifest_records)}")


def apply_extreme_darkness_and_dirt(img: np.ndarray) -> np.ndarray:
    """
    Applies extreme low contrast, severe darkness (brightness 0.15 - 0.30),
    heavy dirt/road grime, and low-light sensor noise.
    """
    h, w = img.shape[:2]
    res = img.astype(np.float32)

    # 1. Extreme low brightness (0.15 - 0.30)
    dark_factor = random.uniform(0.15, 0.30)
    res = res * dark_factor

    # 2. Low contrast (0.45 - 0.70)
    contrast = random.uniform(0.45, 0.70)
    res = (res - 64.0 * dark_factor) * contrast + (64.0 * dark_factor)

    # 3. Low-light sensor noise / grain
    noise_sigma = random.uniform(4.0, 10.0)
    noise = np.random.normal(0, noise_sigma, (h, w, 3)).astype(np.float32)
    res = res + noise

    return np.clip(res, 0, 255).astype(np.uint8)


def generate_extreme_dark_crops(count: int = 300, seed: int = 42):
    """
    Generates 300 procedural crops with extremely low contrast, heavy dirt,
    and darkness (brightness 0.15-0.30) for extracting characters at visibility limits.
    """
    random.seed(seed)
    np.random.seed(seed)

    print("=" * 70)
    print(f"  OmniPlate-RU: Generating {count} Extreme Dark & Heavy Dirt Crops (Brightness 0.15-0.30)")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    renderer = PlateRenderer()
    rectifier = PlateRectifier()

    manifest_records = {}
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
            for line in lines[1:]:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(";") if ";" in line else line.split(",")
                if len(parts) >= 3:
                    manifest_records[parts[0].strip()] = [p.strip() for p in parts]

    generated_records = []
    # Include targeted plates resembling dark road cases
    dark_target_plates = [
        ("O634YH790", "type1"),
        ("O634YH190", "type1"),
        ("M235PX777", "type1"),
        ("H074HP799", "type1"),
        ("B045KE102", "type1"),
        ("H937AP195", "type1"),
        ("E686XH199", "type1"),
        ("A447AA77", "type1"),
        ("B447OP790", "type1"),
        ("M447CO777", "type1"),
    ]

    for idx in range(1, count + 1):
        if idx <= len(dark_target_plates):
            plate_num, p_type = dark_target_plates[idx - 1]
        else:
            p_type = random.choices(["type1", "type1a", "type1b", "type2"], weights=[0.65, 0.20, 0.10, 0.05])[0]
            reg = random.choice(POPULAR_REGIONS)
            if p_type == "type2":
                l2 = "".join(random.choices(ALLOWED_LETTERS, k=2))
                d4 = "".join(random.choices(DIGITS, k=4))
                plate_num = f"{l2}{d4}{reg}"
            elif p_type == "type1b":
                l2 = "".join(random.choices(ALLOWED_LETTERS, k=2))
                d3 = "".join(random.choices(DIGITS, k=3))
                plate_num = f"{l2}{d3}{reg}"
            else:
                l1 = random.choice(ALLOWED_LETTERS)
                d3 = "".join(random.choices(DIGITS, k=3))
                l2 = "".join(random.choices(ALLOWED_LETTERS, k=2))
                plate_num = f"{l1}{d3}{l2}{reg}"

        # Render corresponding plate geometry
        if p_type == "type2":
            pil_im, _ = renderer.render_type2(plate_num)
            crop = cv2.resize(np.array(pil_im), (160, 36), interpolation=cv2.INTER_LINEAR)
            crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
        elif p_type == "type1a":
            pil_im, _ = renderer.render_type1a(plate_num)
            im_cv = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
            im_rect = cv2.resize(im_cv, (160, 96), interpolation=cv2.INTER_LINEAR)
            top, bot = rectifier.split_type1a(im_rect)
            crop_bgr = rectifier.stitch_type1a_horizontal(top, bot, target_size=(160, 36))
        elif p_type == "type1b":
            pil_im, _ = renderer.render_type1b(plate_num)
            crop = cv2.resize(np.array(pil_im), (160, 36), interpolation=cv2.INTER_LINEAR)
            crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
        else:
            pil_im, _ = renderer.render_type1(plate_num)
            crop = cv2.resize(np.array(pil_im), (160, 36), interpolation=cv2.INTER_LINEAR)
            crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)

        # 1. Apply road mud splatter
        if random.random() < 0.70:
            crop_bgr = apply_mud_splatter(crop_bgr)

        # 2. Apply motion or sensor blur
        if random.random() < 0.60:
            crop_bgr = apply_heavy_blur_and_downsample(crop_bgr)

        # 3. Apply extreme darkness, low contrast, and sensor noise (brightness 0.15 - 0.30)
        crop_bgr = apply_extreme_darkness_and_dirt(crop_bgr)

        fn = f"synth_dark_{idx:04d}_{plate_num}.png"
        out_path = OUTPUT_DIR / fn
        cv2.imwrite(str(out_path), crop_bgr)

        record = [fn, plate_num, p_type, "1.0", "synth_extreme_dark"]
        manifest_records[fn] = record
        generated_records.append(record)

    with open(MANIFEST_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["crop_file", "plate_num", "plate_type", "confidence", "source"])
        for r in manifest_records.values():
            writer.writerow(r)

    print(f"[SUCCESS] Generated {len(generated_records)} extreme dark & heavy dirt crops.")
    print(f"          Total manifest records: {len(manifest_records)}")


def generate_type1b_dataset(count: int = 500, seed: int = 42, purge_invalid: bool = True):
    """
    Regenerates a clean, balanced pool of Type 1B synthetic crops strictly adhering
    to GOST R 50577-2018 format: LL DDD RR / RRR (2 letters + 3 digits + region).
    Optionally purges legacy synthetic crops that were mistakenly generated with Type 1 format.
    """
    random.seed(seed)
    np.random.seed(seed)

    print("=" * 70)
    print(f"  OmniPlate-RU: Regenerating {count} Hard Type 1B Crops (GOST LL DDD RR)")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    renderer = PlateRenderer()

    manifest_records = {}
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
            for line in lines[1:]:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(";") if ";" in line else line.split(",")
                if len(parts) >= 3:
                    manifest_records[parts[0].strip()] = [p.strip() for p in parts]

    # 1. Purge invalid synthetic Type 1B if requested
    purged_count = 0
    if purge_invalid:
        bus_re = re.compile(r"^[ABEKMHOPCTYX]{2}\d{3}\d{2,3}$")
        to_delete = []
        for fn, row in manifest_records.items():
            p_type = row[2] if len(row) > 2 else ""
            p_num = row[1] if len(row) > 1 else ""
            src = row[4] if len(row) > 4 else ""
            if p_type == "type1b" and not bus_re.match(p_num) and (src.startswith("synth_") or "hard" in src or "dark" in src):
                to_delete.append(fn)

        for fn in to_delete:
            del manifest_records[fn]
            file_path = OUTPUT_DIR / fn
            if file_path.exists():
                try:
                    file_path.unlink()
                except Exception:
                    pass
            purged_count += 1
        print(f"[*] Purged {purged_count} legacy non-GOST synthetic Type 1B records.")

    # 2. Generate new clean Type 1B crops
    strategies = ["m_t", "b_8", "o_0", "hybrid_hard", "random"]
    strategy_weights = [0.25, 0.25, 0.25, 0.15, 0.10]
    generated_records = []

    for idx in range(1, count + 1):
        strat = random.choices(strategies, weights=strategy_weights)[0]
        plate_num, _, _, _ = generate_targeted_bus_plate_text(strat)
        pil_im, _ = renderer.render_type1b(plate_num)
        crop = cv2.resize(np.array(pil_im), (160, 36), interpolation=cv2.INTER_LINEAR)
        crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)

        # Apply photorealistic augmentations
        if random.random() < 0.85:
            crop_bgr = apply_deep_shadows(crop_bgr)
        if random.random() < 0.65:
            crop_bgr = apply_heavy_blur_and_downsample(crop_bgr)
        if random.random() < 0.40:
            crop_bgr = apply_mud_splatter(crop_bgr)
        if random.random() < 0.30:
            crop_bgr = apply_specular_glare(crop_bgr)

        fn = f"synth_bus_{idx:04d}_{plate_num}.png"
        out_path = OUTPUT_DIR / fn
        cv2.imwrite(str(out_path), crop_bgr)

        record = [fn, plate_num, "type1b", "1.0", "synth_type1b_gost"]
        manifest_records[fn] = record
        generated_records.append(record)

    with open(MANIFEST_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["crop_file", "plate_num", "plate_type", "confidence", "source"])
        for r in manifest_records.values():
            writer.writerow(r)

    print(f"[SUCCESS] Generated {len(generated_records)} pure GOST Type 1B crops.")
    print(f"          Total manifest records: {len(manifest_records)}")


def generate_targeted_road_type1_dataset(count: int = 1500, seed: int = 42):
    """
    Generates targeted road hard synthetic crops for Type 1 to solve
    dominant real-world errors:
    1. Digits 7/1/2/5/9/0 confusion in digits and region codes.
    2. Letters B/O/H/M/A optical confusions.
    3. Heavy road dirt, mud splatter, motion blur, lighting contrast.
    """
    random.seed(seed)
    np.random.seed(seed)

    print("=" * 70)
    print(f"  OmniPlate-RU: Generating {count} Targeted Road Type 1 Hard Crops")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    renderer = PlateRenderer()

    manifest_records = {}
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
            for line in lines[1:]:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(";") if ";" in line else line.split(",")
                if len(parts) >= 3:
                    manifest_records[parts[0].strip()] = [p.strip() for p in parts]

    generated_records = []
    reg_7_list = ["777", "799", "797", "716", "174", "178", "123", "116", "750", "197", "199", "125", "138", "154", "102"]

    for idx in range(1, count + 1):
        strat = random.choices(["digits_7", "letters_bohma", "zero_confusions", "road_hard_mix"], weights=[0.35, 0.30, 0.20, 0.15])[0]

        if strat == "digits_7":
            # Intensive interweaving of 7, 1, 2, 5, 9, 0
            l1 = random.choice(ALLOWED_LETTERS)
            d3 = random.choice([
                "712", "271", "127", "727", "272", "171", "717", "575", "757", "979", "797",
                "707", "070", "771", "711", "177", "722", "227", "755", "557", "799", "997",
                "071", "170", "270", "720", "571", "175", "971", "179", "705", "507", "735"
            ])
            l2 = "".join(random.choices(ALLOWED_LETTERS, k=2))
            reg = random.choice(reg_7_list)
        elif strat == "letters_bohma":
            # Intensive interweaving of B, O, H, M, A
            l1 = random.choice(["B", "O", "H", "M", "A", "C", "E", "P"])
            d3 = "".join(random.choices(DIGITS, k=3))
            l2 = random.choice([
                "BM", "MB", "BO", "OB", "HM", "MH", "AM", "MA", "AO", "OA",
                "HA", "AH", "HO", "OH", "BA", "AB", "BK", "OK", "BP", "OP",
                "PX", "XP", "YX", "XY", "BC", "CB", "MC", "CM", "MO", "OM"
            ])
            reg = random.choice(POPULAR_REGIONS)
        elif strat == "zero_confusions":
            # 1 vs 0, 5 vs 0, 7 vs 0, 8 vs 0
            l1 = random.choice(["O", "C", "B", "A", "H", "M"])
            d3 = random.choice(["010", "101", "050", "505", "070", "707", "080", "808", "001", "100", "005", "500", "007", "700"])
            l2 = random.choice(["OO", "OB", "BO", "OM", "MO", "OH", "HO", "OA", "AO"])
            reg = random.choice(["102", "150", "750", "790", "154", "777", "799"])
        else:
            # Complex combination
            l1 = random.choice(["B", "M", "H", "A", "O", "T", "P"])
            d3 = random.choice(["712", "727", "808", "050", "393", "272", "979", "575", "171", "656"])
            l2 = random.choice(["BM", "MH", "AO", "BK", "PX", "MA", "OH", "OB", "TO"])
            reg = random.choice(reg_7_list)

        plate_num = f"{l1}{d3}{l2}{reg}"
        pil_im, _ = renderer.render_type1(plate_num)
        crop = cv2.resize(np.array(pil_im), (160, 36), interpolation=cv2.INTER_LINEAR)
        crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)

        # Photorealistic augmentations tailored to road conditions
        # 1. Edge mounting artifacts (screws / frame cutouts)
        if random.random() < 0.70:
            crop_bgr = apply_left_edge_artifact(crop_bgr)
        # 2. Road mud splatter & spray
        if random.random() < 0.65:
            crop_bgr = apply_mud_splatter(crop_bgr)
        # 3. Deep shadows & uneven lighting
        if random.random() < 0.75:
            crop_bgr = apply_deep_shadows(crop_bgr)
        # 4. Motion blur & optical downsampling
        if random.random() < 0.60:
            crop_bgr = apply_heavy_blur_and_downsample(crop_bgr)
        # 5. Specular glare & headlight reflections
        if random.random() < 0.35:
            crop_bgr = apply_specular_glare(crop_bgr)
        # 6. Random low contrast / darkness
        if random.random() < 0.30:
            crop_bgr = apply_extreme_darkness_and_dirt(crop_bgr)

        fn = f"synth_road_t1_{idx:04d}_{plate_num}.png"
        out_path = OUTPUT_DIR / fn
        cv2.imwrite(str(out_path), crop_bgr)

        record = [fn, plate_num, "type1", "1.0", "synth_road_t1"]
        manifest_records[fn] = record
        generated_records.append(record)

    with open(MANIFEST_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["crop_file", "plate_num", "plate_type", "confidence", "source"])
        for r in manifest_records.values():
            writer.writerow(r)

    print(f"[SUCCESS] Generated {len(generated_records)} targeted road Type 1 crops.")
    print(f"          Total manifest records: {len(manifest_records)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate hard synthetic crops")
    parser.add_argument("--count", type=int, default=0, help="Total number of hard standard crops (0 to skip)")
    parser.add_argument("--trailers", type=int, default=0, help="Total number of hard trailer crops (Type 2)")
    parser.add_argument("--confusion", type=int, default=0, help="Total number of hard confusion crops (6<->8, E<->P)")
    parser.add_argument("--edge_artifacts", type=int, default=0, help="Total number of edge-artifact crops (102/152/195/196, K/H)")
    parser.add_argument("--extreme_dark", type=int, default=0, help="Total number of extreme dark & heavy dirt crops (0.15-0.30 brightness)")
    parser.add_argument("--type1b", type=int, default=0, help="Total number of pure GOST Type 1B bus crops (LL DDD RR)")
    parser.add_argument("--purge_invalid_bus", action="store_true", help="Purge legacy non-GOST synthetic Type 1B from manifest")
    parser.add_argument("--road_type1", type=int, default=0, help="Total number of targeted road Type 1 hard crops")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if args.count > 0:
        generate_hard_dataset(total_count=args.count, seed=args.seed)
    if args.trailers > 0:
        generate_trailer_dataset(count=args.trailers, seed=args.seed)
    if args.confusion > 0:
        generate_confusion_dataset(count=args.confusion, seed=args.seed)
    if args.edge_artifacts > 0:
        generate_edge_artifact_and_region_dataset(count=args.edge_artifacts, seed=args.seed)
    if args.extreme_dark > 0:
        generate_extreme_dark_crops(count=args.extreme_dark, seed=args.seed)
    if args.type1b > 0 or args.purge_invalid_bus:
        generate_type1b_dataset(count=max(args.type1b, 0), seed=args.seed, purge_invalid=args.purge_invalid_bus)
    if args.road_type1 > 0:
        generate_targeted_road_type1_dataset(count=args.road_type1, seed=args.seed)