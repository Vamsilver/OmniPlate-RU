#!/usr/bin/env python3
"""
OmniPlate-RU Visual Smoke Test & Verification Script.

Runs end-to-end inference on selected real and synthetic test images,
draws bounding boxes, 4 corner keypoints, and recognized license plate text,
and saves the annotated images into `test_output/`.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Optional
import cv2
import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.pipeline import OmniPlatePipeline, PlateDetection


COLOR_MAP = {
    "type1": (0, 255, 0),      # Bright Green
    "type1a": (255, 200, 0),   # Cyan / Light Blue
    "type1b": (0, 215, 255),   # Golden Yellow
    "other": (0, 0, 255),      # Red
}


def draw_detection(
    image: np.ndarray,
    det: PlateDetection,
    timing_ms: Optional[float] = None,
) -> np.ndarray:
    """Renders bbox, quad polygon, and text badge onto the scene image."""
    vis = image.copy()
    color = COLOR_MAP.get(det.plate_type, (0, 255, 0))

    # 1. Draw BBox
    bx, by, bw, bh = det.bbox
    cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), color, 2)

    # 2. Draw 4 Corner Quad
    if det.quad and len(det.quad) == 8:
        pts = np.array(det.quad, dtype=np.int32).reshape(4, 2)
        cv2.polylines(vis, [pts], isClosed=True, color=(255, 255, 255), thickness=2)
        for idx, (px, py) in enumerate(pts):
            cv2.circle(vis, (px, py), 5, (0, 0, 255), -1)
            cv2.circle(vis, (px, py), 2, (255, 255, 255), -1)

    # 3. Text Badge
    label_text = f"{det.plate_type.upper()}: {det.text or 'N/A'} ({det.confidence:.2f})"
    if timing_ms is not None:
        label_text += f" [{timing_ms:.1f}ms]"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

    badge_y1 = max(0, by - th - 10)
    badge_y2 = by
    badge_x1 = bx
    badge_x2 = bx + tw + 10

    cv2.rectangle(vis, (badge_x1, badge_y1), (badge_x2, badge_y2), color, -1)
    cv2.putText(
        vis,
        label_text,
        (badge_x1 + 5, badge_y2 - baseline - 2),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )

    return vis


def load_meta_lookup() -> dict:
    """Loads image relative path -> {plate_num, plate_type} from meta.csv."""
    lookup = {}
    meta_file = PROJECT_ROOT / "dataset" / "meta.csv"
    if not meta_file.exists():
        return lookup
    try:
        import csv
        with open(meta_file, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f, delimiter=";")
            header = next(reader, None)
            for row in reader:
                if len(row) >= 3:
                    img_rel, plate_num, p_type = row[0].replace("\\", "/"), row[1], row[2]
                    base = Path(img_rel).name
                    lookup[base] = {"plate_num": plate_num, "plate_type": p_type}
    except Exception:
        pass
    return lookup


def main():
    parser = argparse.ArgumentParser(description="OmniPlate Visual Smoke Tester")
    parser.add_argument("--image", type=str, default=None, help="Path to single test image")
    parser.add_argument("--output_dir", type=str, default="test_output", help="Directory to save visual results")
    parser.add_argument("--device", type=str, default="cuda", help="Inference device: 'cuda' or 'cpu'")
    parser.add_argument("--conf", type=float, default=0.25, help="Detector confidence threshold (default: 0.25)")
    parser.add_argument("--limit", type=int, default=12, help="Max sample images to evaluate")
    args = parser.parse_args()

    print("=" * 75)
    print("  Volga IT 2026 - OmniPlate End-to-End Visual Verification")
    print("=" * 75)

    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Initialize Pipeline
    pipeline = OmniPlatePipeline(device=args.device, conf_threshold=args.conf)

    print(f"[+] Pipeline initialized on {pipeline.device}")
    print(f"    - Detector: {pipeline.detector_path}")
    print(f"    - OCR:      {pipeline.ocr_path} (use_onnx={pipeline.use_onnx})")

    # Warmup
    print("[*] Warming up pipeline...")
    pipeline.warmup(iterations=2)

    # Collect Balanced Test Images
    test_files: List[Path] = []
    meta_lookup = load_meta_lookup()

    if args.image:
        test_files.append(Path(args.image))
    else:
        synth_dir = PROJECT_ROOT / "dataset" / "images" / "synthetic"
        real_dir = PROJECT_ROOT / "dataset" / "images" / "real"

        # 1. Synthetic samples (Type 1, Type 1A, Type 1B)
        if synth_dir.exists():
            for p_type, max_c in [("type1", 2), ("type1a", 2), ("type1b", 2)]:
                found = 0
                for f in synth_dir.glob("synth_*.jpg"):
                    info = meta_lookup.get(f.name, {})
                    if info.get("plate_type") == p_type:
                        test_files.append(f)
                        found += 1
                        if found >= max_c:
                            break

        # 2. Real samples (Type 1A, Type 1B, Other)
        if real_dir.exists():
            for pattern, max_c in [
                ("real_type1a_*.jpg", 2),
                ("real_type1b_*.jpg", 2),
                ("real_other_*.jpg", 1),
            ]:
                matches = list(real_dir.glob(pattern))
                if matches:
                    test_files.extend(matches[:max_c])

    test_files = test_files[:args.limit]
    if not test_files:
        print("[-] No test images found.")
        sys.exit(1)

    print(f"\n[*] Evaluating {len(test_files)} balanced sample images (conf={args.conf})...")
    print("-" * 90)
    print(f"{'Image File':<24} | {'Type':<8} | {'Pred Text':<12} | {'GT Text':<12} | {'Match':<7} | {'Time (ms)':<9}")
    print("-" * 90)

    for img_path in test_files:
        if not img_path.exists():
            continue

        raw_img = cv2.imread(str(img_path))
        if raw_img is None:
            continue

        t0 = time.perf_counter()
        detections, timings = pipeline.predict_with_timing(raw_img)
        dur_ms = (time.perf_counter() - t0) * 1000.0

        vis_img = raw_img.copy()
        gt_info = meta_lookup.get(img_path.name, {})
        gt_text = gt_info.get("plate_num", "-")
        gt_type = gt_info.get("plate_type", "-")

        if not detections:
            match_status = "PASS (neg)" if gt_type == "other" or gt_text == "-" else "MISS"
            print(f"{img_path.name:<24} | {gt_type:<8} | {'(no plate)':<12} | {gt_text:<12} | {match_status:<7} | {dur_ms:<9.2f}")
        else:
            for det in detections:
                pred_clean = (det.text or "").strip()
                gt_clean = gt_text.strip()
                is_match = "EXACT" if (pred_clean and pred_clean == gt_clean) else ("CLOSE" if (pred_clean and gt_clean and sum(c1 == c2 for c1, c2 in zip(pred_clean, gt_clean)) >= len(gt_clean) - 1) else "DIFF")
                if gt_type == "other":
                    is_match = "N/A"
                print(f"{img_path.name:<24} | {det.plate_type:<8} | {pred_clean or '(empty)':<12} | {gt_text:<12} | {is_match:<7} | {dur_ms:<9.2f}")
                vis_img = draw_detection(vis_img, det, timing_ms=dur_ms)

        # Save annotated image
        save_path = out_dir / f"vis_{img_path.name}"
        cv2.imwrite(str(save_path), vis_img)

    print("-" * 90)
    print(f"\n[SUCCESS] Visualized images saved to: {out_dir}")


if __name__ == "__main__":
    main()
