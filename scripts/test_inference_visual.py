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


def main():
    parser = argparse.ArgumentParser(description="OmniPlate Visual Smoke Tester")
    parser.add_argument("--image", type=str, default=None, help="Path to single test image")
    parser.add_argument("--output_dir", type=str, default="test_output", help="Directory to save visual results")
    parser.add_argument("--device", type=str, default="cuda", help="Inference device: 'cuda' or 'cpu'")
    parser.add_argument("--limit", type=int, default=6, help="Max sample images to evaluate")
    args = parser.parse_args()

    print("=" * 65)
    print("  Volga IT 2026 - OmniPlate End-to-End Visual Verification")
    print("=" * 65)

    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Initialize Pipeline
    pipeline = OmniPlatePipeline(device=args.device)
    print(f"[+] Pipeline initialized on {pipeline.device}")
    print(f"    - Detector: {pipeline.detector_path}")
    print(f"    - OCR:      {pipeline.ocr_path} (use_onnx={pipeline.use_onnx})")

    # Warmup
    print("[*] Warming up pipeline...")
    pipeline.warmup(iterations=2)

    # Collect Test Images
    test_files: List[Path] = []
    if args.image:
        test_files.append(Path(args.image))
    else:
        # Pick representative samples from each class
        real_dir = PROJECT_ROOT / "dataset" / "images" / "real"
        synth_dir = PROJECT_ROOT / "dataset" / "images" / "synthetic"

        if real_dir.exists():
            # Try to grab 1a, 1b, other
            for pattern in ["real_type1a_*.jpg", "real_type1b_*.jpg", "real_other_*.jpg"]:
                matches = list(real_dir.glob(pattern))
                if matches:
                    test_files.extend(matches[:2])

        if synth_dir.exists():
            synth_matches = list(synth_dir.glob("synth_*.jpg"))
            if synth_matches:
                test_files.extend(synth_matches[:2])

    test_files = test_files[:args.limit]
    if not test_files:
        print("[-] No test images found.")
        sys.exit(1)

    print(f"\n[*] Evaluating {len(test_files)} sample images...")
    print("-" * 75)
    print(f"{'Image File':<25} | {'Type':<8} | {'Plate Text':<12} | {'Conf':<6} | {'Time (ms)':<9}")
    print("-" * 75)

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

        if not detections:
            print(f"{img_path.name:<25} | {'None':<8} | {'(no plate)':<12} | {'-':<6} | {dur_ms:<9.2f}")
        else:
            for det in detections:
                print(f"{img_path.name:<25} | {det.plate_type:<8} | {det.text or '(empty)':<12} | {det.confidence:<6.2f} | {dur_ms:<9.2f}")
                vis_img = draw_detection(vis_img, det, timing_ms=dur_ms)

        # Save annotated image
        save_path = out_dir / f"vis_{img_path.name}"
        cv2.imwrite(str(save_path), vis_img)

    print("-" * 75)
    print(f"\n[SUCCESS] Visualized images saved to: {out_dir}")


if __name__ == "__main__":
    main()
