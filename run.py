#!/usr/bin/env python3
"""
OmniPlate-RU — Main Competition Inference CLI Entrypoint.
Volga IT 2026: Automatic Vehicle License Plate Recognition (Types 1, 1A, 1B).

Usage:
    python run.py --input /path/to/images --output results.csv
    python run.py --input_dir /path/to/images --output_file results.csv --device cuda

Output format:
    CSV file (UTF-8, delimiter=';') with columns:
    image;plate_num;plate_type;confidence
"""

import argparse
import csv
import os
import sys
import time

# Strict offline mode guarantees for Volga IT 2026 competition runner
os.environ["YOLO_AUTOINSTALL"] = "0"
os.environ["ULTRALYTICS_AUTOINSTALL"] = "0"
os.environ["YOLO_OFFLINE"] = "1"
os.environ["YOLO_SYNC"] = "0"

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.pipeline import OmniPlatePipeline, PlateDetection

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def collect_images(input_path: Path) -> List[Path]:
    """Collects all image files from a single path or directory."""
    if input_path.is_file():
        if input_path.suffix.lower() in IMAGE_EXTENSIONS:
            return [input_path]
        return []

    if not input_path.is_dir():
        return []

    images = []
    for item in sorted(input_path.iterdir()):
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(item)
    return images


def annotate_image(
    image: np.ndarray,
    detections: List[PlateDetection],
) -> np.ndarray:
    """Draws visual boxes, quads, and text badges on image."""
    vis = image.copy()
    color_map = {
        "type1": (0, 255, 0),      # Green
        "type1a": (255, 200, 0),   # Cyan
        "type1b": (0, 215, 255),   # Yellow
        "other": (0, 0, 255),      # Red
    }

    for det in detections:
        color = color_map.get(det.plate_type, (0, 255, 0))
        bx, by, bw, bh = det.bbox

        # 1. BBox
        cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), color, 2)

        # 2. Quad polygon
        if det.quad and len(det.quad) == 8:
            pts = np.array(det.quad, dtype=np.int32).reshape(4, 2)
            cv2.polylines(vis, [pts], isClosed=True, color=(255, 255, 255), thickness=2)
            for px, py in pts:
                cv2.circle(vis, (px, py), 4, (0, 0, 255), -1)

        # 3. Label Badge
        text_label = f"{det.plate_type.upper()}: {det.text or 'N/A'} ({det.confidence:.2f})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.55
        thick = 2
        (tw, th), base = cv2.getTextSize(text_label, font, scale, thick)
        y1 = max(0, by - th - 8)
        y2 = by
        x1 = bx
        x2 = bx + tw + 8
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, -1)
        cv2.putText(vis, text_label, (x1 + 4, y2 - base - 2), font, scale, (0, 0, 0), thick, cv2.LINE_AA)

    return vis


def run_inference(
    input_path: Path,
    output_csv: Path,
    device: str = "cuda",
    conf_threshold: float = 0.12,
    iou_threshold: float = 0.45,
    save_vis_dir: Optional[Path] = None,
    verbose: bool = False,
) -> Tuple[int, int, float]:
    """
    Executes end-to-end plate recognition over input image directory
    and writes results to output_csv.
    """
    image_files = collect_images(input_path)
    if not image_files:
        print(f"[!] No valid image files found at: {input_path}")
        # Create empty CSV with required header
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(["image", "plate_num", "plate_type", "confidence"])
        return 0, 0, 0.0

    print("=" * 65)
    print("  OmniPlate-RU — Competition Inference Runner (Volga IT 2026)")
    print(f"  Input:       {input_path} ({len(image_files)} images)")
    print(f"  Output CSV:  {output_csv}")
    print(f"  Device:      {device}")
    print(f"  Threshold:   conf={conf_threshold}, iou={iou_threshold}")
    if save_vis_dir:
        print(f"  Visuals Dir: {save_vis_dir}")
        save_vis_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 65)

    # Initialize End-to-End Pipeline
    pipeline = OmniPlatePipeline(
        device=device,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
    )
    print(f"[+] Active Pipeline on {pipeline.device}")
    print(f"    - Detector: {pipeline.detector_path}")
    print(f"    - OCR:      {pipeline.ocr_path} (use_onnx={pipeline.use_onnx})")

    # Warmup
    print("[*] Warming up pipeline...")
    pipeline.warmup(iterations=2)

    # Process images
    total_detections = 0
    t_start = time.perf_counter()

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["image", "plate_num", "plate_type", "confidence"])

        for idx, img_file in enumerate(image_files, start=1):
            img_bgr = cv2.imread(str(img_file))
            if img_bgr is None:
                if verbose:
                    print(f"[-] [{idx}/{len(image_files)}] Failed to load: {img_file.name}")
                continue

            t0 = time.perf_counter()
            detections = pipeline.predict(img_bgr)
            dur_ms = (time.perf_counter() - t0) * 1000.0

            if len(detections) > 1:
                def det_sort_key(d: PlateDetection):
                    is_target = 1 if d.plate_type in ("type1", "type1a", "type1b") else 0
                    real_chars = len([c for c in (d.text or "") if c != "#"])
                    return (is_target, real_chars > 0, real_chars, d.ocr_confidence, d.confidence)
                detections = sorted(detections, key=det_sort_key, reverse=True)

            for det in detections:
                conf_val = round(det.confidence, 2)
                # Strictly adhere to competition classes: type1, type1a, type1b, other
                out_type = "other" if det.plate_type in ("type2", "other") else det.plate_type
                raw_text = (det.text or "").strip().upper()
                plate_text = raw_text if raw_text else ("" if out_type == "other" else "########")

                writer.writerow([
                    img_file.name,
                    plate_text,
                    out_type,
                    f"{conf_val:.2f}",
                ])
                total_detections += 1

            if verbose or idx % 50 == 0 or idx == len(image_files):
                found_str = f"{len(detections)} plate(s)" if detections else "0 plates"
                print(f"[{idx:>4}/{len(image_files)}] {img_file.name:<30} -> {found_str:<12} ({dur_ms:5.1f} ms)")

            # Optional visualization save
            if save_vis_dir is not None and detections:
                vis_img = annotate_image(img_bgr, detections)
                cv2.imwrite(str(save_vis_dir / f"vis_{img_file.name}"), vis_img)

    total_time = time.perf_counter() - t_start
    avg_fps = len(image_files) / max(0.001, total_time)
    avg_ms = (total_time / max(1, len(image_files))) * 1000.0

    print("=" * 65)
    print(f"[SUCCESS] Inference Complete!")
    print(f"  Processed:       {len(image_files)} images")
    print(f"  Detections:      {total_detections} plates written to {output_csv.name}")
    print(f"  Total Duration:  {total_time:.2f} s")
    print(f"  Average Speed:   {avg_fps:.1f} FPS ({avg_ms:.1f} ms/frame)")
    print("=" * 65)

    return len(image_files), total_detections, total_time


def resolve_device(device_arg: str) -> str:
    """Auto-detects CUDA hardware if 'auto' is specified, otherwise honors explicit choice."""
    arg_lower = device_arg.strip().lower()
    if arg_lower == "auto":
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"
    return arg_lower


def main():
    parser = argparse.ArgumentParser(
        description="OmniPlate-RU: Volga IT 2026 Vehicle License Plate Recognition",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", "--input_dir", "-i",
        dest="input_path",
        type=str,
        required=True,
        help="Path to directory containing images or single image file",
    )
    parser.add_argument(
        "--output", "--output_file", "-o",
        dest="output_file",
        type=str,
        default="results.csv",
        help="Path to output CSV file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Inference device ('auto' detects CUDA if available, 'cuda', or 'cpu')",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.12,
        help="Detector confidence threshold",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="NMS IOU threshold",
    )
    parser.add_argument(
        "--visualize", "--save_vis",
        dest="save_vis",
        type=str,
        default=None,
        help="Optional directory to save annotated preview images",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print per-frame timing and detection details",
    )

    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_csv = Path(args.output_file)
    vis_dir = Path(args.save_vis) if args.save_vis else None
    resolved_device = resolve_device(args.device)

    if not input_path.exists():
        print(f"[-] Error: Input path does not exist: {input_path}")
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(["image", "plate_num", "plate_type", "confidence"])
        sys.exit(1)

    run_inference(
        input_path=input_path,
        output_csv=output_csv,
        device=resolved_device,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        save_vis_dir=vis_dir,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
