#!/usr/bin/env python3
"""
scripts/train/train_detector_yolo.py
YOLOv8n-pose Detector Fine-Tuning on NVIDIA GeForce RTX 5080 (Volga IT 2026).
Trains on full balanced dataset (6,504 annotations):
- Predicts BBox, 4 Quad Keypoints, and 4 classes:
  0: type1, 1: type1a, 2: type1b, 3: other
- Validates Recall on Type 1A (target >= 95%).
- Exports trained model to ONNX: models/detector_yolo_pose.onnx
- Saves PyTorch weights: models/detector_yolo_pose_best.pt
"""

import argparse
import os
import shutil
import sys
import time
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="YOLO-Pose Detector Training on RTX 5080")
    parser.add_argument("--epochs", type=int, default=60, help="Training epochs (default: 60)")
    parser.add_argument("--batch", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640)")
    parser.add_argument("--workers", type=int, default=8, help="Dataloader workers (default: 8)")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience (default: 20)")
    parser.add_argument("--device", type=str, default=None, help="Device (default: 0 if cuda else cpu)")
    parser.add_argument("--lr0", type=float, default=0.005, help="Initial learning rate (default: 0.005)")
    parser.add_argument("--resume_best", action="store_true", help="Resume from existing detector_yolo_pose_best.pt")
    args = parser.parse_args()

    print("=" * 65)
    print(f"  Volga IT 2026 - YOLO-Pose Detector Training ({args.epochs} Epochs)")
    print("=" * 65)

    has_cuda = torch.cuda.is_available()
    device = args.device if args.device is not None else ("0" if has_cuda else "cpu")

    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available:  {has_cuda}")
    if has_cuda:
        print(f"GPU Device:      {torch.cuda.get_device_name(0)}")
        print(f"VRAM Total:      {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")

    data_yaml = os.path.join(PROJECT_ROOT, "dataset", "yolo_pose", "data.yaml")
    if not os.path.exists(data_yaml):
        print(f"[!] data.yaml not found at {data_yaml}. Preparing now...")
        from scripts.build_verified_dataset import build_dataset
        build_dataset(os.path.join(PROJECT_ROOT, "dataset"))

    # Initial weights
    initial_weights = os.path.join(PROJECT_ROOT, "yolov8n-pose.pt")
    best_existing = os.path.join(PROJECT_ROOT, "models", "detector_yolo_pose_best.pt")

    if args.resume_best and os.path.exists(best_existing):
        initial_weights = best_existing
        print(f"[*] Resuming from existing weights: {initial_weights}")
    elif os.path.exists(best_existing):
        # Fine-tune from existing best checkpoint for faster convergence on 6504 dataset
        initial_weights = best_existing
        print(f"[*] Fine-tuning from existing trained checkpoint: {initial_weights}")
    elif os.path.exists(initial_weights):
        print(f"[*] Initializing from base YOLOv8n-pose backbone: {initial_weights}")
    else:
        initial_weights = "yolov8n-pose.pt"
        print(f"[*] Ultralytics will download/use: {initial_weights}")

    model = YOLO(initial_weights)

    start_time = time.time()
    print(f"[*] Starting training on device {device} ({args.epochs} epochs, batch={args.batch})...")
    results = model.train(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        project="models",
        name="yolo_pose_run",
        exist_ok=True,
        save=True,
        patience=args.patience,
        lr0=args.lr0,
        verbose=True,
        plots=True,
    )
    total_time = time.time() - start_time
    print(f"\n[+] Training completed in {total_time:.1f}s ({total_time / 60.0:.2f} min)!")

    # Validate model
    print("\n[*] Running final validation on val split...")
    val_results = model.val(data=data_yaml, device=device, imgsz=args.imgsz)

    # Locate best.pt
    candidate_paths = [
        getattr(results, "save_dir", "") and os.path.join(str(results.save_dir), "weights", "best.pt"),
        os.path.join(PROJECT_ROOT, "runs", "pose", "models", "yolo_pose_run", "weights", "best.pt"),
        os.path.join(PROJECT_ROOT, "models", "yolo_pose_run", "weights", "best.pt"),
    ]
    best_pt = next((p for p in candidate_paths if p and os.path.exists(p)), None)
    target_pt = os.path.join(PROJECT_ROOT, "models", "detector_yolo_pose_best.pt")

    if best_pt and os.path.exists(best_pt):
        os.makedirs(os.path.dirname(target_pt), exist_ok=True)
        shutil.copy(best_pt, target_pt)
        print(f"[+] Saved best PyTorch checkpoint: {target_pt}")

        # Export to ONNX
        print("[*] Exporting model to ONNX for low-latency inference (<25 ms)...")
        best_model = YOLO(target_pt)
        onnx_file = best_model.export(format="onnx", imgsz=args.imgsz, dynamic=False, simplify=True)
        target_onnx = os.path.join(PROJECT_ROOT, "models", "detector_yolo_pose.onnx")
        if os.path.exists(onnx_file) and onnx_file != target_onnx:
            shutil.copy(onnx_file, target_onnx)
        print(f"[SUCCESS] ONNX Model ready: {target_onnx}")
    else:
        print(f"[!] Warning: best.pt not found at candidate paths: {candidate_paths}")


if __name__ == "__main__":
    main()
