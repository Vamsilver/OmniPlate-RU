#!/usr/bin/env python3
"""
YOLO-Pose Training Script for Volga IT 2026.
Trains YOLOv8n-pose / YOLO11n-pose on NVIDIA RTX 5080 with CUDA acceleration.
Outputs best model checkpoint and exports to ONNX for <= 100 ms inference.
"""

import argparse
import os
import shutil
import sys
import time
import torch
sys.path.insert(0, os.path.dirname(__file__))
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="YOLO-Pose Detector Training")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs (default: 100)")
    parser.add_argument("--batch", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640)")
    parser.add_argument("--workers", type=int, default=8, help="Dataloader workers (default: 8)")
    parser.add_argument("--patience", type=int, default=25, help="Early stopping patience (default: 25)")
    parser.add_argument("--device", type=str, default=None, help="Device (default: 0 if cuda else cpu)")
    parser.add_argument("--resume_best", action="store_true", help="Fine-tune from existing detector_yolo_pose_best.pt")
    args = parser.parse_args()

    print("=" * 65)
    print(f"  Volga IT 2026 - YOLO-Pose Plate & Quad Detector Training ({args.epochs} Epochs)")
    print("=" * 65)

    has_cuda = torch.cuda.is_available()
    device = args.device if args.device is not None else ("0" if has_cuda else "cpu")

    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available:  {has_cuda}")
    if has_cuda:
        print(f"GPU Device:      {torch.cuda.get_device_name(0)}")
        print(f"CUDA Capability: {torch.cuda.get_device_capability(0)}")
        print(f"VRAM Total:      {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")

    data_yaml = os.path.abspath("dataset/yolo_pose/data.yaml")
    if not os.path.exists(data_yaml):
        print(f"[!] data.yaml not found at {data_yaml}. Preparing now...")
        from prepare_yolo_dataset import prepare_yolo_pose
        prepare_yolo_pose("dataset")

    initial_weights = "yolov8n-pose.pt"
    if args.resume_best and os.path.exists("models/detector_yolo_pose_best.pt"):
        initial_weights = "models/detector_yolo_pose_best.pt"
        print(f"[*] Resuming from existing weights: {initial_weights}")
    else:
        print(f"[*] Initializing fresh YOLOv8 nano pose backbone: {initial_weights}")

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
        verbose=True,
    )
    total_time = time.time() - start_time
    print(f"\n[+] Training completed in {total_time:.1f}s ({total_time / 60.0:.2f} min)!")

    candidate_paths = [
        getattr(results, "save_dir", "") and os.path.join(str(results.save_dir), "weights", "best.pt"),
        os.path.join("runs", "pose", "models", "yolo_pose_run", "weights", "best.pt"),
        os.path.join("models", "yolo_pose_run", "weights", "best.pt"),
    ]
    best_pt = next((p for p in candidate_paths if p and os.path.exists(p)), None)
    target_pt = os.path.join("models", "detector_yolo_pose_best.pt")

    if best_pt and os.path.exists(best_pt):
        os.makedirs("models", exist_ok=True)
        shutil.copy(best_pt, target_pt)
        print(f"[+] Saved best PyTorch model checkpoint: {target_pt}")

        # Export to ONNX
        print("[*] Exporting model to ONNX for low-latency inference...")
        best_model = YOLO(target_pt)
        onnx_file = best_model.export(format="onnx", imgsz=args.imgsz, dynamic=False, simplify=True)
        target_onnx = os.path.join("models", "detector_yolo_pose.onnx")
        if os.path.exists(onnx_file) and onnx_file != target_onnx:
            shutil.copy(onnx_file, target_onnx)
        print(f"[SUCCESS] ONNX Model ready: {target_onnx}")
    else:
        print(f"[!] Warning: best.pt not found at {best_pt}")


if __name__ == "__main__":
    main()
