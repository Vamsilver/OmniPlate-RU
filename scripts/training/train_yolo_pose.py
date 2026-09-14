#!/usr/bin/env python3
"""
YOLO-Pose Training Script for Volga IT 2026.
Trains YOLOv8n-pose / YOLO11n-pose on NVIDIA RTX 5080 with CUDA acceleration.
Outputs best model checkpoint and exports to ONNX for <= 100 ms inference.
"""

import os
import shutil
import sys
import time
import torch
sys.path.insert(0, os.path.dirname(__file__))
from ultralytics import YOLO


def main():
    print("=" * 65)
    print("  Volga IT 2026 - YOLO-Pose Plate & Quad Detector Training")
    print("=" * 65)

    has_cuda = torch.cuda.is_available()
    device = "0" if has_cuda else "cpu"

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

    print(f"\n[*] Initializing YOLOv8 nano pose backbone...")
    model = YOLO("yolov8n-pose.pt")

    start_time = time.time()
    print(f"[*] Starting training on {device}...")
    results = model.train(
        data=data_yaml,
        epochs=30,
        imgsz=640,
        batch=32,
        device=device,
        workers=8,
        project="models",
        name="yolo_pose_run",
        exist_ok=True,
        save=True,
        patience=10,
        verbose=True
    )
    total_time = time.time() - start_time
    print(f"\n[+] Training completed in {total_time:.1f}s!")

    best_pt = os.path.join("models", "yolo_pose_run", "weights", "best.pt")
    target_pt = os.path.join("models", "detector_yolo_pose_best.pt")

    if os.path.exists(best_pt):
        os.makedirs("models", exist_ok=True)
        shutil.copy(best_pt, target_pt)
        print(f"[+] Saved best PyTorch model checkpoint: {target_pt}")

        # Export to ONNX
        print("[*] Exporting model to ONNX for low-latency inference...")
        best_model = YOLO(target_pt)
        onnx_file = best_model.export(format="onnx", imgsz=640, dynamic=False, simplify=True)
        target_onnx = os.path.join("models", "detector_yolo_pose.onnx")
        if os.path.exists(onnx_file) and onnx_file != target_onnx:
            shutil.copy(onnx_file, target_onnx)
        print(f"[SUCCESS] ONNX Model ready: {target_onnx}")
    else:
        print(f"[!] Warning: best.pt not found at {best_pt}")


if __name__ == "__main__":
    main()
