@echo off
title VolgaIT - Train YOLO-Pose Detector on RTX 5080 (100 Epochs)
cd /d D:\AIProjects\VolgaIT

echo ========================================================
echo Preparing Combined Dataset (5000 Synthetic + 545 Real)...
echo ========================================================
.venv\Scripts\python.exe scripts\training\prepare_yolo_dataset.py --dataset_dir dataset --val_ratio 0.15

echo.
echo ========================================================
echo Starting YOLO-Pose Training on RTX 5080 (100 Epochs)
echo ========================================================
.venv\Scripts\python.exe -u scripts\training\train_yolo_pose.py --epochs 100 --batch 32 --workers 8 --patience 25 2>&1 | powershell -command "$input | Tee-Object -FilePath train_yolo.log"

echo.
echo ========================================================
echo Detector Training and ONNX Export Finished.
echo ========================================================
pause
