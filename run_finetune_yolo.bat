@echo off
title VolgaIT - Fine-Tune YOLO-Pose Detector on RTX 5080 (50 Epochs, lr=0.002)
cd /d D:\AIProjects\VolgaIT

echo ========================================================
echo Rebuilding Cleaned Combined Dataset (5000 Synth + 545 Real)...
echo ========================================================
.venv\Scripts\python.exe scripts\training\prepare_yolo_dataset.py --dataset_dir dataset --val_ratio 0.15

echo.
echo ========================================================
echo Starting YOLO-Pose Fine-Tuning from Best Checkpoint (RTX 5080)
echo ========================================================
.venv\Scripts\python.exe -u scripts\training\train_yolo_pose.py --epochs 50 --batch 32 --workers 8 --patience 20 --lr0 0.002 --resume_best 2>&1 | powershell -command "$input | Tee-Object -FilePath train_yolo.log"

echo.
echo ========================================================
echo Fine-Tuning and ONNX Export Finished.
echo ========================================================
pause
