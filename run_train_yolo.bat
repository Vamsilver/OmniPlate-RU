@echo off
title VolgaIT - Train YOLO-Pose on RTX 5080
cd /d D:\AIProjects\VolgaIT
echo ========================================================
echo Preparing YOLO-Pose dataset manifest...
echo ========================================================
.venv\Scripts\python.exe scripts\training\prepare_yolo_dataset.py --dataset_dir dataset
echo.
echo ========================================================
echo Training YOLO-Pose Detector on NVIDIA RTX 5080 (CUDA)...
echo ========================================================
.venv\Scripts\python.exe scripts\training\train_yolo_pose.py 2>&1 | powershell -command "$input | Tee-Object -FilePath train_error.log"
pause
