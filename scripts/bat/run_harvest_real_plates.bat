@echo off
title VolgaIT - Harvest Real Plates from Wikimedia Commons
cd /d D:\AIProjects\VolgaIT
echo ========================================================
echo Harvesting Real Russian Plates under CC BY-SA...
echo Auto Face Blur (OpenCV YuNet ONNX) + YOLO-Pose Auto-Annotator Active
echo ========================================================
.venv\Scripts\python.exe scripts\collector\fetch_commons_real_plates.py --type all
echo.
echo ========================================================
echo Running Dataset Validator...
echo ========================================================
.venv\Scripts\python.exe scripts\validate_dataset.py
pause
