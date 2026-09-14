@echo off
title VolgaIT - Generate Synthetic Dataset
cd /d D:\AIProjects\VolgaIT
echo ========================================================
echo Generating 5000 Synthetic License Plates...
echo Multi-core CPU + SIMD fast noise
echo ========================================================
.venv\Scripts\python.exe dataset\generator\generate_synthetic.py --count 5000 --overwrite
echo.
echo Running dataset validator...
.venv\Scripts\python.exe scripts\validate_dataset.py
pause
