@echo off
title Update PyTorch for RTX 5080 (Blackwell sm_120)
cd /d D:\AIProjects\VolgaIT
echo ========================================================
echo Installing PyTorch with Blackwell (RTX 5080 / CUDA 12.8)...
echo ========================================================
.venv\Scripts\python.exe -m pip install --upgrade --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
echo.
echo Re-installing ultralytics dependencies...
.venv\Scripts\python.exe -m pip install --upgrade ultralytics
echo.
echo Testing GPU after installation:
.venv\Scripts\python.exe scripts\test_gpu.py
pause
