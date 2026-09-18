@echo off
title Test GPU CUDA
cd /d D:\AIProjects\VolgaIT
echo Testing default:
.venv\Scripts\python.exe scripts\test_gpu.py
echo.
echo Testing with set CUDA_FORCE_PTX_JIT=1:
set CUDA_FORCE_PTX_JIT=1
.venv\Scripts\python.exe scripts\test_gpu.py
pause
