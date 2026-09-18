@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
title VolgaIT - Train LPRNet OCR on RTX 5080
cd /d D:\AIProjects\VolgaIT

echo ========================================================
echo Ensuring ONNX export dependencies (onnxscript)...
echo ========================================================
.venv\Scripts\python.exe -m pip install onnxscript --quiet

echo.
echo ========================================================
echo Starting LPRNet Micro Fine-Tuning (Targeted Hard Synth + Nomeroff + Real, RTX 5080, ~4 min)
echo ========================================================
.venv\Scripts\python.exe -u scripts\train_ocr.py --epochs 8 --batch_size 128 --lr 1e-4 --resume 2>&1 | powershell -NoProfile -Command "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; $OutputEncoding = [System.Text.Encoding]::UTF8; $input | Tee-Object -FilePath train_ocr.log"

echo.
echo ========================================================
echo Training and ONNX Export Finished.
echo ========================================================
pause
