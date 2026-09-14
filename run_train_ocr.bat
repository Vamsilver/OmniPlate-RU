@echo off
title VolgaIT - Train LPRNet OCR on RTX 5080
cd /d D:\AIProjects\VolgaIT

echo ========================================================
echo Ensuring ONNX export dependencies (onnxscript)...
echo ========================================================
.venv\Scripts\python.exe -m pip install onnxscript --quiet

echo.
echo ========================================================
echo Starting Deep High-Precision LPRNet Training (150 Epochs, Heavy Augmentations, RTX 5080)
echo ========================================================
.venv\Scripts\python.exe -u scripts\train_ocr.py --epochs 150 --batch_size 64 --workers 0 --resume 2>&1 | powershell -command "$input | Tee-Object -FilePath train_ocr.log"

echo.
echo ========================================================
echo Training and ONNX Export Finished.
echo ========================================================
pause
