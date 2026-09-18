@echo off
title VolgaIT - Harvest Verified Russian Real Plates
cd /d D:\AIProjects\VolgaIT
echo ========================================================
echo Harvesting Clean Verified Russian Plates (Zero Duplicates, In-flight YOLO)
echo ========================================================
.venv\Scripts\python.exe scripts\collector\harvest_verified_real_plates.py --type all
echo.
echo ========================================================
echo Validating Dataset Schema & Quotas...
echo ========================================================
.venv\Scripts\python.exe scripts\validate_dataset.py
pause
