@echo off
title VolgaIT - Pure Real License Plate Harvester (Zero False Positives)
cd /d D:\AIProjects\VolgaIT

echo ========================================================
echo Harvesting PURE Verified Russian Plates (Type 1B, 1A, Other)
echo In-flight PlateQualityVerifier: Zero Scaffolding/Windows/LED
echo ========================================================
.venv\Scripts\python.exe scripts\collector\harvest_pure_plates.py

echo.
echo ========================================================
echo Validating Dataset Schema & Quotas...
echo ========================================================
.venv\Scripts\python.exe scripts\validate_dataset.py
pause
