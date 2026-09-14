@echo off
title VolgaIT - Build Certified Real Plate Dataset (TZ Compliant)
cd /d D:\AIProjects\VolgaIT

echo ========================================================
echo Building Certified Real Dataset (Type 1B, Type 1A, Other)
echo ========================================================
.venv\Scripts\python.exe scripts\collector\build_verified_tz_dataset.py

echo.
echo ========================================================
echo Running Official Dataset Validator...
echo ========================================================
.venv\Scripts\python.exe scripts\validate_dataset.py
pause
