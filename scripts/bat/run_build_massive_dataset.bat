@echo off
title VolgaIT - Build Massive Verified Real Plate Dataset
cd /d D:\AIProjects\VolgaIT

echo ========================================================
echo Checking dependencies (pyarrow)...
echo ========================================================
.venv\Scripts\python.exe -m pip install --quiet pyarrow

echo ========================================================
echo Building Massive Verified Dataset (783 Real Open + Buses)
echo ========================================================
.venv\Scripts\python.exe scripts\collector\build_massive_verified_dataset.py

echo.
echo ========================================================
echo Running Official Dataset Validator...
echo ========================================================
.venv\Scripts\python.exe scripts\validate_dataset.py
pause
