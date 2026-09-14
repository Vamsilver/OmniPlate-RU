@echo off
title VolgaIT - Audit Real Dataset Quality
cd /d D:\AIProjects\VolgaIT
echo ========================================================
echo Auditing Real Dataset Quality with YOLO-Pose + HSV/Geom...
echo ========================================================
.venv\Scripts\python.exe scripts\collector\clean_and_audit_real.py
echo.
pause
