@echo off
title VolgaIT - Host Worker (RTX 5080)
cd /d D:\AIProjects\VolgaIT
echo ========================================================
echo Starting Volga IT Host Task Runner on RTX 5080...
echo ========================================================
if exist .venv\Scripts\python.exe (
    .venv\Scripts\python.exe scripts\host_runner.py
) else (
    python scripts\host_runner.py
)
pause
