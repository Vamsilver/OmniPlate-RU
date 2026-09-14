@echo off
title Volga IT 2026 - Visual Inference Smoke Test
cd /d D:\AIProjects\VolgaIT
echo ========================================================
echo Running Visual Inference Smoke Test on RTX 5080...
echo ========================================================
.venv\Scripts\python.exe scripts\test_inference_visual.py --device cuda --limit 6
echo.
echo Annotated images saved in test_output\
pause
