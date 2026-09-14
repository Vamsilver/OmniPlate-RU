@echo off
title Volga IT 2026 - CLI Inference Runner
cd /d D:\AIProjects\VolgaIT
echo ========================================================
echo Running OmniPlate-RU Inference CLI on RTX 5080...
echo ========================================================
if not exist test_output mkdir test_output
.venv\Scripts\python.exe run.py --input dataset\images\real --output results.csv --device cuda --visualize test_output\cli_vis --verbose
echo.
echo Results saved to results.csv
pause
