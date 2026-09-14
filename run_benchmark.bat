@echo off
title Volga IT 2026 - Benchmark Latency and VRAM
cd /d D:\AIProjects\VolgaIT
echo ========================================================
echo Running OmniPlate End-to-End Latency Benchmark on RTX 5080...
echo ========================================================
.venv\Scripts\python.exe scripts\benchmark_latency.py --device cuda --runs 100 --warmup 10
echo.
echo Benchmark completed! See docs\report\latency_benchmark.md for full report.
pause
