@echo off
chcp 65001 > nul
echo =====================================================================
echo  OmniPlate-RU: Packaging Dataset Archive for Volga IT 2026 Jury
echo =====================================================================
.venv\Scripts\python.exe scripts\package_dataset.py %*
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Dataset packaging failed with exit code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)
echo.
echo [SUCCESS] Dataset archive ready in submission\
pause
