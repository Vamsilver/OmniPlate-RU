@echo off
chcp 65001 > nul
echo =====================================================================
echo  OmniPlate-RU: Packaging Solution Archive for Volga IT 2026 Jury
echo =====================================================================
.venv\Scripts\python.exe scripts\package_submission.py %*
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Packaging failed with exit code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)
echo.
echo [SUCCESS] Package ready in submission\
pause
