@echo off
chcp 65001 > nul
cd /d "D:\AIProjects\VolgaIT"

echo =======================================================
echo   OmniPlate-RU — Запуск локального сервера аудита
echo =======================================================
echo.
echo Запуск сервера на http://localhost:8080 ...
echo Браузер откроется автоматически. Все правки сохраняются на диск сразу!
echo.

.venv\Scripts\python.exe scripts\run_audit_server.py %*

pause
