@echo off
title Image2Splat - Dashboard
cd /d "%~dp0"

REM Self-locate (this BAT lives in the hot-folder root, alongside the daemon BAT).
set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

REM Translate Windows path to WSL (/mnt/c/Users/...)
for /f "usebackq tokens=*" %%i in (`wsl wslpath -u "%ROOT%"`) do set "WSL_ROOT=%%i"

echo ============================================================
echo   IMAGE2SPLAT - Dashboard
echo ============================================================
echo.
echo   Hot-folder: %ROOT%
echo.
echo   Open on this PC:
echo     http://localhost:8080/
echo.
echo   From phone / laptop on the same WiFi:
echo     Pick the 192.x or 10.x address below, then open
echo     http://^<that-ip^>:8080/ on your device.
echo.
ipconfig | findstr /R /C:"IPv4 Address"
echo.
echo   JSON API:  http://localhost:8080/api/state
echo.
echo   This tracker reads the daemon log + dataset folders only.
echo   It does NOT use the GPU - safe to leave running alongside the daemon.
echo.
echo   Press Ctrl+C in this window to stop the dashboard.
echo ============================================================
echo.

REM Auto-open browser after a short delay so the server has time to bind
start /min cmd /c "timeout /t 3 /nobreak >nul && start http://localhost:8080/"

REM Launch the dashboard via WSL - stdlib-only Python, no conda env needed
wsl.exe -e bash -lic "python3 ~/projects/Image2Splat/scripts/dashboard.py --hotfolder '%WSL_ROOT%' --port 8080"

echo.
echo ============================================================
echo   Dashboard stopped. Press any key to close this window.
echo ============================================================
pause >nul
