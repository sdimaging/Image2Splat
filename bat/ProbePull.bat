@echo off
title Image2Splat - Probe Pull
cd /d "%~dp0"

REM Self-locate: this BAT lives in the hot-folder root.
set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"
for /f "usebackq tokens=*" %%i in (`wsl wslpath -u "%ROOT%"`) do set "WSL_ROOT=%%i"

echo ============================================================
echo   IMAGE2SPLAT - Probe Pull
echo ============================================================
echo.
echo   Hot-folder: %ROOT%
echo.
echo   Scans datasets\ for every folder that has a probe\ subdir
echo   (real datasets; grouping folders like Pre/Post are skipped),
echo   reads your kept probe selects, and copies each asset's source
echo   from completed\ into the matching inbox\T#_seed#\ batch folder.
echo.
echo   Multi-cell selects are renamed _1/_2/_3. Re-running is safe.
echo   After this, run the daemon in BATCH mode to render them.
echo ============================================================
echo.

wsl.exe -e bash -lic "python3 ~/projects/Image2Splat/scripts/probepull.py --hotfolder '%WSL_ROOT%'"

echo.
echo ============================================================
echo   Done. Press any key to close this window.
echo ============================================================
pause >nul
