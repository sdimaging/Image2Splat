@echo off
title Image2Splat - Hot Folder Daemon
cd /d "%~dp0"

echo ============================================================
echo   IMAGE2SPLAT - Hot Folder Daemon
echo ============================================================
echo.
echo   Drop image files into:        inbox\
echo   Successful sources go to:     completed\
echo   Failed sources + logs go to:  failed\
echo   COLMAP / Nerfstudio datasets: datasets\
echo.
echo   Workflow:
echo     1. Drop images into inbox\
echo     2. (Optional) Run Autocrop.bat first to tight-crop subjects
echo     3. (Optional) Run Upscale.bat in Desktop\UPSCALE\ for AuraSR
echo     4. Daemon prompts for seed / HDRI / sampler tier on launch
echo     5. Pick tier 1-5 (production) or 6 (probe — compare all tiers)
echo.
echo   Close this window or press Ctrl+C to stop the daemon.
echo ============================================================
echo.

wsl.exe -e bash -lic "~/projects/Image2Splat/scripts/run_hotfolder.sh"

echo.
echo ============================================================
echo   Daemon stopped. Press any key to close this window.
echo ============================================================
pause >nul
