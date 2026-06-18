@echo off
title Image2Splat - Hot Folder Daemon (resilient)
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
echo   On launch it prompts for the run type:
echo     1-5  single-tier production
echo     6    probe (all 5 tiers; enter a seed COUNT for random,
echo          or a comma-list like 222,74964,91766 to pin seeds)
echo     B    batch (run pre-staged T#_seed#\ folders in inbox)
echo.
echo   The run then SELF-HEALS: auto-restarts on OOM / GPU resets,
echo   resumes finished work via skip-existing, and quarantines any
echo   image that fails 3 times to failed\ so the run keeps going.
echo   If the PC reboots, just re-launch and it offers to resume.
echo.
echo   Close this window or press Ctrl+C to stop.
echo ============================================================
echo.

wsl.exe -e bash -lic "~/projects/Image2Splat/scripts/run_hotfolder_watchdog.sh --ask"

echo.
echo ============================================================
echo   Daemon stopped. Press any key to close this window.
echo ============================================================
pause >nul
