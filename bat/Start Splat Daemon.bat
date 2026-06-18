@echo off
title Image2Splat - Hot Folder Daemon (resilient)
cd /d "%~dp0"

REM Self-locate: this BAT lives in the hot-folder root. Derive the WSL path so
REM the daemon, dashboard, and watchdog all operate on THIS folder.
set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"
for /f "usebackq tokens=*" %%i in (`wsl wslpath -u "%ROOT%"`) do set "WSL_ROOT=%%i"

echo ============================================================
echo   IMAGE2SPLAT - Hot Folder Daemon
echo ============================================================
echo.
echo   Hot-folder: %ROOT%
echo     inbox\  processing\  completed\  failed\  datasets\
echo.
echo   On launch it prompts for the run type:
echo     1-5  single-tier production
echo     6    probe (all 5 tiers; enter a seed COUNT for random,
echo          or a comma-list like 222,74964,91766 to pin seeds)
echo     B    batch (run pre-staged T#_seed#\ folders in inbox)
echo.
echo   The run SELF-HEALS: auto-restarts on OOM / GPU resets, resumes
echo   finished work via skip-existing, and quarantines any image that
echo   fails 3 times to failed\ so the run keeps going. If the PC
echo   reboots, just re-launch and it offers to resume.
echo.
echo   Close this window or press Ctrl+C to stop.
echo ============================================================
echo.

wsl.exe -e bash -lic "~/projects/Image2Splat/scripts/run_hotfolder_watchdog.sh --hotfolder '%WSL_ROOT%' --ask"

echo.
echo ============================================================
echo   Daemon stopped. Press any key to close this window.
echo ============================================================
pause >nul
