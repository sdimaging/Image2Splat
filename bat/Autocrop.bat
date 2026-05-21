@echo off
title Image2Splat - AutoCrop
setlocal

REM Use this BAT's parent folder as the hot-folder root.
REM This BAT MUST live in the hot-folder (alongside Start Splat Daemon.bat).
set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

REM Translate Windows path to WSL (/mnt/c/Users/...)
for /f "usebackq tokens=*" %%i in (`wsl wslpath -u "%ROOT%"`) do set "WSL_ROOT=%%i"

REM Ensure inbox exists
if not exist "%ROOT%\inbox" mkdir "%ROOT%\inbox"

echo ============================================================
echo   Image2Splat - AutoCrop
echo ============================================================
echo.
echo   Hot-folder: %ROOT%
echo   Operating on:  %ROOT%\inbox
echo.
echo   Crops images to a tight 40px-bordered bounding box.
echo   Originals are auto-backed up to inbox\.original_backups\
echo.
echo   This eliminates wasted negative space, increasing subject
echo   pixel density at Pixal3D's image conditioning stage.
echo.
echo   Run BEFORE the splat daemon for best results.
echo ============================================================
echo.

wsl.exe -e bash -lic "~/projects/Image2Splat/scripts/run_autocrop.sh '%WSL_ROOT%/inbox'"

echo.
echo ============================================================
echo   AutoCrop finished. Press any key to close.
echo ============================================================
pause >nul
