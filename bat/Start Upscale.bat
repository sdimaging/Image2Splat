@echo off
title Image2Splat - AuraSR 4x Upscale
setlocal enabledelayedexpansion

REM Use this BAT's parent folder as the UPSCALE root
set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

REM Ensure Input + Output subfolders exist
if not exist "%ROOT%\Input"  mkdir "%ROOT%\Input"
if not exist "%ROOT%\Output" mkdir "%ROOT%\Output"

REM Translate Windows path to WSL (/mnt/c/Users/...)
for /f "usebackq tokens=*" %%i in (`wsl wslpath -u "%ROOT%"`) do set "WSL_ROOT=%%i"

echo ============================================================
echo   Image2Splat - AuraSR 4x Upscale
echo ============================================================
echo.
echo   Drop images into:          %ROOT%\Input
echo   Upscaled PNGs land in:     %ROOT%\Output
echo.
echo   Each image gets a 4x generative upscale (~6-10s per image
echo   on RTX 5090). Existing outputs are skipped.
echo.
echo   Close this window or press Ctrl+C to stop.
echo ============================================================
echo.

wsl.exe -e bash -lic "~/projects/Image2Splat/upscale/run_upscale.sh '%WSL_ROOT%/Input' '%WSL_ROOT%/Output'"

echo.
echo ============================================================
echo   Upscale finished. Press any key to close this window.
echo ============================================================
pause >nul
