@echo off
title Image2Splat - AutoCrop
setlocal

REM This BAT runs autocrop.py on the daemon's inbox folder. It detects the
REM bounding box of subject pixels (anything not near-white, near-black, or
REM transparent), expands by 40px on all sides, and overwrites the originals
REM in place. Originals are auto-backed up to inbox/.original_backups/ for
REM safety — delete that folder once you're confident the crops are correct.

echo ============================================================
echo   Image2Splat - AutoCrop
echo ============================================================
echo.
echo   Crops images in inbox/ to a tight 40px-bordered bounding box.
echo   Originals are auto-backed up to inbox\.original_backups\
echo.
echo   This eliminates wasted negative space, increasing subject
echo   pixel density at Pixal3D's image conditioning stage.
echo.
echo   Run BEFORE the splat daemon for best results.
echo ============================================================
echo.

wsl.exe -e bash -lic "~/projects/Image2Splat/scripts/run_autocrop.sh"

echo.
echo ============================================================
echo   AutoCrop finished. Press any key to close.
echo ============================================================
pause >nul
